"""Batch network benchmarking over one error-stamped dataset CSV, with optional
link-level purified hops.

    python -m qetwork.topologies.runner.run_nb <incsv> <outcsv> \\
        [--purification] [--purification-rounds R] [--protocol seq|par] \\
        [--samples N] [--jobs J]

Reads every row of <incsv>, rebuilds each path as a linear chain (per-node
gates/source/T1-T2 from the pipe-lists, per-edge fiber, and the endpoint
MZI+SNSPD detectors from the d1_*/d2_* scalars), runs the event-driven
NetworkBenchmark (Helsen & Wehner, arXiv:2103.01165), and writes ONE output CSV
row per path:

    DatasetID, PathID, PathString, path_fidelity, avg_time_us

  * path_fidelity = F_path = (1 + f)/2 from the b_m = A*f^m fit (SPAM-robust:
    detector noise lands in A, not f).
  * avg_time_us   = ( sum_m time_m / m ) / n_m, in microseconds, where time_m is
    the virtual clock the whole m-group of sequences consumed (_TimedNB below).

Flags:
  --purification            purify every hop's link (LinkPumpSession: standard
                            repeated DEJMPS with per-pair source-phase calibration)
  --purification-rounds R   pump level, 1..5 consecutive successes (default 1);
                            only valid together with --purification
  --protocol seq|par        distribution protocol handed to the benchmark's
                            `mode`. RESERVED for the hop-by-hop bounce: validated
                            and recorded, no behavioral difference yet (future:
                            seq = pump links on demand, par = pre-pump a sweep's
                            links concurrently).
  --samples N               RB sequences per m (default 40; the detector readout
                            is one click per sequence, so raise it for clean fits)
  --jobs J                  worker processes (default 1; 0 = all cores). Every row
                            is an independent simulation with a deterministic seed
                            (SEED + row index), so the result SET is identical for
                            any J; with J > 1 rows land in completion order and
                            DatasetID/PathID identify them. Submission uses a
                            bounded in-flight window, so huge CSVs are streamed.

Fixed settings (module constants below): m = M_MIN..M_MAX bounces, base SEED,
detector readout USE_DETECTOR, calibration CALIBRATE.
"""

import argparse
import copy
import csv
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait

import numpy as np

from qetwork.kernel.timeline import Timeline
from qetwork.topologies.topology_spec import TopologySpec
from qetwork.topologies.topology_generator import (
    DEFAULT_NODE, DEFAULT_EDGE, DEFAULT_NETWORK, DEFAULT_DETECTOR,
)
from qetwork.protocols.e_dist_swap import SEQUENTIAL, PARALLEL
from qetwork.protocols.net_benchmarking import NetworkBenchmark, fit_decay

M_MIN, M_MAX = 1, 8            # bounce range of the RB fit
SEED = 1                       # base seed; row i runs with SEED + i
USE_DETECTOR = False            # MZI+SNSPD readout (False = exact readout)
CALIBRATE = True               # per-pair source-phase calibration in purified hops
MAX_ROUNDS = 5

# --- CSV pipe-list column -> where it lands in a node/edge spec ---
PER_NODE = {
    "node_t1": ("t1",), "node_t2": ("t2",),
    "node_p_depol_1q": ("gates", "p_depol_1q"),
    "node_p_depol_2q": ("gates", "p_depol_2q"),
    # node_p_depol_swap in older CSVs is ignored: SWAP error is derived in hardware
    # as SWAP_DEPOL_FACTOR (1.3) * p_depol_2q -- see roles.RepeaterNode
    "node_coh1_angle": ("gates", "coherent_1q", "angle"),
    "node_coh2_zz_angle": ("gates", "coherent_2q", "zz_angle"),
    "node_src_visibility": ("source", "visibility"),
    "node_src_phase": ("source", "phase"),
}
PER_EDGE = {
    "edge_length": "length",
    "edge_attenuation": "attenuation",
    "edge_insertion_loss_db": "insertion_loss_db",
}
# --- detector: d1_<suffix> scalar -> path inside one detector cfg (source node) ---
DET = {
    "coupling_1": ("coupling_1",), "coupling_2": ("coupling_2",),
    "mzi_phase_error": ("mzi", "phase_error"),
    "mzi_loss_short": ("mzi", "loss_short"), "mzi_loss_long": ("mzi", "loss_long"),
    "bs1_reflectivity": ("mzi", "bs1", "reflectivity"), "bs1_loss": ("mzi", "bs1", "loss"),
    "bs2_reflectivity": ("mzi", "bs2", "reflectivity"), "bs2_loss": ("mzi", "bs2", "loss"),
    "snspd1_efficiency": ("snspd_1", "efficiency"), "snspd1_jitter": ("snspd_1", "jitter_fwhm"),
    "snspd1_dark": ("snspd_1", "dark_count_rate"), "snspd1_dead": ("snspd_1", "dead_time"),
    "snspd2_efficiency": ("snspd_2", "efficiency"), "snspd2_jitter": ("snspd_2", "jitter_fwhm"),
    "snspd2_dark": ("snspd_2", "dark_count_rate"), "snspd2_dead": ("snspd_2", "dead_time"),
}


def _floats(cell):
    return [float(x) for x in cell.split("|")]


def _set(d, path, value):
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value


def _detector_cfg(row, prefix):
    """One TimeEnergyDetector cfg from the <prefix>_* scalar columns (defaults fill the rest)."""
    cfg = copy.deepcopy(DEFAULT_DETECTOR)
    for suffix, path in DET.items():
        val = float(row[f"{prefix}_{suffix}"])
        _set(cfg, path, int(round(val)) if suffix.endswith("_dead") else val)
    return cfg


def build_chain_spec(row, use_detector=True):
    """A linear-chain TopologySpec dict for one dataset path (c0 = source, c{N-1} = dest)."""
    ids = row["PathString"].split("->")
    n = len(ids)
    pernode = {c: _floats(row[c]) for c in PER_NODE}
    peredge = {c: _floats(row[c]) for c in PER_EDGE}
    for c, v in pernode.items():
        if len(v) != n:
            raise ValueError(f"{c}: expected {n} values, got {len(v)}")
    for c, v in peredge.items():
        if len(v) != n - 1:
            raise ValueError(f"{c}: expected {n - 1} values, got {len(v)}")

    nodes = {}
    for i in range(n):
        nd = copy.deepcopy(DEFAULT_NODE)
        nd["coord"] = [0, i]
        nd["detectors"] = {}
        for col, path in PER_NODE.items():
            _set(nd, path, pernode[col][i])
        nodes[f"c{i}"] = nd
    if use_detector:                                  # source node carries d1 -> NB reads it
        nodes["c0"]["detectors"] = {"d1": _detector_cfg(row, "d1")}
        nodes[f"c{n-1}"]["detectors"] = {"d2": _detector_cfg(row, "d2")}

    edges = {
        f"e{i}": {"u": f"c{i}", "v": f"c{i+1}", "n": DEFAULT_EDGE["n"],
                  **{dst: peredge[col][i] for col, dst in PER_EDGE.items()}}
        for i in range(n - 1)
    }
    return {
        "schema": "qetwork-topology/5",
        "name": f"path-D{row['DatasetID']}-P{row['PathID']}",
        "provenance": {}, "network": dict(DEFAULT_NETWORK),
        "roles": {"source": "c0", "destination": f"c{n-1}"},
        "nodes": nodes, "edges": edges,
    }


class _TimedNB(NetworkBenchmark):
    """NetworkBenchmark + per-m clock accounting, so we can form the avg_time metric.

    Only measurement/bookkeeping is added; the physics (per-hop bounce, optional
    link-level purification) is entirely the base class's. measure() mirrors run()
    but splits the virtual clock per m-group instead of returning one total.
    """

    def measure(self):
        rng = self.tl.rng
        self._cleanup()
        raw, time_m = {}, {}
        for m in range(self.m_min, self.m_max + 1):
            t0 = self.tl.now()
            vals = [self._one_sequence(m, rng) for _ in range(self.n_samples)]
            time_m[m] = self.tl.now() - t0                       # clock this m-group burned
            raw[m] = [v for v in vals if v is not None]          # post-select clicks
        decay = {m: (float(np.mean(v)) if v else 0.0) for m, v in raw.items()}
        _f, f_path, _a = fit_decay(decay, spam=self.detector is not None)
        avg_time = sum(time_m[m] / m for m in time_m) / len(time_m)   # (sum_m time_m/m)/n_m, ps
        return f_path, avg_time / 1e6                             # us


def run_one(row, *, rounds, mode, samples, seed):
    net = TopologySpec(build_chain_spec(row, USE_DETECTOR)).materialize(Timeline(seed=seed))
    nb = _TimedNB(net, net.path(), m_min=M_MIN, m_max=M_MAX, n_samples=samples,
                  purify_rounds=rounds, calibrate=CALIBRATE, mode=mode)
    return nb.measure()


def _run_row(job):
    """Pool worker: one dataset row -> a flat result tuple (top-level so it pickles)."""
    row, cfg = job
    ident = (row["DatasetID"], row["PathID"], row["PathString"])
    try:
        f_path, avg_us = run_one(row, rounds=cfg["rounds"], mode=cfg["mode"],
                                 samples=cfg["samples"], seed=cfg["seed"])
        return ident + (f_path, avg_us, None)
    except Exception as e:                            # one bad path never kills the batch
        return ident + (None, None, f"{type(e).__name__}: {e}")


def _jobs_stream(incsv, base_cfg):
    """Yield (row, cfg) work items. The seed is per-row deterministic (SEED + row
    index), order-free, so any --jobs count reproduces the identical result set."""
    with open(incsv, newline="") as fin:
        for idx, row in enumerate(csv.DictReader(fin)):
            yield row, {**base_cfg, "seed": SEED + idx}


def main():
    ap = argparse.ArgumentParser(
        description="Batch network benchmarking over one dataset CSV, optionally with "
                    "link-level purified hops.")
    ap.add_argument("incsv", help="input *_datasets.csv file")
    ap.add_argument("outcsv", help="output results CSV")
    ap.add_argument("--purification", action="store_true",
                    help="purify every hop's link (standard repeated DEJMPS pumping)")
    ap.add_argument("--purification-rounds", type=int, default=None,
                    help=f"pump level, 1..{MAX_ROUNDS} consecutive successes (default 1); "
                         f"requires --purification")
    ap.add_argument("--protocol", choices=("seq", "par"), default="seq",
                    help="distribution protocol handed to the benchmark (reserved for the bounce)")
    ap.add_argument("--samples", type=int, default=40,
                    help="RB sequences per m (detector readout is one click/seq; raise for clean fits)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes (0 = all cores); >1 writes rows in completion order")
    args = ap.parse_args()

    if not os.path.isfile(args.incsv):
        ap.error(f"input CSV not found: {args.incsv}")
    if args.purification_rounds is not None and not args.purification:
        ap.error("--purification-rounds requires --purification")
    rounds = 0
    if args.purification:
        rounds = 1 if args.purification_rounds is None else args.purification_rounds
        if not 1 <= rounds <= MAX_ROUNDS:
            ap.error(f"--purification-rounds must be in 1..{MAX_ROUNDS}, got {rounds}")
    mode = SEQUENTIAL if args.protocol == "seq" else PARALLEL
    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)

    base_cfg = {"rounds": rounds, "mode": mode, "samples": args.samples}
    cols = ["DatasetID", "PathID", "PathString", "path_fidelity", "avg_time_us"]

    n_ok = n_err = 0
    with open(args.outcsv, "w", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=cols)
        w.writeheader()

        def emit(res):
            nonlocal n_ok, n_err
            did, pid, pstr, f_path, avg_us, err = res
            if err is not None:
                n_err += 1
                print(f"  !! D{did} P{pid}: {err}")
                return
            w.writerow({"DatasetID": did, "PathID": pid, "PathString": pstr,
                        "path_fidelity": f"{f_path:.6f}", "avg_time_us": f"{avg_us:.6f}"})
            fout.flush()
            n_ok += 1
            print(f"[{n_ok}] D{did} P{pid} ({len(pstr.split('->')) - 1} hops)  "
                  f"F={f_path:.4f}  t={avg_us:.2f}us")

        work = _jobs_stream(args.incsv, base_cfg)
        if jobs == 1:                                 # in-process: no pool overhead, plain tracebacks
            for job in work:
                emit(_run_row(job))
        else:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                pending = set()
                for job in work:                      # bounded window: stream, never load the CSV whole
                    pending.add(ex.submit(_run_row, job))
                    if len(pending) >= jobs * 4:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for fut in done:
                            emit(fut.result())
                for fut in as_completed(pending):
                    emit(fut.result())
    print(f"done: {n_ok} rows -> {args.outcsv}  ({n_err} errors)")


if __name__ == "__main__":
    main()
