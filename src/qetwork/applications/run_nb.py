"""Batch network benchmarking over error-stamped dataset CSVs: a thin driver
that feeds each CSV path to protocols.net_benchmarking and writes the outputs.

    python -m qetwork.applications.run_nb [in] [--out-root DIR] \\
        [--purification] [--purification-rounds R] [--protocol seq|par|tad] \\
        [--samples N] [--jobs J] [--fresh]

[in] is one *_datasets.csv or a directory of them (existing *_results.csv
files are skipped as inputs); omit it to read every CSV from the package's
input_dir/ directory (../input_dir next to this script). Every X.csv lands as X_results.csv inside a
directory named <topology>_<protocol>: the topology is recovered from the
input's filename (<name>_{prior,test,train_ds<k>}_datasets.csv -> <name>;
any other stem is used as-is) and the protocol tag is seq|par|tad, extended to
e.g. seq-pur2 when purifying so raw and purified runs never collide. The
directory is created under --out-root (default: run_nb_res/ next to this
script).

Reads every row of each input CSV, rebuilds each path as a linear chain (per-node
gates/source/T1-T2 from the pipe-lists, per-edge fiber, and the endpoint
MZI+SNSPD detectors from the d1_*/d2_* scalars), runs NetworkBenchmark.measure()
(Helsen & Wehner over EntanglementDistribution transport -- see the protocol
module for the physics), and writes ONE output CSV row per path:

    DatasetID, PathID, PathString, path_fidelity, avg_time_us

Flags:
  --purification            purify-then-swap distribution (PurifiedDistribution):
                            pump every edge's link (repeated DEJMPS), swap the
                            kept pairs into the end-to-end pair. Works with every
                            --protocol (the mode schedules the pump starts)
  --purification-rounds R   pump level, 1..5 consecutive successes (default 1);
                            only valid together with --purification
  --protocol seq|par|tad    distribution mode: sequential absorb->emit baton (or
                            chained edge pumps), parallel emission at t=0, or tad
                            time-aligned staggered starts
  --samples N               RB sequences per m (default 40; the detector readout
                            is one click per sequence, so raise it for clean fits)
  --jobs J                  worker processes (default 1; 0 = all cores). Every row
                            is an independent simulation with a deterministic seed
                            (SEED + row index), so the result SET is identical for
                            any J; with J > 1 rows land in completion order and
                            DatasetID/PathID identify them. Submission uses a
                            bounded in-flight window, so huge CSVs are streamed.
  --fresh                   recompute from scratch, overwriting any existing
                            results instead of resuming them

Crash recovery: every finished row is flushed AND fsynced, so a crash loses at
most the rows in flight. On rerun an existing results CSV is resumed, not
redone: its (DatasetID, PathID) keys are skipped (a torn last line is dropped
via an atomic rewrite first) and only the missing rows run, with their original
row-index seeds -- the resumed file is identical to a never-interrupted run.
A <out>.meta.json sidecar records the config fingerprint; resuming under a
different config (or onto a results file with no sidecar) refuses with a
pointer to --fresh. Failed rows are never written, so they retry on resume.

Fixed settings (module constants below): m = M_MIN..M_MAX bounces, base SEED,
detector readout USE_DETECTOR, calibration CALIBRATE.
"""

import argparse
import copy
import csv
import json
import os
import re
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait

from qetwork.kernel.timeline import Timeline
from qetwork.topologies.topology_spec import TopologySpec
from qetwork.topologies.topos.topology_generator import (
    DEFAULT_NODE, DEFAULT_EDGE, DEFAULT_NETWORK, DEFAULT_DETECTOR,
)
from qetwork.protocols.e_dist_swap import SEQUENTIAL, PARALLEL, TAD
from qetwork.protocols.net_benchmarking import NetworkBenchmark

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_nb_res")
INPUT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "input_dir"))

M_MIN, M_MAX = 1, 8            # bounce range of the RB fit
SEED = 1                       # base seed; row i runs with SEED + i
USE_DETECTOR = False            # MZI+SNSPD readout (False = exact readout)
CALIBRATE = True               # per-pair source-phase calibration in purified hops
MAX_ROUNDS = 5
COLS = ["DatasetID", "PathID", "PathString", "path_fidelity", "avg_time_us"]

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


def run_one(row, *, rounds, mode, samples, seed):
    net = TopologySpec(build_chain_spec(row, USE_DETECTOR)).materialize(Timeline(seed=seed))
    nb = NetworkBenchmark(net, net.path(), m_min=M_MIN, m_max=M_MAX, n_samples=samples,
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


def _result_name(incsv):
    """grid-10x10-seed7_prior.csv -> grid-10x10-seed7_prior_results.csv"""
    return os.path.splitext(os.path.basename(incsv))[0] + "_results.csv"


_PART_RE = re.compile(r"_(prior|test|train_ds\d+)$")


def _topology_name(incsv):
    """ws-100-k4-p0.4-seed7_train_ds3_datasets.csv -> ws-100-k4-p0.4-seed7;
    a stem without the dataset suffixes is used as-is."""
    stem = os.path.splitext(os.path.basename(incsv))[0]
    stem = stem.removesuffix("_datasets")
    return _PART_RE.sub("", stem)


def _out_dir(incsv, root, tag):
    """<root>/<topology>_<tag>, created on demand."""
    d = os.path.join(root, f"{_topology_name(incsv)}_{tag}")
    os.makedirs(d, exist_ok=True)
    return d


def _discover(indir):
    """The dataset CSVs of one directory, sorted; *_results.csv outputs are
    skipped so out-dir == in-dir round-trips cleanly."""
    return [os.path.join(indir, n) for n in sorted(os.listdir(indir))
            if n.endswith(".csv") and not n.endswith("_results.csv")]


def _meta_path(outcsv):
    return os.path.splitext(outcsv)[0] + ".meta.json"


def _count_rows(incsv):
    with open(incsv, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _meta(incsv, base_cfg):
    """The config fingerprint a results file must match to be resumable.

    --jobs is deliberately absent: the per-row seeds make the result set
    invariant to the worker count."""
    return {
        "schema": 1,
        "input": os.path.basename(incsv),
        "input_rows": _count_rows(incsv),
        "benchmark": base_cfg.get("benchmark", "hop-by-hop"),
        "rounds": base_cfg["rounds"],
        "mode": base_cfg["mode"],
        "samples": base_cfg["samples"],
        "seed": SEED, "m_min": M_MIN, "m_max": M_MAX,
        "use_detector": USE_DETECTOR, "calibrate": CALIBRATE,
    }


def _recover(outcsv):
    """Parse a possibly crash-torn results CSV.

    Returns (rows, torn): the valid result rows in file order, and whether
    anything invalid (torn last line, bad header) had to be dropped."""
    rows, torn = [], False
    with open(outcsv, newline="") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames != COLS:            # header never made it to disk whole
            return [], True
        for row in rdr:
            try:
                ok = all(row[c] not in (None, "") for c in COLS)
                if ok:
                    float(row["path_fidelity"])
                    float(row["avg_time_us"])
            except (KeyError, TypeError, ValueError):
                ok = False
            if ok:
                rows.append({c: row[c] for c in COLS})
            else:
                torn = True
    return rows, torn


def _rewrite(outcsv, rows):
    """Replace outcsv with exactly rows, atomically (crash-safe recovery)."""
    tmp = outcsv + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, outcsv)


def process_file(incsv, outcsv, base_cfg, jobs, fresh=False, worker=_run_row):
    """All rows of one dataset CSV -> one results CSV. Returns (n_ok, n_err).

    Seeds restart at SEED + row index for every file, so each file's result
    set is identical whether it runs alone or inside a directory batch.
    An existing results CSV is resumed (its keys are skipped) unless fresh.
    worker maps one (row, cfg) job to a flat result tuple; other benchmark
    architectures (run_nb1) reuse this pipeline by passing their own."""
    meta, mpath = _meta(incsv, base_cfg), _meta_path(outcsv)
    done = set()
    if not fresh and os.path.exists(outcsv):
        if not os.path.exists(mpath):
            raise RuntimeError(
                f"{outcsv} exists without {os.path.basename(mpath)}, so its config "
                f"cannot be verified; rerun with --fresh to recompute it")
        with open(mpath) as f:
            prev = json.load(f)
        if prev != meta:
            differs = sorted(k for k in {*prev, *meta} if prev.get(k) != meta.get(k))
            raise RuntimeError(
                f"{outcsv} was produced under a different config "
                f"(differs: {', '.join(differs)}); rerun with --fresh to recompute it")
        rows, torn = _recover(outcsv)
        if torn:
            _rewrite(outcsv, rows)
        done = {(r["DatasetID"], r["PathID"]) for r in rows}
        print(f"  resuming: {len(done)}/{meta['input_rows']} rows already done")
        out_mode = "a"
    else:
        with open(mpath, "w") as f:
            json.dump(meta, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        out_mode = "w"

    n_ok = n_err = 0
    with open(outcsv, out_mode, newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=COLS)
        if out_mode == "w":
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
            os.fsync(fout.fileno())           # a crash loses only rows in flight
            n_ok += 1
            print(f"[{n_ok}] D{did} P{pid} ({len(pstr.split('->')) - 1} hops)  "
                  f"F={f_path:.4f}  t={avg_us:.2f}us")

        work = (job for job in _jobs_stream(incsv, base_cfg)
                if (job[0]["DatasetID"], job[0]["PathID"]) not in done)
        if jobs == 1:                                 # in-process: no pool overhead, plain tracebacks
            for job in work:
                emit(worker(job))
        else:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                pending = set()
                for job in work:                      # bounded window: stream, never load the CSV whole
                    pending.add(ex.submit(worker, job))
                    if len(pending) >= jobs * 4:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for fut in done:
                            emit(fut.result())
                for fut in as_completed(pending):
                    emit(fut.result())
    return n_ok, n_err


def main():
    ap = argparse.ArgumentParser(
        description="Batch network benchmarking over dataset CSVs, optionally with "
                    "link-level purified hops.")
    ap.add_argument("incsv", nargs="?", default=None,
                    help="input *_datasets.csv file, or a directory of them "
                         "(default: the input_dir/ package directory)")
    ap.add_argument("--out-root", default=None,
                    help="where the <topology>_<protocol> results directory is "
                         "created (default: run_nb_res/ next to this script)")
    ap.add_argument("--purification", action="store_true",
                    help="purify every hop's link (standard repeated DEJMPS pumping)")
    ap.add_argument("--purification-rounds", type=int, default=None,
                    help=f"pump level, 1..{MAX_ROUNDS} consecutive successes (default 1); "
                         f"requires --purification")
    ap.add_argument("--protocol", choices=("seq", "par", "tad"), default="seq",
                    help="distribution protocol handed to the benchmark (reserved for the bounce)")
    ap.add_argument("--samples", type=int, default=40,
                    help="RB sequences per m (detector readout is one click/seq; raise for clean fits)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes (0 = all cores); >1 writes rows in completion order")
    ap.add_argument("--fresh", action="store_true",
                    help="recompute from scratch instead of resuming existing results")
    args = ap.parse_args()

    if args.purification_rounds is not None and not args.purification:
        ap.error("--purification-rounds requires --purification")
    rounds = 0
    if args.purification:
        rounds = 1 if args.purification_rounds is None else args.purification_rounds
        if not 1 <= rounds <= MAX_ROUNDS:
            ap.error(f"--purification-rounds must be in 1..{MAX_ROUNDS}, got {rounds}")
    mode = {"seq": SEQUENTIAL, "par": PARALLEL, "tad": TAD}[args.protocol]
    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)

    base_cfg = {"rounds": rounds, "mode": mode, "samples": args.samples,
                "benchmark": "e2e-swap"}
    tag = args.protocol + (f"-pur{rounds}" if rounds else "")

    root = args.out_root or OUT_ROOT
    incsv = args.incsv if args.incsv is not None else INPUT_DIR
    if os.path.isdir(incsv):
        files = _discover(incsv)
        if not files:
            ap.error(f"no dataset *.csv files in {incsv}")
        pairs = [(f, os.path.join(_out_dir(f, root, tag), _result_name(f)))
                 for f in files]
    elif os.path.isfile(incsv):
        pairs = [(incsv,
                  os.path.join(_out_dir(incsv, root, tag), _result_name(incsv)))]
    else:
        ap.error(f"input not found: {incsv}")

    bad = []
    for i, (incsv, outcsv) in enumerate(pairs):
        if len(pairs) > 1:
            print(f"=== [{i + 1}/{len(pairs)}] {incsv}")
        try:
            n_ok, n_err = process_file(incsv, outcsv, base_cfg, jobs, fresh=args.fresh)
        except Exception as e:                        # one bad file never kills the batch
            bad.append(incsv)
            print(f"  !! {incsv}: {type(e).__name__}: {e}")
            continue
        print(f"done: {n_ok} rows -> {outcsv}  ({n_err} errors)")
    if len(pairs) > 1:
        print(f"all done: {len(pairs) - len(bad)}/{len(pairs)} files"
              + (f"  (failed: {', '.join(bad)})" if bad else ""))


if __name__ == "__main__":
    main()
