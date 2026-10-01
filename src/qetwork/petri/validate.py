"""Cross-check the entanglement Petri net against the event-driven simulator it models.

    python -m qetwork.petri.validate [--lengths 20000,35000,10000] [--runs 300] \\
        [--modes sequential,parallel,tad] [--algebra density|bell] [--t1 inf] [--t2 1e9] \\
        [--p1q 1e-3] [--p2q 1e-2] [--visibility 0.974] [--attenuation 0.2] [--seed 1]
    python -m qetwork.petri.validate --csv src/qetwork/input_dir/tester.csv --row 0 [--runs 300]

Both sides get the same linear chain: a qetwork TopologySpec is materialized and run
through EntanglementDistribution, and the same network is read into ChainParams for
the net. The chain is either a synthetic one (--lengths and the hardware flags) or one
dataset path (--csv/--row, built exactly as applications.run_nb builds it, exact readout).
Two checks per mode:

  lossless    the chain with every fiber made lossless. Timing is then deterministic in
              both models: latency and attempts must agree to the picosecond on every run.
              Fidelity must agree to float precision when it is deterministic too, and
              statistically otherwise (BSM outcomes can steer it -- noisy correction gates,
              T1, coherent errors -- and the two models draw them from different rngs).
  lossy       the chain as configured, over --runs independent seeds: mean +- SE of
              latency, attempts and fidelity (|z| > 4 flags a mismatch) and the two-sample
              KS test on latency (p < 1e-3 flags a mismatch).

Exit status 1 when any check flags a mismatch. With --algebra bell, finite T1 makes
the fidelity rows measure the Pauli-twirl approximation (see petri.bell) instead.
"""

import argparse
import bisect
import copy
import csv
import math
import statistics
import sys

import numpy as np

from qetwork.applications.run_nb import build_chain_spec
from qetwork.kernel.timeline import Timeline
from qetwork.petri.entanglement import ChainParams, simulate
from qetwork.petri.pair_state import ALGEBRAS
from qetwork.protocols.e_dist_swap import (
    PARALLEL, SEQUENTIAL, TAD, EntanglementDistribution, bell_fidelity,
)
from qetwork.topologies.topology_spec import TopologySpec
from qetwork.topologies.topos.topology_generator import DEFAULT_EDGE, DEFAULT_NETWORK, DEFAULT_NODE

DURATIONS = {"gate_1q": 10_000, "gate_2q": 100_000, "measure": 1_000_000}   # ps
LOSSLESS_RUNS = 20


def chain_spec(lengths, *, attenuation: float = 0.2, t1="inf", t2="inf",
               p1q: float = 0.0, p2q: float = 0.0, visibility: float = 0.974,
               durations: dict | None = None) -> dict:
    """A linear-chain topology dict c0 (source) -> ... -> c{K} (destination), one fiber
    length in metres per edge; every node shares the same hardware parameters."""
    lengths = list(lengths)
    hops = len(lengths)
    if hops < 1:
        raise ValueError("a chain needs at least one edge length")
    nodes = {}
    for i in range(hops + 1):
        nd = copy.deepcopy(DEFAULT_NODE)
        nd["coord"] = [0, i]
        nd["t1"], nd["t2"] = t1, t2
        nd["gates"]["p_depol_1q"], nd["gates"]["p_depol_2q"] = p1q, p2q
        nd["gates"]["durations"] = dict(durations or DURATIONS)
        nd["source"]["visibility"] = visibility
        nodes[f"c{i}"] = nd
    edges = {f"e{i}": {"u": f"c{i}", "v": f"c{i + 1}", "length": lengths[i],
                       "attenuation": attenuation, "insertion_loss_db": 0.0, "n": DEFAULT_EDGE["n"]}
             for i in range(hops)}
    return {"schema": "qetwork-topology/5", "name": f"chain-{hops}", "provenance": {},
            "network": dict(DEFAULT_NETWORK), "roles": {"source": "c0", "destination": f"c{hops}"},
            "nodes": nodes, "edges": edges}


def lossless(spec: dict) -> dict:
    out = copy.deepcopy(spec)
    for e in out["edges"].values():
        e["attenuation"] = 0.0
        e["insertion_loss_db"] = 0.0
    return out


def run_seed(base: int, side: int, k: int) -> int:
    """Independent per-run seed: distinct (base, side, run) never share a stream, so
    two invocations with different --seed draw disjoint samples (base + k would overlap)."""
    return int(np.random.SeedSequence([base, side, k]).generate_state(1)[0])


def run_simulator(spec: dict, mode: str, seed: int) -> tuple[int, float, tuple[int, ...]]:
    tl = Timeline(seed=seed)
    net = TopologySpec(spec).materialize(tl)
    path = net.path()
    res = EntanglementDistribution(net, path, mode=mode).run()
    if res is None:
        raise RuntimeError(f"simulator drained before delivery (mode {mode}, seed {seed})")
    f = bell_fidelity(tl.state_tracker, res.source_key, res.dest_key)
    return res.latency, f, tuple(res.attempts[n] for n in path[:-1])


def run_net(spec: dict, mode: str, seed: int, algebra: str) -> tuple[int, float, tuple[int, ...]]:
    net = TopologySpec(spec).materialize(Timeline(seed=seed))
    r = simulate(ChainParams.from_network(net), mode, seed=seed, algebra=ALGEBRAS[algebra]())
    return r.latency, r.fidelity, r.attempts


def ks_2samp(a, b) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and its asymptotic p-value."""
    # CITE ks-asymptotic | Q_KS(lam) = 2 sum_j (-1)^(j-1) exp(-2 j^2 lam^2), lam = (sqrt(Ne) + 0.12 + 0.11/sqrt(Ne)) D | Press et al., "Numerical Recipes", 3rd ed., §14.3.3
    a, b = sorted(a), sorted(b)
    n, m = len(a), len(b)
    d = max(abs(bisect.bisect_right(a, x) / n - bisect.bisect_right(b, x) / m) for x in set(a) | set(b))
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 0.2:
        return d, 1.0
    q = 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101))
    return d, min(1.0, max(0.0, q))


def _mean_se(xs) -> tuple[float, float]:
    return statistics.fmean(xs), (statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0)


def _compare(label: str, xs, ys, scale: float = 1.0, unit: str = "") -> bool:
    """Means agree: to float precision when both samples are constant, else |z| <= 4."""
    (mx, sx), (my, sy) = _mean_se(xs), _mean_se(ys)
    se = math.hypot(sx, sy)
    if se == 0:
        ok, how = abs(mx - my) <= 1e-12 * max(1.0, abs(mx)), "exact"
    else:
        z = (mx - my) / se
        ok, how = abs(z) <= 4, f"z={z:+.2f}"
    print(f"    {label:<9} sim {mx * scale:14.6f} +- {sx * scale:.6f}  net {my * scale:14.6f} "
          f"+- {sy * scale:.6f} {unit:<2} {how:<8} {'ok' if ok else 'MISMATCH'}")
    return ok


def check_lossless(spec: dict, mode: str, args) -> bool:
    spec = lossless(spec)
    n = min(args.runs, LOSSLESS_RUNS)
    sim = [run_simulator(spec, mode, run_seed(args.seed, 2, k)) for k in range(n)]
    net = [run_net(spec, mode, run_seed(args.seed, 3, k), args.algebra) for k in range(n)]
    timing = {(r[0], r[2]) for r in sim} | {(r[0], r[2]) for r in net}
    ok_t = len(timing) == 1
    lat, att = next(iter(timing)) if ok_t else (None, None)
    print(f"  lossless ({n} runs each)")
    print(f"    timing    {'identical on every run: latency ' + str(lat) + ' ps, attempts ' + str(att) if ok_t else 'MISMATCH ' + str(sorted(timing))}")
    return _compare("fidelity", [r[1] for r in sim], [r[1] for r in net]) and ok_t


def check_lossy(spec: dict, mode: str, args) -> bool:
    sim = [run_simulator(spec, mode, run_seed(args.seed, 0, k)) for k in range(args.runs)]
    net = [run_net(spec, mode, run_seed(args.seed, 1, k), args.algebra) for k in range(args.runs)]
    print(f"  lossy ({args.runs} runs each)")
    ok = _compare("latency", [r[0] for r in sim], [r[0] for r in net], 1e-6, "us")
    ok &= _compare("fidelity", [r[1] for r in sim], [r[1] for r in net])
    ok &= _compare("attempts", [sum(r[2]) for r in sim], [sum(r[2]) for r in net])
    d, p = ks_2samp([r[0] for r in sim], [r[0] for r in net])
    flag = p >= 1e-3
    print(f"    KS latency D={d:.4f} p={p:.3f}  {'ok' if flag else 'MISMATCH'}")
    return ok and flag


def _time_arg(s: str):
    return "inf" if s == "inf" else float(s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lengths", type=lambda s: [float(x) for x in s.split(",")],
                    default=[20_000.0, 35_000.0, 10_000.0],
                    help="comma-separated fiber length per edge, metres (unequal edges exercise tad)")
    ap.add_argument("--csv", help="dataset CSV (run_nb input format); overrides the synthetic chain")
    ap.add_argument("--row", type=int, default=0, help="row index within --csv")
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--modes", default=f"{SEQUENTIAL},{PARALLEL},{TAD}")
    ap.add_argument("--algebra", choices=sorted(ALGEBRAS), default="density")
    ap.add_argument("--t1", type=_time_arg, default="inf", help="ps, or inf")
    ap.add_argument("--t2", type=_time_arg, default=1e9, help="ps, or inf")
    ap.add_argument("--p1q", type=float, default=1e-3)
    ap.add_argument("--p2q", type=float, default=1e-2)
    ap.add_argument("--visibility", type=float, default=0.974)
    ap.add_argument("--attenuation", type=float, default=0.2, help="dB/km")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    if args.csv:
        with open(args.csv, newline="") as f:
            rows = list(csv.DictReader(f))
        if not 0 <= args.row < len(rows):
            ap.error(f"--row {args.row} out of range: {args.csv} has {len(rows)} rows")
        spec = build_chain_spec(rows[args.row], use_detector=False)
        what = f"{args.csv} row {args.row}: {rows[args.row]['PathString']}"
    else:
        spec = chain_spec(args.lengths, attenuation=args.attenuation, t1=args.t1, t2=args.t2,
                          p1q=args.p1q, p2q=args.p2q, visibility=args.visibility)
        what = f"edges {'/'.join(f'{x:g}' for x in args.lengths)} m"

    all_ok = True
    for mode in args.modes.split(","):
        print(f"[{mode}] {what}  ({args.algebra} pair state)")
        all_ok &= check_lossless(spec, mode, args)
        all_ok &= check_lossy(spec, mode, args)
    print("consistent" if all_ok else "MISMATCH")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
