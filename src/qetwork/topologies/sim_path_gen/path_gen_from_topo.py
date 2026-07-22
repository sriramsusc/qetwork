"""Path dataset generation from a topology.

Two generators, callable separately, plus one driver that writes three CSVs:

max_flow_paths: a distinct S-T path through every coverable edge. Per named
edge (u, v), a max-flow construction decides whether an S-T path can traverse
it: a super-source feeds one unit into S and one into T, both endpoints drain
into a super-sink, the edge itself is removed, and in-edges of S and T are cut
so neither appears mid-path. Flow 2 means edge-disjoint segments S->u and T->v
exist; stitching them across (u, v) is the path. No two edges may receive the
same path: solves that could collide are penalized against the colliding
paths, and an edge forced onto a taken path steals it while the old owner
re-solves (cycle-guarded); an edge with no distinct path left gets None.

generate_paths: nx.shortest_simple_paths (Yen) yields simple S-T paths in
nondecreasing hop order; the first max_paths are kept. No longer used by
generate_datasets (Yen cannot reach long hop counts without exhausting the
combinatorially many shorter paths first); kept as a standalone generator.

sample_paths_by_length: up to per_length distinct simple S-T paths per hop
count, sampled by seeded self-avoiding walks pruned by distance-to-target
(plus parity on bipartite graphs), so every hop count is reachable directly.

generate_datasets: runs max_flow_paths and sample_paths_by_length on one
topology and writes, under <name>_raw_paths/ in out_dir:
    <name>_prior.csv    every assigned max-flow path
    <name>_train.csv    the sampled paths minus the test picks
    <name>_test.csv     1 in TEST_SHARE of every hop-count bucket (seeded)
The per-bucket split keeps train and test on the same hop-count distribution;
the sampled hop range comes from HOP_RANGE[<generator kind>] unless overridden.
All CSVs are PathID,PathString rows: 1,n94->n84->...->n63"""

from itertools import islice
from pathlib import Path

import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path

from qetwork.seed import make_rng
from qetwork.topologies.topology_spec import TopologySpec

_SRC, _SINK = "S_prime", "T_prime"
_BIG = 10**6          # penalty per use of an already-assigned path's edge
_RETRIES = 4          # penalized re-solves before declaring an edge forced
MAX_PATHS = 220
TEST_SHARE = 9        # one path in TEST_SHARE of every hop-count bucket goes to the test set
HOP_RANGE = {"grid": (11, 19)}   # per generator kind: sampled hop counts, inclusive
PER_LENGTH = 44       # paths wanted per hop count; grid: 5 odd counts * (40 train + 4 test) = 220
_DRY_STREAK = 5000    # sampler attempts without a new distinct path before a bucket closes


def _as_spec(topology) -> TopologySpec:
    """Accept a TopologySpec or a path to a topology JSON."""
    if isinstance(topology, TopologySpec):
        return topology
    return TopologySpec.from_json(topology)


def _directed_unit(G: nx.Graph) -> nx.DiGraph:
    """Node-split digraph: w:in -> w:out (capacity 1) forces each node to be
    transited at most once, so the two flow segments share no nodes and every
    stitched path is simple."""
    D = nx.DiGraph()
    for w in G.nodes:
        D.add_edge(f"{w}:in", f"{w}:out", capacity=1)
    for u, v in G.edges:
        D.add_edge(f"{u}:out", f"{v}:in", capacity=1)
        D.add_edge(f"{v}:out", f"{u}:in", capacity=1)
    return D


def _transform_for_edge(D: nx.DiGraph, u: str, v: str, s: str, t: str) -> nx.DiGraph:
    H = D.copy()
    H.remove_edges_from(list(H.in_edges(f"{s}:in")))
    H.remove_edges_from(list(H.in_edges(f"{t}:in")))
    for a, b in ((u, v), (v, u)):
        if H.has_edge(f"{a}:out", f"{b}:in"):
            H.remove_edge(f"{a}:out", f"{b}:in")
    H.add_edge(_SRC, f"{s}:in", capacity=1)
    H.add_edge(_SRC, f"{t}:in", capacity=1)
    H.add_edge(f"{u}:out", _SINK, capacity=1)
    H.add_edge(f"{v}:out", _SINK, capacity=1)
    return H


def _segments(flow_dict: dict, s: str, t: str) -> tuple[list[str], list[str]]:
    """Decompose the 2-unit flow; collapse w:in / w:out back to w."""
    F = nx.DiGraph((a, b) for a, nbrs in flow_dict.items()
                   for b, f in nbrs.items() if f > 0)
    segs = {}
    for _ in range(2):
        p = nx.shortest_path(F, _SRC, _SINK)
        F.remove_edges_from(zip(p, p[1:]))
        segs[p[1]] = [x[:-3] for x in p[1:-1] if x.endswith(":in")]
    return segs[f"{s}:in"], segs[f"{t}:in"]


def _uses_edge(path: tuple[str, ...], u: str, v: str) -> bool:
    return any(hop in ((u, v), (v, u)) for hop in zip(path, path[1:]))


def _solve(D: nx.DiGraph, u: str, v: str, s: str, t: str,
           avoid: list[list[str]]) -> list[str] | None:
    """One S-T path through (u, v), steering around the paths in `avoid`.

    Empty `avoid` keeps the plain fast solve (collision impossible there).
    A path listed twice in `avoid` is penalized twice as hard."""
    H = _transform_for_edge(D, u, v, s, t)
    if not avoid:
        flow_val, flow_dict = nx.maximum_flow(
            H, _SRC, _SINK, flow_func=shortest_augmenting_path)
    else:
        usage: dict[tuple[str, str], int] = {}
        for p in avoid:
            for a, b in zip(p, p[1:]):
                usage[(a, b)] = usage.get((a, b), 0) + 1
                usage[(b, a)] = usage.get((b, a), 0) + 1
        for a, b in H.edges:
            if a.endswith(":out") and b.endswith(":in"):
                H[a][b]["weight"] = 1 + _BIG * usage.get((a[:-4], b[:-3]), 0)
            else:
                H[a][b]["weight"] = 1
        flow_dict = nx.max_flow_min_cost(H, _SRC, _SINK)
        flow_val = sum(flow_dict[_SRC].values())
    if flow_val < 2:
        return None
    seg_s, seg_t = _segments(flow_dict, s, t)
    return seg_s + seg_t[::-1]                # seam is exactly the edge (u, v)


def max_flow_paths(topology) -> dict[str, list[str] | None]:
    """A distinct S-T path per named edge; None if uncoverable or out of paths."""
    spec = _as_spec(topology)
    s, t = spec.roles["source"], spec.roles["destination"]
    D = _directed_unit(spec.graph())
    assigned: dict[str, list[str] | None] = {}
    owner: dict[tuple[str, ...], str] = {}    # path -> edge that holds it

    for ename in spec.edges:
        cur, visited = ename, set()
        while cur is not None:                # steal chain, iterative
            visited.add(cur)
            u, v = spec.edges[cur]["u"], spec.edges[cur]["v"]
            avoid = [list(p) for p in owner if _uses_edge(p, u, v)]
            key = None
            for _ in range(_RETRIES):
                path = _solve(D, u, v, s, t, avoid)
                if path is None:              # no S-T path through this edge at all
                    key = None
                    break
                key = tuple(path)
                if key not in owner:          # fresh path found
                    break
                avoid.append(path)            # recurring collider: double its penalty
            if key is None:
                assigned[cur] = None
                cur = None
            elif key not in owner:            # claim the fresh path
                owner[key] = cur
                assigned[cur] = list(key)
                cur = None
            else:                             # forced onto a taken path
                rival = owner[key]
                if rival in visited:          # steal chain closed a cycle: give up
                    assigned[cur] = None
                    cur = None
                else:                         # steal it; rival re-acquires next turn
                    owner[key] = cur
                    assigned[cur] = list(key)
                    del assigned[rival]
                    cur = rival
    return {ename: assigned[ename] for ename in spec.edges}


def generate_paths(topology, max_paths: int = MAX_PATHS) -> list[list[str]]:
    """The max_paths shortest simple S-T paths, in nondecreasing hop order."""
    spec = _as_spec(topology)
    if max_paths < 1:
        raise ValueError(f"max_paths must be positive, got {max_paths}")
    s, t = spec.roles["source"], spec.roles["destination"]
    gen = nx.shortest_simple_paths(spec.graph(), s, t)
    return [list(p) for p in islice(gen, max_paths)]


def sample_paths_by_length(topology, *, min_hops: int, max_hops: int,
                           per_length: int = PER_LENGTH,
                           seed: int | None = None) -> dict[int, list[list[str]]]:
    """Up to per_length distinct simple S-T paths per hop count in
    [min_hops, max_hops], sampled by seeded self-avoiding walks.

    A walk of target length L only steps to nodes it can still finish from:
    unvisited w with dist(w, T) within the remaining budget, parity-compatible
    (bipartite graphs only, where every S-T path length has the same parity),
    and T itself only on the last hop — so a completed walk has exactly L hops.
    Walled-in walks restart; a bucket closes after _DRY_STREAK consecutive
    attempts without a new distinct path, so scarce hop counts come up short
    instead of spinning. Provably empty hop counts (below the S-T distance, or
    parity-infeasible) are skipped and absent from the returned dict."""
    spec = _as_spec(topology)
    if not 1 <= min_hops <= max_hops:
        raise ValueError(f"need 1 <= min_hops <= max_hops, got {min_hops}..{max_hops}")
    if per_length < 1:
        raise ValueError(f"per_length must be positive, got {per_length}")
    s, t = spec.roles["source"], spec.roles["destination"]
    G = spec.graph()
    dist = nx.single_source_shortest_path_length(G, t)
    if s not in dist:
        raise ValueError(f"no path from {s} to {t} in the topology")
    if dist[s] > max_hops:
        raise ValueError(f"shortest {s}->{t} path is {dist[s]} hops, above max_hops={max_hops}")
    fixed_parity = nx.is_bipartite(G)
    rng = make_rng(seed)

    buckets: dict[int, list[list[str]]] = {}
    for L in range(min_hops, max_hops + 1):
        if L < dist[s] or (fixed_parity and (L - dist[s]) % 2):
            continue
        found: dict[tuple[str, ...], None] = {}      # insertion-ordered dedup
        dry = 0
        while len(found) < per_length and dry < _DRY_STREAK:
            path, visited, budget = [s], {s}, L
            while budget:
                opts = [w for w in G[path[-1]]
                        if w not in visited and dist[w] <= budget - 1
                        and not (fixed_parity and (budget - 1 - dist[w]) % 2)
                        and (w != t or budget == 1)]
                if not opts:
                    break
                w = opts[int(rng.integers(len(opts)))]
                path.append(w)
                visited.add(w)
                budget -= 1
            key = tuple(path)
            if path[-1] == t and key not in found:
                found[key] = None
                dry = 0
            else:
                dry += 1
        buckets[L] = [list(p) for p in found]
    return buckets


def write_paths_csv(paths: list[list[str]], out_path) -> None:
    """PathID,PathString rows, IDs numbered 1..len(paths) in list order."""
    with open(out_path, "w") as f:
        f.write("PathID,PathString\n")
        for i, p in enumerate(paths, 1):
            f.write(f"{i},{'->'.join(p)}\n")


def _validate(paths: list[list[str]], spec: TopologySpec, label: str) -> None:
    """Every path must run source->destination over real edges, no duplicates."""
    s, t = spec.roles["source"], spec.roles["destination"]
    adj = {frozenset((e["u"], e["v"])) for e in spec.edges.values()}
    for p in paths:
        if len(p) < 2 or p[0] != s or p[-1] != t:
            raise RuntimeError(f"{label}: path endpoints wrong: {p}")
        for hop in zip(p, p[1:]):
            if frozenset(hop) not in adj:
                raise RuntimeError(f"{label}: hop {hop} is not a topology edge")
    if len({tuple(p) for p in paths}) != len(paths):
        raise RuntimeError(f"{label}: duplicate paths")


def generate_datasets(topology, *, per_length: int = PER_LENGTH,
                      hop_range: tuple[int, int] | None = None,
                      seed: int | None = None, out_dir=None) -> dict[str, Path]:
    """Write <name>_raw_paths/<name>_{prior,train,test}.csv under out_dir;
    return the CSV paths.

    train/test are per_length sampled paths per hop count in hop_range
    (default: HOP_RANGE[<generator kind>] from the spec's provenance), split
    1 in TEST_SHARE within every hop-count bucket so both sets keep the same
    hop-count distribution; buckets that cannot fill their quota contribute
    what exists, and the shortfall is reported. out_dir defaults to the
    topology file's directory (cwd if a TopologySpec object was passed
    instead of a file); the <name>_raw_paths subdirectory is created if
    missing."""
    spec = _as_spec(topology)
    if hop_range is None:
        kind = spec.provenance.get("generator")
        hop_range = HOP_RANGE.get(kind)
        if hop_range is None:
            raise ValueError(f"no hop range configured for generator {kind!r}; "
                             f"add it to HOP_RANGE or pass hop_range=")
    lo_h, hi_h = hop_range
    if per_length < TEST_SHARE:
        raise ValueError(f"per_length={per_length} cannot fund a stratified test "
                         f"split; need at least TEST_SHARE={TEST_SHARE} per bucket")
    if out_dir is None:
        out_dir = Path.cwd() if isinstance(topology, TopologySpec) \
            else Path(topology).resolve().parent
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise ValueError(f"out_dir {out_dir} is not an existing directory")
    raw_dir = out_dir / f"{spec.name}_raw_paths"
    raw_dir.mkdir(exist_ok=True)

    per_edge = max_flow_paths(spec)
    prior = [p for p in per_edge.values() if p is not None]
    missing = sorted(e for e, p in per_edge.items() if p is None)
    _validate(prior, spec, "prior")

    buckets = sample_paths_by_length(spec, min_hops=lo_h, max_hops=hi_h,
                                     per_length=per_length, seed=seed)
    sampled = [p for L in sorted(buckets) for p in buckets[L]]
    _validate(sampled, spec, "sampled")
    rng = make_rng(seed)
    train, test = [], []
    for L in sorted(buckets):
        b = buckets[L]
        n_test = len(b) // TEST_SHARE
        picked = {int(i) for i in rng.choice(len(b), size=n_test, replace=False)} \
            if n_test else set()
        train += [p for i, p in enumerate(b) if i not in picked]
        test += [p for i, p in enumerate(b) if i in picked]
    if not test:
        raise ValueError(f"sampled only {len(sampled)} path(s) across hops "
                         f"{lo_h}..{hi_h}; no bucket reached TEST_SHARE={TEST_SHARE}")

    outs = {"prior": raw_dir / f"{spec.name}_prior.csv",
            "train": raw_dir / f"{spec.name}_train.csv",
            "test": raw_dir / f"{spec.name}_test.csv"}
    write_paths_csv(prior, outs["prior"])
    write_paths_csv(train, outs["train"])
    write_paths_csv(test, outs["test"])

    print(f"{spec.name}: prior {len(prior)}/{len(per_edge)} edges"
          + (f" (no path for {missing})" if missing else ""))
    print(f"sampled {len(sampled)} paths ({per_length} wanted per hop count "
          f"{lo_h}..{hi_h}); split train {len(train)} / test {len(test)}")
    infeasible = [L for L in range(lo_h, hi_h + 1) if L not in buckets]
    if infeasible:
        print(f"  no possible path at hop counts {infeasible}")
    short = {L: len(b) for L, b in buckets.items() if len(b) < per_length}
    if short:
        print("  !! under quota: "
              + ", ".join(f"{L} hops: {n}" for L, n in sorted(short.items())))
    for kind, p in outs.items():
        print(f"  {kind}: {p}")
    return outs


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    generate_datasets(here / "grid10x10.json")
