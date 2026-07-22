"""topology_generator: build named connectivity graphs and write fully-resolved topology files.

Every randomness (edge lengths, role pick) resolves HERE, seeded and recorded in
provenance; the written file mentions every parameter of every materialized component."""

import copy
import json

import networkx as nx

from qetwork.seed import make_rng
from qetwork.topologies.topology_spec import TopologySpec


DEFAULT_DETECTOR = {                  # TimeEnergyDetector — MZI + 2 SNSPDs, every knob
    "kind": "time-energy",
    "coupling_1": 1.0,
    "coupling_2": 1.0,
    "mzi": {
        "delta_t": 400,               # ps; FSR = 1000/delta_t = 2.5 GHz
        "phase": 0.0,
        "phase_error": 0.0,
        "loss_short": 0.0,
        "loss_long": 0.0,
        "band": [1520, 1570],
        "bs1": {"reflectivity": 0.5, "loss": 0.0, "convention": "real", "band": [1520, 1570]},
        "bs2": {"reflectivity": 0.5, "loss": 0.0, "convention": "real", "band": [1520, 1570]},
    },
    "snspd_1": {"efficiency": 0.90, "jitter_fwhm": 15.0, "dark_count_rate": 1e-12,
                "dead_time": 20000, "band": [1400, 1700]},
    "snspd_2": {"efficiency": 0.90, "jitter_fwhm": 15.0, "dark_count_rate": 1e-12,
                "dead_time": 20000, "band": [1400, 1700]},
}


DEFAULT_NODE = {
    "t1": "inf",
    "t2": "inf",
    "gates": {
        "p_depol_1q": 0.0,
        "p_depol_2q": 0.0,
        "coherent_1q": {"axis": "z", "angle": 0.0},
        "coherent_2q": {"zz_angle": 0.0},
        "durations": {"gate_1q": 0, "gate_2q": 0, "measure": 0},   # ps, ints
    },
    "source": {
        "signal_wavelength": 1530.0,
        "idler_wavelength": 1570.0,
        "visibility": 0.974,
        "phase": 0.0,
        "encoding": "energy-time",
    },
    "memory": {
        "emission_encoding": "energy-time",
        "emission_wavelength": None,
    },
    "detectors": {},
}

DEFAULT_EDGE = {                      # QFiber — all of its knobs except the drawn length
    "attenuation": 0.2,
    "insertion_loss_db": 0.0,
    "n": 1.468,
}

DEFAULT_NETWORK = {                   # CFiber mesh — lengths are derived, these are the inputs
    "cfiber_latency": 0,
    "cfiber_n": 1.468,
}


def grid(rows: int, cols: int) -> nx.Graph:
    if rows < 1 or cols < 1:
        raise ValueError(f"grid needs positive dims, got {rows}x{cols}")
    G = nx.Graph()

    def name(r, c):
        return f"n{r * cols + c + 1}"

    for r in range(rows):
        for c in range(cols):
            G.add_node(name(r, c), coord=(r, c))

    e = 0
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:                          # right neighbour
                e += 1
                G.add_edge(name(r, c), name(r, c + 1), name=f"e{e}")
            if r + 1 < rows:                          # down neighbour
                e += 1
                G.add_edge(name(r, c), name(r + 1, c), name=f"e{e}")
    return G


_GENERATORS = {"grid": grid}


def _merge(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def generate(kind: str, *, seed: int, out_path: str | None = None,
             qlink_length_range: tuple[float, float] = (1_000.0, 20_000.0),
             roles: tuple[str, str] | None = None,
             node_overrides: dict | None = None, edge_overrides: dict | None = None,
             network: dict | None = None, name: str = "", **kind_kwargs) -> dict:
    """Build a graph, resolve all randomness, and return (optionally write) the full spec dict.

    roles pins (source, destination) to the named nodes instead of drawing them."""
    gen = _GENERATORS.get(kind)
    if gen is None:
        raise ValueError(f"unknown generator kind {kind!r}, expected one of {sorted(_GENERATORS)}")
    lo, hi = qlink_length_range
    if not 0 <= lo <= hi:
        raise ValueError(f"qlink_length_range must be 0 <= lo <= hi, got {qlink_length_range}")

    G = gen(**kind_kwargs)
    if G.number_of_nodes() < 2:
        raise ValueError(f"topology needs at least 2 nodes, got {G.number_of_nodes()}")
    unknown_nodes = sorted(set(node_overrides or {}) - set(G.nodes))
    if unknown_nodes:
        raise ValueError(f"node_overrides name unknown nodes {unknown_nodes}")
    edge_names = {d["name"] for _, _, d in G.edges(data=True)}
    unknown_edges = sorted(set(edge_overrides or {}) - edge_names)
    if unknown_edges:
        raise ValueError(f"edge_overrides name unknown edges {unknown_edges}")
    if roles is not None:
        src, dst = roles
        unknown_roles = sorted({src, dst} - set(G.nodes))
        if unknown_roles:
            raise ValueError(f"roles name unknown nodes {unknown_roles}")
        if src == dst:
            raise ValueError(f"roles must be two distinct nodes, got {src!r} twice")
    rng = make_rng(seed)

    ids = list(G.nodes)
    # drawn even when roles pins the pick: skipping it would shift the RNG stream
    # and change every edge length drawn under the same seed
    i, j = (int(k) for k in rng.choice(len(ids), size=2, replace=False))
    if roles is not None:
        i, j = ids.index(roles[0]), ids.index(roles[1])

    nodes = {}
    for nid in ids:
        entry = _merge(DEFAULT_NODE, (node_overrides or {}).get(nid))
        if nid == ids[i] and not entry["detectors"]:
            entry["detectors"] = {"d1": copy.deepcopy(DEFAULT_DETECTOR),
                                  "d2": copy.deepcopy(DEFAULT_DETECTOR)}
        coord = G.nodes[nid].get("coord")
        entry["coord"] = list(coord) if coord is not None else None
        nodes[nid] = entry

    edges = {}
    for u, v, data in G.edges(data=True):
        ename = data["name"]
        entry = _merge(DEFAULT_EDGE, (edge_overrides or {}).get(ename))
        length = entry.pop("length", None)            # override may pin a length
        if length is None:
            length = float(rng.uniform(lo, hi))
        edges[ename] = {"u": u, "v": v, "length": length, **entry}

    spec = {
        "schema": "qetwork-topology/5",
        "name": name or f"{kind}-{'x'.join(str(v) for v in kind_kwargs.values())}-seed{seed}",
        "provenance": {"generator": kind, **kind_kwargs, "seed": seed,
                       "qlink_length_range": [lo, hi],
                       **({"roles_override": list(roles)} if roles else {})},
        "network": _merge(DEFAULT_NETWORK, network),
        "roles": {"source": ids[i], "destination": ids[j]},
        "nodes": nodes,
        "edges": edges,
    }
    TopologySpec(spec)     # round-trip through the strict loader: an invalid file can never be born
    if out_path is not None:
        with open(out_path, "w") as f:
            json.dump(spec, f, indent=2, allow_nan=False)
    return spec

