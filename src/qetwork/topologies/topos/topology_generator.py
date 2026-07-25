"""topology_generator: build named connectivity graphs and write fully-resolved topology files.

Every randomness (edge lengths, role pick) resolves HERE, seeded and recorded in
provenance; the written file mentions every parameter of every materialized component."""

import copy
import inspect
import json
import math

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


def watts_strogatz(n: int, k: int, p: float, seed: int) -> nx.Graph:
    """Connected Watts-Strogatz ring: k nearest neighbours, each edge rewired with prob p.

    Nodes are n1..nN placed on a ring (coord on a radius-1000 circle); edges
    spanning more than k/2 ring positions are tagged shortcut=True for rendering.
    Raises nx.NetworkXError if 100 rewiring attempts all come out disconnected.
    """
    if k % 2 or not 2 <= k < n:
        raise ValueError(f"k must be even and 2 <= k < n, got k={k}, n={n}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got p={p}")
    ws = nx.connected_watts_strogatz_graph(n, k, p, tries=100, seed=seed)

    G = nx.Graph()
    for i in range(n):
        theta = 2 * math.pi * i / n
        G.add_node(f"n{i + 1}", coord=(round(1000 * math.cos(theta)),
                                       round(1000 * math.sin(theta))))
    for e, (u, v) in enumerate(ws.edges, 1):
        ring_dist = min(abs(u - v), n - abs(u - v))
        G.add_edge(f"n{u + 1}", f"n{v + 1}", name=f"e{e}", shortcut=ring_dist > k // 2)
    return G


_GENERATORS = {"grid": grid, "watts_strogatz": watts_strogatz}


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
             network: dict | None = None, name: str = "",
             image_path: str | None = None, **kind_kwargs) -> dict:
    """Build a graph, resolve all randomness, and return (optionally write) the full spec dict.

    roles pins (source, destination) to the named nodes instead of drawing them.
    image_path additionally renders the connectivity graph to a PNG."""
    gen = _GENERATORS.get(kind)
    if gen is None:
        raise ValueError(f"unknown generator kind {kind!r}, expected one of {sorted(_GENERATORS)}")
    lo, hi = qlink_length_range
    if not 0 <= lo <= hi:
        raise ValueError(f"qlink_length_range must be 0 <= lo <= hi, got {qlink_length_range}")

    gen_kwargs = dict(kind_kwargs)
    if "seed" in inspect.signature(gen).parameters:  # graph structure itself is random
        gen_kwargs["seed"] = seed
    G = gen(**gen_kwargs)
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
    if image_path is not None:
        save_image(G, image_path, title=spec["name"], roles=spec["roles"])
    return spec


_IMG_SURFACE = "#fcfcfb"
_IMG_INK = "#0b0b0b"
_IMG_MUTED = "#52514e"
_IMG_EDGE = "#c3c2b7"
_IMG_ACCENT = "#2a78d6"


def save_image(G: nx.Graph, out_path: str, *, title: str = "",
               roles: dict | None = None) -> None:
    """Render the connectivity graph to a PNG, laid out by node coord.

    Edges tagged shortcut=True (Watts-Strogatz rewires) are drawn in the accent
    color; source/destination roles are highlighted and labelled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pos = {nid: (c[1], -c[0]) for nid, c in G.nodes(data="coord")}
    plain = [(u, v) for u, v, s in G.edges(data="shortcut") if not s]
    shortcuts = [(u, v) for u, v, s in G.edges(data="shortcut") if s]
    role_ids = list((roles or {}).values())

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(_IMG_SURFACE)
    ax.set_facecolor(_IMG_SURFACE)
    ax.set_axis_off()

    nx.draw_networkx_edges(G, pos, edgelist=plain, ax=ax,
                           edge_color=_IMG_EDGE, width=1.2)
    nx.draw_networkx_edges(G, pos, edgelist=shortcuts, ax=ax,
                           edge_color=_IMG_ACCENT, width=2.0)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=90, node_color=_IMG_MUTED,
                           edgecolors=_IMG_SURFACE, linewidths=1.5)
    if role_ids:
        nx.draw_networkx_nodes(G, pos, nodelist=role_ids, ax=ax, node_size=160,
                               node_color=_IMG_ACCENT, edgecolors=_IMG_SURFACE,
                               linewidths=1.5)
        nx.draw_networkx_labels(G, pos, labels={r: r for r in role_ids}, ax=ax,
                                font_size=8, font_color=_IMG_INK)

    ax.set_title(title, color=_IMG_INK, fontsize=13, pad=14)
    handles = [Line2D([], [], color=_IMG_ACCENT, marker="o", lw=0,
                      label="source / destination")]
    if shortcuts:
        handles = [Line2D([], [], color=_IMG_EDGE, lw=1.2, label="lattice edge"),
                   Line2D([], [], color=_IMG_ACCENT, lw=2.0, label="rewired shortcut"),
                   *handles]
    ax.legend(handles=handles, loc="lower right", frameon=False,
              labelcolor=_IMG_MUTED, fontsize=9)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=_IMG_SURFACE)
    plt.close(fig)

