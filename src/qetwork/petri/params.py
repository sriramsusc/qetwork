"""Chain parameters the entanglement net's arc expressions read: per-node hardware and
per-edge timing/loss, either written by hand or read off a materialized QuantumNetwork."""

import math
from dataclasses import dataclass

from qetwork.protocols.e_dist_swap import validate_path


def _is_tick(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def t2_star(t1: float, t2: float) -> float:
    """Pure-dephasing time 1/T2* = 1/T2 - 1/(2 T1), as MemoryQubit._t2_star: T1 already
    decays coherences by e^(-t/2T1), so feeding T2 to the dephasing channel would count it twice."""
    if math.isinf(t1):
        return t2
    inv = 1.0 / t2 - 1.0 / (2.0 * t1)
    return math.inf if inv <= 0 else 1.0 / inv


@dataclass(slots=True, frozen=True)
class NodeParams:
    t1: float = math.inf            # ps
    t2: float = math.inf            # ps
    p_depol_1q: float = 0.0
    p_depol_2q: float = 0.0
    coherent_1q_axis: str = "z"     # residual rotation after every 1q gate (roles._coherent_1q)
    coherent_1q_angle: float = 0.0
    zz_angle: float = 0.0           # residual ZZ after every 2q gate (roles._coherent_2q)
    visibility: float = 1.0         # this node's pair source; it emits on its downstream edge
    source_phase: float = 0.0
    bsm_duration: int = 0           # ps
    correct_duration: int = 0       # ps

    def __post_init__(self) -> None:
        for name in ("t1", "t2"):
            t = getattr(self, name)
            if math.isnan(t) or t <= 0:
                raise ValueError(f"{name} must be positive (inf allowed), got {t}")
        if self.t2 > 2 * self.t1:
            raise ValueError(f"unphysical coherence times: T2={self.t2} exceeds 2*T1={2 * self.t1}")
        for name in ("p_depol_1q", "p_depol_2q", "visibility"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1], got {getattr(self, name)}")
        if self.coherent_1q_axis not in ("x", "y", "z"):
            raise ValueError(f"coherent_1q_axis must be x/y/z, got {self.coherent_1q_axis!r}")
        for name in ("coherent_1q_angle", "zz_angle", "source_phase"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)}")
        for name in ("bsm_duration", "correct_duration"):
            if not _is_tick(getattr(self, name)):
                raise ValueError(f"{name} must be an int >= 0 (ps), got {getattr(self, name)!r}")

    @property
    def coherent(self) -> bool:
        """Any error outside the Pauli channels (a Bell-diagonal colour cannot carry it)."""
        return self.coherent_1q_angle != 0 or self.zz_angle != 0 or self.source_phase != 0


@dataclass(slots=True, frozen=True)
class EdgeParams:
    qdelay: int                     # ps, emitter -> absorber photon flight
    cdelay: int                     # ps, one-way classical delay (symmetric, as the simulator assumes)
    survival: float                 # heralding probability of one attempt

    def __post_init__(self) -> None:
        for name in ("qdelay", "cdelay"):
            if not _is_tick(getattr(self, name)):
                raise ValueError(f"{name} must be an int >= 0 (ps), got {getattr(self, name)!r}")
        if not 0 < self.survival <= 1:
            raise ValueError(f"survival must be in (0,1], got {self.survival}")


@dataclass(slots=True, frozen=True)
class ChainParams:
    nodes: tuple[NodeParams, ...]
    edges: tuple[EdgeParams, ...]
    dest_cdelay: tuple[int, ...]    # ps, classical delay n_r -> n_K for every node (n_K's own is 0)

    def __post_init__(self) -> None:
        if len(self.edges) < 1:
            raise ValueError("a chain needs at least one edge")
        if len(self.nodes) != len(self.edges) + 1:
            raise ValueError(f"{len(self.edges)} edges need {len(self.edges) + 1} nodes, got {len(self.nodes)}")
        if len(self.dest_cdelay) != len(self.nodes):
            raise ValueError(f"dest_cdelay needs one entry per node, got {len(self.dest_cdelay)}")
        if not all(_is_tick(d) for d in self.dest_cdelay):
            raise ValueError(f"dest_cdelay entries must be ints >= 0 (ps), got {self.dest_cdelay}")

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def rtt(self, i: int) -> int:
        return self.edges[i].qdelay + self.edges[i].cdelay

    @classmethod
    def from_network(cls, net, path=None) -> "ChainParams":
        """Read a materialized QuantumNetwork path. Coherent-error settings come from the
        spec (nodes keep only the built matrices); delays and survival from the live fibers."""
        path = validate_path(net, path if path is not None else net.path())
        dest_id = path[-1]
        nodes = []
        for nid in path:
            nd = net.nodes[nid]
            gates = net.spec.nodes[nid]["gates"]
            nodes.append(NodeParams(
                t1=nd.t1, t2=nd.t2,
                p_depol_1q=nd.p_depol_1q, p_depol_2q=nd.p_depol_2q,
                coherent_1q_axis=gates["coherent_1q"]["axis"],
                coherent_1q_angle=gates["coherent_1q"]["angle"],
                zz_angle=gates["coherent_2q"]["zz_angle"],
                visibility=nd.source.visibility, source_phase=nd.source.phase,
                bsm_duration=nd.bsm_duration, correct_duration=nd.correct_duration))
        edges = []
        for u, v in zip(path, path[1:]):
            A, B = net.nodes[u], net.nodes[v]
            qf = A.ports[f"q:{v}"].qfiber
            fwd, back = A.cfibers[v], B.cfibers[u]
            cd_fwd, cd_back = fwd.delay + fwd.latency, back.delay + back.latency
            if cd_fwd != cd_back:
                raise ValueError(f"classical delay {u!r}<->{v!r} is asymmetric ({cd_fwd} vs {cd_back})")
            edges.append(EdgeParams(qdelay=qf.delay, cdelay=cd_back, survival=qf.survival))
        dest_cd = []
        for nid in path:
            cf = net.nodes[nid].cfibers.get(dest_id)
            dest_cd.append(0 if nid == dest_id else cf.delay + cf.latency)
        return cls(tuple(nodes), tuple(edges), tuple(dest_cd))
