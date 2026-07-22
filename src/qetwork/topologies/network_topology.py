"""QuantumNetwork: materialize a TopologySpec into live nodes and fibers on a timeline.

Deterministic by construction — every random choice (lengths, roles) was resolved
when the spec file was written, so building consumes no timeline rng draws."""

from dataclasses import dataclass

import networkx as nx

from qetwork.components.utils.roles import RepeaterNode, SourceNode, DestinationNode
from qetwork.components.fiber import QFiber, CFiber
from qetwork.components.port import Port


@dataclass(slots=True, frozen=True)
class QEdge:
    """One materialized duplex edge: two nodes, their ports, two antiparallel fibers."""
    name: str
    u: str
    v: str
    length: float
    port_u: Port
    port_v: Port
    fiber_uv: QFiber
    fiber_vu: QFiber


class QuantumNetwork:
    def __init__(self, spec, timeline) -> None:
        self.spec = spec
        self.timeline = timeline
        self.graph = spec.graph()
        self.source_id = spec.roles["source"]
        self.dest_id = spec.roles["destination"]

        self.nodes: dict[str, RepeaterNode] = {}
        for nid, cfg in spec.nodes.items():
            role = (SourceNode if nid == self.source_id
                    else DestinationNode if nid == self.dest_id
                    else RepeaterNode)
            self.nodes[nid] = role(nid, timeline,
                                   t1=cfg["t1"], t2=cfg["t2"],
                                   gate_cfg=dict(cfg["gates"]),
                                   source_cfg=dict(cfg["source"]),
                                   memory_cfg=dict(cfg["memory"]),
                                   detector_cfg=dict(cfg["detectors"]),
                                    )
        self.qedges: dict[str, QEdge] = {}
        for ename, e in spec.edges.items():
            u, v = e["u"], e["v"]
            port_u = self.nodes[u].add_edge_port(v)
            port_v = self.nodes[v].add_edge_port(u)
            fiber_uv = QFiber(timeline, port_u, port_v, e["length"], attenuation=e["attenuation"],
                              insertion_loss_db=e["insertion_loss_db"], n=e["n"])
            fiber_vu = QFiber(timeline, port_v, port_u, e["length"], attenuation=e["attenuation"],
                              insertion_loss_db=e["insertion_loss_db"], n=e["n"])
            self.qedges[ename] = QEdge(ename, u, v, e["length"], port_u, port_v, fiber_uv, fiber_vu)

        # classical mesh: all-to-all, distance = shortest qfiber-path distance
        dist = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight="length"))
        self.cfibers: dict[tuple[str, str], CFiber] = {}
        for a in self.nodes:
            for b in self.nodes:
                if a == b:
                    continue
                self.cfibers[(a, b)] = CFiber(timeline, self.nodes[a], self.nodes[b], dist[a][b],
                                              n=spec.network["cfiber_n"],
                                              latency=spec.network["cfiber_latency"])

    def path(self, u: str | None = None, v: str | None = None,
             weight: str | None = "length") -> list[str]:
        """Linear node sequence; default source -> destination, min total fiber length.
        weight=None gives the min-hop path instead (fewer swaps, longer fibers)."""
        u = self.source_id if u is None else u
        v = self.dest_id if v is None else v
        return nx.shortest_path(self.graph, u, v, weight=weight)
    
    def start_dark_counts(self) -> None:
        """Poisson dark clicks on every detector in the network. Self-perpetuating events:
        use a finite stop_time, or drive protocols via step()-bounded runs."""
        for node in self.nodes.values():
            for det in node.detectors.values():
                det.start_dark_counts()

    def stop_dark_counts(self) -> None:
        for node in self.nodes.values():
            for det in node.detectors.values():
                det.stop_dark_counts()

