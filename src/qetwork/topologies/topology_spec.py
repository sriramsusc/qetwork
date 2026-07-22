"""TopologySpec: load, validate, and materialize fully-resolved topology files.

Strict both ways: a missing parameter and an unknown parameter are both errors,
so files can neither under-specify a component nor hide a typo. Structural checks
live here; physical range checks stay in the components' own constructors."""

import json
import math

import networkx as nx

_SCHEMA = "qetwork-topology/5"

_TOP_KEYS     = {"schema", "name", "provenance", "network", "roles", "nodes", "edges"}
_NETWORK_KEYS = {"cfiber_latency", "cfiber_n"}
_ROLES_KEYS   = {"source", "destination"}
_NODE_KEYS  = {"coord", "t1", "t2", "gates", "source", "memory", "detectors"}
_DET_KEYS   = {"kind", "coupling_1", "coupling_2", "mzi", "snspd_1", "snspd_2"}
_MZI_KEYS   = {"delta_t", "phase", "phase_error", "loss_short", "loss_long", "band", "bs1", "bs2"}
_BS_KEYS    = {"reflectivity", "loss", "convention", "band"}
_SNSPD_KEYS = {"efficiency", "jitter_fwhm", "dark_count_rate", "dead_time", "band"}
_GATES_KEYS  = {"p_depol_1q", "p_depol_2q", "coherent_1q", "coherent_2q", "durations"}
_COH1_KEYS   = {"axis", "angle"}
_COH2_KEYS   = {"zz_angle"}
_DURATIONS_KEYS = {"gate_1q", "gate_2q", "measure"}
_SOURCE_KEYS = {"signal_wavelength", "idler_wavelength", "visibility", "phase", "encoding"}
_MEMORY_KEYS  = {"emission_encoding", "emission_wavelength"}
_EDGE_KEYS    = {"u", "v", "length", "attenuation", "insertion_loss_db", "n"}


def _check_keys(mapping, required: set, ctx: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{ctx} must be a mapping, got {type(mapping).__name__}")
    missing = sorted(required - mapping.keys())
    if missing:
        raise ValueError(f"{ctx} is missing required keys {missing}")
    unknown = sorted(mapping.keys() - required)
    if unknown:
        raise ValueError(f"{ctx} has unknown keys {unknown}")


def _parse_time(value, ctx: str) -> float:
    """'inf' -> math.inf; a plain number stays itself. Strict JSON has no Infinity."""
    if value == "inf":
        return math.inf
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"{ctx} must be a number or 'inf', got {value!r}")


class TopologySpec:
    def __init__(self, data: dict) -> None:
        _check_keys(data, _TOP_KEYS, "topology file")
        if data["schema"] != _SCHEMA:
            raise ValueError(f"unsupported schema {data['schema']!r}, expected {_SCHEMA!r}")
        _check_keys(data["network"], _NETWORK_KEYS, "network block")
        _check_keys(data["roles"], _ROLES_KEYS, "roles block")

        self.name = data["name"]
        self.provenance = dict(data["provenance"])   # informational only, never validated further
        self.network = dict(data["network"])
        self.roles = dict(data["roles"])

        if len(data["nodes"]) < 2:
            raise ValueError(f"topology needs at least 2 nodes, got {len(data['nodes'])}")
        self.nodes: dict[str, dict] = {}
        for nid, entry in data["nodes"].items():
            _check_keys(entry, _NODE_KEYS, f"node {nid!r}")
            _check_keys(entry["source"], _SOURCE_KEYS, f"node {nid!r} source")
            _check_keys(entry["memory"], _MEMORY_KEYS, f"node {nid!r} memory")
            _check_keys(entry["gates"], _GATES_KEYS, f"node {nid!r} gates")
            _check_keys(entry["gates"]["coherent_1q"], _COH1_KEYS, f"node {nid!r} gates coherent_1q")
            _check_keys(entry["gates"]["coherent_2q"], _COH2_KEYS, f"node {nid!r} gates coherent_2q")
            _check_keys(entry["gates"]["durations"], _DURATIONS_KEYS, f"node {nid!r} gates durations")

            for dname, det in entry["detectors"].items():
                ctx = f"node {nid!r} detector {dname!r}"
                _check_keys(det, _DET_KEYS, ctx)
                if det["kind"] != "time-energy":
                    raise ValueError(f"{ctx} has unknown kind {det['kind']!r}")
                _check_keys(det["mzi"], _MZI_KEYS, f"{ctx} mzi")
                _check_keys(det["mzi"]["bs1"], _BS_KEYS, f"{ctx} mzi bs1")
                _check_keys(det["mzi"]["bs2"], _BS_KEYS, f"{ctx} mzi bs2")
                _check_keys(det["snspd_1"], _SNSPD_KEYS, f"{ctx} snspd_1")
                _check_keys(det["snspd_2"], _SNSPD_KEYS, f"{ctx} snspd_2")

            parsed = dict(entry)
            parsed["t1"] = _parse_time(entry["t1"], f"node {nid!r} t1")
            parsed["t2"] = _parse_time(entry["t2"], f"node {nid!r} t2")
            self.nodes[nid] = parsed

        self.edges: dict[str, dict] = {}
        seen_pairs = set()
        for ename, entry in data["edges"].items():
            _check_keys(entry, _EDGE_KEYS, f"edge {ename!r}")
            u, v = entry["u"], entry["v"]
            if isinstance(entry["length"], bool) or not isinstance(entry["length"], (int, float)) or not math.isfinite(entry["length"]) or entry["length"] < 0: raise
            for endpoint in (u, v):
                if endpoint not in self.nodes:
                    raise ValueError(f"edge {ename!r} references unknown node {endpoint!r}")
            if u == v:
                raise ValueError(f"edge {ename!r} is a self-loop on {u!r}")
            pair = frozenset((u, v))
            if pair in seen_pairs:
                raise ValueError(f"edge {ename!r} duplicates an edge between {u!r} and {v!r}")
            seen_pairs.add(pair)
            length = entry["length"]
            if (isinstance(length, bool) or not isinstance(length, (int, float))
                    or not math.isfinite(length) or length < 0):
                raise ValueError(f"edge {ename!r} length must be a finite non-negative number, got {length!r}")

            self.edges[ename] = dict(entry)

        for role, nid in self.roles.items():
            if nid not in self.nodes:
                raise ValueError(f"role {role!r} names unknown node {nid!r}")
        if self.roles["source"] == self.roles["destination"]:
            raise ValueError("source and destination must be different nodes")
        if not nx.is_connected(self.graph()):
            raise ValueError("topology must be connected: classical distances are undefined otherwise")

    @classmethod
    def from_json(cls, path) -> "TopologySpec":
        with open(path) as f:
            return cls(json.load(f))

    def to_dict(self) -> dict:
        nodes = {}
        for nid, e in self.nodes.items():
            out = dict(e)
            out["t1"] = "inf" if math.isinf(e["t1"]) else e["t1"]
            out["t2"] = "inf" if math.isinf(e["t2"]) else e["t2"]
            nodes[nid] = out
        return {"schema": _SCHEMA, "name": self.name, "provenance": self.provenance,
                "network": self.network, "roles": self.roles,
                "nodes": nodes, "edges": self.edges}

    def to_json(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, allow_nan=False)

    def graph(self) -> nx.Graph:
        G = nx.Graph()
        for nid, entry in self.nodes.items():
            G.add_node(nid, coord=entry["coord"])
        for ename, e in self.edges.items():
            G.add_edge(e["u"], e["v"], name=ename, length=e["length"])
        return G

    def materialize(self, timeline):
        from qetwork.topologies.network_topology import QuantumNetwork   # local: avoids import cycle
        return QuantumNetwork(self, timeline)
