"""Role nodes: repeater / source / destination — hardware and hooks only, no protocol logic.

Every node carries ONE SFWM source, routed per emission attempt to an edge port.
Link memories track the attached edges by construction: one dedicated pair-half
memory per neighbor, bound when the topology wires the edge. Source/destination
add one extra unbound memory (the data / hold slot) in their constructors.
"""

import math
import numpy as np

from collections.abc import Callable

from qetwork.operations.gates import apply_gate, CNOT, H, X, Z, SWAP, rx, ry, rz
from qetwork.operations.measurement import measure

from qetwork.components.node import Node
from qetwork.components.port import Port
from qetwork.components.mem_qubit import MemoryQubit
from qetwork.components.sfwm import SFWMSource
from qetwork.components.beamsplitter import BeamSplitter
from qetwork.components.mzi import MZI
from qetwork.components.snspd import SNSPD
from qetwork.components.detector import TimeEnergyDetector


_GATE_DEFAULTS = {
    "p_depol_1q": 0.0,
    "p_depol_2q": 0.0,
    "p_depol_swap": 0.0,       # SWAP ~ 3 CNOTs; deserves its own incoherent knob
    "coherent_1q": {"axis": "z", "angle": 0.0},
    "coherent_2q": {"zz_angle": 0.0},
    "durations": {"gate_1q": 0, "gate_2q": 0, "measure": 0},   # ps, ints
}


def _build_detector(timeline, cfg) -> TimeEnergyDetector:
    if cfg["kind"] != "time-energy":
        raise ValueError(f"unknown detector kind {cfg['kind']!r}, expected 'time-energy'")
    m = cfg["mzi"]
    bs1 = BeamSplitter(reflectivity=m["bs1"]["reflectivity"], loss=m["bs1"]["loss"],
                       convention=m["bs1"]["convention"], band=tuple(m["bs1"]["band"]))
    bs2 = BeamSplitter(reflectivity=m["bs2"]["reflectivity"], loss=m["bs2"]["loss"],
                       convention=m["bs2"]["convention"], band=tuple(m["bs2"]["band"]))
    mzi = MZI(delta_t=m["delta_t"], phase=m["phase"], bs1=bs1, bs2=bs2,
              loss_short=m["loss_short"], loss_long=m["loss_long"],
              phase_error=m["phase_error"], band=tuple(m["band"]))
    snspds = [SNSPD(efficiency=s["efficiency"], jitter_fwhm=s["jitter_fwhm"],
                    dark_count_rate=s["dark_count_rate"], dead_time=s["dead_time"],
                    band=tuple(s["band"]))
              for s in (cfg["snspd_1"], cfg["snspd_2"])]
    return TimeEnergyDetector(timeline, mzi, snspds[0], snspds[1], coupling_1=cfg["coupling_1"], coupling_2=cfg["coupling_2"])


def _coherent_1q(cfg) -> np.ndarray | None:
    """Residual rotation after every ideal 1q gate; angle 0 -> None (skip)."""
    axis, angle = cfg["axis"], cfg["angle"]
    if axis not in ("x", "y", "z"):
        raise ValueError(f"coherent_1q axis must be x/y/z, got {axis!r}")
    if angle == 0:
        return None
    return {"x": rx, "y": ry, "z": rz}[axis](angle)


def _coherent_2q(cfg) -> np.ndarray | None:
    """Residual ZZ rotation exp(-i theta/2 Z@Z) after every ideal 2q gate; angle 0 -> None."""
    # CITE zz-crosstalk | always-on ZZ coupling as the canonical coherent two-qubit error | Krantz et al., Appl. Phys. Rev. 6, 021318 (2019)
    theta = cfg["zz_angle"]
    if theta == 0:
        return None
    e = np.exp(1j * theta / 2)
    return np.diag([e.conjugate(), e, e, e.conjugate()])   # basis order |00>,|01>,|10>,|11>



class RepeaterNode(Node):
    """A link station: per-edge memories, one routable pair source, gate-based BSM."""

    @property
    def bsm_duration(self) -> int:
        return self.t_gate_2q + self.t_gate_1q + 2 * self.t_measure

    @property
    def correct_duration(self) -> int:
        return 2 * self.t_gate_1q      # worst case; hardware allocates the slot regardless of bits

    def __init__(self, node_id: str, timeline,
                 t1: float = math.inf, t2: float = math.inf,
                 gate_cfg: dict | None = None,
                 source_cfg: dict | None = None,
                 memory_cfg: dict | None = None,
                 detector_cfg: dict | None = None) -> None:
        super().__init__(node_id, timeline)
        for name, t in (("t1", t1), ("t2", t2)):
            if t <= 0:
                raise ValueError(f"{name} must be positive, got {t}")
        unknown = sorted(set(gate_cfg or {}) - set(_GATE_DEFAULTS))
        if unknown:
            raise ValueError(f"unknown gate_cfg keys {unknown}")
        gates = {**_GATE_DEFAULTS, **(gate_cfg or {})}
        for name in ("p_depol_1q", "p_depol_2q", "p_depol_swap"):
            if not 0 <= gates[name] <= 1:
                raise ValueError(f"{name} must be in [0,1], got {gates[name]}")
        dur = gates["durations"]
        for name, d in dur.items():
            if not isinstance(d, int) or isinstance(d, bool) or d < 0:
                raise ValueError(f"duration {name!r} must be a non-negative int (ps), got {d!r}")
        self.t_gate_1q = dur["gate_1q"]
        self.t_gate_2q = dur["gate_2q"]
        self.t_measure = dur["measure"]

        self.t1 = t1
        self.t2 = t2
        for name, t in (("t1", t1), ("t2", t2)):
            if math.isnan(t) or t <= 0:
                raise ValueError(f"{name} must be positive (inf allowed), got {t}")

        self.p_depol_1q = gates["p_depol_1q"]
        self.p_depol_2q = gates["p_depol_2q"]
        self.p_depol_swap = gates["p_depol_swap"]
        self.coherent_1q = _coherent_1q(gates["coherent_1q"])
        self.coherent_2q = _coherent_2q(gates["coherent_2q"])
        self.source = SFWMSource(timeline, owner=self, **(source_cfg or {}))
        self.memory_cfg = dict(memory_cfg or {})
        self.memories: list[MemoryQubit] = []        # the node's full bank
        self.detectors: dict[str, TimeEnergyDetector] = {
            name: _build_detector(timeline, dcfg)
            for name, dcfg in (detector_cfg or {}).items()
        }

        self.link_mems: dict[str, MemoryQubit] = {}  # neighbor_id -> pair-half memory
        self._named_mems: dict[str, MemoryQubit] = {}     # protocol-owned working slots
        self.absorb_hooks: dict[str, Callable] = {}  # neighbor_id -> protocol hook, per edge
        self.on_absorb: Callable | None = None       # protocol hook: called with neighbor_id

    @property
    def purify_duration(self) -> int:
        return 2 * self.t_gate_1q + self.t_gate_2q + self.t_measure

    @property
    def move_duration(self) -> int:
        return 3 * self.t_gate_2q      # SWAP ~ 3 CNOTs, mirroring p_depol_swap

    @property
    def calibration_duration(self) -> int:
        return self.t_gate_1q

    # --- keep named slots out of the data_mem/hold_mem[0] path (the gotcha) ---
    def unbound_mems(self) -> list[MemoryQubit]:
        reserved = set(self.link_mems.values()) | set(self._named_mems.values())
        return [m for m in self.memories if m not in reserved]

    # --- protocol-allocated named memory (your requested capability) ---
    def add_memory(self, name: str) -> MemoryQubit:
        if name in self._named_mems:
            raise ValueError(f"node {self.node_id} already has a memory named {name!r}")
        mem = MemoryQubit(name, self.timeline, owner=self, t1=self.t1, t2=self.t2, **self.memory_cfg)
        self.memories.append(mem)
        self._named_mems[name] = mem
        return mem

    def memory(self, name: str) -> MemoryQubit:
        mem = self._named_mems.get(name)
        if mem is None:
            raise ValueError(f"node {self.node_id} has no memory named {name!r}")
        return mem

    def ensure_memory(self, name: str) -> MemoryQubit:
        """Get-or-create a protocol-owned named slot (idempotent across deliveries)."""
        mem = self._named_mems.get(name)
        return mem if mem is not None else self.add_memory(name)

    # --- noisy local move (generalizes DestinationNode.transfer_to_hold) ---
    def move(self, src: MemoryQubit, dst: MemoryQubit) -> None:
        """SWAP a qubit from src into empty dst, freeing src. Transfers entanglement."""
        if src.is_empty():   raise ValueError(f"source memory {src.name} is empty")
        if not dst.is_empty(): raise ValueError(f"dest memory {dst.name} is not empty")
        src.decohere()
        dst.initialize()                                  # |0> ancilla gets a key
        apply_gate(self.timeline.state_tracker, (src.key, dst.key), SWAP,
                self.p_depol_swap, coherent=self.coherent_2q)
        src.reset()                                       # trace out the emptied half
        
    # --- cancel a KNOWN source phase on one half of a raw pair (local knowledge:
    #     the emitter owns the source whose phase it corrects) ---
    def calibrate_phase(self, mem: MemoryQubit, phase: float) -> None:
        if mem.owner is not self:
            raise ValueError(f"calibrate_phase is local: {mem.name} not in node {self.node_id}")
        if mem.is_empty():
            raise ValueError(f"memory {mem.name} is empty; nothing to calibrate")
        mem.decohere()
        apply_gate(self.timeline.state_tracker, (mem.key,), rz(-phase),
                   self.p_depol_1q, coherent=self.coherent_1q)

    # --- one endpoint's half of a DEJMPS round (near-clone of bsm) ---
    def purify_local(self, kept: MemoryQubit, sac: MemoryQubit, sign: int, rotate: bool) -> int:
        if sign not in (+1, -1):
            raise ValueError(f"sign must be +1 (source) or -1 (dest), got {sign}")
        for mem in (kept, sac):
            if mem.owner is not self:
                raise ValueError(f"purify_local is local: {mem.name} not in node {self.node_id}")
            if mem.is_empty():
                raise ValueError(f"memory {mem.name} is empty; purify needs two loaded halves")
        kept.decohere(); sac.decohere()
        tracker = self.timeline.state_tracker
        if rotate:                                   # phase round; bit round skips the rotation
            theta = sign * (np.pi / 2)
            apply_gate(tracker, (kept.key,), rx(theta), self.p_depol_1q, coherent=self.coherent_1q)
            apply_gate(tracker, (sac.key,),  rx(theta), self.p_depol_1q, coherent=self.coherent_1q)
        apply_gate(tracker, (kept.key, sac.key), CNOT, self.p_depol_2q, coherent=self.coherent_2q)
        o = measure(tracker, sac.key, self.timeline.rng.random(), basis="Z")
        sac.reset()
        return 0 if o > 0 else 1


    def unrotate(self, kept: MemoryQubit, sign: int) -> None:
        """Undo the DEJMPS rotation on a surviving kept half (rx(-sign*pi/2))."""
        kept.decohere()
        apply_gate(self.timeline.state_tracker, (kept.key,),
                rx(-sign * np.pi / 2), self.p_depol_1q, coherent=self.coherent_1q)
        


    def _ensure_link_mem(self, neighbor_id: str) -> MemoryQubit:
        mem = self.link_mems.get(neighbor_id)
        if mem is None:
            mem = MemoryQubit(f"{self.node_id}:mem:{neighbor_id}", self.timeline,
                              owner=self, t1=self.t1, t2=self.t2, **self.memory_cfg)
            self.memories.append(mem)
            self.link_mems[neighbor_id] = mem
        return mem

    # --- wiring verb (called by the topology layer, once per incident edge) ---

    def add_edge_port(self, neighbor_id: str) -> Port:
        """Bind this edge's memory and open its port: outgoing fiber attaches here,
        incoming photons absorb into the same memory."""
        mem = self._ensure_link_mem(neighbor_id)
        port = self.add_port(f"q:{neighbor_id}")

        def absorb(photon) -> None:
            mem.get(photon)
            hook = self.absorb_hooks.get(neighbor_id)
            if hook is not None:
                hook(neighbor_id)

        port.attach(absorb)
        return port

    # --- primitive verbs (driven by protocols later) ---

    def attempt_link(self, neighbor_id: str) -> tuple[int, int]:
        """One on-demand pair emission along an edge; retry policy is the protocol's job."""
        port = self.ports.get(f"q:{neighbor_id}")
        if port is None:
            raise ValueError(f"node {self.node_id} has no edge toward {neighbor_id!r}")
        mem = self.link_mems[neighbor_id]
        if not mem.is_empty():
            raise ValueError(f"link memory toward {neighbor_id!r} is occupied; reset it before a new attempt")
        return self.source.emit(signal_to=port.send, idler_to=mem.get)

    def bsm(self, mem_ctrl: MemoryQubit, mem_tgt: MemoryQubit) -> tuple[int, int]:
        """Gate-based Bell measurement on two local memories; consumes both.

        Returns bits (m1, m2); the surviving far half needs X^m2 then Z^m1."""
        # CITE teleport | BSM outcomes (m1,m2) fix the far half up to X^m2 Z^m1 | Bennett et al., PRL 70, 1895 (1993)
        if mem_ctrl is mem_tgt:
            raise ValueError("bsm needs two distinct memories")
        for mem in (mem_ctrl, mem_tgt):
            if mem.owner is not self:
                raise ValueError(f"bsm is a local operation: memory {mem.name} does not sit in node {self.node_id}")
            if mem.is_empty():
                raise ValueError(f"memory {mem.name} is empty; bsm needs two loaded memories")
        mem_ctrl.decohere()
        mem_tgt.decohere()
        tracker = self.timeline.state_tracker
        rng = self.timeline.rng
        apply_gate(tracker, (mem_ctrl.key, mem_tgt.key), CNOT,
                   self.p_depol_2q, coherent=self.coherent_2q)   # ctrl first: key order IS control/target
        apply_gate(tracker, (mem_ctrl.key,), H, self.p_depol_1q, coherent=self.coherent_1q)
        o1 = measure(tracker, mem_ctrl.key, rng.random(), basis="Z")
        o2 = measure(tracker, mem_tgt.key, rng.random(), basis="Z")

        mem_ctrl.reset()   # measured keys are lone collapsed states; reset() -> discard removes them
        mem_tgt.reset()
        return (0 if o1 > 0 else 1), (0 if o2 > 0 else 1)

    def correct(self, mem: MemoryQubit, m1: int, m2: int) -> None:
        """Pauli-correct the surviving half held in a local memory: X if m2, then Z if m1."""
        for name, bit in (("m1", m1), ("m2", m2)):
            if bit not in (0, 1):
                raise ValueError(f"{name} must be a classical bit 0/1, got {bit!r}")
        if mem.owner is not self:
            raise ValueError(f"correct is a local operation: memory {mem.name} does not sit in node {self.node_id}")
        if mem.is_empty():
            raise ValueError(f"memory {mem.name} is empty; nothing to correct")
        mem.decohere()
        tracker = self.timeline.state_tracker
        if m2:
            apply_gate(tracker, (mem.key,), X, self.p_depol_1q, coherent=self.coherent_1q)
        if m1:
            apply_gate(tracker, (mem.key,), Z, self.p_depol_1q, coherent=self.coherent_1q)



class SourceNode(RepeaterNode):
    """End node that injects |psi>: repeater hardware plus one data slot."""

    def __init__(self, node_id: str, timeline, **kwargs) -> None:
        super().__init__(node_id, timeline, **kwargs)
        self.memories.append(MemoryQubit(f"{node_id}:hold", timeline, owner=self,
                                         t1=self.t1, t2=self.t2, **self.memory_cfg))


    @property
    def data_mem(self) -> MemoryQubit:
        free = self.unbound_mems()
        if not free:
            raise ValueError(f"node {self.node_id} has no free data slot")
        return free[0]


class DestinationNode(RepeaterNode):
    """End node that receives |psi>: repeater hardware plus one hold slot."""

    def __init__(self, node_id: str, timeline, **kwargs) -> None:
        super().__init__(node_id, timeline, **kwargs)
        self.memories.append(MemoryQubit(f"{node_id}:hold", timeline,
                                         owner=self, t1=self.t1, t2=self.t2))
        
    
    def transfer_to_hold(self, neighbor_id: str) -> MemoryQubit:
        """Noisy local SWAP moving the received state off the link memory into an empty hold slot."""
        link = self.link_mems.get(neighbor_id)
        if link is None:
            raise ValueError(f"node {self.node_id} has no link memory bound for {neighbor_id!r}")
        if link.is_empty():
            raise ValueError(f"link memory for {neighbor_id!r} is empty; nothing to transfer")
        holds = [m for m in self.unbound_mems() if m.is_empty()]
        if not holds:
            raise ValueError(f"node {self.node_id} has no empty hold slot for the transfer")
        hold = holds[0]
        link.decohere()
        hold.initialize()   # |0> ancilla gets a fresh key
        apply_gate(self.timeline.state_tracker, (link.key, hold.key), SWAP,
                   self.p_depol_swap, coherent=self.coherent_2q)
        # SWAP merged the two keys into one cluster; reset() -> discard traces the link key
        # out, leaving the hold slot the correct reduced rho even under noise.
        link.reset()
        return hold


    @property
    def hold_mem(self) -> MemoryQubit:
        free = self.unbound_mems()
        if not free:
            raise ValueError(f"node {self.node_id} has no free hold slot")
        return free[0]
