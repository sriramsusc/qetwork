"""Memory qubit: a qubit that stores quantum state with finite coherence, noise will be applied through events as it involves density manipulation"""
# Customizable:
# CITE yb-coherence | 171Yb+ hyperfine qubit: T2≈5500 s, T1 effectively unbounded | P. Wang et al., "Single 171Yb+ ion qubit with estimated coherence time exceeding one hour", Nat. Commun. 12, 233 (2021)

import math

from qetwork.components.photon import Photon
from qetwork.operations.err_channels import T1, T2
from qetwork.components.node import Node
from qetwork.operations.measurement import discard
from qetwork.components.utils.encoding import TIME_BIN, validate



class MemoryQubit:
    def __init__(self, name:str, timeline, owner: Node | None = None, 
                 t1: float = math.inf, t2: float = math.inf,
                 emission_encoding: str = TIME_BIN,
                 emission_wavelength: float |None = None) -> None:
        if owner is None:
            raise ValueError(f"Memory qubit{name} requires an owner to sit in")
        if not isinstance(owner, Node):
            raise TypeError(f"owner must be a node, got {type(owner).__name__}")
        if owner.timeline is not timeline:
            raise ValueError(f"memory {name}'s owner {owner.node_id} lives on a different "
                             f"timeline - state keys are per tracker and would silently alias")
        
        self.name = name
        self.timeline = timeline
        self.owner = owner
        self.t1 = t1
        self.t2 = t2
        self.key: int | None = None
        self._load_time: int |None = None
        self.emission_encoding = validate(emission_encoding)
        self.emission_wavelength = emission_wavelength
        # CITE t2-le-2t1 | physical bound T2 <= 2*T1, from 1/T2 = 1/(2 T1) + 1/Tphi with Tphi >= 0 | Krantz et al., "A Quantum Engineer's Guide to Superconducting Qubits", Appl. Phys. Rev. 6, 021318 (2019), §II.C
        if math.isnan(t1) or t1 <= 0:
            raise ValueError(f"T1 must be positive (inf allowed), got {t1}")
        if math.isnan(t2) or t2 <= 0:
            raise ValueError(f"T2 must be positive (inf allowed), got {t2}")
        if t2 > 2 * t1:
            raise ValueError(f"unphysical coherence times: T2={t2} exceeds 2*T1={2 * t1} (from 1/T2 = 1/(2 T1) + 1/Tphi with Tphi >= 0)")


    def _require_empty(self) -> None:
        if not self.is_empty():
            raise ValueError(f"memory {self.name} already holds a qubit and is not empty")

    def initialize(self, state=None) -> None:
        self._require_empty()
        self.key = self.timeline.state_tracker.new(state)
        self._load_time = self.timeline.now()
    
    def get(self, photon: Photon) -> None:
        # CITE no-cloning | a quantum state has exactly one holder — keys transfer, never copy | Wootters & Zurek, "A single quantum cannot be cloned", Nature 299, 802-803 (1982)
        if photon.key is None:
            raise ValueError("cannot absorb photon that carries no quantum state")
        if photon.timeline is not self.timeline:
            raise ValueError(f"memory {self.name} received a photon from a different simulation; "
                             f"its key would alias an unrelated local state")
        self._require_empty()
        self._load_time = self.timeline.now()
        self.key = photon.take_key()

    def emit(self, port = None) -> Photon:
        if self.is_empty():
            raise ValueError(f"memory {self.name} is empty, nothing to emit")
        self.decohere() 
        photon = Photon(key=self.key, encoding=self.emission_encoding,
                        wavelength=self.emission_wavelength, timeline=self.timeline)
        self.key = None
        self._load_time = None
        if port is not None:
            port.send(photon)
        return photon

    def is_empty(self) -> bool:
        return self.key is None
    
    def _t2_star(self) -> float:
        """Pure dephasing time: 1/T2* = 1/T2 - 1/(2*T1).

        T1 already decays coherences by e^(-t/2T1); feeding T2 straight into the
        dephasing channel counts that twice. T2* = T2 exactly when T1 = inf.
        """
        if math.isinf(self.t1):
            return self.t2
        inv = 1.0 / self.t2 - 1.0 / (2.0 * self.t1)
        return math.inf if inv <= 0 else 1.0 / inv
    
    def decohere(self) -> None:
        if self.is_empty():
            return
        elapsed = self.timeline.now() - self._load_time
        tracker = self.timeline.state_tracker
        T1(tracker, (self.key,), elapsed, self.t1)
        T2(tracker, (self.key,), elapsed, self._t2_star())
        self._load_time = self.timeline.now()

    def reset(self) -> None:
        """Clear the memory, tracing out its qubit if one is held.

        Used when a link attempt fails: the local half was loaded but the remote
        half never arrived, so this qubit must be discarded and the slot freed.
        A no-op on an already-empty memory (idempotent — safe to call on timeout
        regardless of whether the ack already cleared it).
        """
        if self.is_empty():
            return
        discard(self.timeline.state_tracker, self.key)
        self.key = None
        self._load_time = None
