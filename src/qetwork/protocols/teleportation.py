"""Teleportation over a distributed Bell pair: one extra BSM at the source + a Pauli correction.

Teleportation is entanglement swapping with a data state as one input. This runs an
EntanglementDistribution (sequential or parallel -- caller's choice) to deliver an
end-to-end pair, then the source Bell-measures its data qubit against its half of the pair
and ships the two bits to the destination, which applies X^m2 Z^m1 to land |psi> in its
memory. Reuses the RepeaterNode verbs (bsm / correct) and the classical layer; no new physics.
"""

from dataclasses import dataclass

import numpy as np

from qetwork.components.utils.roles import SourceNode
from qetwork.protocols.e_dist_swap import EntanglementDistribution, DistributionResult, SEQUENTIAL, PARALLEL
from qetwork.protocols.e_purification import deliver, DeliverySpec

TELEPORT_RESULT = "teleport_result"   # source -> destination: BSM byproduct bits (m1, m2)


def _density(psi) -> np.ndarray:
    """A single-qubit state as a density matrix. Accepts a length-2 statevector (normalized
    into |psi><psi|) or a 2x2 density matrix (passed through; validated on load)."""
    a = np.asarray(psi, dtype=complex)
    if a.shape == (2,):
        a = a / np.linalg.norm(a)
        return np.outer(a, a.conj())
    if a.shape == (2, 2):
        return a
    raise ValueError(f"psi must be a length-2 statevector or 2x2 density matrix, got shape {a.shape}")


@dataclass(slots=True, kw_only=True)
class TeleportResult:
    """The teleported qubit: where it landed, how faithfully, and the pair it consumed."""
    dest_key: int
    fidelity: float
    latency: int
    distribution: DistributionResult | None = None      # None for LINK-level delivery



class Teleportation:
    def __init__(self, net, path=None, spec=None, psi=None):
        self.net = net; self.timeline = net.timeline
        self.spec = spec or DeliverySpec()
        self.path = path if path is not None else net.path()
        src = net.nodes[self.path[0]]
        if not isinstance(src, SourceNode):
            raise ValueError(f"teleportation needs a data slot at {self.path[0]!r}")
        self.psi = _density(psi if psi is not None else np.array([1.0, 1.0]))
        self.delivery = None; self.result = None
        net.nodes[self.path[-1]].register_handler(TELEPORT_RESULT, self._on_teleport_result, replace=True)

    def run(self):
        self.delivery = deliver(self.net, self.path, self.spec)   # distribute (+purify)
        self._teleport()
        self.timeline.run()
        return self.result

    def _teleport(self):
        src = self.net.nodes[self.path[0]]
        data = src.data_mem                              # unbound_mems() fix keeps this correct
        data.initialize(self.psi)
        m1, m2 = src.bsm(data, self.delivery.source_mem)
        src.send_to(self.path[-1], TELEPORT_RESULT, m1=m1, m2=m2)

    def _on_teleport_result(self, message):
        dest = self.net.nodes[self.path[-1]]
        dest.correct(self.delivery.dest_mem, message.payload["m1"], message.payload["m2"])
        self.result = TeleportResult(
            dest_key=self.delivery.dest_mem.key,
            fidelity=state_fidelity(self.timeline.state_tracker, self.delivery.dest_mem.key, self.psi),
            latency=self.timeline.now(), distribution=self.delivery.distribution)


def state_fidelity(tracker, key: int, psi) -> float:
    """<psi|rho|psi> for the lone single-qubit state at `key` (Tr(|psi><psi| @ rho))."""
    state = tracker.get(key)
    if state.keys != (key,):
        raise ValueError(f"key {key} is not a lone single-qubit state (covers {state.keys})")
    return float(np.trace(_density(psi) @ state.matrix).real)
