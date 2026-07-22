"""Entanglement distribution over a linear path: per-node agents, message-passing only.

Design: docs/e_distribution_ir.md. One RepeaterProtocol agent per node on the path.
Both modes share the same agent; they differ only in *when* a node first fires its
downstream link and in the retry cadence:

  * sequential -- only the source fires at t=0; each node fires its downstream link when
    it absorbs its upstream half (the absorb->emit baton), and retries as soon as it
    knows a link failed (one round trip later).
  * parallel   -- every emitter fires at t=0 (round 1); failed links retry together on a
    shared round grid (the slowest edge's round trip) until all links are up.

Swaps are as-ready in both modes: an interior swaps the instant its own two links are up.
Corrections defer to the destination as one Pauli frame (XOR of the BSM bits). The
deliverable is a bare end-to-end Bell pair held in the source and destination link
memories. Agents drive the RepeaterNode verbs (attempt_link / on_absorb / bsm / correct)
and the classical layer (send_to / register_handler); no density-matrix math lives here.
"""

from dataclasses import dataclass

import numpy as np

from qetwork.protocols.errors import ProtocolError
from qetwork.events.priority import PROTOCOL

SEQUENTIAL = "sequential"
PARALLEL   = "parallel"

LINK_ACK    = "link_ack"      # absorber -> emitter: "your downstream half landed"
SWAP_RESULT = "swap_result"   # interior -> destination: BSM byproduct bits (m1, m2)


@dataclass(slots=True)
class DistributionResult:
    """The delivered end-to-end pair and the effort it took."""
    source_key: int
    dest_key: int
    latency: int                    # ps from t=0 to the destination's correction
    attempts: dict[str, int]        # emitter node_id -> number of link attempts it made
    rounds: int                     # max attempts over all edges (parallel: synchronized rounds run)

def validate_path(net, path) -> list[str]:
    """Shared end-to-end path validation: real nodes, real edges, no revisits,
    endpoints matching the network's source/destination roles."""
    if len(path) < 2:
        raise ValueError(f"path needs at least 2 nodes, got {path}")
    missing = [nid for nid in path if nid not in net.nodes]
    if missing:
        raise ValueError(f"path references unknown nodes {missing}")
    if len(set(path)) != len(path):
        raise ValueError(f"path revisits a node: {path}")
    for a, b in zip(path, path[1:]):
        if not net.graph.has_edge(a, b):
            raise ValueError(f"path hop {a!r}->{b!r} has no quantum edge")
    if path[0] != net.source_id or path[-1] != net.dest_id:
        raise ValueError(f"path endpoints ({path[0]!r}, {path[-1]!r}) do not match "
                         f"network roles ({net.source_id!r}, {net.dest_id!r})")
    return path


class RepeaterProtocol:
    """One node's agent. Reacts only to its local on_absorb hook and classical messages."""

    SOURCE, INTERIOR, DEST = "source", "interior", "dest"

    def __init__(self, node, role, up_id, down_id, dest_id, num_edges, mode, on_done=None):
        self.node = node
        self.timeline = node.timeline
        self.role = role
        self.up_id = up_id
        self.down_id = down_id
        self.dest_id = dest_id
        self.num_edges = num_edges
        self.mode = mode
        self.on_done = on_done

        self.rtt = self._round_trip() if down_id is not None else None
        self.round_period: int | None = None   # parallel retry grid; set by the coordinator
        self.attempts = 0

        self.up_ready = False
        self.down_ready = False
        self.swapped = False
        self.done = False
        self.down_gen = 0

        self.acc_m1 = 0
        self.acc_m2 = 0
        self.results = 0

    # --- wiring ---

    def install(self) -> None:
        """Attach this agent's local hook and message handlers to its node."""
        if self.up_id is not None:                      # only absorbers watch an edge
            self.node.absorb_hooks[self.up_id] = self.on_absorb
        if self.down_id is not None:                    # source + interiors emit -> receive ACKs
            self.node.register_handler(LINK_ACK, self.handle_ack, replace=True)
        if self.role == self.DEST:
            self.node.register_handler(SWAP_RESULT, self.handle_swap_result, replace=True)


    def _round_trip(self) -> int:
        """qdelay(downstream photon) + cdelay(ACK back). Classical distance is symmetric,
        so the reverse ACK delay equals this node's own outgoing classical delay."""
        qdelay = self.node.ports[f"q:{self.down_id}"].qfiber.delay
        cfiber = self.node.cfibers[self.down_id]
        return qdelay + cfiber.delay + cfiber.latency

    # --- emitter side: downstream link generation + retry ---

    def attempt_downstream(self) -> None:
        """Emit one pair on the downstream edge and arm the retry check.

        The retry cadence is the whole sequential/parallel difference: sequential re-fires
        as soon as it knows it failed (rtt+1); parallel snaps every retry to the shared
        round grid (round_period) so all failed edges re-attempt in lockstep.
        """
        self.node.link_mems[self.down_id].reset()       # clear a stale idler on retry (no-op if empty)
        self.down_gen += 1
        self.down_ready = False
        self.attempts += 1
        self.node.attempt_link(self.down_id)
        delay = self.round_period if self.mode == PARALLEL else self.rtt + 1
        self.timeline.schedule(self._retry_if_failed, self.down_gen,
                               at=self.timeline.now() + delay, priority=PROTOCOL)

    def _retry_if_failed(self, gen: int) -> None:
        """One round-trip (sequential) or one round (parallel) after an attempt: if the
        link never heralded, re-emit; otherwise self-cancel via the gen / down_ready guard."""
        if gen != self.down_gen or self.down_ready:
            return                                       # stale, or already heralded
        self.attempt_downstream()                        # photon was lost: reset + retry

    def handle_ack(self, message) -> None:
        if self.down_ready:
            return
        self.down_ready = True
        if self.role == self.INTERIOR:
            self.try_swap()

    # --- absorber side: upstream herald ---

    def on_absorb(self, neighbor: str) -> None:
        if neighbor != self.up_id:
            return                                       # a different (off-path) edge on this node
        self.up_ready = True
        self.node.send_to(self.up_id, LINK_ACK)
        if self.role == self.INTERIOR:
            if self.mode == SEQUENTIAL:
                self.attempt_downstream()                # sequential baton: absorb -> emit next hop
            self.try_swap()                              # parallel already emitted at t=0 (round grid)
        elif self.role == self.DEST:
            self.try_complete()

    # --- interior: entanglement swap (as-ready in both modes) ---

    def try_swap(self) -> None:
        if self.swapped or not (self.up_ready and self.down_ready):
            return
        self.swapped = True
        m1, m2 = self.node.bsm(self.node.link_mems[self.up_id],
                               self.node.link_mems[self.down_id])
        self.timeline.schedule(self.node.send_to, self.dest_id, SWAP_RESULT,
                               at=self.timeline.now() + self.node.bsm_duration,
                               priority=PROTOCOL, m1=m1, m2=m2)


    # --- destination: collect corrections + finish ---

    def handle_swap_result(self, message) -> None:
        self.acc_m1 ^= message.payload["m1"]
        self.acc_m2 ^= message.payload["m2"]
        self.results += 1
        self.try_complete()

    def try_complete(self) -> None:
        if self.done or not self.up_ready or self.results < self.num_edges - 1:
            return
        self.done = True
        delay = 0
        if self.num_edges > 1:
            self.node.correct(self.node.link_mems[self.up_id], self.acc_m1, self.acc_m2)
            delay = self.node.correct_duration
        if self.on_done is not None:
            if delay:
                self.timeline.schedule(self.on_done, at=self.timeline.now() + delay, priority=PROTOCOL)
            else:
                self.on_done()



class EntanglementDistribution:
    """Builds one agent per path node, kicks off, and collects the delivered pair.

    Sequential vs parallel is one knob. Sequential fires only the source at t=0 and lets
    the absorb->emit baton walk the chain, retrying each edge on its own round trip.
    Parallel fires every emitter at t=0 and retries failed links together on a shared round
    grid until all are up. Swaps are as-ready in both.
    """

    def __init__(self, net, path=None, mode=SEQUENTIAL):
        self.net = net
        self.timeline = net.timeline
        self.path = validate_path(net, path if path is not None else net.path())
        self.mode = mode
        if mode not in (SEQUENTIAL, PARALLEL):
            raise ValueError(f"mode must be {SEQUENTIAL!r} or {PARALLEL!r}, got {mode!r}")
        missing = [nid for nid in self.path if nid not in net.nodes]
        if missing:
            raise ValueError(f"path references unknown nodes {missing}")
        if len(set(self.path)) != len(self.path):
            raise ValueError(f"path revisits a node: {self.path}")
        for a, b in zip(self.path, self.path[1:]):
            if not net.graph.has_edge(a, b):
                raise ValueError(f"path hop {a!r}->{b!r} has no quantum edge")
        if self.path[0] != net.source_id or self.path[-1] != net.dest_id:
            raise ValueError(f"path endpoints ({self.path[0]!r}, {self.path[-1]!r}) do not match "
                             f"network roles ({net.source_id!r}, {net.dest_id!r})")

        self.num_edges = len(self.path) - 1
        self.result: DistributionResult | None = None

        # pass 1: one agent per node on the path
        k = self.num_edges
        dest_id = self.path[-1]
        self.agents: dict[str, RepeaterProtocol] = {}
        for i, nid in enumerate(self.path):
            role = (RepeaterProtocol.SOURCE if i == 0
                    else RepeaterProtocol.DEST if i == k
                    else RepeaterProtocol.INTERIOR)
            up_id   = self.path[i - 1] if i > 0 else None
            down_id = self.path[i + 1] if i < k else None
            self.agents[nid] = RepeaterProtocol(net.nodes[nid], role, up_id, down_id, dest_id,
                                                self.num_edges, self.mode, on_done=self._finish)

        # pass 2: shared round period (the slowest edge's round trip, +1 so every herald is
        # in before the next round fires) + wire each agent's hook/handlers
        rtts = [a.rtt for a in self.agents.values() if a.rtt is not None]
        self.round_period = (max(rtts) + 1) if rtts else 0
        for a in self.agents.values():
            if self.mode == PARALLEL:
                a.round_period = self.round_period
            a.install()

    def start(self) -> None:
        """Kickoff: sequential fires only the source; parallel fires every emitter (round 1)."""
        starters = [self.path[0]] if self.mode == SEQUENTIAL else self.path[:-1]
        now = self.timeline.now()
        for nid in starters:
            self.timeline.schedule(self.agents[nid].attempt_downstream, at=now, priority=PROTOCOL)

    def run(self) -> DistributionResult | None:
        self.start()
        while self.result is None and self.timeline.step():
            pass
        return self.result
    
    def _finish(self) -> None:
        """Called once the destination has corrected. Settle both endpoint halves'
        decoherence up to now -- the source half is never consumed by a swap, so nothing
        else would (single-qubit T1/T2 commutes with the swaps that happened meanwhile) --
        then record the delivered pair."""
        source = self.net.nodes[self.path[0]]
        dest   = self.net.nodes[self.path[-1]]
        source.link_mems[self.path[1]].decohere()        # source half: waited the whole run
        dest.link_mems[self.path[-2]].decohere()         # dest half: already settled by correct(), no-op
        attempts = {nid: a.attempts for nid, a in self.agents.items() if a.down_id is not None}
        self.result = DistributionResult(
            source_key=source.link_mems[self.path[1]].key,
            dest_key=dest.link_mems[self.path[-2]].key,
            latency=self.timeline.now(),
            attempts=attempts,
            rounds=max(attempts.values(), default=0),
        )

    def reset(self) -> None:
        """Re-arm every agent for another pair over the same path (handlers stay)."""
        self.result = None
        for a in self.agents.values():
            a.up_ready = a.down_ready = a.swapped = a.done = False
            a.down_gen = a.acc_m1 = a.acc_m2 = a.results = a.attempts = 0
        for a, b in zip(self.path, self.path[1:]):      # only THIS path's edge memories
            self.net.nodes[a].link_mems[b].reset()
            self.net.nodes[b].link_mems[a].reset()

class LinkGeneration:
    """Heralded raw-pair production on ONE directed edge: emitter A drives, absorber B
    heralds, retry-until-success. One pair per arm(); reusable for successive pairs.

    Ordering contract (I1): _on_absorb dispatches the LINK_ACK BEFORE invoking
    on_b_ready, so any message that callback sends on the same tick reaches A after
    the ACK (same fiber, FIFO tie-break). Race-freedom: a successful attempt's ACK
    lands at t_emit + rtt, strictly before its timeout at t_emit + rtt + 1.
    """

    def __init__(self, net, a_id, b_id, on_b_ready=None, on_a_ready=None):
        if not net.graph.has_edge(a_id, b_id):
            raise ValueError(f"no quantum edge between {a_id!r} and {b_id!r}")
        self.net = net
        self.timeline = net.timeline
        self.a_id, self.b_id = a_id, b_id
        self.A = net.nodes[a_id]
        self.B = net.nodes[b_id]
        self.on_b_ready = on_b_ready
        self.on_a_ready = on_a_ready
        qd = self.A.ports[f"q:{b_id}"].qfiber.delay
        cf = self.A.cfibers[b_id]                       # classical distance is symmetric
        self.rtt = qd + cf.delay + cf.latency
        self.gen = 0
        self.acked = False
        self.pending = False
        self.attempts = 0                               # emissions, incl. lost photons
        self.pairs = 0                                  # heralded pairs

    def install(self) -> None:
        self.A.register_handler(LINK_ACK, self._handle_ack, replace=True)
        self.B.absorb_hooks[self.a_id] = self._on_absorb

    def uninstall(self) -> None:
        self.A.unregister_handler(LINK_ACK, self._handle_ack)
        if self.B.absorb_hooks.get(self.a_id) == self._on_absorb:   # ==, not is: bound methods
            del self.B.absorb_hooks[self.a_id]

    def arm(self) -> None:
        """Begin producing ONE pair; completes at A's LINK_ACK (on_a_ready)."""
        if self.pending:
            raise ProtocolError(f"link {self.a_id}->{self.b_id}: arm() while a pair is pending")
        self.pending = True
        self.acked = False
        self._attempt()

    def _attempt(self) -> None:
        self.A.link_mems[self.b_id].reset()             # clear a lost attempt's idler; no-op first time
        self.gen += 1
        self.attempts += 1
        self.A.attempt_link(self.b_id)
        self.timeline.schedule(self._timeout, self.gen,
                               at=self.timeline.now() + self.rtt + 1, priority=PROTOCOL)

    def _timeout(self, gen: int) -> None:
        if gen != self.gen or self.acked:
            return                                      # stale, or already heralded
        self._attempt()

    def _on_absorb(self, neighbor: str) -> None:
        self.pairs += 1
        self.B.send_to(self.a_id, LINK_ACK)             # ACK FIRST -- the I1 contract
        if self.on_b_ready is not None:
            self.on_b_ready()

    def _handle_ack(self, message) -> None:
        if message.src != self.b_id:
            raise ProtocolError(f"link {self.a_id}->{self.b_id}: LINK_ACK from {message.src!r}")
        if self.acked:
            return
        self.acked = True
        self.pending = False
        if self.on_a_ready is not None:
            self.on_a_ready()


def bell_fidelity(tracker, key_a: int, key_b: int, phase: float = 0.0) -> float:
    """<Phi+|rho|Phi+> for the joint state over (key_a, key_b). |Phi+> is swap-symmetric,
    so the tensor-factor order of the two keys inside the state doesn't matter."""
    state = tracker.get(key_a)
    if set(state.keys) != {key_a, key_b}:
        raise ValueError(f"keys {key_a},{key_b} are not the sole occupants of their state "
                         f"(covers {state.keys}); the end-to-end pair is not isolated")
    amp = 1 / np.sqrt(2)
    phi = np.array([amp, 0, 0, amp * np.exp(1j * phase)], dtype=complex)
    return float((phi.conj() @ state.matrix @ phi).real)
