"""Entanglement purification: standard repeated DEJMPS at PATH or LINK level.

Circuit (Deutsch et al., PRL 77, 2818 (1996)): every round, source-side rx(+pi/2)
on both its qubits, dest-side rx(-pi/2) on both, CNOT kept -> sacrificial, Z-measure
the sacrificial, keep on coincidence. Same circuit every round, no unrotation.
`rounds` = CONSECUTIVE successes required on the kept pair; a mismatch rebuilds it.

PATH level: the kept pair spans the endpoints and every sacrificial is a full
end-to-end distribution (blocking driver; no calibration -- the composite phase of
a swapped chain is out of scope here). LINK level: every edge pumps its own pair
via LinkPumpSession (concurrent or chained per spec.mode), then interiors swap the
purified kept pairs as-ready and the destination applies the accumulated Pauli
frame. Drives node verbs only; no density math lives here.
"""

from dataclasses import dataclass
from functools import partial

from qetwork.events.priority import PROTOCOL
from qetwork.protocols.errors import ProtocolError
from qetwork.protocols.e_dist_swap import (
    EntanglementDistribution, DistributionResult, bell_fidelity, validate_path,
    SEQUENTIAL, PARALLEL, SWAP_RESULT,
)
from qetwork.protocols.link_pump import LinkPumpSession, EdgeStats

LINK, PATH = "link", "path"


@dataclass(slots=True)
class DeliverySpec:
    """One knob for every caller: distribution mode, purification level and depth."""
    mode: str = SEQUENTIAL         # SEQUENTIAL | PARALLEL
    rounds: int = 0                # consecutive DEJMPS successes required; 0 => no purification
    level: str = PATH              # PATH | LINK
    calibrate: bool = True         # LINK only: cancel each source's known phase per raw pair


@dataclass(slots=True, kw_only=True)
class DeliveryResult:
    source_key: int
    dest_key: int
    source_mem: object             # the MemoryQubit holding each delivered half
    dest_mem: object
    fidelity: float
    latency: int
    rounds_run: int                # pump rounds attempted (all edges, LINK)
    successes: int                 # keep verdicts issued (all edges, LINK)
    raw_pairs: int                 # heralded elementary pairs consumed
    distribution: DistributionResult | None = None      # PATH/plain only
    edges: dict | None = None      # LINK only: (emitter, absorber) -> EdgeStats


class Purification:
    """PATH-level pumping: kept pair in named endpoint slots, each sacrificial is a
    fresh end-to-end distribution. Blocking driver (ed.run() + stepped waits)."""

    def __init__(self, net, path=None, spec=None):
        self.net = net
        self.timeline = net.timeline
        self.spec = spec or DeliverySpec()
        if self.spec.level != PATH:
            raise ValueError("Purification is the PATH-level pump; use deliver() for LINK")
        self.ed = EntanglementDistribution(net, path, self.spec.mode)   # reused via reset()
        self.path = self.ed.path
        self.src = net.nodes[self.path[0]]
        self.dst = net.nodes[self.path[-1]]
        self.kept_src = self.src.ensure_memory("purify:kept")
        self.kept_dst = self.dst.ensure_memory("purify:kept")
        self.kept_src.reset()
        self.kept_dst.reset()
        self.raw_pairs = 0
        self.last_dist = None

    def _classical_rtt(self) -> int:
        c = self.src.cfibers[self.path[-1]]
        return 2 * (c.delay + c.latency)

    def _wait(self, delay: int) -> None:
        if delay <= 0:
            return
        target = self.timeline.now() + delay
        self.timeline.schedule(lambda: None, at=target, priority=PROTOCOL)
        while self.timeline.now() < target and self.timeline.step():
            pass

    def _make_kept(self) -> None:
        """Distribute one pair, move both halves into the kept slots (frees link mems)."""
        for m in (self.kept_src, self.kept_dst):
            if not m.is_empty():
                m.reset()
        self.ed.reset()
        self.last_dist = self.ed.run()
        self.raw_pairs += 1
        self.src.move(self.src.link_mems[self.path[1]], self.kept_src)
        self.dst.move(self.dst.link_mems[self.path[-2]], self.kept_dst)
        self._wait(max(self.src.move_duration, self.dst.move_duration))

    def _round(self) -> bool:
        """One standard DEJMPS round against a freshly distributed sacrificial."""
        self.ed.reset()
        self.last_dist = self.ed.run()
        self.raw_pairs += 1
        sac_s = self.src.link_mems[self.path[1]]
        sac_d = self.dst.link_mems[self.path[-2]]
        bit_s = self.src.purify_local(self.kept_src, sac_s, sign=+1, rotate=True)
        bit_d = self.dst.purify_local(self.kept_dst, sac_d, sign=-1, rotate=True)
        self._wait(max(self.src.purify_duration, self.dst.purify_duration)
                   + self._classical_rtt())
        if bit_s == bit_d:
            return True
        self.kept_src.reset()
        self.kept_dst.reset()
        return False

    def run(self) -> DeliveryResult:
        self._make_kept()
        level = attempts = keeps = 0
        while level < self.spec.rounds:
            attempts += 1
            if self._round():
                level += 1
                keeps += 1
            else:
                level = 0
                self._make_kept()
        tracker = self.timeline.state_tracker
        self.kept_src.decohere()               # settle the final wait (purify + rtt) into the pair
        self.kept_dst.decohere()
        return DeliveryResult(
            source_key=self.kept_src.key, dest_key=self.kept_dst.key,
            source_mem=self.kept_src, dest_mem=self.kept_dst,
            fidelity=bell_fidelity(tracker, self.kept_src.key, self.kept_dst.key),
            latency=self.timeline.now(), rounds_run=attempts,
            successes=keeps, raw_pairs=self.raw_pairs, distribution=self.last_dist)


class LinkPurificationDistribution:
    """LINK-level delivery: pump every edge with a LinkPumpSession (all at t=0 in
    PARALLEL; chained absorber-READY -> next session in SEQUENTIAL), swap the
    purified kept pairs as-ready at interiors, accumulate the Pauli frame at the
    destination, correct, report."""

    def __init__(self, net, path=None, spec=None):
        self.net = net
        self.timeline = net.timeline
        self.spec = spec or DeliverySpec()
        self.path = validate_path(net, path if path is not None else net.path())
        self.k = len(self.path) - 1
        self.dest_id = self.path[-1]
        self.dest_node = net.nodes[self.dest_id]

        self.sessions: list[LinkPumpSession] = []
        for i in range(self.k):
            self.sessions.append(LinkPumpSession(
                net, self.path[i], self.path[i + 1],
                rounds=self.spec.rounds, calibrate=self.spec.calibrate,
                on_a_ready=partial(self._edge_ready, i, "a"),
                on_b_ready=partial(self._edge_ready, i, "b")))

        self.up_ready = [False] * (self.k + 1)     # node j: edge j-1 absorber side READY
        self.down_ready = [False] * (self.k + 1)   # node j: edge j emitter side READY
        self.swapped = [False] * (self.k + 1)
        self.acc_m1 = 0
        self.acc_m2 = 0
        self.results = 0
        self.completed = False
        self.result: DeliveryResult | None = None

    def _edge_ready(self, i: int, side: str) -> None:
        if side == "a":                            # emitter side of edge i = node i
            self.down_ready[i] = True
            if i > 0:                              # the source (i == 0) just holds its half
                self._maybe_swap(i)
            return
        j = i + 1                                  # absorber side of edge i = node i+1
        self.up_ready[j] = True
        if self.spec.mode == SEQUENTIAL and i + 1 < self.k:
            self.sessions[i + 1].start()           # the baton: this node emits edge i+1
        if j < self.k:
            self._maybe_swap(j)
        else:
            self._try_complete()

    def _maybe_swap(self, j: int) -> None:
        if self.swapped[j] or not (self.up_ready[j] and self.down_ready[j]):
            return
        self.swapped[j] = True
        node = self.net.nodes[self.path[j]]
        kept_up = node.memory(f"purify:kept:{self.path[j - 1]}")
        kept_down = node.memory(f"purify:kept:{self.path[j + 1]}")
        m1, m2 = node.bsm(kept_up, kept_down)      # ctrl = upstream, the existing convention
        self.timeline.schedule(node.send_to, self.dest_id, SWAP_RESULT,
                               at=self.timeline.now() + node.bsm_duration,
                               priority=PROTOCOL, m1=m1, m2=m2)

    def _handle_swap_result(self, message) -> None:
        self.acc_m1 ^= message.payload["m1"]
        self.acc_m2 ^= message.payload["m2"]
        self.results += 1
        self._try_complete()

    def _try_complete(self) -> None:
        if self.completed or not self.up_ready[self.k] or self.results < self.k - 1:
            return
        self.completed = True
        if self.k > 1:
            kept = self.dest_node.memory(f"purify:kept:{self.path[-2]}")
            self.dest_node.correct(kept, self.acc_m1, self.acc_m2)
            self.timeline.schedule(self._finish, priority=PROTOCOL,
                                   at=self.timeline.now() + self.dest_node.correct_duration)
        else:
            self._finish()

    def _finish(self) -> None:
        kept_src = self.net.nodes[self.path[0]].memory(f"purify:kept:{self.path[1]}")
        kept_dst = self.dest_node.memory(f"purify:kept:{self.path[-2]}")
        kept_src.decohere()                        # waited since its edge went READY
        kept_dst.decohere()                        # settled by correct() when k > 1; no-op then
        for s in self.sessions:
            s.uninstall()
        self.dest_node.unregister_handler(SWAP_RESULT, self._handle_swap_result)
        edges = {(s.a_id, s.b_id): s.stats for s in self.sessions}
        stats = list(edges.values())
        self.result = DeliveryResult(
            source_key=kept_src.key, dest_key=kept_dst.key,
            source_mem=kept_src, dest_mem=kept_dst,
            fidelity=bell_fidelity(self.timeline.state_tracker, kept_src.key, kept_dst.key),
            latency=self.timeline.now(),
            rounds_run=sum(st.pump_rounds for st in stats),
            successes=sum(st.pump_rounds - (st.epochs - 1) for st in stats),
            raw_pairs=sum(st.heralded_pairs for st in stats),
            distribution=None, edges=edges)

    def run(self) -> DeliveryResult:
        for s in self.sessions:
            s.install()
        self.dest_node.register_handler(SWAP_RESULT, self._handle_swap_result, replace=True)
        if self.spec.mode == PARALLEL:
            for s in self.sessions:
                s.start()
        else:
            self.sessions[0].start()
        while self.result is None and self.timeline.step():
            pass
        if self.result is None:
            raise ProtocolError("link-level delivery: timeline drained before completion")
        return self.result


def deliver(net, path=None, spec=None) -> DeliveryResult:
    """The one entry point every caller uses: distribute, optionally purify."""
    spec = spec or DeliverySpec()
    if not isinstance(spec.rounds, int) or isinstance(spec.rounds, bool) or spec.rounds < 0:
        raise ValueError(f"spec.rounds must be an int >= 0, got {spec.rounds!r}")
    if spec.level not in (PATH, LINK):
        raise ValueError(f"spec.level must be {PATH!r} or {LINK!r}, got {spec.level!r}")
    if spec.mode not in (SEQUENTIAL, PARALLEL):
        raise ValueError(f"spec.mode must be {SEQUENTIAL!r} or {PARALLEL!r}, got {spec.mode!r}")
    if not isinstance(spec.calibrate, bool):
        raise ValueError(f"spec.calibrate must be a bool, got {spec.calibrate!r}")
    if spec.rounds == 0:                               # plain distribution, no purification
        ed = EntanglementDistribution(net, path, spec.mode)
        d = ed.run()
        src_mem = net.nodes[ed.path[0]].link_mems[ed.path[1]]
        dst_mem = net.nodes[ed.path[-1]].link_mems[ed.path[-2]]
        f = bell_fidelity(net.timeline.state_tracker, d.source_key, d.dest_key)
        return DeliveryResult(source_key=d.source_key, dest_key=d.dest_key,
                              source_mem=src_mem, dest_mem=dst_mem, fidelity=f,
                              latency=d.latency, rounds_run=0, successes=0,
                              raw_pairs=1, distribution=d)
    if spec.level == PATH:
        return Purification(net, path, spec).run()
    return LinkPurificationDistribution(net, path, spec).run()
