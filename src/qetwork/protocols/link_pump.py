"""LinkPumpSession: DEJMPS entanglement pumping on ONE directed edge.

Standard repeated DEJMPS (Deutsch et al., PRL 77, 2818 (1996)): every round the
emitter applies rx(+pi/2) to both its qubits and the absorber rx(-pi/2) to both of
its qubits, each CNOTs kept -> sacrificial, Z-measures the sacrificial, and the pair
is kept on coincident outcomes. The SAME circuit every round, NO unrotation -- the
map's Bell-coefficient permutation is part of the recurrence. `rounds` counts
CONSECUTIVE successes on the current kept pair (its pump level); a mismatch discards
the kept pair, bumps the epoch, and rebuilds from a fresh raw pair.

Calibration (calibrate=True): the emitter applies rz(-source.phase) to its retained
half of EVERY raw pair (kept-maker included), cancelling its own source's known
phase so raw pairs are exactly Werner and the Deutsch recurrence is the exact
reference. calibrate=False leaves the phase in as a coherent error (research mode).

Hardware model (documented idealizations):
  * communication + storage memories: the per-edge link memory is the only optically
    coupled slot; kept pairs are SWAPped out via node.move() (noisy, timed by
    move_duration); sacrificial pairs are consumed in the link memories directly;
  * each edge-interface has independent local gate hardware -- concurrent pumps at
    one node never contend; operations on ONE edge-side serialize via busy clocks;
  * the classical channel is lossless.

Timing convention: state math applies inside the triggering event; durations gate
DEPENDENT actions (message sends, the next arm) -- the bsm/SWAP_RESULT convention.
The arm rule  t_arm = max(now, a_free, b_free - qd + 1)  guarantees the next photon
lands strictly after the absorber is hardware-free AND has processed the previous
round's verdict; that makes the absorber's "kept empty => kept-maker" test
unambiguous and keeps the (epoch, level) counters in lockstep without extra
messages. Counter mismatches raise ProtocolError -- desync is a bug, never noise.

At most ONE active session per edge, one delivery in flight per network.
"""

from dataclasses import dataclass

from qetwork.events.priority import PROTOCOL
from qetwork.protocols.errors import ProtocolError
from qetwork.protocols.e_dist_swap import (
    LinkGeneration, validate_path, expected_completion,
    SEQUENTIAL, PARALLEL, TAD,
)

PURIFY_BIT = "purify_bit"          # absorber -> emitter: outcome bit + (epoch, level) stamps
PURIFY_VERDICT = "purify_verdict"  # emitter -> absorber: keep/discard + (epoch, level) echo

MAKE_KEPT, WAIT_ROUND, DONE = "make_kept", "wait_round", "done"


@dataclass(slots=True)
class EdgeStats:
    emitter: str
    absorber: str
    emission_attempts: int = 0     # emissions incl. lost photons
    heralded_pairs: int = 0        # pairs that survived the fiber (kept-makers + sacrificials)
    pump_rounds: int = 0           # purification rounds attempted
    epochs: int = 1                # kept pairs built (1 + discards)
    a_ready_time: int | None = None
    b_ready_time: int | None = None


class LinkPumpSession:
    """Pump one directed edge (emitter -> absorber) to `rounds` consecutive successes.

    Delivers the purified pair in emitter's `purify:kept:{absorber}` and absorber's
    `purify:kept:{emitter}` storage slots. Direction-agnostic: any adjacent ordered
    pair works, including reverse traversal of an edge.
    """

    def __init__(self, net, emitter_id, absorber_id, rounds, calibrate=True,
                 on_a_ready=None, on_b_ready=None, on_done=None):
        if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
            raise ValueError(f"rounds must be an int >= 1, got {rounds!r}")
        self.net = net
        self.timeline = net.timeline
        self.a_id, self.b_id = emitter_id, absorber_id
        self.A = net.nodes[emitter_id]
        self.B = net.nodes[absorber_id]
        self.rounds = rounds
        self.calibrate = calibrate
        self.on_a_ready = on_a_ready   # emitter-side READY (its kept half is final)
        self.on_b_ready = on_b_ready   # absorber-side READY (fires last)
        self.on_done = on_done

        self.kept_a = self.A.ensure_memory(f"purify:kept:{absorber_id}")
        self.kept_b = self.B.ensure_memory(f"purify:kept:{emitter_id}")

        # static delay/duration constants -- wiring-time config, legitimately shared
        self.qd = self.A.ports[f"q:{absorber_id}"].qfiber.delay
        cf_ab = self.A.cfibers[absorber_id]
        cf_ba = self.B.cfibers[emitter_id]
        self.cd_ab = cf_ab.delay + cf_ab.latency
        self.cd_ba = cf_ba.delay + cf_ba.latency
        self.pd_a = self.A.purify_duration
        self.pd_b = self.B.purify_duration
        self.cal_a = self.A.calibration_duration if calibrate else 0
        self.mv_a = self.A.move_duration
        self.mv_b = self.B.move_duration
        self.phi = self.A.source.phase

        self.linkgen = LinkGeneration(net, emitter_id, absorber_id,
                                      on_b_ready=self._b_pair_arrived,
                                      on_a_ready=self._a_pair_acked)

        # per-side counters, kept in lockstep by the message chain + arm rule.
        # Fields suffixed _a are touched only by emitter-side events, _b only by
        # absorber-side events; the session object is just their shared container.
        self.epoch_a = 0
        self.level_a = 0
        self.epoch_b = 0
        self.level_b = 0
        self.phase = MAKE_KEPT         # emitter-side machine: MAKE_KEPT | WAIT_ROUND | DONE
        self.bit_a = None              # emitter's own outcome, current round
        self.bit_b = None              # absorber's outcome as received, current round
        self.t_abs = None              # absorb instant of the current round's pair
        self.a_done_at = 0             # when the emitter's local circuit completes
        self.a_ready = False
        self.b_ready = False
        self.done = False
        self.stats = EdgeStats(emitter=emitter_id, absorber=absorber_id)

    # -- lifecycle --

    def install(self) -> None:
        self.linkgen.install()
        self.A.register_handler(PURIFY_BIT, self._handle_bit, replace=True)
        self.B.register_handler(PURIFY_VERDICT, self._handle_verdict, replace=True)

    def uninstall(self) -> None:
        self.linkgen.uninstall()
        self.A.unregister_handler(PURIFY_BIT, self._handle_bit)
        self.B.unregister_handler(PURIFY_VERDICT, self._handle_verdict)

    def start(self) -> None:
        """Clear stale kept slots and kick off the first (kept-maker) pair."""
        self.kept_a.reset()
        self.kept_b.reset()
        self.timeline.schedule(self.linkgen.arm, at=self.timeline.now(), priority=PROTOCOL)

    def run(self) -> EdgeStats:
        """Blocking convenience for single-session callers (e.g. benchmarking):
        install, pump to completion, uninstall."""
        self.install()
        self.start()
        while not self.done and self.timeline.step():
            pass
        self.uninstall()
        if not self.done:
            raise ProtocolError(f"pump {self.a_id}->{self.b_id}: timeline drained before completion")
        return self.stats

    # -- absorber side (all inside B-local events) --

    def _b_pair_arrived(self) -> None:
        """Photon absorbed at B; the LINK_ACK is already dispatched (I1)."""
        now = self.timeline.now()
        self.stats.heralded_pairs += 1
        link = self.B.link_mems[self.a_id]
        if self.kept_b.is_empty():                       # kept-maker: promote to storage
            self.B.move(link, self.kept_b)               # B busy until now + mv_b (A models it)
            return
        # pump round: B's DEJMPS half, immediately -- absorption proves both halves exist
        bit = self.B.purify_local(self.kept_b, link, sign=-1, rotate=True)
        self.timeline.schedule(self.B.send_to, self.a_id, PURIFY_BIT,
                               at=now + self.pd_b, priority=PROTOCOL,
                               bit=bit, epoch=self.epoch_b, level=self.level_b)

    def _handle_verdict(self, message) -> None:
        if message.src != self.a_id:
            raise ProtocolError(f"pump {self.a_id}->{self.b_id}: verdict from {message.src!r}")
        p = message.payload
        if p["epoch"] != self.epoch_b or p["level"] != self.level_b:
            raise ProtocolError(
                f"pump {self.a_id}->{self.b_id}: verdict stamps ({p['epoch']},{p['level']}) "
                f"!= absorber counters ({self.epoch_b},{self.level_b})")
        if p["keep"]:
            self.level_b += 1                            # success costs B no local gate
            if self.level_b == self.rounds:
                self.b_ready = True
                self.stats.b_ready_time = self.timeline.now()
                if self.on_b_ready is not None:
                    self.on_b_ready()
                self._maybe_done()
        else:
            self.kept_b.reset()
            self.level_b = 0
            self.epoch_b += 1

    # -- emitter side (all inside A-local events) --

    def _a_pair_acked(self) -> None:
        """LINK_ACK at A: the raw pair provably exists at both ends. Purifying any
        earlier would CNOT the kept pair against a possibly-lost half's I/2 remnant."""
        now = self.timeline.now()                        # = t_abs + cd_ba
        t_abs = now - self.cd_ba
        link = self.A.link_mems[self.b_id]
        if self.phase == MAKE_KEPT:
            if self.calibrate:
                self.A.calibrate_phase(link, self.phi)   # kept-maker is calibrated too
            self.A.move(link, self.kept_a)
            self.phase = WAIT_ROUND
            self._arm_at(a_free=now + self.cal_a + self.mv_a,
                         b_free=t_abs + self.mv_b)
            return
        if self.phase != WAIT_ROUND:
            raise ProtocolError(f"pump {self.a_id}->{self.b_id}: ACK in phase {self.phase!r}")
        # pump round: calibrate the sacrificial, then A's DEJMPS half
        if self.calibrate:
            self.A.calibrate_phase(link, self.phi)
        self.bit_a = self.A.purify_local(self.kept_a, link, sign=+1, rotate=True)
        self.a_done_at = now + self.cal_a + self.pd_a
        self.t_abs = t_abs
        self.stats.pump_rounds += 1

    def _handle_bit(self, message) -> None:
        if message.src != self.b_id:
            raise ProtocolError(f"pump {self.a_id}->{self.b_id}: bit from {message.src!r}")
        p = message.payload
        if p["epoch"] != self.epoch_a or p["level"] != self.level_a:
            raise ProtocolError(
                f"pump {self.a_id}->{self.b_id}: bit stamps ({p['epoch']},{p['level']}) "
                f"!= emitter counters ({self.epoch_a},{self.level_a})")
        if self.phase != WAIT_ROUND or self.bit_b is not None:
            raise ProtocolError(f"pump {self.a_id}->{self.b_id}: unexpected bit "
                                f"(phase {self.phase!r})")
        if self.bit_a is None:                           # I1 violated: bit beat the ACK
            raise ProtocolError(f"pump {self.a_id}->{self.b_id}: bit arrived before ACK")
        self.bit_b = p["bit"]
        now = self.timeline.now()
        if now >= self.a_done_at:                        # local circuit already finished
            self._decide()
        else:                                            # bit beat the slower local circuit
            self.timeline.schedule(self._decide, at=self.a_done_at, priority=PROTOCOL)

    def _decide(self) -> None:
        """Runs at t_dec = max(bit arrival, emitter circuit completion)."""
        now = self.timeline.now()
        keep = (self.bit_a == self.bit_b)
        round_epoch, round_level = self.epoch_a, self.level_a   # pre-round stamps to echo
        self.bit_a = self.bit_b = None
        if keep:
            self.level_a += 1
        else:
            self.kept_a.reset()
            self.level_a = 0
            self.epoch_a += 1
            self.stats.epochs += 1
            self.phase = MAKE_KEPT
        self.A.send_to(self.b_id, PURIFY_VERDICT,
                       keep=keep, epoch=round_epoch, level=round_level)
        if keep and self.level_a == self.rounds:
            self.phase = DONE
            self.a_ready = True
            self.stats.a_ready_time = now
            if self.on_a_ready is not None:
                self.on_a_ready()
            self._maybe_done()
            return
        # next pair: B free after its purify AND after processing this verdict
        self._arm_at(a_free=now,
                     b_free=max(self.t_abs + self.pd_b, now + self.cd_ab))

    # -- shared --

    def _arm_at(self, a_free: int, b_free: int) -> None:
        """THE arm rule: next photon (t_arm + qd) lands strictly after b_free."""
        t = max(self.timeline.now(), a_free, b_free - self.qd + 1)
        self.timeline.schedule(self.linkgen.arm, at=t, priority=PROTOCOL)

    def _maybe_done(self) -> None:
        if self.done or not (self.a_ready and self.b_ready):
            return
        self.done = True
        self.stats.emission_attempts = self.linkgen.attempts
        if self.on_done is not None:
            self.on_done()


# --- purified end-to-end distribution: many sessions on one path, scheduled by mode ---

def _noop(*_args, **_kwargs) -> None:
    """A do-nothing event: advances the virtual clock to a target instant."""


def _dejmps_step(fidelity: float) -> tuple[float, float]:
    """One DEJMPS round on a Werner-parameterized Bell-diagonal input of the given
    Phi+ fidelity: (keep probability, kept-pair fidelity). Standard Deutsch et al.
    recurrence -- an APPROXIMATION to purify_local's exact circuit, used only to size
    tad start offsets and validated against empirical keep rates, never a fidelity claim."""
    a = fidelity
    b = c = d = (1.0 - fidelity) / 3.0
    p_keep = (a + d) ** 2 + (b + c) ** 2
    f_new = (a * a + d * d) / p_keep if p_keep > 0 else 0.0
    return p_keep, f_new


def expected_pump_pairs(rounds: int, f0: float) -> float:
    """Expected heralded pairs to build a kept pair to `rounds` consecutive DEJMPS
    successes from raw fidelity f0. A failed round discards the kept pair (back to a
    fresh kept-maker), so this is the restart-to-zero recurrence
        E_R = 0;  E_l = 1 + q_l E_{l+1} + (1-q_l)(1 + E_0)
    solved by an affine backward pass (E_l = a_l + b_l E_0). +1 counts the kept-maker."""
    qs = []
    f = f0
    for _ in range(rounds):
        q, f = _dejmps_step(f)
        qs.append(q)
    a_coef, b_coef = 0.0, 0.0                        # E_R = 0
    for q in reversed(qs):                           # l = R-1 .. 0
        a_coef = (2.0 - q) + q * a_coef
        b_coef = (1.0 - q) + q * b_coef
    e0 = a_coef / (1.0 - b_coef)
    return 1.0 + e0


def expected_pump_completion(rtt: int, p_link: float, rounds: int, f0: float) -> float:
    """Expected wall-time to pump one edge to `rounds`: expected heralded pairs times
    the expected time per heralded pair (the raw-link statistic). Ignores per-round
    classical/gate overhead -- second-order when pair generation dominates."""
    return expected_pump_pairs(rounds, f0) * expected_completion(rtt, p_link)


@dataclass(slots=True)
class PurifiedResult:
    """The delivered purified end-to-end pair and per-edge pump effort."""
    source_key: int
    dest_key: int
    latency: int                   # ps from t=0 to the destination's frame correction
    edge_stats: dict               # emitter node_id -> EdgeStats
    offsets: dict                  # emitter node_id -> scheduled start delay in ps (0 outside tad)


class PurifiedDistribution:
    """Purify-then-swap end-to-end distribution: one LinkPumpSession per path edge,
    pumped to `rounds` consecutive DEJMPS successes, then swapped into one end-to-end
    pair with a single XOR Pauli frame corrected at the path destination. Delivers the
    pair in the source's `purify:kept:{next}` and destination's `purify:kept:{prev}` slots.

    Mode schedules only the pump STARTS -- every session free-runs its own arm cadence:
      * sequential -- edge k+1 starts when edge k finishes (chained on_done); reproduces
        the historical edge-by-edge blocking loop.
      * parallel   -- every edge starts at t=0 and pumps concurrently on the shared
        timeline; the barrier is the slowest edge, not the sum of all edges.
      * tad        -- every edge starts at an expectation-aligned offset (slowest expected
        edge at 0) so edges finish pumping together, minimizing kept-pair idle before swap.

    Swaps are barrier-synchronized in all modes: every edge done, then one swap cascade.
    """

    def __init__(self, net, path, rounds, calibrate=True, mode=SEQUENTIAL):
        if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
            raise ValueError(f"rounds must be an int >= 1, got {rounds!r}")
        if mode not in (SEQUENTIAL, PARALLEL, TAD):
            raise ValueError(f"mode must be {SEQUENTIAL!r}, {PARALLEL!r} or {TAD!r}, got {mode!r}")
        self.net = net
        self.timeline = net.timeline
        self.path = validate_path(net, path)
        self.rounds = rounds
        self.calibrate = calibrate
        self.mode = mode
        self.num_edges = len(self.path) - 1
        self.edges = list(zip(self.path, self.path[1:]))     # (emitter, absorber) per hop
        self.result: PurifiedResult | None = None

        # static tad offsets: align expected pump completions; slowest expected edge anchors
        # at 0. p and rtt are read from the same qfiber/cfiber the session would use.
        self.offsets = {u: 0 for u, _ in self.edges}
        if mode == TAD:
            ets = {}
            for u, v in self.edges:
                A = net.nodes[u]
                cf = A.cfibers[v]
                rtt = A.ports[f"q:{v}"].qfiber.delay + cf.delay + cf.latency
                p_link = A.ports[f"q:{v}"].qfiber.survival
                if not 0.0 < p_link <= 1.0:
                    raise ValueError(f"edge {u!r}->{v!r} has survival {p_link!r}; "
                                     f"expected a probability in (0, 1]")
                f0 = (1.0 + 3.0 * A.source.visibility) / 4.0
                ets[u] = expected_pump_completion(rtt, p_link, rounds, f0)
            et_max = max(ets.values())
            self.offsets = {u: round(et_max - e) for u, e in ets.items()}

    def _cdelay(self, A, B) -> int:
        cf = A.cfibers[B.node_id]
        return cf.delay + cf.latency

    def run(self) -> PurifiedResult | None:
        tl = self.timeline
        now0 = tl.now()
        order = [u for u, _ in self.edges]
        sessions = {u: LinkPumpSession(self.net, u, v, rounds=self.rounds,
                                       calibrate=self.calibrate)
                    for u, v in self.edges}
        pending = set(order)

        def _make_on_done(u, idx):
            def _done():
                pending.discard(u)
                if self.mode == SEQUENTIAL and idx + 1 < len(order):
                    tl.schedule(sessions[order[idx + 1]].start, at=tl.now(), priority=PROTOCOL)
            return _done

        for idx, u in enumerate(order):
            sessions[u].on_done = _make_on_done(u, idx)
            sessions[u].install()

        if self.mode == SEQUENTIAL:                          # only the first edge starts now
            tl.schedule(sessions[order[0]].start, at=now0, priority=PROTOCOL)
        else:                                                # parallel/tad: each at its offset
            for u in order:
                tl.schedule(sessions[u].start, at=now0 + self.offsets[u], priority=PROTOCOL)

        while pending and tl.step():                         # pump phase -> barrier
            pass
        if pending:
            for s in sessions.values():
                s.uninstall()
            raise ProtocolError(f"purified distribution {self.path}: "
                                f"timeline drained before all edges pumped")

        # swap cascade + one Pauli frame at the path destination (barrier-synchronized)
        dest = self.net.nodes[self.path[-1]]
        if self.num_edges > 1:
            acc1 = acc2 = 0
            latest = tl.now()
            for i in range(1, self.num_edges):
                node = self.net.nodes[self.path[i]]
                m1, m2 = node.bsm(node.memory(f"purify:kept:{self.path[i - 1]}"),
                                  node.memory(f"purify:kept:{self.path[i + 1]}"))
                acc1 ^= m1
                acc2 ^= m2
                latest = max(latest, tl.now() + node.bsm_duration + self._cdelay(node, dest))
            tl.schedule(dest.correct, dest.memory(f"purify:kept:{self.path[-2]}"),
                        acc1, acc2, at=latest, priority=PROTOCOL)
            tl.schedule(_noop, at=latest + dest.correct_duration, priority=PROTOCOL)
        while tl.step():                                     # flush cascade + stale timeouts
            pass
        for s in sessions.values():
            s.uninstall()

        src_half = self.net.nodes[self.path[0]].memory(f"purify:kept:{self.path[1]}")
        dst_half = self.net.nodes[self.path[-1]].memory(f"purify:kept:{self.path[-2]}")
        src_half.decohere()                                  # settle the endpoint halves to now
        dst_half.decohere()                                  # (dst already settled by correct())
        self.result = PurifiedResult(
            source_key=src_half.key,
            dest_key=dst_half.key,
            latency=tl.now(),
            edge_stats={u: s.stats for u, s in sessions.items()},
            offsets=dict(self.offsets),
        )
        return self.result

    def reset(self) -> None:
        """Re-arm for another delivery over the same path (offsets are static)."""
        self.result = None
