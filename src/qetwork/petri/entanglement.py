"""Entanglement generation + swapping on a linear repeater chain as a timed coloured
Petri net: the place/transition structure of protocols.e_dist_swap.EntanglementDistribution.

ONE folded net serves every chain length. Nodes, edges and attempt generations are
colours, so the hop count K shows up only in the colour sets and the initial marking;
per-node and per-edge physics (delays, loss, T1/T2, gate error) enter through arc
expressions that read ChainParams. Design notes: docs/petri_net.md.

Path n_0 .. n_K; edge i joins emitter n_i -> absorber n_{i+1}. Every node keeps one
link memory per path side (`up` toward n_0, `down` toward n_K).

Colour sets
  Slot     = Node x {up, down}                 a link memory
  Arm      = Edge x Gen                        emitter of edge i, Gen attempts made so far
  Attempt  = Edge x Gen x Time                 one emitted pair (idler stored, emitted at Time)
  Tag      = Edge x Gen                        control knowledge about attempt (i, g)
  Pair     = Node x Node x Time x Time x State entangled pair (left, right, per-half settle
                                               times, two-qubit state -- see pair_state)
  SwapBits = Node x Bit x Bit                  BSM outcome reported by interior node r
  Frame    = Int x Bit x Bit                   destination's Pauli-frame accumulator

Places -- physical (what the hardware holds) | knowledge (what a node has learned)
  phys  FreeMem    Slot      empty link memories
  know  Armed      Arm       emitter may fire its next attempt
  phys  InFlight   Attempt   idler at n_i, signal photon in fiber i
  phys  Dangling   Attempt   idler at n_i whose photon was lost
  phys  Entangled  Pair      heralded pairs in memory (swaps merge them)
  know  Pending    Tag       emitter waits for LINK_ACK of attempt g
  know  Timer      Tag       emitter's retry timer for attempt g
  know  AckChan    Tag       LINK_ACK on the classical fiber n_{i+1} -> n_i
  know  Receipt    Tag       emitter saw the ACK (complement of Pending: retires the timer)
  know  DownReady  int       interior n_r knows its downstream link is up
  know  SwapMsg    SwapBits  SWAP_RESULT on the classical fiber n_r -> n_K
  know  Frame      Frame     swaps reported to n_K so far, XOR of their bits
  phys  Delivered  Pair      end-to-end pair after the frame correction

Transitions (priority tiers mirror events.priority: ARRIVAL < OPERATION < PROTOCOL)
  Emit      Armed(i,n), FreeMem(i,down)                 -> InFlight(i,n+1,now)@+qd(i), Pending(i,n+1),
                                                            Timer(i,n+1)@+timeout(i)
  Herald    InFlight(i,g,te), FreeMem(i+1,up)  [w=surv] -> Entangled(i,i+1,te,now,raw(n_i)),
                                                            AckChan(i,g)@+cd(i) [, Armed(i+1,0) seq baton]
  Lose      InFlight(i,g,te), FreeMem(i+1,up)  [w=1-surv] -> Dangling(i,g,te), FreeMem(i+1,up)
  Ack       AckChan(i,g), Pending(i,g)                  -> Receipt(i,g) [, DownReady(i) if interior]
  Timeout   Timer(i,g), Pending(i,g), Dangling(i,g,_)   -> Armed(i,g), FreeMem(i,down)
  Cancel    Timer(i,g), Receipt(i,g)                    -> (nothing)
  Swap      DownReady(r), Entangled(a,r), Entangled(r,b) -> Entangled(a,b), SwapMsg(r,m1,m2)@+bsm(r)+cd(r,K),
                                                            FreeMem(r,up)@+bsm(r), FreeMem(r,down)@+bsm(r)
  Collect   SwapMsg(r,m1,m2), Frame(n,a1,a2)            -> Frame(n+1, a1^m1, a2^m2)
  Complete  Frame(K-1,a1,a2), Entangled(0,K)            -> Delivered(0,K)@+corr

Herald and Lose share their preset (Lose hands the absorber's slot straight back), an
extended-free-choice conflict, so the GSPN switch with weights surv, 1-surv is exactly
the photon's Bernoulli survival at the receiving port.

Modes change only the initial marking, the timeout and the baton -- as in the agent:
  sequential  Armed(0,0)@0; Herald on edge i arms edge i+1 (absorb -> emit baton);
              timeout = rtt(i)+1
  parallel    Armed(i,0)@0 for every edge; timeout = max rtt + 1 (shared round grid)
  tad         Armed(i,0)@offset(i), offsets aligning expected completions; timeout = rtt(i)+1

Correctness of the retry race needs no inhibitor arc: a heralded attempt's ACK lands at
t_emit + rtt, strictly before its timer, so Ack consumes Pending first and Cancel later
retires the timer against the Receipt. Timeout can only ever bind a lost attempt.

Timing convention (the simulator's): state math happens in the firing; durations delay
only DEPENDENT tokens (the SWAP_RESULT, the freed memories, the delivered pair). Memory
decoherence is lazy, as in MemoryQubit: each half carries the instant it was last
settled and is brought up to date only when a gate touches it (Swap, Complete) --
exact, because idle noise on one qubit commutes with every operation on the others.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

from qetwork.events.priority import ARRIVAL, OPERATION, PROTOCOL
from qetwork.petri.pair_state import LEFT, RIGHT, DensityMatrix, PairAlgebra
from qetwork.petri.params import ChainParams, EdgeParams, NodeParams
from qetwork.petri.tcpn import In, Out, TimedColoredPetriNet
from qetwork.protocols.e_dist_swap import PARALLEL, SEQUENTIAL, TAD, expected_completion
from qetwork.protocols.errors import ProtocolError

__all__ = ["ChainParams", "EdgeParams", "NodeParams", "ChainResult", "build_net", "simulate",
           "start_offsets", "Slot", "Arm", "Attempt", "Tag", "Pair", "SwapBits", "Frame"]

UP, DOWN = "up", "down"


# --- colour sets ---

class Slot(NamedTuple):
    node: int
    side: str


class Arm(NamedTuple):
    edge: int
    gen: int


class Attempt(NamedTuple):
    edge: int
    gen: int
    t_emit: int


class Tag(NamedTuple):
    edge: int
    gen: int


class Pair(NamedTuple):
    left: int
    right: int
    t_left: int          # left half's state is settled up to this instant
    t_right: int
    state: object        # colour of the pair algebra (density matrix or Bell vector)


class SwapBits(NamedTuple):
    node: int
    m1: int
    m2: int


class Frame(NamedTuple):
    count: int
    m1: int
    m2: int


# --- the net ---

def start_offsets(params: ChainParams, mode: str) -> tuple[int, ...]:
    """First-shot delay per edge: 0 except under tad, where the slowest expected edge
    anchors at 0 and the rest wait so all expected completions coincide."""
    k = params.num_edges
    if mode != TAD:
        return (0,) * k
    ets = [expected_completion(params.rtt(i), params.edges[i].survival) for i in range(k)]
    et_max = max(ets)
    return tuple(round(et_max - et) for et in ets)


def build_net(params: ChainParams, mode: str = SEQUENTIAL, seed: int | None = None,
              record: bool = False, algebra: PairAlgebra | None = None) -> TimedColoredPetriNet:
    """The folded net with its initial marking. `algebra` picks the pair-state colour
    (default DensityMatrix, exact against the simulator)."""
    if mode not in (SEQUENTIAL, PARALLEL, TAD):
        raise ValueError(f"mode must be {SEQUENTIAL!r}, {PARALLEL!r} or {TAD!r}, got {mode!r}")
    alg = DensityMatrix() if algebra is None else algebra
    alg.check(params)
    K = params.num_edges
    N, E = params.nodes, params.edges
    if mode == PARALLEL:
        period = max(params.rtt(i) for i in range(K)) + 1
        timeout = [period] * K
    else:
        timeout = [params.rtt(i) + 1 for i in range(K)]
    offsets = start_offsets(params, mode)

    net = TimedColoredPetriNet(f"entanglement-chain-{K}hop-{mode}", seed=seed, record=record)
    net.add_place("FreeMem", Slot, "empty link memories")
    net.add_place("Armed", Arm, "emitter may fire its next attempt")
    net.add_place("InFlight", Attempt, "idler stored at the emitter, signal photon in the fiber")
    net.add_place("Dangling", Attempt, "idler stored at the emitter, photon lost")
    net.add_place("Entangled", Pair, "heralded pairs held in memory")
    net.add_place("Pending", Tag, "emitter waits for the LINK_ACK of attempt g")
    net.add_place("Timer", Tag, "emitter's retry timer for attempt g")
    net.add_place("AckChan", Tag, "LINK_ACK in flight on the classical fiber")
    net.add_place("Receipt", Tag, "emitter saw the ACK; retires the stale timer")
    net.add_place("DownReady", int, "interior node knows its downstream link is up")
    net.add_place("SwapMsg", SwapBits, "SWAP_RESULT in flight to the destination")
    net.add_place("Frame", Frame, "destination's Pauli frame: swaps reported, XOR of bits")
    net.add_place("Delivered", Pair, "end-to-end pair after the frame correction")

    # -- link generation --

    def emit(b, ctx):
        i, g = b["arm"].edge, b["arm"].gen + 1
        return (Out("InFlight", Attempt(i, g, ctx.now), E[i].qdelay),
                Out("Pending", Tag(i, g)),
                Out("Timer", Tag(i, g), timeout[i]))

    net.add_transition(
        "Emit",
        inputs=(In("Armed", "arm", inscription="(i,n)"),
                In("FreeMem", "slot", lambda b, s: s == (b["arm"].edge, DOWN), "(i,down)")),
        outputs={"InFlight": "(i,n+1,now)@+qd(i)", "Pending": "(i,n+1)",
                 "Timer": "(i,n+1)@+timeout(i)"},
        action=emit, priority=PROTOCOL,
        doc="n_i emits one pair on edge i: idler into its down memory, signal into the fiber")

    def absorber_slot(b, s):
        return s == (b["att"].edge + 1, UP)

    def herald(b, ctx):
        att = b["att"]
        i = att.edge
        outs = [Out("Entangled", Pair(i, i + 1, att.t_emit, ctx.now, alg.raw(N[i]))),
                Out("AckChan", Tag(i, att.gen), E[i].cdelay)]
        if mode == SEQUENTIAL and i + 1 < K:
            outs.append(Out("Armed", Arm(i + 1, 0)))
        return outs

    net.add_transition(
        "Herald",
        inputs=(In("InFlight", "att", inscription="(i,g,te)"),
                In("FreeMem", "slot", absorber_slot, "(i+1,up)")),
        outputs={"Entangled": "(i,i+1,te,now,raw(n_i))", "AckChan": "(i,g)@+cd(i)",
                 "Armed": "(i+1,0) [sequential, interior]"},
        action=herald, weight=lambda b: E[b["att"].edge].survival, priority=ARRIVAL,
        doc="photon survives the fiber and is absorbed: the pair now exists at both ends")

    net.add_transition(
        "Lose",
        inputs=(In("InFlight", "att", inscription="(i,g,te)"),
                In("FreeMem", "slot", absorber_slot, "(i+1,up)")),
        outputs={"Dangling": "(i,g,te)", "FreeMem": "(i+1,up)"},
        action=lambda b, ctx: (Out("Dangling", b["att"]), Out("FreeMem", b["slot"])),
        weight=lambda b: 1.0 - E[b["att"].edge].survival, priority=ARRIVAL,
        doc="photon lost in the fiber; the emitter's idler dangles until its timer fires")

    def ack(b, ctx):
        tag = b["msg"]
        outs = [Out("Receipt", tag)]
        if tag.edge > 0:                              # the source has nothing to swap
            outs.append(Out("DownReady", tag.edge))
        return outs

    net.add_transition(
        "Ack",
        inputs=(In("AckChan", "msg", inscription="(i,g)"),
                In("Pending", "pend", lambda b, p: p == b["msg"], "(i,g)")),
        outputs={"Receipt": "(i,g)", "DownReady": "i [if i > 0]"},
        action=ack, priority=ARRIVAL,
        doc="LINK_ACK reaches the emitter: it now knows the link is up")

    net.add_transition(
        "Timeout",
        inputs=(In("Timer", "tm", inscription="(i,g)"),
                In("Pending", "pend", lambda b, p: p == b["tm"], "(i,g)"),
                In("Dangling", "lost", lambda b, d: (d.edge, d.gen) == b["tm"], "(i,g,te)")),
        outputs={"Armed": "(i,g)", "FreeMem": "(i,down)"},
        action=lambda b, ctx: (Out("Armed", Arm(*b["tm"])), Out("FreeMem", Slot(b["tm"].edge, DOWN))),
        priority=PROTOCOL,
        doc="no ACK by the deadline: discard the idler, re-arm the emitter")

    net.add_transition(
        "Cancel",
        inputs=(In("Timer", "tm", inscription="(i,g)"),
                In("Receipt", "rc", lambda b, r: r == b["tm"], "(i,g)")),
        outputs={},
        action=lambda b, ctx: (),
        priority=PROTOCOL,
        doc="a heralded attempt's timer expires harmlessly")

    # -- swapping and delivery --

    def swap(b, ctx):
        r, lp, rp = b["r"], b["lp"], b["rp"]
        node = N[r]
        vl = alg.idle(lp.state, RIGHT, node, ctx.now - lp.t_right)     # inner halves decohere up to the BSM
        vr = alg.idle(rp.state, LEFT, node, ctx.now - rp.t_left)
        v, m1, m2 = alg.swap(vl, vr, node, ctx.rng)
        done = node.bsm_duration
        return (Out("Entangled", Pair(lp.left, rp.right, lp.t_left, rp.t_right, v)),
                Out("SwapMsg", SwapBits(r, m1, m2), done + params.dest_cdelay[r]),
                Out("FreeMem", Slot(r, UP), done),
                Out("FreeMem", Slot(r, DOWN), done))

    net.add_transition(
        "Swap",
        inputs=(In("DownReady", "r", inscription="r"),
                In("Entangled", "lp", lambda b, p: p.right == b["r"], "(a,r,..)"),
                In("Entangled", "rp", lambda b, p: p.left == b["r"], "(r,b,..)")),
        outputs={"Entangled": "(a,b,..)", "SwapMsg": "(r,m1,m2)@+bsm(r)+cd(r,K)",
                 "FreeMem": "(r,up),(r,down)@+bsm(r)"},
        action=swap, priority=OPERATION,
        doc="interior n_r Bell-measures its two halves as soon as both links are known up")

    net.add_transition(
        "Collect",
        inputs=(In("SwapMsg", "msg", inscription="(r,m1,m2)"),
                In("Frame", "f", inscription="(n,a1,a2)")),
        outputs={"Frame": "(n+1,a1^m1,a2^m2)"},
        action=lambda b, ctx: (Out("Frame", Frame(b["f"].count + 1,
                                                  b["f"].m1 ^ b["msg"].m1,
                                                  b["f"].m2 ^ b["msg"].m2)),),
        priority=ARRIVAL,
        doc="the destination folds one SWAP_RESULT into its Pauli frame")

    def complete(b, ctx):
        f, p = b["f"], b["p"]
        src, dst = N[0], N[K]
        v, t_right, delay = p.state, p.t_right, 0
        if K > 1:                                     # X^a2 then Z^a1 on the destination half
            v = alg.idle(v, RIGHT, dst, ctx.now - t_right)   # settle first: T1 does not commute with X
            v = alg.correct(v, dst, f.m1, f.m2)
            t_right, delay = ctx.now, dst.correct_duration
        t = ctx.now + delay
        v = alg.idle(alg.idle(v, LEFT, src, t - p.t_left), RIGHT, dst, t - t_right)
        return (Out("Delivered", Pair(0, K, t, t, v), delay),)

    net.add_transition(
        "Complete",
        inputs=(In("Frame", "f", lambda b, f: f.count == K - 1, "(K-1,a1,a2)"),
                In("Entangled", "p", lambda b, p: p.left == 0 and p.right == K, "(0,K,..)")),
        outputs={"Delivered": "(0,K,t,t,v)@+corr"},
        action=complete, priority=OPERATION,
        doc="every swap reported and the last link heralded: apply the frame, deliver")

    # -- initial marking --
    for j in range(K + 1):
        if j < K:
            net.put("FreeMem", Slot(j, DOWN))
        if j > 0:
            net.put("FreeMem", Slot(j, UP))
    for i in ([0] if mode == SEQUENTIAL else range(K)):
        net.put("Armed", Arm(i, 0), at=offsets[i])
    net.put("Frame", Frame(0, 0, 0))
    return net


@dataclass(slots=True, frozen=True)
class ChainResult:
    latency: int                  # ps from t=0 to the delivered pair
    fidelity: float               # Phi+ fidelity of the delivered pair
    state: object                 # its colour: density matrix or Bell vector
    attempts: tuple[int, ...]     # emissions per edge, lost photons included
    offsets: tuple[int, ...]      # first-shot delay per edge (0 outside tad)
    firings: int
    net: TimedColoredPetriNet = field(repr=False, compare=False)


def simulate(params: ChainParams, mode: str = SEQUENTIAL, seed: int | None = None,
             record: bool = False, algebra: PairAlgebra | None = None) -> ChainResult:
    """Play the net until the end-to-end pair is delivered."""
    alg = DensityMatrix() if algebra is None else algebra
    net = build_net(params, mode, seed=seed, record=record, algebra=alg)
    attempts = [0] * params.num_edges

    def count(firing) -> None:
        if firing.transition == "Emit":
            attempts[firing.binding["arm"].edge] += 1

    net.observers.append(count)
    steps = net.run(stop=lambda n: bool(n.tokens("Delivered")))
    delivered = net.tokens("Delivered")
    if not delivered:
        raise ProtocolError(f"{net.name}: dead marking at t={net.clock} before delivery")
    tok = delivered[0]
    return ChainResult(latency=tok.time, fidelity=alg.fidelity(tok.color.state),
                       state=tok.color.state, attempts=tuple(attempts),
                       offsets=start_offsets(params, mode), firings=steps, net=net)
