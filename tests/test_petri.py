"""Tests for the timed coloured Petri net kernel and the entanglement net built on it.

Run:  uv run pytest tests/test_petri.py
"""

import math
import statistics

import numpy as np
import pytest

from qetwork.kernel.timeline import Timeline
from qetwork.petri import bell
from qetwork.petri.entanglement import ChainParams, EdgeParams, NodeParams, build_net, simulate
from qetwork.petri.pair_state import LEFT, RIGHT, BellDiagonal, DensityMatrix
from qetwork.petri.tcpn import In, Out, TimedColoredPetriNet
from qetwork.petri.validate import chain_spec, lossless, run_simulator
from qetwork.protocols.e_dist_swap import PARALLEL, SEQUENTIAL, TAD
from qetwork.topologies.topology_spec import TopologySpec

MODES = (SEQUENTIAL, PARALLEL, TAD)

# Bell basis in label order I, Z, X, Y -> Phi+, Phi-, Psi+, Psi-
_S = 1 / math.sqrt(2)
BELL_BASIS = np.array([[_S, 0, 0, _S], [_S, 0, 0, -_S], [0, _S, _S, 0], [0, _S, -_S, 0]], dtype=complex)


def to_density(v) -> np.ndarray:
    return sum(p * np.outer(b, b.conj()) for p, b in zip(v, BELL_BASIS))


def bell_coeffs(rho) -> np.ndarray:
    return np.array([(b.conj() @ rho @ b).real for b in BELL_BASIS])


def random_bell(rng) -> tuple:
    return tuple(rng.dirichlet(np.ones(4)))


# --- kernel ---

def _net(**kw):
    net = TimedColoredPetriNet("t", seed=kw.pop("seed", 0))
    net.add_place("A", int)
    net.add_place("B", int)
    return net


def test_token_waits_for_its_timestamp():
    net = _net()
    net.add_transition("t", [In("A", "x")], {"B": "x"}, lambda b, c: [Out("B", b["x"], 5)])
    net.put("A", 1, at=10)
    f = net.step()
    assert (f.time, net.clock) == (10, 10)
    assert net.tokens("B")[0].time == 15


def test_lower_priority_value_fires_first_on_a_tie():
    net = _net()
    net.add_place("C", int)
    net.add_transition("late", [In("A", "x")], {"B": "x"}, lambda b, c: [Out("B", b["x"])], priority=20)
    net.add_transition("early", [In("C", "x")], {"B": "x"}, lambda b, c: [Out("B", b["x"])], priority=0)
    net.put("A", 1)
    net.put("C", 2)
    assert [net.step().transition for _ in range(2)] == ["early", "late"]


def test_until_leaves_the_clock_alone():
    net = _net()
    net.add_transition("t", [In("A", "x")], {}, lambda b, c: [])
    net.put("A", 1, at=100)
    assert net.step(until=99) is None and net.clock == 0
    assert net.step(until=100) is not None and net.clock == 100


def test_two_arcs_on_one_place_take_distinct_tokens():
    net = _net()
    net.add_transition("pair", [In("A", "x"), In("A", "y")], {"B": "x+y"},
                       lambda b, c: [Out("B", b["x"] + b["y"])])
    net.put("A", 3)
    assert net.step() is None                      # one token cannot bind both arcs
    net.put("A", 4)
    net.step()
    assert net.colors("B") == [7] and net.colors("A") == []


def test_arc_filter_joins_on_earlier_variables():
    net = _net()
    net.add_transition("match", [In("A", "x"), In("B", "y", lambda b, y: y == b["x"])], {}, lambda b, c: [])
    net.put("A", 1)
    net.put("B", 2)
    assert net.step() is None
    net.put("B", 1)
    assert net.step() is not None and net.colors("B") == [2]


def test_weighted_conflict_is_a_bernoulli_switch():
    net = _net(seed=7)
    net.add_place("Hit", int)
    net.add_place("Miss", int)
    net.add_transition("hit", [In("A", "x")], {"Hit": "x"}, lambda b, c: [Out("Hit", 1)], weight=lambda b: 0.3)
    net.add_transition("miss", [In("A", "x")], {"Miss": "x"}, lambda b, c: [Out("Miss", 1)], weight=lambda b: 0.7)
    n = 20_000
    for _ in range(n):
        net.put("A", 0)
        net.step()
    k = len(net.tokens("Hit"))
    assert abs(k - 0.3 * n) < 4 * math.sqrt(n * 0.3 * 0.7)


def test_zero_weight_disables():
    net = _net()
    net.add_transition("never", [In("A", "x")], {}, lambda b, c: [], weight=lambda b: 0.0)
    net.put("A", 1)
    assert net.step() is None


def test_structure_and_colour_errors_raise():
    net = _net()
    with pytest.raises(ValueError):
        net.add_transition("bad", [In("Nope", "x")], {}, lambda b, c: [])
    with pytest.raises(TypeError):
        net.put("A", "not an int")
    with pytest.raises(ValueError):
        net.put("A", 1, at=-1)
    net.add_transition("leak", [In("A", "x")], {"B": "x"}, lambda b, c: [Out("A", 1)])
    net.put("A", 1)
    with pytest.raises(ValueError, match="undeclared"):
        net.step()


# --- pair-state algebras ---

def _node(**kw) -> NodeParams:
    return NodeParams(**kw)


def test_bell_swap_and_bsm_noise_match_the_density_circuit():
    """Pauli noise only: the gate-level BSM + frame correction must leave exactly the
    convolved Bell vector, for every Born-sampled outcome."""
    rng = np.random.default_rng(3)
    node = _node(p_depol_1q=0.02, p_depol_2q=0.05)
    ideal_dst = _node()                                  # noiseless correction: isolates the BSM
    bd, dm = BellDiagonal(), DensityMatrix()
    for _ in range(20):
        p, q = random_bell(rng), random_bell(rng)
        expect, _, _ = bd.swap(p, q, node, rng)
        rho, m1, m2 = dm.swap(to_density(p), to_density(q), node, rng)
        rho = dm.correct(rho, ideal_dst, m1, m2)
        assert np.allclose(rho, to_density(expect), atol=1e-12)


def test_bell_correction_noise_matches_density():
    """Depolarizing commutes with the Pauli it follows, so a noisy X^m2 Z^m1 is the ideal
    correction followed by one depolarizing channel per applied gate -- the Bell model."""
    rng = np.random.default_rng(4)
    dm, noisy = DensityMatrix(), _node(p_depol_1q=0.03)
    for m1 in (0, 1):
        for m2 in (0, 1):
            rho = to_density(random_bell(rng))
            ideal = bell_coeffs(dm.correct(rho, _node(), m1, m2))
            got = bell_coeffs(dm.correct(rho, noisy, m1, m2))
            expect = BellDiagonal().correct(tuple(ideal), noisy, m1, m2)
            assert np.allclose(got, expect, atol=1e-12)


def test_dephasing_idle_is_exact_in_the_bell_model():
    rng = np.random.default_rng(5)
    node = _node(t2=3e8)
    for side in (LEFT, RIGHT):
        v = random_bell(rng)
        got = DensityMatrix().idle(to_density(v), side, node, 1e8)
        assert np.allclose(got, to_density(BellDiagonal().idle(v, side, node, 1e8)), atol=1e-12)


def test_one_damping_step_keeps_phi_plus_fidelity_exact_under_the_twirl():
    node = _node(t1=1e9, t2=5e8)
    v = bell.werner(0.9)
    for side in (LEFT, RIGHT):
        f_dm = DensityMatrix().fidelity(DensityMatrix().idle(to_density(v), side, node, 7e8))
        f_bd = bell.fidelity(BellDiagonal().idle(v, side, node, 7e8))
        assert f_dm == pytest.approx(f_bd, abs=1e-12)


def test_bell_algebra_refuses_coherent_physics():
    params = _chain(nodes=_node(source_phase=0.3))
    with pytest.raises(ValueError, match="coherent"):
        build_net(params, algebra=BellDiagonal())


# --- the entanglement net ---

def _chain(k=3, nodes=None, survival=0.5) -> ChainParams:
    nd = nodes or _node(t2=1e9, p_depol_2q=0.01, visibility=0.97, bsm_duration=1000, correct_duration=20)
    qd = [50_000_000, 80_000_000, 30_000_000][:k] + [40_000_000] * max(0, k - 3)
    edges = tuple(EdgeParams(qdelay=d, cdelay=d, survival=survival) for d in qd)
    dest = tuple(sum(qd[r:]) for r in range(k + 1))
    return ChainParams((nd,) * (k + 1), edges, dest)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("lengths", [[20_000.0], [20_000.0, 35_000.0], [20_000.0, 35_000.0, 10_000.0]])
@pytest.mark.parametrize("algebra", [DensityMatrix, BellDiagonal])
def test_lossless_chain_equals_the_simulator_exactly(mode, lengths, algebra):
    """Pauli noise + lossless fibers: both models are deterministic, so latency,
    attempts and the delivered fidelity must agree exactly."""
    spec = lossless(chain_spec(lengths, t2=1e9, p1q=0.0, p2q=0.02, visibility=0.97))
    lat_s, f_s, att_s = run_simulator(spec, mode, seed=1)
    params = ChainParams.from_network(TopologySpec(spec).materialize(Timeline(seed=1)))
    r = simulate(params, mode, seed=2, algebra=algebra())
    assert (r.latency, r.attempts) == (lat_s, att_s)
    assert r.fidelity == pytest.approx(f_s, abs=1e-12)


def _slot_census(net, k):
    """Memory P-invariant: every link memory is free, or holds exactly one half."""
    count = {}
    for s in net.colors("FreeMem"):
        count[tuple(s)] = count.get(tuple(s), 0) + 1
    for place in ("InFlight", "Dangling"):
        for a in net.colors(place):
            count[(a.edge, "down")] = count.get((a.edge, "down"), 0) + 1
    for place in ("Entangled", "Delivered"):
        for p in net.colors(place):
            count[(p.left, "down")] = count.get((p.left, "down"), 0) + 1
            count[(p.right, "up")] = count.get((p.right, "up"), 0) + 1
    slots = {(j, "down") for j in range(k)} | {(j, "up") for j in range(1, k + 1)}
    return count, slots


@pytest.mark.parametrize("mode", MODES)
def test_memory_invariant_holds_and_the_net_drains_cleanly(mode):
    k = 4
    params = _chain(k, survival=0.4)
    for seed in range(15):
        net = build_net(params, mode, seed=seed, algebra=BellDiagonal())

        def check(_firing, net=net):
            count, slots = _slot_census(net, k)
            assert set(count) == slots and all(c == 1 for c in count.values()), count

        net.observers.append(check)
        net.run()                                     # past delivery, to the dead marking
        assert len(net.tokens("Delivered")) == 1
        for place in ("Armed", "InFlight", "Dangling", "Pending", "Timer", "AckChan",
                      "Receipt", "DownReady", "SwapMsg", "Entangled", "Frame"):
            assert net.tokens(place) == (), place
        assert len(net.tokens("FreeMem")) == 2 * (k - 1)     # every interior memory freed by its swap


def test_single_hop_latency_matches_the_geometric_closed_form():
    """K=1 sequential: latency = qd + (N-1)(rtt+1), N ~ Geometric(p)."""
    p = 0.35
    params = _chain(1, survival=p)
    qd, rtt = params.edges[0].qdelay, params.rtt(0)
    xs = [simulate(params, SEQUENTIAL, seed=s, algebra=BellDiagonal()).latency for s in range(3000)]
    expect = qd + (1 / p - 1) * (rtt + 1)
    se = statistics.stdev(xs) / math.sqrt(len(xs))
    assert abs(statistics.fmean(xs) - expect) < 4 * se


def test_tad_offsets_align_expected_completions():
    params = _chain(3, survival=0.5)
    r = simulate(params, TAD, seed=0, algebra=BellDiagonal())
    from qetwork.protocols.e_dist_swap import expected_completion
    ets = [expected_completion(params.rtt(i), 0.5) + r.offsets[i] for i in range(3)]
    assert max(ets) - min(ets) <= 1 and min(r.offsets) == 0


def test_from_network_reads_delays_and_survival():
    spec = chain_spec([20_000.0, 10_000.0], attenuation=0.2)
    net = TopologySpec(spec).materialize(Timeline(seed=0))
    params = ChainParams.from_network(net)
    c0, c1 = net.nodes["c0"], net.nodes["c1"]
    assert params.edges[0].qdelay == c0.ports["q:c1"].qfiber.delay
    assert params.edges[0].survival == pytest.approx(10 ** (-0.4))
    assert params.dest_cdelay == (c0.cfibers["c2"].delay, c1.cfibers["c2"].delay, 0)
    assert params.nodes[1].bsm_duration == c1.bsm_duration


def test_exports_render_every_place_and_transition():
    net = build_net(_chain(2), SEQUENTIAL)
    dot, mer = net.to_dot(), net.to_mermaid()
    for name in list(net.places) + list(net.transitions):
        assert name in dot and name in mer
