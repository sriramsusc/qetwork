"""Network benchmarking (Helsen & Wehner, arXiv:2103.01165), fully event-driven.

Transport is the entanglement-distribution stack: every teleport of the data
qubit rides on an end-to-end pair delivered by EntanglementDistribution (per-node
agents walking the path link by link -- generation, herald, loss + retry, swaps
as-ready, one deferred Pauli frame at the destination). One BOUNCE runs that
distribution in the two directions one after the other: Clifford at the source,
distribute + teleport S->D, Clifford at the destination, distribute + teleport
D->S. The data qubit only ever lives at the two ends; interior gate noise enters
through the swaps, and the data qubit pays memory decoherence for every leg's
full distribution latency.

mode is EntanglementDistribution's: sequential (absorb->emit baton), parallel
(round-grid emission) or tad (expectation-aligned staggered starts, own-rtt
retries). purify_rounds >= 1 switches the distribution to
purify-then-swap: every edge is pumped to that level (LinkPumpSession, repeated
DEJMPS), the kept pairs are swapped into the end-to-end pair; edges pump one
after another (sequential mode only) and earlier kept pairs decohere while later
edges pump -- that cost is real.

Exact readout + perfect prep => no SPAM => the survival fits b_m = f^m and
F_path = (1 + f) / 2; with the detector readout SPAM lands in A, not f.
measure() is the scalar interface: samples + bounce range + path in,
(path_fidelity, avg_time_us) out.
"""

from dataclasses import dataclass

import numpy as np

from qetwork.operations.gates import random_clifford, X, apply_gate
from qetwork.events.priority import PROTOCOL
from qetwork.protocols.errors import ProtocolError
from qetwork.protocols.e_dist_swap import EntanglementDistribution, SEQUENTIAL, PARALLEL, TAD
from qetwork.protocols.link_pump import LinkPumpSession


@dataclass(slots=True)
class BenchmarkResult:
    path: list[str]
    hops: int
    f: float
    F_path: float
    A: float
    decay: dict[int, float]
    raw: dict[int, list[float]]
    clock_ps: int



class NetworkBenchmark:
    """Event-driven network benchmarking over a fixed path; the data qubit
    teleports over end-to-end pairs delivered by EntanglementDistribution."""

    def __init__(self, net, path=None, m_min=1, m_max=8, n_samples=40, n_shots=0,
                 purify_rounds=0, calibrate=True, mode=SEQUENTIAL):
        if mode not in (SEQUENTIAL, PARALLEL, TAD):
            raise ValueError(f"mode must be {SEQUENTIAL!r}, {PARALLEL!r} or {TAD!r}, got {mode!r}")
        self.mode = mode
        self.net = net
        self.tl = net.timeline
        self.path = list(path) if path is not None else net.path()
        if len(self.path) < 2:
            raise ValueError(f"path needs at least 2 nodes, got {self.path}")
        self.nodes = [net.nodes[nid] for nid in self.path]
        self.detector = next(iter(self.nodes[0].detectors.values()), None)   # source node's detector, if any
        self.K = len(self.nodes)
        self.m_min, self.m_max, self.n_samples, self.n_shots = m_min, m_max, n_samples, n_shots
        if not isinstance(purify_rounds, int) or isinstance(purify_rounds, bool) or purify_rounds < 0:
            raise ValueError(f"purify_rounds must be an int >= 0, got {purify_rounds!r}")
        if purify_rounds >= 1 and mode != SEQUENTIAL:
            raise ValueError("purified distribution pumps edges sequentially; "
                             f"only {SEQUENTIAL!r} mode supports it, got {mode!r}")
        self.purify_rounds = purify_rounds
        self.calibrate = calibrate
        for end in (self.nodes[0], self.nodes[-1]):
            if not end.unbound_mems():
                raise ValueError(f"path endpoint {end.node_id!r} needs a spare (unbound) memory "
                                 f"to park the data qubit at the turnaround")
        # pump sessions replace the raw agents' hooks, so the two transports are
        # mutually exclusive per benchmark instance
        self.dist = (EntanglementDistribution(net, self.path, mode=mode)
                     if purify_rounds < 1 else None)
        self.s_half_id = self.path[1]     # source's pair half lives in link_mems[here]
        self.d_half_id = self.path[-2]    # destination's half

    # --- small verbs ---

    def _cdelay(self, A, B):
        cf = A.cfibers[B.node_id]
        return cf.delay + cf.latency

    def _noop(self):
        pass

    def _do_correct(self, B, mem, m1, m2):
        B.correct(mem, m1, m2)

    def _park(self, node, data):                     # lossless key move (no SWAP, no error)
        spare = next(m for m in node.unbound_mems() if m.is_empty())
        spare.key, spare._load_time = data.key, data._load_time
        data.key, data._load_time = None, None
        return spare

    def _apply_clifford(self, node, data, u_total, rng):
        u = random_clifford(rng)
        apply_gate(self.tl.state_tracker, (data.key,), u, node.p_depol_1q, coherent=node.coherent_1q)
        return u @ u_total

    def _cleanup(self):
        for node in self.nodes:
            for mem in list(node.memories):
                mem.reset()

    # --- one leg: distribute an end-to-end pair, teleport `data` over it ---

    def _teleport(self, A, B, data):
        """Distribute one end-to-end pair (raw or purified), then teleport `data`
        from A to B over it.

        The BSM settles `data`'s decoherence over the whole distribution wait; the
        correction lands after bsm_duration + the classical A->B flight."""
        tl = self.tl
        a_half_id = self.s_half_id if A is self.nodes[0] else self.d_half_id
        b_half_id = self.d_half_id if B is self.nodes[-1] else self.s_half_id
        if self.purify_rounds >= 1:
            data, pair_a, pair_b = self._purified_pair(A, B, a_half_id, b_half_id, data)
        else:
            if data is A.link_mems[a_half_id]:    # dist.reset() would wipe the data qubit
                data = self._park(A, data)
            self.dist.reset()
            if self.dist.run() is None:
                raise ProtocolError("entanglement distribution starved before delivering")
            pair_a, pair_b = A.link_mems[a_half_id], B.link_mems[b_half_id]
        m1, m2 = A.bsm(data, pair_a)
        tl.schedule(self._do_correct, B, pair_b, m1, m2,
                    at=tl.now() + A.bsm_duration + self._cdelay(A, B),
                    priority=PROTOCOL)
        tl.run()                                  # bits fly, correction settles pair_b
        return pair_b

    def _purified_pair(self, A, B, a_half_id, b_half_id, data):
        """Purify-then-swap: pump every edge to purify_rounds kept pairs, swap the
        kept pairs into one end-to-end pair (single XOR Pauli frame, corrected at
        the path destination). Returns (data, A's half, B's half); `data` moves to
        a spare first when it occupies a slot the pumping would recycle."""
        tl = self.tl
        kept_a = A.ensure_memory(f"purify:kept:{a_half_id}")
        if data is A.link_mems[a_half_id] or data is kept_a:
            data = self._park(A, data)
        for u, v in zip(self.path, self.path[1:]):       # pump edge by edge
            LinkPumpSession(self.net, u, v, rounds=self.purify_rounds,
                            calibrate=self.calibrate).run()
        dest = self.nodes[-1]
        kept_dest = dest.memory(f"purify:kept:{self.d_half_id}")
        if self.K > 2:                                   # swap cascade + one frame
            acc1 = acc2 = 0
            latest = tl.now()
            for i in range(1, self.K - 1):
                node = self.nodes[i]
                m1, m2 = node.bsm(node.memory(f"purify:kept:{self.path[i - 1]}"),
                                  node.memory(f"purify:kept:{self.path[i + 1]}"))
                acc1 ^= m1
                acc2 ^= m2
                latest = max(latest, tl.now() + node.bsm_duration + self._cdelay(node, dest))
            tl.schedule(self._do_correct, dest, kept_dest, acc1, acc2,
                        at=latest, priority=PROTOCOL)
            tl.schedule(self._noop, at=latest + dest.correct_duration, priority=PROTOCOL)
            tl.run()                                     # frame lands, pair is standard
        return data, A.memory(f"purify:kept:{a_half_id}"), B.memory(f"purify:kept:{b_half_id}")

    # --- one RB sequence of m bounces ---

    def _one_sequence(self, m, rng):
        tracker = self.tl.state_tracker
        S, D = self.nodes[0], self.nodes[-1]
        data = next(mm for mm in S.unbound_mems() if mm.is_empty())
        data.initialize()
        u_total = np.eye(2, dtype=complex)
        for _ in range(m):
            u_total = self._apply_clifford(S, data, u_total, rng)
            data = self._teleport(S, D, data)
            u_total = self._apply_clifford(D, data, u_total, rng)
            data = self._teleport(D, S, data)
        p = int(rng.integers(0, 2))
        g_inv = (X if p else np.eye(2, dtype=complex)) @ u_total.conj().T
        apply_gate(tracker, (data.key,), g_inv, S.p_depol_1q, coherent=S.coherent_1q)
        if self.detector is None:                          # exact readout: no SPAM
            rho00 = float(tracker.get(data.key).matrix[0, 0].real)
            p_correct = rho00 if p == 0 else 1.0 - rho00
            b = 2.0 * p_correct - 1.0
            if self.n_shots > 0:
                std = np.sqrt(max(0.0, (1.0 - b * b) / self.n_shots))
                b = float(np.clip(b + rng.normal(0.0, std), -1.0, 1.0))
            self._cleanup()
            return b
        # realistic readout: emit the recovered qubit -> MZI + SNSPD
        photon = data.emit()
        before = len(self.detector.detections)
        self.detector.get(photon)                          # MZI analyze + Z measure + SNSPD click
        self.tl.run()                                      # the click fires (or not)
        new = self.detector.detections[before:]
        self._cleanup()
        if not new:
            return None                                    # no click (loss/efficiency) -> post-select
        arm = new[-1][0]
        return 1.0 if arm == ("eig1" if p == 0 else "eig2") else -1.0

    # --- drivers ---

    def run(self) -> BenchmarkResult:
        rng = self.tl.rng
        self._cleanup()
        raw = {m: [b for b in (self._one_sequence(m, rng) for _ in range(self.n_samples)) if b is not None]
               for m in range(self.m_min, self.m_max + 1)}
        decay = {m: (float(np.mean(v)) if v else 0.0) for m, v in raw.items()}
        f, F_path, A = fit_decay(decay, spam=self.detector is not None)
        return BenchmarkResult(self.path, self.K - 1, f, F_path, A, decay, raw, self.tl.now())

    def measure(self):
        """The scalar interface: (path_fidelity, avg_time_us).

        path_fidelity = F_path = (1 + f)/2 from the b_m = A*f^m fit (SPAM-robust);
        avg_time_us = (sum_m time_m / m) / n_m where time_m is the virtual clock
        the whole m-group of sequences consumed."""
        rng = self.tl.rng
        self._cleanup()
        raw, time_m = {}, {}
        for m in range(self.m_min, self.m_max + 1):
            t0 = self.tl.now()
            vals = [self._one_sequence(m, rng) for _ in range(self.n_samples)]
            time_m[m] = self.tl.now() - t0                       # clock this m-group burned
            raw[m] = [v for v in vals if v is not None]          # post-select clicks
        decay = {m: (float(np.mean(v)) if v else 0.0) for m, v in raw.items()}
        _f, f_path, _a = fit_decay(decay, spam=self.detector is not None)
        avg_time = sum(time_m[m] / m for m in time_m) / len(time_m)   # (sum_m time_m/m)/n_m, ps
        return f_path, avg_time / 1e6                             # us



def fit_decay(decay, spam=False):
    """b_m = A*f^m. Exact readout: A=1. Detector readout (SPAM): fit A in [0,1]."""
    ms = np.array(sorted(decay), dtype=float)
    ys = np.array([decay[m] for m in sorted(decay)], dtype=float)

    def best(fs):
        b = (np.inf, 1.0, 0.0)
        for f in fs:
            fm = f ** ms
            d = float(np.dot(fm, fm))
            a = float(np.clip(np.dot(ys, fm) / d, 0.0, 1.0)) if (spam and d > 0) else 1.0
            r = float(np.sum((ys - a * fm) ** 2))
            if r < b[0]:
                b = (r, a, f)
        return b

    _, a, f = best(np.linspace(0.0, 1.0, 1001))
    _, a, f = best(np.linspace(max(0.0, f - 0.001), min(1.0, f + 0.001), 400))
    return float(f), 0.5 * (1.0 + float(f)), float(a)
