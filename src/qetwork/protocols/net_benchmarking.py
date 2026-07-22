"""Network benchmarking (Helsen & Wehner, arXiv:2103.01165), fully event-driven.

Every hop advances the clock through the same machinery as EntanglementDistribution:
`attempt_link` schedules the photon over the fiber, the `on_absorb` herald confirms it,
loss triggers a real retry, and the teleport correction is a scheduled classical message.
`timeline.run()` drains each hop's events, so memory T1/T2 decoherence accrues over the
real elapsed storage time (fiber flight + retries + classical delay). Exact readout + perfect
prep => no SPAM => the survival fits b_m = f^m and F_path = (1 + f) / 2.
"""

from dataclasses import dataclass

import numpy as np

from qetwork.operations.gates import random_clifford, X, apply_gate
from qetwork.events.priority import PROTOCOL
from qetwork.protocols.e_dist_swap import SEQUENTIAL, PARALLEL
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
    """Hop-by-hop, event-driven network benchmarking over a fixed path."""

    def __init__(self, net, path=None, m_min=1, m_max=8, n_samples=40, n_shots=0,
                 purify_rounds=0, calibrate=True, mode=SEQUENTIAL):
        if mode not in (SEQUENTIAL, PARALLEL):
            raise ValueError(f"mode must be {SEQUENTIAL!r} or {PARALLEL!r}, got {mode!r}")
        # RESERVED knob, validated + stored but with NO behavioral difference yet: the
        # hop-by-hop bounce pumps each link on demand either way. Intended future meaning:
        # sequential = pump the next link when the data qubit needs it (current behavior);
        # parallel = pre-pump all of a sweep's links concurrently before the data hops.
        self.mode = mode
        self.net = net
        self.tl = net.timeline
        self.path = list(path) if path is not None else net.path()
        if len(self.path) < 2:
            raise ValueError(f"path needs at least 2 nodes, got {self.path}")
        self.nodes = [net.nodes[nid] for nid in self.path]                    # <-- restore
        self.detector = next(iter(self.nodes[0].detectors.values()), None)   # source node's detector, if any
        self.K = len(self.nodes)
        self.m_min, self.m_max, self.n_samples, self.n_shots = m_min, m_max, n_samples, n_shots
        if not isinstance(purify_rounds, int) or isinstance(purify_rounds, bool) or purify_rounds < 0:
            raise ValueError(f"purify_rounds must be an int >= 0, got {purify_rounds!r}")
        self.purify_rounds = purify_rounds
        self.calibrate = calibrate        
        for end in (self.nodes[0], self.nodes[-1]):
            if not end.unbound_mems():
                raise ValueError(f"path endpoint {end.node_id!r} needs a spare (unbound) memory "
                                 f"to park the data qubit at the turnaround")

    # --- fiber delays, read straight off the wired components ---

    def _qdelay(self, A, B):
        return A.ports[f"q:{B.node_id}"].qfiber.delay

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

    # --- one hop, fully event-driven (photon over the fiber, herald, retry, correction) ---

    def _generate_link(self, A, B):
        tl = self.tl
        A_id, B_id = A.node_id, B.node_id
        rtt = self._qdelay(A, B) + self._cdelay(B, A)
        fired = []
        B.absorb_hooks[A_id] = lambda nbr: fired.append(nbr)
        while not fired:                             # real loss + retry over the fiber
            A.link_mems[B_id].reset()
            t0 = tl.now()
            A.attempt_link(B_id)                     # schedules the photon at t0 + qdelay
            tl.schedule(self._noop, at=t0 + rtt + 1, priority=PROTOCOL)   # sender timeout tick
            tl.run()                                 # photon arrives (herald or loss) + timeout fires
        B.absorb_hooks.pop(A_id, None)

    def _raw_hop(self, A, B, data):
        tl = self.tl
        if data is A.link_mems[B.node_id]:           # turnaround: park off the slot we need
            data = self._park(A, data)
        self._generate_link(A, B)                    # clock now at t0 + rtt + 1
        m1, m2 = A.bsm(data, A.link_mems[B.node_id]) # decoheres data + idler over real storage
        link_b = B.link_mems[A.node_id]
        tl.schedule(self._do_correct, B, link_b, m1, m2,
                    at=tl.now() + self._cdelay(A, B), priority=PROTOCOL)
        tl.run()                                     # bits fly cdelay, then correct (decoheres signal)
        return link_b

    # --- one RB sequence of m bounces ---

    def _apply_clifford(self, node, data, u_total, rng):
        u = random_clifford(rng)
        apply_gate(self.tl.state_tracker, (data.key,), u, node.p_depol_1q, coherent=node.coherent_1q)
        return u @ u_total

    def _cleanup(self):
        for node in self.nodes:
            for mem in list(node.memories):
                mem.reset()

    def _one_sequence(self, m, rng):
        tracker = self.tl.state_tracker
        a1 = self.nodes[0]
        data = next(mm for mm in a1.unbound_mems() if mm.is_empty())
        data.initialize()
        u_total = np.eye(2, dtype=complex)
        for _ in range(m):
            for k in range(self.K - 1):
                u_total = self._apply_clifford(self.nodes[k], data, u_total, rng)
                data = self._hop(self.nodes[k], self.nodes[k + 1], data)
            for k in range(self.K - 1, 0, -1):
                u_total = self._apply_clifford(self.nodes[k], data, u_total, rng)
                data = self._hop(self.nodes[k], self.nodes[k - 1], data)
        p = int(rng.integers(0, 2))
        g_inv = (X if p else np.eye(2, dtype=complex)) @ u_total.conj().T
        apply_gate(tracker, (data.key,), g_inv, a1.p_depol_1q, coherent=a1.coherent_1q)
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


    def run(self) -> BenchmarkResult:
        rng = self.tl.rng
        self._cleanup()
        raw = {m: [b for b in (self._one_sequence(m, rng) for _ in range(self.n_samples)) if b is not None]
               for m in range(self.m_min, self.m_max + 1)}
        decay = {m: (float(np.mean(v)) if v else 0.0) for m, v in raw.items()}
        f, F_path, A = fit_decay(decay, spam=self.detector is not None)
        return BenchmarkResult(self.path, self.K - 1, f, F_path, A, decay, raw, self.tl.now())

    def _hop(self, A, B, data):
        if self.purify_rounds < 1:
            return self._raw_hop(A, B, data)
        return self._purified_hop(A, B, data)

    def _purified_hop(self, A, B, data):
        """Teleport `data` over a link-purified pair instead of a raw one. Memory
        decoherence on `data` accrues over the full pump time -- that is the point."""
        tl = self.tl
        kept_a_name = f"purify:kept:{B.node_id}"
        # turnaround: park the data qubit off any slot this hop's session will use
        if data is A.link_mems[B.node_id] or data is A.ensure_memory(kept_a_name):
            data = self._park(A, data)
        session = LinkPumpSession(self.net, A.node_id, B.node_id,
                                  rounds=self.purify_rounds, calibrate=self.calibrate)
        session.run()                                # install -> pump -> uninstall (blocking)
        kept_a = A.memory(kept_a_name)
        kept_b = B.memory(f"purify:kept:{A.node_id}")
        m1, m2 = A.bsm(data, kept_a)                 # decoheres data over the real pump time
        tl.schedule(self._do_correct, B, kept_b, m1, m2,
                    at=tl.now() + self._cdelay(A, B), priority=PROTOCOL)
        tl.run()
        return kept_b



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
