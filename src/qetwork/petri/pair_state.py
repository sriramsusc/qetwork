"""Pair-state colour algebras: what an entangled-pair token carries and how the net's
arc expressions evolve it. The net structure is the same for both; only the colour
of Pair.state and these five operations change.

  DensityMatrix  the pair's 4x4 density matrix, evolved by the simulator's OWN
                 operations (T1/T2 channels, apply_gate, measure) on a scratch
                 StateTracker -- exact for every physics knob qetwork has (finite T1,
                 source phase, coherent gate errors). BSM bits are Born-sampled; the
                 state stays in the UNCORRECTED frame until the destination's X^m2 Z^m1.
  BellDiagonal   four Pauli-frame probabilities (see petri.bell) -- the textbook model,
                 exact for Pauli noise, transparent enough for closed forms. BSM bits are
                 uniform (true for Bell-diagonal inputs); the frame is tracked implicitly.

Qubit order inside a pair is (left, right) = (upstream end, downstream end).
"""

import math
from typing import Protocol

import numpy as np

from qetwork.components.sfwm import _werner
from qetwork.components.utils.roles import _coherent_1q, _coherent_2q
from qetwork.kernel.state_tracker.tracker import StateTracker
from qetwork.operations.err_channels import T1, T2
from qetwork.operations.gates import CNOT, H, SWAP, X, Z, apply_gate
from qetwork.operations.measurement import discard, measure
from qetwork.petri import bell
from qetwork.petri.params import ChainParams, NodeParams, t2_star

LEFT, RIGHT = 0, 1

_PHI_PLUS = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)


class PairAlgebra(Protocol):
    name: str

    def check(self, params: ChainParams) -> None: ...
    def raw(self, node: NodeParams): ...
    def idle(self, state, side: int, node: NodeParams, elapsed: int): ...
    def swap(self, left, right, node: NodeParams, rng: np.random.Generator) -> tuple[object, int, int]: ...
    def correct(self, state, node: NodeParams, m1: int, m2: int): ...
    def fidelity(self, state) -> float: ...


class BellDiagonal:
    name = "bell"

    def check(self, params: ChainParams) -> None:
        bad = [i for i, n in enumerate(params.nodes) if n.coherent]
        if bad:
            raise ValueError(f"nodes {bad} have coherent errors or a source phase; the Bell-diagonal "
                             f"colour set cannot carry them -- use DensityMatrix")

    def raw(self, node: NodeParams) -> bell.Bell:
        return bell.werner(node.visibility)

    def idle(self, state, side, node, elapsed):
        return bell.convolve(state, bell.memory_channel(elapsed, node.t1, node.t2))

    def swap(self, left, right, node, rng):
        v = bell.apply(bell.convolve(left, right), bell.bsm_channel(node.p_depol_1q, node.p_depol_2q))
        return v, int(rng.integers(2)), int(rng.integers(2))

    def correct(self, state, node, m1, m2):
        for _ in range(m1 + m2):                 # each applied correction gate carries its error
            state = bell.convolve(state, bell.depolarizing_1q(node.p_depol_1q))
        return state

    def fidelity(self, state) -> float:
        return bell.fidelity(state)


def _frozen(m: np.ndarray) -> np.ndarray:
    m = np.array(m, dtype=complex)               # private copy: tokens must never alias tracker storage
    m.flags.writeable = False
    return m


class DensityMatrix:
    name = "density"

    def check(self, params: ChainParams) -> None:
        pass                                     # every NodeParams knob is representable

    @staticmethod
    def _load(*pairs) -> tuple[StateTracker, list[int]]:
        """Scratch tracker holding the given pairs, two fresh keys per pair, in order."""
        tr = StateTracker()
        keys: list[int] = []
        for rho in pairs:
            k = (tr.new(), tr.new())
            tr.set(k, np.array(rho, dtype=complex))
            keys.extend(k)
        return tr, keys

    @staticmethod
    def _read(tr: StateTracker, a: int, b: int) -> np.ndarray:
        st = tr.get(a)
        if set(st.keys) != {a, b}:
            raise ValueError(f"pair ({a},{b}) is not isolated: its state covers {st.keys}")
        m = st.matrix if st.keys == (a, b) else SWAP @ st.matrix @ SWAP
        return _frozen(m)

    def raw(self, node):
        # |00> + e^(i phi)|11> and I/4 are symmetric under qubit exchange, so the
        # source's (signal, idler) order needs no permutation into (left, right)
        return _frozen(_werner(node.visibility, node.source_phase))

    def idle(self, state, side, node, elapsed):
        ts = t2_star(node.t1, node.t2)
        if elapsed == 0 or (math.isinf(node.t1) and math.isinf(ts)):
            return state                         # identity channel, as MemoryQubit.decohere skips it
        tr, keys = self._load(state)
        k = (keys[side],)
        if not math.isinf(node.t1):              # same split, same order as MemoryQubit.decohere
            T1(tr, k, elapsed, node.t1)
        if not math.isinf(ts):
            T2(tr, k, elapsed, ts)
        return self._read(tr, *keys)

    def swap(self, left, right, node, rng):
        """RepeaterNode.bsm on the two inner halves: CNOT(up -> down), H(up), Z-measure both."""
        tr, (a, r_up, r_down, b) = self._load(left, right)
        c1 = _coherent_1q({"axis": node.coherent_1q_axis, "angle": node.coherent_1q_angle})
        c2 = _coherent_2q({"zz_angle": node.zz_angle})
        apply_gate(tr, (r_up, r_down), CNOT, node.p_depol_2q, coherent=c2)
        apply_gate(tr, (r_up,), H, node.p_depol_1q, coherent=c1)
        o1 = measure(tr, r_up, rng.random(), basis="Z")
        o2 = measure(tr, r_down, rng.random(), basis="Z")
        discard(tr, r_up)
        discard(tr, r_down)
        return self._read(tr, a, b), (0 if o1 > 0 else 1), (0 if o2 > 0 else 1)

    def correct(self, state, node, m1, m2):
        """RepeaterNode.correct on the right (destination) half: X if m2, then Z if m1."""
        if not (m1 or m2):
            return state
        tr, (a, b) = self._load(state)
        c1 = _coherent_1q({"axis": node.coherent_1q_axis, "angle": node.coherent_1q_angle})
        if m2:
            apply_gate(tr, (b,), X, node.p_depol_1q, coherent=c1)
        if m1:
            apply_gate(tr, (b,), Z, node.p_depol_1q, coherent=c1)
        return self._read(tr, a, b)

    def fidelity(self, state) -> float:
        return float((_PHI_PLUS.conj() @ state @ _PHI_PLUS).real)


ALGEBRAS = {"density": DensityMatrix, "bell": BellDiagonal}
