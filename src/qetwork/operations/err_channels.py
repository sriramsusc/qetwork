import math
import numpy as np

from qetwork.operations.core import _embed
from qetwork.operations.core import _apply_kraus

Z = np.array([[1, 0], [0, -1]], dtype=complex)

def _validate_time(elapsed: int, t: float) -> None:
    if elapsed < 0:
        raise ValueError(f"elapsed time must be non-negative, got {elapsed}")
    if t <= 0:
        raise ValueError(f"relaxation time must be positive, got {t}")

def T1(tracker, keys: tuple[int, ...], elapsed: int, t1: float) -> None:
    """Apply T1 relaxation to a single qubit, joint or lone."""
    # CITE amp-damp-t1 | amplitude damping (T1): A0=[[1,0],[0,sqrt(1-g)]], A1=[[0,sqrt(g)],[0,0]], g(t)=1-e^(-t/T1) | Nielsen & Chuang (2010), §8.3.5 "Amplitude damping"
    _validate_time(elapsed, t1)
    if len(keys) != 1:
        raise ValueError("T1 channel can only be applied to a single qubit")

    gamma = 1 - math.exp(-elapsed / t1)
    A0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    A1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)

    state = tracker.get(keys[0])
    idx = state.keys.index(keys[0])
    tracker.set(state.keys, _apply_kraus(state.matrix, (A0, A1), (idx,)))


def T2(tracker, keys: tuple[int, ...], elapsed: int, t2: float) -> None:
    """Apply T2 pure dephasing to a single qubit, joint or lone."""
    # CITE dephasing-t2 | pure dephasing: off-diagonals x e^(-t/Tdp), i.e. (1-lam) rho + lam Z rho Z with lam=(1-e^(-t/Tdp))/2 | Nielsen & Chuang (2010), §8.3.3 "Phase flip"
    _validate_time(elapsed, t2)
    if len(keys) != 1:
        raise ValueError("T2 channel can only be applied to a single qubit")
    dephase(tracker, keys, math.exp(-elapsed / t2))


def dephase(tracker, keys: tuple[int, ...], factor: float) -> None:
    """Scale qubit's coherence by `factor`; 1.0 = no ops, 0.0 = fully dephased"""
    # CITE phase-flip | partial dephasing (1-lam)rho + lam Z rho Z with lam=(1-f)/2 scales the off-diagonals by exactly f | Nielsen & Chuang, "Quantum Computation and Quantum Information", 10th Anniv. Ed., §8.3.3 "Phase flip"
    if not 0 <= factor <= 1:
        raise ValueError(f"factor must be in [0,1], got {factor}")
    if len(keys) != 1:
        raise ValueError(f"dephase acts on a single qubit, got {keys}")
    state = tracker.get(keys[0])
    joint = state.keys
    idx = joint.index(keys[0])
    Z_i = _embed(Z, (idx, ), len(joint))
    lam = (1 - factor) / 2
    rho = state.matrix
    tracker.set(joint, (1-lam) * rho + lam * (Z_i @ rho @ Z_i))

from itertools import product

from qetwork.operations.core import _embed, _apply_kraus

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _paulis(k: int):
    """All 4^k k-qubit Pauli products, identity-first (index 0 is I^(x)k)."""
    out = []
    for combo in product((I, X, Y, Z), repeat=k):
        M = combo[0]
        for P in combo[1:]:
            M = np.kron(M, P)
        out.append(M)
    return out


def _depol_kraus(k: int, p: float):
    """Kraus set for the k-qubit depolarizing channel."""
    # CITE depolarizing | k-qubit depolarizing channel: rho -> (1-p) rho + p/(4^k-1) sum_{P != I} P rho P; Kraus ops sqrt(1-p) I and sqrt(p/(4^k-1)) P are complete since P^dag P = I | Nielsen & Chuang (2010), §8.3.4 "Depolarizing channel"
    paulis = _paulis(k)
    w = p / (4**k - 1)
    return [np.sqrt(1 - p) * paulis[0]] + [np.sqrt(w) * P for P in paulis[1:]]


def depolarize(tracker, keys: tuple[int, ...], p: float) -> None:
    """Apply the joint k-qubit depolarizing channel to the qubits at `keys`."""
    keys = tuple(keys)
    if not keys:
        raise ValueError("depolarize requires at least one key")
    if len(set(keys)) != len(keys):
        raise ValueError(f"keys must be unique, got {keys}")
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0,1], got {p}")
    missing = [k for k in keys if k not in tracker.states]
    if missing:
        raise ValueError(f"keys not in tracker (allocate via new() first: {missing})")
    if p == 0:
        return

    joint = tracker.combine(keys)
    indices = tuple(joint.index(key) for key in keys)
    rho = tracker.get(keys[0]).matrix
    tracker.set(joint, _apply_kraus(rho, _depol_kraus(len(keys), p), indices))


def depol_p_from_rb(r: float, k: int) -> float:
    """Convert an RB average error-per-gate `r` to the depolarizing parameter `p`."""
    # CITE rb-to-depol | our channel weights non-identity Paulis by p/(d^2-1), so F_pro = 1-p and
    #   F_avg = (d(1-p)+1)/(d+1); with r = 1-F_avg this is r = p*d/(d+1), i.e. p = r*(d+1)/d
    #   (k=1 -> 1.5r, k=2 -> 1.25r). Valid only while p<=1, i.e. r <= d/(d+1).
    #   | M. A. Nielsen, Phys. Lett. A 303, 249-252 (2002); Magesan et al., PRL 106, 180504 (2011)
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive int, got {k!r}")

    if not 0 <= r <= 1:
        raise ValueError(f"r must be in [0,1], got {r}")
    d = 2**k
    p = r * (d + 1) / d
    if p > 1:
        raise ValueError(
            f"r={r} exceeds the depolarizable maximum d/(d+1)={d / (d + 1):.4f} for k={k}: "
            f"implied p={p:.4f} is not a valid channel probability"
        )
    return p

