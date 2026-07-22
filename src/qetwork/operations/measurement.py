"""Measurements: Projective, Bell, SNSPD detection and mem qubit collapse"""

import numpy as np 

from qetwork.operations.core import _embed, _eigenprojectors
from qetwork.operations.partial_trace import partial_trace
from qetwork.operations.gates import I, X, Y, Z

_PAULI = {"X" : X, "x": X, 
          "Y" : Y, "y": Y, 
          "Z" : Z, "z": Z
          }

def _observable(basis: np.ndarray | str) -> np.ndarray:
    """Resolve a measurement basis to a single-qubit Hermitian observable."""
    if isinstance(basis, str):
        if basis not in _PAULI:
            raise ValueError(f"unknown Pauli basis {basis!r}, expected one of {sorted(_PAULI)}")
        return _PAULI[basis]
    A = np.asarray(basis)
    if A.shape != (2, 2):
        raise ValueError(f"observable must be a 2x2 matrix, got shape {A.shape}")
    if not np.allclose(A, A.conj().T):
        raise ValueError("observable must be Hermitian (A == A^dag)")
    w = np.linalg.eigvalsh(A)
    if np.isclose(w[0], w[1]):
        raise ValueError(f"observable is degenerate (eigenvalues {w}); a single-qubit "
                         f"measurement needs two distinct eigenvalues")
    return A



def _select(probs: list[float], sample: float) -> int:
    if not 0 <= sample < 1:
        raise ValueError(f"sample must be in [0,1), got {sample}")
    clamped = [p if p > 0 else 0.0 for p in probs]
    total = sum(clamped)
    if total <= 1e-9:
        raise ValueError(f"total outcome probability is effectively zero: {total}")
    cumulative = 0.0
    for m, p in enumerate(clamped):
        cumulative += p / total
        if sample < cumulative:
            return m
    # roundoff at the very top of [0,1): fall back to the heaviest real outcome
    return max(range(len(clamped)), key=lambda i: clamped[i])


def _measure(rho: np.ndarray, idx: int, projectors: list[np.ndarray], sample: float):
    # CITE born-collapse | ...
    n = rho.shape[0].bit_length() - 1
    embedded = [_embed(P, (idx,), n) for P in projectors]
    probs = [np.trace(E @ rho).real for E in embedded]
    m = _select(probs, sample)
    pm = probs[m]
    if pm <= 1e-12:
        raise ValueError(f"selected outcome {m} has effectively zero probability {pm}")
    E = embedded[m]
    return m, E @ rho @ E.conj().T / pm


def measure(tracker, key: int, sample: float, basis: np.ndarray | str = "Z"):
    state = tracker.get(key)
    joint_keys = state.keys
    idx = joint_keys.index(key)
    outcomes, projectors = _eigenprojectors(_observable(basis))
    m, rho2 = _measure(state.matrix, idx, projectors, sample)
    n = len(joint_keys)
    if n == 1:
        tracker.set(joint_keys, rho2)
    else:
        rho_rest, rho_meas = partial_trace(rho2, (idx,), n)
        rest = tuple(k for k in joint_keys if k != key)
        tracker.split([((key,), rho_meas), (rest, rho_rest)])

    return outcomes[m]


def discard(tracker, key: int) -> None:
    """Trace out and drop a lost/absorbed qubit"""
    state = tracker.get(key)
    joint = state.keys
    if len(joint) > 1:
        idx = joint.index(key)
        rho_rest, rho_key = partial_trace(state.matrix, (idx, ), len(joint))
        rest = tuple(k for k in joint if k !=key)
        tracker.split([((key, ), rho_key), (rest, rho_rest)])
    tracker.remove(key)
