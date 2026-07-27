"""Core primitives shared by all operations."""

from functools import lru_cache

import numpy as np

_I2 = np.eye(2, dtype=complex)   # shared identity factor for embedding; read-only, never mutate

def _embed(op: np.ndarray, indices: tuple[int, ...], n: int) -> np.ndarray:
    """Embed a k-qubit operator op into the n qubit space: it acts on the qubits at indices and I on rest"""

    k = len(indices)
    if op.shape != (2**k, 2**k):
        raise ValueError(f"op must be 2^{k} x 2^{k} for {k} target qubits, got {op.shape}")
    if len(set(indices)) != k or any(i < 0 or i>= n for i in indices):
        raise ValueError(f"indices must be unique and in [0,{n}), got {indices}")
    
    out_lbl = list(range(n))
    in_lbl = list(range(n, 2 * n))
    o = op.reshape([2] * k +[2] * k)
    operands = [o, [out_lbl[i] for i in indices]+ [in_lbl[i] for i in indices]]
    for i in range(n):
        if i not in indices:
            operands += [_I2, [out_lbl[i], in_lbl[i]]]
    
    return np.einsum(*operands, out_lbl + in_lbl).reshape(2**n, 2**n)

def _apply_unitary(rho: np.ndarray, U: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    n = rho.shape[0].bit_length() -1 
    if rho.shape != (2**n, 2**n):
        raise ValueError(f"rho must be a 2^n square matrix matrix, got {rho.shape}")
        
    E = _embed(U, indices, n)

    return E @ rho @ E.conj().T

@lru_cache(maxsize=None)
def _eigenprojectors_cached(shape: tuple, data: bytes):
    A = np.frombuffer(data, dtype=complex).reshape(shape)
    eigvals, eigvecs = np.linalg.eigh(A)
    outcomes, projectors, used =[], [], np.zeros(len(eigvals), dtype = bool)
    for i in range(len(eigvals)):
        if used[i]:
            continue
        group = np.isclose(eigvals, eigvals[i])
        used |= group
        V = eigvecs[:, group]
        projectors.append(V @ V.conj().T)
        outcomes.append(eigvals[i].real)
    return outcomes, projectors


def _eigenprojectors(A: np.ndarray):
    """Spectral decomposition of a Hermitian observable into (eigenvalues, projectors).

    Memoized on the matrix content: projective measurements repeat the same few
    observables (Pauli Z overwhelmingly), so the eigendecomposition runs once per
    distinct A. The returned lists are SHARED across calls -- read-only downstream."""
    Ac = np.ascontiguousarray(A, dtype=complex)
    return _eigenprojectors_cached(Ac.shape, Ac.tobytes())

def _apply_kraus(rho: np.ndarray, kraus, indices: tuple[int, ...]) -> np.ndarray:
    """Apply a Kraus channel: rho -> sum_i K_i rho K_i^dag, embedded at `indices`.

    Trace preservation (sum K^dag K == I) is a PRECONDITION, validated where each
    Kraus set is built -- a constant of the channel -- not re-checked on every
    application: _depol_kraus verifies its cached set per (k, p), and T1's set is
    analytically complete for every gamma. Callers must pass a complete set."""
    # CITE kraus-opsum | operator-sum representation: a CPTP map is sum_i K_i rho K_i^dag with sum_i K_i^dag K_i = I | Nielsen & Chuang, "Quantum Computation and Quantum Information", 10th Anniv. Ed., §8.2.3 "Operator-sum representation", Cambridge Univ. Press (2010)
    n = rho.shape[0].bit_length() - 1
    if rho.shape != (2**n, 2**n):
        raise ValueError(f"rho must be a 2^n square matrix, got {rho.shape}")

    kraus = tuple(kraus)
    if not kraus:
        raise ValueError("a channel requires at least one Kraus operator")
    k = len(indices)
    for K in kraus:
        if K.shape != (2**k, 2**k):
            raise ValueError(f"each Kraus op must be 2^{k} x 2^{k} for {k} targets, got {K.shape}")

    out = np.zeros(rho.shape, dtype=np.result_type(rho.dtype, complex))
    for K in kraus:
        E = _embed(K, indices, n)
        out += E @ rho @ E.conj().T
    return out
