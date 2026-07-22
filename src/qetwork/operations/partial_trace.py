"""Partial trace operation for quantum state manipulation."""

import numpy as np

def _reduce(rho: np.ndarray, kept: list[int], n: int) -> np.ndarray:

    t = rho.reshape([2] * n + [2] * n)
    labels = list(range(2 *( n)))
    for a in range(n):
        if a not in kept:
            labels[n+a] = labels[a]
    out = [a for a in kept] + [n+a for a in kept]
    d = 2 ** len(kept)
    return np.einsum(t, labels, out).reshape(d,d)

def partial_trace(rho: np.ndarray, traced: tuple[int, ...], n: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be a positive int, got {n!r}")
    if rho.shape != (2**n, 2**n):
        raise ValueError(f"rho must be 2^{n} square for {n} qubits, got {rho.shape}")
    traced = tuple(traced)
    if not all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n for i in traced):
        raise ValueError(f"traced indices must be ints in [0,{n}), got {traced}")
    traced_set = set(traced)
    if len(traced_set) != len(traced):
        raise ValueError(f"traced indices must be unique, got {traced}")
    kept = [j for j in range(n) if j not in traced_set]
    return _reduce(rho, kept, n), _reduce(rho, sorted(traced_set), n)
