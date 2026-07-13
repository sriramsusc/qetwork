"""State: one quantum state(density matrix) covering one or more qubits."""

from dataclasses import dataclass

import numpy as np

@dataclass(slots=True)
class State:
    matrix: np.ndarray
    keys: tuple[int, ...]

    def __post_init__(self) -> None:
        m = self.matrix
        n = len(self.keys)
        if m.ndim !=2 or m.shape[0] != m.shape[1]:
            raise ValueError(f"State matrix must be square, got shape {m.shape}")
        if m.shape[0] != 2**n:
            raise ValueError(f"State matrix size {m.shape[0]} does not match number of qubits {n}")
        if not np.allclose(m, m.conj().T):
            raise ValueError("State matrix must be Hermitian")
        if not np.isclose(np.trace(m), 1):
            raise ValueError("State matrix must have trace 1")
        min_eigenvalue = np.linalg.eigvalsh(m)[0]
        if min_eigenvalue < -1e-10:
            raise ValueError(f"State matrix must be positive semidefinite, got min eigenvalue {min_eigenvalue}")

