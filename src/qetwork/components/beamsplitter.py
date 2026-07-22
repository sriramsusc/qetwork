"""Beam splitter: mixes the two path modes of one photon (a dual rail path qubit)"""

import numpy as np

def _assert_unitary(U: np.ndarray) -> None:
    if not np.allclose(U.conj().T @ U, np.eye(U.shape[0], dtype=complex)):
        raise ValueError(f"beam splitter matrix is not unitary: \n{U}")
    
class BeamSplitter:
    # CITE bs-unitary | lossless 2x2 beam splitter is unitary; symmetric [[t,ir],[ir,t]] and real [[t,r],[r,-t]] conventions differ by local phases | Gerry & Knight, "Introductory Quantum Optics", Cambridge Univ. Press (2005), §6.2 "The beam splitter"
    # CITE dual-rail | one photon across two modes is a dual-rail qubit; a beam splitter acts on it as a single-qubit unitary | Kok et al., "Linear optical quantum computing with photonic qubits", Rev. Mod. Phys. 79, 135 (2007), §II.A

    _CONVENTIONS = ("symmetric", "real")

    def __init__(self, reflectivity: float = 0.5, loss: float = 0.0,
                 convention: str = "symmetric", band: tuple[float, float] = (1520,1570)) -> None:
        if not 0 <= reflectivity <=1:
            raise ValueError(f"reflectivity must be in [0,1], got {reflectivity}")
        if loss < 0:
            raise ValueError(f"Loss must be non negative, got {loss}")
        if convention not in self._CONVENTIONS:
            raise ValueError(f"convention must be one of {self._CONVENTIONS}, got {convention}")
        lo, hi = band
        if lo > hi:
            raise ValueError(f"band must be low, high, got {band}")
        
        self.reflectivity = reflectivity
        self.loss = loss
        self.convention = convention
        self.band = band
        _assert_unitary(self.matrix())

    @property
    def transmissivity(self) -> float:
        return 1.0 - self.reflectivity
    
    @property
    def survival(self) -> float:
        """Fraction surviving the excess loss; 1.0 when loss = 0dB"""
        return 10 ** (-self.loss / 10)
    
    def matrix(self) -> np.ndarray:
        """The 2x2 unitary on the path qubit, basis order (port_0, port_1)"""
        t = np.sqrt(self.transmissivity)
        r = np.sqrt(self.reflectivity)
        if self.convention == "symmetric":
            return np.array([[t, 1j * r], [1j * r, t]], dtype= complex)
        return np.array([[t, r], [r, -t]], dtype= complex)

    def split_ratio(self) -> tuple[float, float]:
        return self.transmissivity, self.reflectivity
    
    def survives(self, sample: float) -> bool:
        """Excess loss check against a pre drawn sample in [0,1)"""
        if not 0 <= sample <1:
            raise ValueError(f"sample must be in [0,1), got {sample}")
        return sample < self.survival
    
    @classmethod
    def balanced(cls, **kwargs) -> "BeamSplitter":
        """Ideal 50/50 coupler"""
        return cls(reflectivity = 0.5, **kwargs)
    def __repr__(self) -> str:
        return (f"Beamspliiter(R={self.reflectivity}, loss={self.loss} dB, convention={self.convention})")