"""An unbalanced Mach Zhender interferometer- the energy time analyzer"""

import math
import numpy as np

from qetwork.components.beamsplitter import BeamSplitter
from qetwork.operations.gates import apply_unitary
from qetwork.operations.err_channels import dephase
from qetwork.components.utils.encoding import ENERGY_TIME
from qetwork.operations.measurement import discard

class MZI:
    # CITE franson-mzi | energy-time entanglement analysed with one unbalanced MZI per photon, tau_c << dT << tau_pump; only the central coincidence peak interferes | Franson, "Bell inequality for position and time", Phys. Rev. Lett. 62, 2205 (1989)
    # CITE kylia-mint | MINT/WT-MINT: FSR>=2.5GHz, IL<=2.0dB, ER>=18dB (typ 24dB), separate short/long arm losses IL1/IL2, OWR 1520-1570nm | Kylia, "Delay Line Interferometers: MINT and WT-MINT", datasheet V1.2

    def __init__(self, delta_t: int, phase: float = 0.0, bs1: BeamSplitter | None = None, bs2: BeamSplitter | None = None,
                 loss_short: float = 0.0, loss_long: float = 0.0, phase_error: float = 0.0, band: tuple[float, float] = (1520, 1570)) -> None:
        if delta_t <= 0:
            raise ValueError(f"delta_t must be positive, got {delta_t}")
        for name, loss in (("loss_short", loss_short), ("loss_long", loss_long)):
            if loss < 0:
                raise ValueError(f"{name} must be non negative, got {loss}")
        lo, hi = band
        if lo > hi:
            raise ValueError(f"band must be (low, high), got {band}")
        
        self.delta_t = delta_t
        if not isinstance(delta_t, int) or isinstance(delta_t, bool):
            raise TypeError(f"delta_t must be an int (picoseconds), got {type(delta_t).__name__}: {delta_t!r}")
        if delta_t <= 0:
            raise ValueError(f"delta_t must be positive, got {delta_t}")

        self.phase = phase
        self.phase_error = phase_error
        self.bs1 = bs1 if bs1 is not None else BeamSplitter.balanced(convention="real")
        self.bs2 = bs2 if bs2 is not None else BeamSplitter.balanced(convention="real")
        for name, bs in (("bs1", self.bs1), ("bs2", self.bs2)):
            if not isinstance(bs, BeamSplitter):
                raise TypeError(f"{name} must be a BeamSplitter, got {type(bs).__name__}")
        if self.bs1.convention != self.bs2.convention:
            raise ValueError(f"both beam splitters must share a convention, got "
                             f"bs1={self.bs1.convention!r}, bs2={self.bs2.convention!r}")
        for name, bs in (("bs1", self.bs1), ("bs2", self.bs2)):
            if not (bs.band[0] <= lo and hi <= bs.band[1]):
                raise ValueError(f"MZI band {band} not contained in {name} band {bs.band}")
        self.loss_short = loss_short
        self.loss_long = loss_long
        self.band = band



    def _arm_weights(self) -> tuple[float, float]:
        """Unnormalized (short, long) detection weights: BS1 split x arm transmission."""
        s_short = 10 ** (-self.loss_short / 10)
        s_long = 10 ** (-self.loss_long / 10)
        return self.bs1.transmissivity * s_short, self.bs1.reflectivity * s_long


    def get(self, tracker, photon, arm_sample: float, surv_sample: float) -> int | None:
        """Photon entry point for the MZI, validate the photon, then apply the MZI"""
        if photon.key is None:
            raise ValueError("Cannot analyze a photon carrying no quantum state")
        if photon.encoding != ENERGY_TIME:
            raise ValueError(f"MZI analysis {ENERGY_TIME} qubits, got {photon.encoding}")
        if photon.wavelength is not None:
            lo, hi = self.band
            if not lo <= photon.wavelength <= hi:
                raise ValueError(f"photon wavelength {photon.wavelength} nm outside the MZI band {self.band}")
        if not self.survives(surv_sample):
            discard(tracker, photon.key)     # from qetwork.operations.measurement
            photon.destroy()
            return None
        
        return self.analyze(tracker, photon.key, arm_sample)

    def set_phase(self, phi: float) -> None:
        self.phase = phi


    def free_spectral_range(self) -> float:
        """FSR in GHz(delta_t is in ps): FSR = 1000 / deltas_t"""
        return 1000 / self.delta_t
    

    def phase_shift(self) -> np.ndarray:
        """P(phi): the long arm's phase, acting on |long> = |1>"""
        phi = self.phase + self.phase_error
        return np.array([[1,0],[0, np.exp(1j * phi)]], dtype= complex)


    def unitary(self) -> np.ndarray:
        """BS2 . P(phi) . BS1, the coherent action on path qubit"""
        return self.bs2.matrix() @ self.phase_shift() @ self.bs1.matrix()


    def visibility(self) -> float:
        """Fringe contrast from arm amplitude imbalance: V= 2*sqrt(Is*Il)/(Is+Il)"""
        i_s = 10 ** (-self.loss_short / 10)
        i_l = 10 ** (-self.loss_long / 10)
        return 2 * math.sqrt(i_s * i_l) / (i_s +i_l)
    

    def survival(self) -> float:
        """P(photon exits and is timed): BS excess losses x arm transmission (marginal)."""
        w_short, w_long = self._arm_weights()
        return self.bs1.survival * self.bs2.survival * (w_short + w_long)

    
    def survives(self, sample: float) -> bool:
        if not 0 <= sample < 1:
            raise ValueError(f"sample must be in [0,1), got {sample}")
        return sample < self.survival()


    def arm_delay(self, arm_sample: float) -> int:
        """Short/long click-time offset, conditioned on survival."""
        if not 0 <= arm_sample < 1:
            raise ValueError(f"arm sample must be in [0,1), got {arm_sample}")
        w_short, w_long = self._arm_weights()
        total = w_short + w_long
        if total == 0:
            raise ValueError("both arms fully attenuated; no surviving photon to time")
        return 0 if arm_sample < w_short / total else self.delta_t
    

    def analyze(self, tracker, key: int, arm_sample: float) -> int:
        """Apply the interferometer to the photon's path qubit; retuth the click time offset 
        Does not measure- the SNSPD collapses it doenstream"""
        apply_unitary(tracker, (key, ), self.phase_shift() @ self.bs1.matrix())
        v = self.visibility()
        if v < 1.0:
            dephase(tracker, (key, ), v)
        apply_unitary(tracker, (key, ), self.bs2.matrix())
        return self.arm_delay(arm_sample)
    

    def __repr__(self) -> str:
        return (f"MZI(delta_t={self.delta_t} ps, phase={self.phase:.4f} rad, "
                f"FSR={self.free_spectral_range():.3f} GHz, V={self.visibility():.4f})")
