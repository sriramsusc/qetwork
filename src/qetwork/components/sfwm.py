"""SFWMSource: CW-SFWM energy-time entangled pair source — on-demand emission only.

A stateless pair factory: each emit() produces one signal/idler pair in a Werner
state; WHEN pairs are produced is the caller's policy (a protocol attempt, a
characterization sweep). The energy-time qubit is the Franson two-level subspace
analyzed against an MZI delay; validity assumes tau_c << delta_t << pump
coherence time — neither coherence scale is modeled explicitly."""

import numpy as np

from qetwork.components.photon import Photon
from qetwork.components.utils.encoding import ENERGY_TIME, validate
from qetwork.operations.measurement import discard


def _werner(visibility: float, phase: float) -> np.ndarray:
    # CITE werner-chsh | (keep as-is)
    amp = 1 / np.sqrt(2)
    phi = np.array([amp, 0, 0, amp * np.exp(1j * phase)], dtype=complex)
    bell = np.outer(phi, phi.conj())
    return visibility * bell + (1 - visibility) * np.eye(4, dtype=complex) / 4

class SFWMSource:
    # CITE afrl-sfwm | Si spiral-waveguide CW-SFWM PIC: 1550nm pump, signal/idler 1530/1570nm, ~10-20k pairs/s, CAR 75, visibility 97.4%, CHSH S=2.717 | Sheridan et al., arXiv:2508.01030 (2025)
    # CITE werner-chsh | Werner state rho=V|Phi><Phi|+(1-V)I/4: two-photon fringe visibility = V, optimal CHSH S = 2*sqrt(2)*V | Werner, PRA 40, 4277 (1989)

    def __init__(self, timeline, owner=None, signal_port=None, idler_port=None,
                 signal_wavelength: float = 1530.0, idler_wavelength: float = 1570.0,
                 visibility: float = 0.974, phase: float = 0.0,
                 encoding: str = ENERGY_TIME) -> None:
        if not 0 <= visibility <= 1:
            raise ValueError(f"visibility must be in [0,1], got {visibility}")
        for name, wl in (("signal_wavelength", signal_wavelength),
                         ("idler_wavelength", idler_wavelength)):
            if not np.isfinite(wl) or wl <= 0:
                raise ValueError(f"{name} must be positive and finite, got {wl}")
        if not np.isfinite(phase):
            raise ValueError(f"phase must be finite, got {phase}")
        self.timeline = timeline
        self.owner = owner
        self.signal_port = signal_port      # default receivers; emit() may override per call
        self.idler_port = idler_port
        self.signal_wavelength = signal_wavelength
        self.idler_wavelength = idler_wavelength
        self.visibility = visibility
        self.phase = phase
        self.encoding = validate(encoding)
        self.pair_count = 0

    def emit(self, signal_to=None, idler_to=None) -> tuple[int, int]:
        """Emit one pair now, delivering both photons; keys are allocated only
        once both receivers are known, so no state can leak."""
        sig_out = signal_to if signal_to is not None else self.signal_port
        idl_out = idler_to if idler_to is not None else self.idler_port
        if sig_out is None or idl_out is None:
            raise ValueError("emit() needs both signal and idler receivers; "
                             "an unreceived photon would leak a tracker state")
        tracker = self.timeline.state_tracker
        k_s = tracker.new()
        k_i = tracker.new()
        tracker.set((k_s, k_i), _werner(self.visibility, self.phase))   # order: (signal, idler)
        sig = Photon(key=k_s, wavelength=self.signal_wavelength,
                     encoding=self.encoding, timeline=self.timeline)
        idl = Photon(key=k_i, wavelength=self.idler_wavelength,
                     encoding=self.encoding, timeline=self.timeline)
        self.pair_count += 1

        try:
            sig_out(sig)
        except BaseException:
            discard(tracker, k_s)      # nothing was delivered: drop both halves
            discard(tracker, k_i)
            sig.destroy(); idl.destroy()
            raise
        try:
            idl_out(idl)
        except BaseException:
            if idl.key is not None:    # receiver may have taken the key before failing
                discard(tracker, k_i)  # signal is already in flight: it continues, mixed
            idl.destroy()
            raise
        return k_s, k_i
