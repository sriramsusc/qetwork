"""SNSPD: one superconducting-nanowire channel — click/no-click + timestamp; basis-blind, tracker-blind."""

import math
from qetwork.kernel.timeline import nsec, sec

class SNSPD:

    _FWHM_TO_SIGMA = 1 / (2*math.sqrt(2 * math.log(2)))

    def __init__(self, efficiency: float = 0.90, jitter_fwhm:float = 15.0,
                 dark_count_rate: float = 1/sec, dead_time: int = 20*nsec,
                 band: tuple[float, float] = (1400, 1700)) -> None:
        
        if not math.isfinite(jitter_fwhm) or jitter_fwhm < 0:
            raise ValueError(f"jitter_fwhm must be finite and non-negative, got {jitter_fwhm}")
        if not math.isfinite(dark_count_rate) or dark_count_rate < 0:
            raise ValueError(f"dark count rate must be finite and non-negative, got {dark_count_rate}")
        if not isinstance(dead_time, int) or isinstance(dead_time, bool) or dead_time < 0:
            raise ValueError(f"dead_time must be a non-negative int (ps), got {dead_time!r}")
        if band[0] > band[1]:
            raise ValueError(f"band must be (low, high), got {band}")

        self.efficiency = efficiency
        self.jitter_fwhm = jitter_fwhm
        self.dead_time = dead_time
        self.dark_count_rate = dark_count_rate
        self.band = band
        self._last_click_time: int | None = None


    def _register(self, t: int) -> int | None:

        """A readout pulse at time t: suppressed if within dead time, else latches it. Returns t or none."""
        if self._last_click_time is not None and t < self._last_click_time + self.dead_time:
            return None
        self._last_click_time = t
        
        return t


    def try_click(self, t_true, samp_eff:float, samp_jitter: float) -> int | None:

        """Photon at t_true -> reported click time or None(lost or dead- photon absorbed either way)"""
        if not 0 <= samp_eff <1:
            raise ValueError(f"samp_eff must be in [0,1), got {samp_eff}")
        if samp_eff >=self.efficiency:

            return None
        fired = self._register(t_true)
        if fired is None:

            return None
        
        return fired + round(samp_jitter * self.jitter_fwhm *self._FWHM_TO_SIGMA)
        

    def dark_click(self, window_start: int, window_len: int, samp: float) -> int | None:
        """Atmost one dark count in the window (exact for rate*window << 1, our ~1e-9 regime)"""
        # TODO make scheduled poisson events for darkcounts

        if not 0 <= samp < 1:
            raise ValueError(f"samp must be in [0,1), got {samp}")
        if window_len < 0:
            raise ValueError(f"window length should be non negative, got {window_len}")
        p = 1 - math.exp(-self.dark_count_rate * window_len)   # P(>=1 dark count), bounded <1
        if samp < p:
            t = window_start + int((samp / p) * window_len)

            return self._register(t)     # suppressed inside dead time; arms it otherwise
        
        return None

