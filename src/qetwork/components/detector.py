"""A detector module that contains a polarization photon detector with a pbs and 2 SNSPD modules."""

from qetwork.components.snspd import SNSPD
from qetwork.components.pbs import PBS
from qetwork.operations.measurement import measure
from qetwork.components.utils.encoding import POLARIZATION
from qetwork.events.priority import ARRIVAL


class _TwoArmDetector:
    def __init__(self, timeline, arms, coupling):
        for name, c in coupling.items():
            if not 0 <= c <= 1:
                raise ValueError(f"coupling {name!r} must be in [0,1], got {c}")
        self.timeline = timeline
        self.arms = arms
        self.coupling = coupling
        self.detections: list[tuple[str, int]] = []
        self._dark_gen = 0

    # dark counts: Poisson per arm, event-driven, same dead-time gate as photon clicks.
    # Untagged in `detections` on purpose — a real detector cannot tell them apart.
    def start_dark_counts(self) -> None:
        """Self-perpetuating events: only start with a finite stop_time, or run() never drains."""
        self._dark_gen += 1
        for arm, snspd in self.arms.items():
            if snspd.dark_count_rate > 0:
                self._schedule_dark(arm, self._dark_gen)

    def stop_dark_counts(self) -> None:
        self._dark_gen += 1

    def _schedule_dark(self, arm: str, gen: int) -> None:
        tl = self.timeline
        dt = round(tl.rng.exponential(1.0 / self.arms[arm].dark_count_rate))
        tl.schedule(self._dark_fire, arm, gen, at=tl.now() + dt, priority=ARRIVAL)

    def _dark_fire(self, arm: str, gen: int) -> None:
        if gen != self._dark_gen:
            return
        t = self.timeline.now()
        if self.arms[arm]._register(t) is not None:
            self.detections.append((arm, t))
        self._schedule_dark(arm, gen)


    def _click(self, arm: str, t_true: int, wavelength) -> tuple[str, int] | None:
        """Couple → band-gate → nanowire click → record. Assumes `arm` is a real port."""
        rng = self.timeline.rng
        if self.coupling[arm] < 1.0 and rng.random() >= self.coupling[arm]:
            return None
        snspd = self.arms[arm]
        if wavelength is not None and not (snspd.band[0] <= wavelength <= snspd.band[1]):
            return None
        t_report = snspd.try_click(t_true, rng.random(), rng.standard_normal())
        if t_report is None:
            return None
        result = (arm, t_report)
        self.detections.append(result)
        return result


class PolarizationDetector(_TwoArmDetector):
    _BASES = ("X", "x", "Y", "y", "Z", "z")
    def __init__(self, timeline, pbs=None, snspd_t=None, snspd_r=None,
                 basis="Z", coupling_t=1.0, coupling_r=1.0):
        if basis not in self._BASES:
            raise ValueError(f"detector basis must be a Pauli letter (±1 eigenvalues make "
                             f"sign-routing valid), got {basis!r}")
        self.pbs = pbs or PBS()
        super().__init__(timeline,
            arms={"transmitted": snspd_t or SNSPD(), "reflected": snspd_r or SNSPD()},
            coupling={"transmitted": coupling_t, "reflected": coupling_r})
        self.basis = basis

    def set_basis(self, basis):
        if basis not in self._BASES:
            raise ValueError(f"detector basis must be a Pauli letter, got {basis!r}")
        self.basis = basis

    def get(self, photon):
        if photon.timeline is not self.timeline:
            raise ValueError("detector received a photon from a different simulation; "
                             "its key would alias an unrelated local state")
        if photon.key is None:
            raise ValueError("cannot detect a photon carrying no quantum state")
        if photon.encoding != POLARIZATION:
            raise ValueError(f"polarization detector got {photon.encoding!r} photon; "
                             f"expected {POLARIZATION}")
        if photon.wavelength is not None:
            lo, hi = self.pbs.band
            if not lo <= photon.wavelength <= hi:
                raise ValueError(f"photon wavelength {photon.wavelength} nm outside PBS band {self.pbs.band}")

        tracker, rng, t, key = self.timeline.state_tracker, self.timeline.rng, self.timeline.now(), photon.key
        outcome = measure(tracker, key, rng.random(), self.basis)
        ideal_port = "transmit" if outcome > 0 else "reflect"
        arm = self.pbs.route(ideal_port, rng.random())
        result = self._click(arm, t, photon.wavelength) if arm is not None else None
        tracker.remove(key); photon.destroy()
        return result


class TimeEnergyDetector(_TwoArmDetector):
    def __init__(self, timeline, mzi, snspd_1=None, snspd_2=None,
                 coupling_1=1.0, coupling_2=1.0):
        self.mzi = mzi
        super().__init__(timeline,
            arms={"eig1": snspd_1 or SNSPD(), "eig2": snspd_2 or SNSPD()},
            coupling={"eig1": coupling_1, "eig2": coupling_2})

    def set_phase(self, phi): self.mzi.set_phase(phi)   # φ IS the analyzer basis

    def get(self, photon):
        tracker, rng, t, key = (self.timeline.state_tracker, self.timeline.rng,
                                self.timeline.now(), photon.key)
        surv_sample = rng.random()
        arm_sample = rng.random()
        offset = self.mzi.get(tracker, photon, arm_sample, surv_sample)
        if offset is None:
            return None            # lost in the interferometer — MZI already discarded + destroyed

        outcome = measure(tracker, key, rng.random(), "Z")   # phi set the basis; Z reads the port
        arm = "eig1" if outcome > 0 else "eig2"
        wavelength = photon.wavelength      # capture BEFORE destroy() clears it
        tracker.remove(key)
        photon.destroy()
        # dead-time correctness: the nanowire must see clicks in time order, so the
        # delayed-arm click fires as an event AT t+offset, not synchronously now
        return self.timeline.schedule(self._click, arm, t + offset, wavelength,
                                      at=t + offset, priority=ARRIVAL)

