"""Photon: a light carrier that transports a qubit's state key between devices, has a wavelength."""

from dataclasses import dataclass

@dataclass(slots= True, kw_only= True)
class Photon:
    key: int | None = None
    wavelength: float | None = None
    encoding: str
    timeline: object | None = None    # the simulation whose tracker owns this key
    depart_time: int | None = None
    in_flight: bool = False
    _spent: bool = False


    def take_key(self) -> int:
        """Transfer the carried qubit to a new owner; the photon is spent afterward."""
        if self._spent or self.key is None:
            raise RuntimeError("photon already consumed/transferred (single-owner violated)")
        key, self.key, self._spent = self.key, None, True
        return key

    def destroy(self) -> None:
        self.key = None
        self.wavelength = None
        self.depart_time = None
        self.timeline = None
        self.in_flight = False
