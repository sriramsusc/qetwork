"""Port: a quantum endpoint on a node — attaches a fiber for outgoing photons, routes incoming ones to a handler."""

from collections.abc import Callable

from qetwork.operations.measurement import discard


class Port:
    def __init__(self, name: str, node) -> None:
        self.name = name
        self.node = node
        self.qfiber = None
        self.handler: Callable | None = None

    def __repr__(self) -> str:
        return f"Port({self.node.node_id}.{self.name})"

    def attach(self, handler: Callable) -> None:
        self.handler = handler

    def send(self, photon) -> None:
        if self.qfiber is None:
            raise ValueError(f"{self!r} has no quantum fiber attached")
        if photon.in_flight:
            raise ValueError(f"{self!r} cannot send a photon that is already in flight")
        if photon.key is None:
            raise ValueError(f"{self!r} cannot send a spent photon carrying no state")
        photon.in_flight = True
        photon.depart_time = self.node.timeline.now()
        self.qfiber.transmit(photon)

    def receive(self, photon, fiber) -> None:
        photon.in_flight = False
        if self.handler is None:
            raise ValueError(f"{self!r} received a photon but has no handler attached")
        tl = self.node.timeline
        if tl.rng.random() >= fiber.survival:
            if photon.key is not None:
                discard(tl.state_tracker, photon.key)   # trace out if entangled
            photon.destroy()
            return
        # TODO: in-flight noise over (tl.now() - photon.depart_time) — needs a fiber channel
        self.handler(photon)
