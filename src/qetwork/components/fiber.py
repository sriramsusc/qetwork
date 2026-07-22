"""Fiber: an optical channel — quantum between ports, classical between nodes."""

import math
from qetwork.events.priority import ARRIVAL


class Fiber:
    _C = 299_792_458

    def __init__(self, timeline, length, n=1.468):
        for name, value in (("length", length), ("n", n)):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if length < 0:
            raise ValueError(f"length of fiber should be non negative, got {length}")
        if n < 1:
            raise ValueError(f"group index below 1 implies faster-than-vacuum propagation, got {n}")
        self.timeline = timeline
        self.length = length
        self.n = n
        self.delay = round(length / (self._C / n) * 1e12)

    def transmit(self, payload) -> None:
        raise NotImplementedError


class QFiber(Fiber):
    """Unidirectional quantum channel between two ports; loss is sampled at the receiving port."""

    def __init__(self, timeline, src_port, dst_port, length: float,
                 attenuation: float = 0.2, insertion_loss_db: float = 0.0,
                 n: float = 1.468) -> None:
        super().__init__(timeline, length, n)
        if src_port is None or dst_port is None:
            raise ValueError("a quantum fiber requires both source and destination port")
        for port, role in ((src_port, "src_port"), (dst_port, "dst_port")):
            if port.node.timeline is not timeline:
                raise ValueError(f"{role} belongs to a different timeline - state keys are "
                                 f"per tracker and would silently alias across simulations")
        for name, value in (("attenuation", attenuation),
                            ("insertion_loss_db", insertion_loss_db)):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if src_port.qfiber is not None:
            raise ValueError(f"{src_port!r} already has a quantum fiber attached")
        self.src_port = src_port
        self.dst_port = dst_port
        self.attenuation = attenuation
        self.insertion_loss_db = insertion_loss_db
        self.survival = 10 ** (-(attenuation * length / 1000 + insertion_loss_db) / 10)
        src_port.qfiber = self

    def transmit(self, photon) -> None:
        tl = self.timeline
        tl.schedule(self.dst_port.receive, photon, self, at=tl.now() + self.delay, priority=ARRIVAL)


class CFiber(Fiber):
    """Unidirectional classical channel between two nodes; port-less — delivery is node dispatch."""

    def __init__(self, timeline, src_node, dst_node, length: float,
                 n: float = 1.468, latency: int = 0) -> None:
        super().__init__(timeline, length, n)
        if src_node is None or dst_node is None:
            raise ValueError("a classical fiber requires both source and destination node")
        if src_node is dst_node:
            raise ValueError(f"classical fiber endpoints must differ, got {src_node!r} twice")
        for node, role in ((src_node, "src_node"), (dst_node, "dst_node")):
            if node.timeline is not timeline:
                raise ValueError(f"{role} belongs to a different timeline")
        if not isinstance(latency, int) or isinstance(latency, bool):
            raise TypeError(f"latency must be an int (picoseconds), got {type(latency).__name__}")
        if latency < 0:
            raise ValueError(f"latency must be non-negative, got {latency}")
        if dst_node.node_id in src_node.cfibers:
            raise ValueError(f"{src_node!r} already has a classical fiber to {dst_node.node_id!r}")
        self.src_node = src_node
        self.dst_node = dst_node
        self.latency = latency
        src_node.cfibers[dst_node.node_id] = self

    def transmit(self, message) -> None:
        tl = self.timeline
        tl.schedule(self.dst_node.receive_message, message,
                    at=tl.now() + self.delay + self.latency, priority=ARRIVAL)
