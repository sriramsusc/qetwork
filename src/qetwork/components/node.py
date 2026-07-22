"""Node: a named station owning quantum ports, hardware, and classical message dispatch."""

from collections.abc import Callable

from qetwork.components.port import Port
from qetwork.events.message import Message


class Node:
    def __init__(self, node_id: str, timeline) -> None:
        self.node_id = node_id
        self.timeline = timeline
        self.ports: dict[str, Port] = {}   # quantum ports only
        self.cfibers: dict = {}            # dst node_id -> outgoing CFiber (filled by CFiber itself)
        self.msg_handlers: dict[str, Callable] = {}

    def __repr__(self) -> str:
        return f"Node({self.node_id})"

    def add_port(self, name: str) -> Port:
        if name in self.ports:
            raise ValueError(f"node {self.node_id} already has a port named {name!r}")
        port = Port(name, self)
        self.ports[name] = port
        return port

    def send_to(self, dst_id: str, kind: str, **payload) -> Message:
        fiber = self.cfibers.get(dst_id)
        if fiber is None:
            raise ValueError(f"node {self.node_id} has no classical fiber to {dst_id!r}")
        message = Message(kind=kind, src=self.node_id, dst=dst_id,
                          payload=payload, created_at=self.timeline.now())
        fiber.transmit(message)
        return message

    def receive_message(self, message) -> None:
        if message.dst != self.node_id:
            raise ValueError(f"node {self.node_id} received message addressed to {message.dst!r}")
        handler = self.msg_handlers.get(message.kind)
        if handler is None:
            raise ValueError(f"node {self.node_id} has no handler for kind {message.kind!r}")
        handler(message)

    def register_handler(self, kind: str, handler: Callable, replace: bool = False) -> None:
        if kind in self.msg_handlers and not replace:
            raise ValueError(f"node {self.node_id} already handles message kind {kind!r}")
        self.msg_handlers[kind] = handler

    def unregister_handler(self, kind: str, handler: Callable) -> None:
        """Remove a handler only if it matches the installed one (ownership check);
        no-op otherwise -- a replace=True takeover may have superseded it. Equality,
        not identity: each `self.method` access builds a fresh bound-method object,
        so `is` never matches; `==` compares __self__ and __func__."""
        if self.msg_handlers.get(kind) == handler:
            del self.msg_handlers[kind]
