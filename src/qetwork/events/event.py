"""Event: a single action scheduled to run at a virtual time."""

from dataclasses import dataclass, field
from types import MappingProxyType
from collections.abc import Callable, Mapping

@dataclass(slots=True, frozen=True, eq=False)
class Event:
    time: int
    action: Callable
    args: tuple = ()
    kwargs: Mapping = field(default_factory=dict)
    priority: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", MappingProxyType(dict(self.kwargs)))

    def run(self) -> None:
        self.action(*self.args, **self.kwargs)
