"""Event: a single action scheduled to run at a virtual time."""

from collections.abc import Callable
from dataclasses import dataclass, field

@dataclass(slots=True)
class Event:
    time: int
    action: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    priority: int = 0
    
    def run(self) -> None:
        self.action(*self.args, **self.kwargs)

    