"""A classical payload travelling on a Cfiber beteween nodes"""

from dataclasses import dataclass, field
from types import MappingProxyType
from collections.abc import Mapping

@dataclass(slots= True, kw_only= True, frozen= True)
class Message:
    kind: str
    src: str    # TODO update str to node/port instance
    dst: str    # TODO update str to node/port instance
    payload: Mapping = field(default_factory=dict)
    created_at: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
