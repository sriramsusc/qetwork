"""Discrete-event simulation timeline for the quantum network simulator."""

import math
import heapq
from collections.abc import Callable

from qetwork.seed import make_rng
from qetwork.events.event import Event
from qetwork.kernel.state_tracker.tracker import StateTracker

"""Time units: the simulation tick is one integer picosecond."""
psec   = 1
nsec   = 1_000
micsec = 1_000_000
milsec = 1_000_000_000
sec    = 1_000_000_000_000

class Timeline:
    def __init__(self, stop_time:float = math.inf, seed: int | None = None) -> None:
        self.time: int = 0
        self.stop_time = stop_time
        self.is_running: bool = False
        self.run_counter: int = 0
        self.queue: list = []
        self._counter: int = 0
        self.state_tracker = StateTracker()
        self.rng = make_rng(seed)

    def now(self) -> int:
        return self.time
    
    def schedule(self, action: Callable, *args, at: int, delay: int = 0, priority: int = 0, **kwargs) -> Event:
        for name, value in (("at", at), ("delay", delay), ("priority",priority)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int (integer picoseconds), got {type(value).__name__}: {value!r}")
        if delay <0:
            raise ValueError(f"delay must be non negative, got{delay}")
        time = at + delay
        if time < self.time:
            raise ValueError(f"cannot schedule event in the past: even time {time} < current time {self.time}")
        event = Event(time, action, args, kwargs, priority)
        heapq.heappush(self.queue, (time, priority, self._counter, event))
        self._counter += 1
        return event
    
    def step(self) -> bool:
        """Run the single earliest event; False when the queue is empty or the next event is past stop_time."""
        if not self.queue or self.queue[0][0] > self.stop_time:
            return False
        time, _, _, event = heapq.heappop(self.queue)
        self.time = event.time
        event.run()
        self.run_counter += 1
        return True

    def run(self) -> None:
        if self.is_running:
            raise RuntimeError("run() is not reentrant; timeline is already running")
        self.is_running = True
        try:
            while self.step():
                pass
        finally:
            self.is_running = False

