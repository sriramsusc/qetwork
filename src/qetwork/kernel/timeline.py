"""Discrete-event simulation timeline for the quantum network simulator."""

import math
import heapq
from collections.abc import Callable

from qetwork.events.event import Event

class Timeline:
    def __init__(self, stop_time:float = math.inf) -> None:
        self.time: int = 0
        self.stop_time = stop_time
        self.is_running: bool = False
        self.run_counter: int = 0
        self.queue: list = []
        self.counter: int = 0

    def now(self) -> int:
        return self.time
    
    def schedule(self, action: Callable, *args, at: int, delay: int = 0, priority: int = 0, **kwargs) -> Event:
        time = at + delay
        event = Event(time, action, args, kwargs, priority)
        heapq.heappush(self.queue, (event.time, event.priority, self.counter, event))
        self.counter += 1
        return event
    
    def run(self) -> None:
        self.is_running = True
        while self.queue:
            *_, event = heapq.heappop(self.queue)
            if event.time > self.stop_time:
                break
            self.time = event.time
            event.run()
            self.run_counter += 1
        self.is_running = False