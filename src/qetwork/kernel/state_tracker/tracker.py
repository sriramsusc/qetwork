"""State tracker: the one shared store oif all quantum states in simulation."""

import numpy as np

from qetwork.kernel.state_tracker.state import State

class StateTracker:
    def __init__(self) -> None:
        self.states: dict[int, State] = {}
        self._counter: int = 0

    def new(self, matrix = None) -> int:
        if matrix is None:
            matrix = np.array([[1,0],[0,0]], dtype=complex)
        key = self._counter
        self._counter += 1
        self.states[key] = State(matrix, (key,))
        return key

    def get(self, key: int) -> State:
        return self.states[key]

    def set(self, keys, matrix) -> None:
        state =  State(matrix, tuple(keys))
        for key in state.keys:
            self.states[key] = state

    def remove(self, key: int) -> None:
        del self.states[key]