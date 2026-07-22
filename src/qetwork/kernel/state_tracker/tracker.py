"""State tracker: the one shared store of all quantum states in simulation."""

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
        if isinstance(key, bool) or not isinstance(key, int):
            raise ValueError(f"key must be an int, got {key!r}")
        state = self.states.get(key)
        if state is None:
            raise ValueError(f"key {key} not in tracker")
        return state


    def set(self, keys, matrix) -> None:
        
        keys = tuple(keys)

        if not keys:
            raise ValueError("set() requires at least one key")
        if not all(isinstance(k, int) and not isinstance(k, bool) for k in keys):
            raise ValueError(f"keys must all be integers, got {keys}")
        if len(set(keys)) != len(keys):
            raise ValueError(f"keys must be unique, got {keys}")
        missing = [k for k in keys if k not in self.states]
        if missing:
            raise ValueError(f"keys not in tracker (allocate via new() first): {missing}")
        
        for k in keys:
            cluster = self.states.get(k)
            if cluster is not None and any(other not in keys for other in cluster.keys):
                raise ValueError(f"set() cannot repoint a partial cluster: key {k} is in a cluster"
                                 f"{cluster.keys}, not all of which are in {keys}; splitting needs a split operation"\
                )
        state =  State(matrix, tuple(keys))
        for key in state.keys:
            self.states[key] = state

    def split(self, groups) -> None:
        groups = list(groups)
        if not groups:
            raise ValueError("split() requires atleast one group")
        for keys, _ in groups:
            if not isinstance(keys, tuple):
                raise ValueError(f"each gtoup's keys must be a tuple, got {type(keys).__name__}: {keys!r}")
            if not keys:
                raise ValueError(f"each group must contain atleast one key")
            if not all(isinstance(k, int) and not isinstance(k, bool) for k in keys):
                raise ValueError(f"keys must all be integers, got {keys}")
            
        all_keys = [k for keys, _ in groups for k in keys]
        if len(set(all_keys)) != len(all_keys):
            raise ValueError(f"split groups overlap: {all_keys}")
        missing = [k for k in all_keys if k not in self.states]
        if missing:
            raise ValueError(f"keys not in tracker {missing}")
        affected = set()
        for k in all_keys:
            affected.update(self.states[k].keys)
        if affected != set(all_keys):
            raise ValueError(f"split must cover affected clusters completely: affected {affected}, given {set(all_keys)}")
        new_states = [(tuple(keys), State(matrix, tuple(keys))) for keys, matrix in groups]        
        for keys, state in new_states:
            for k in keys:
                self.states[k] = state

    def remove(self, key: int) -> None:
        if key not in self.states:
            raise ValueError(f"key {key} not in tracker.")
        cluster = self.states[key]
        if len(cluster.keys) > 1:
            raise ValueError(f"cannot remove key. still entangled in cluster {cluster.keys};"
                             f"trace it out first (a discard/split op) so it is a lone state") 
        del self.states[key]

    def combine(self, keys) -> tuple[int, ...]:
        """Merge the clusters containing `keys` into one shared State"""
        keys = tuple(keys)
        if not keys:
            raise ValueError("combine() requires at least one key")
        if not all(isinstance(k, int) and not isinstance(k, bool) for k in keys):
            raise ValueError(f"keys must all be integers, got {keys}")
        missing = [k for k in keys if k not in self.states]
        if missing:
            raise ValueError(f"keys not in tracker (allocate via new() first): {missing}")

        clusters, seen = [], set()
        for k in keys:
            cluster = self.states[k]
            if cluster.keys not in seen:
                seen.add(cluster.keys)
                clusters.append(cluster)
        if len(clusters) == 1:
            return clusters[0].keys

        matrix = clusters[0].matrix
        merged = clusters[0].keys
        for cluster in clusters[1:]:
            matrix = np.kron(matrix, cluster.matrix)
            merged = merged + cluster.keys
        self.set(merged, matrix)
        return merged

