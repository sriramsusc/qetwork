"""Central randomness config: one seeded generator per simulation, no global state."""

import numpy as np

DEFAULT_SEED = 42


def make_rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(DEFAULT_SEED if seed is None else seed)
