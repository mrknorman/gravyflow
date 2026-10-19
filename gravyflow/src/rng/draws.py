"""
A draw scope for Keras' unseeded draws.

Keras seeds every layer built or loaded without a seed (its initialisers, a
dropout layer's seed generator) from Python's global random. `keras_draws`
gives those draws a seed of the seed tree and leaves the global state alone.
"""

import random
from contextlib import contextmanager


@contextmanager
def keras_draws(seed: int):
    """
    Give Keras' unseeded draws ``seed``, leaving the global random state alone.

    Within this context the draws Keras makes when a layer is built or loaded
    come from ``seed``, and Python's random state is restored afterwards: a
    model built here is the same for the same seed, and building or loading
    one changes no global random state.
    """
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)
