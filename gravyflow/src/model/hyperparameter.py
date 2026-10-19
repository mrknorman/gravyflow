"""
A model's hyperparameters: each setting of a ``gf.Model`` and of its layers is a
``HyperParameter``, a value drawn from a distribution (a constant, unless one is
given) by its own seed.
"""
from dataclasses import dataclass
from typing import Union

import numpy as np

import gravyflow as gf


def search_stream(seed: int, consumer: str) -> np.random.Generator:
    """
    The stream ``search/{consumer}`` of the seed tree rooted at ``seed`` (None:
    the default root, ``gf.Defaults.seed``): the draws of a model's
    hyperparameters, its layers and the seeds Keras builds them from.
    """
    return gf.SeedTree(None if seed is None else int(seed)).direct().stream(f"search/{consumer}")


@dataclass
class HyperParameter:
    distribution: gf.Distribution
    seed : int = None
    value: Union[int, float, str] = None

    def __post_init__(self):

        if self.seed is None:
            self.seed = gf.Defaults.seed

        self.rng = search_stream(self.seed, "hyperparameter")

        if isinstance(self.distribution, HyperParameter):
            self.distribution = self.distribution.distribution
            self.value = self.distribution.value
        elif not isinstance(self.distribution, gf.Distribution):
            self.distribution = gf.Distribution(value=self.distribution, type_=gf.DistributionType.CONSTANT)

        self.randomize()

    def reseed(self, seed):
        """Draw from the stream of ``seed``; the distribution, a specification, is never changed."""
        self.seed = seed
        self.rng = search_stream(self.seed, "hyperparameter")

    def randomize(self):
        """
        Randomizes this hyperparameter based on its possible_values.
        """
        self.value = self._draw()

    def _draw(self):
        """
        One value of the distribution, as a Python value: a sample is a NumPy
        scalar, which Keras refuses for an integer setting (Dense units).
        """
        value = self.distribution.sample(1, self.rng)[0]
        return value.item() if isinstance(value, np.generic) else value
