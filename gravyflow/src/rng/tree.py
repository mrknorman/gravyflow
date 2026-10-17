"""
One seed tree for every random draw the library makes.

A run has one root (an integer seed). Every draw comes from a stream addressed
by a path: the split's name, the consumer's name (what the draw is for), the
epoch and the batch index k. A stream is a pure function of (root, path), so a
batch can be rebuilt from its address alone, in any order and in any process,
and no two consumers, splits, epochs or batches share a stream.

Derivation: the path is hashed into a fixed-arity ``spawn_key`` of a NumPy
``SeedSequence`` whose entropy is the root. The arity is fixed so that no path
is a prefix of another, and the path never enters the entropy, where trailing
zeros would collide. Names are hashed with sha256, never Python's ``hash``,
which is salted per process.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from gravyflow.src.rng.consumers import check_consumer, check_name

#: The tree's version. It is part of every path, so a change of derivation
#: changes every stream rather than silently reusing old ones.
TREE_VERSION = 1

#: The epoch and batch coordinates of a stream that has none.
EPOCH_NONE = 0xFFFFFFFF
K_NONE = 0xFFFFFFFF

#: The split of a draw made outside a dataset (a generator or obtainer called
#: directly). It is reserved: ``check_split`` refuses it for any dataset.
DIRECT_SPLIT = "direct"

_MAX_COORDINATE = 0xFFFFFFFE


@lru_cache(maxsize=4096)
def _name_words(name: str, count: int) -> Tuple[int, ...]:
    """The first ``count`` 32-bit words of the sha256 of ``name``."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return tuple(
        int.from_bytes(digest[4 * i: 4 * i + 4], "little") for i in range(count)
    )


def check_coordinate(value, name: str) -> int:
    """
    An epoch or batch index: an integer in the tree's range. Floats, bools and
    strings are refused rather than truncated, which would alias them to one
    batch.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__} {value!r}.")
    value = int(value)
    if value < 0 or value > _MAX_COORDINATE:
        raise ValueError(f"{name} must be in [0, {_MAX_COORDINATE}], got {value}.")
    return value


def _coordinate_words(value: Optional[int], none: int, name: str) -> Tuple[int, int]:
    """A coordinate (epoch or k) as two 32-bit words; None is the sentinel."""
    if value is None:
        return none, none
    return check_coordinate(value, name) & 0xFFFFFFFF, 0


def check_split(split: str) -> str:
    """
    A dataset's split: one of ``gf.SplitSpec``'s splits (KeyError naming them
    otherwise); the reserved ``direct`` is refused by name.
    """
    from gravyflow.src.dataset.splits import SplitSpec

    if split == DIRECT_SPLIT:
        raise ValueError(
            f"The split '{DIRECT_SPLIT}' is reserved for draws made outside a dataset."
        )
    return SplitSpec.check_name(split)


def check_root(seed) -> int:
    """A root seed: a non-negative integer (0 is a valid seed)."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"A seed must be an integer, got {type(seed).__name__}.")
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"A seed must be non-negative, got {seed}.")
    return seed


class SeedTree:
    """
    The root of every random stream of a run, with an optional consumer prefix.

    ``SeedTree(None)`` uses ``gf.Defaults.seed`` and records that the root was
    defaulted (``source``). ``tree.sub("validator/far")`` is the same root with
    every consumer name prefixed, so a sub-tree's streams can never coincide
    with the parent's.
    """

    __slots__ = ("root", "source", "prefix")

    def __init__(self, seed: Optional[int] = None, *, prefix: str = ""):
        if seed is None:
            from gravyflow.src.dataset.config import Defaults
            self.root = check_root(Defaults.seed)
            self.source = "default"
        else:
            self.root = check_root(seed)
            self.source = "user"
        self.prefix = prefix

    @classmethod
    def of(cls, seed) -> "SeedTree":
        """``seed`` as a tree: a SeedTree is returned as it is, an int or None wrapped."""
        return seed if isinstance(seed, SeedTree) else cls(seed)

    def sub(self, prefix: str) -> "SeedTree":
        """This root, with ``prefix`` prepended to every consumer name."""
        tree = SeedTree.__new__(SeedTree)
        tree.root, tree.source = self.root, self.source
        tree.prefix = _join(self.prefix, prefix)
        return tree

    def spawn_key(
        self,
        split: str,
        consumer: str,
        epoch: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Tuple[int, ...]:
        """The 11-word path of one stream: version, split, consumer, epoch, k."""
        consumer = _join(self.prefix, consumer)
        check_name(split, "split")
        check_consumer(consumer)
        return (
            TREE_VERSION,
            *_name_words(split, 2),
            *_name_words(consumer, 4),
            *_coordinate_words(epoch, EPOCH_NONE, "epoch"),
            *_coordinate_words(k, K_NONE, "k"),
        )

    def seed_sequence(self, split, consumer, epoch=None, k=None) -> np.random.SeedSequence:
        return np.random.SeedSequence(
            entropy=self.root, spawn_key=self.spawn_key(split, consumer, epoch, k)
        )

    def stream(self, split, consumer, epoch=None, k=None) -> np.random.Generator:
        """A NumPy Generator (PCG64) for one stream."""
        return np.random.Generator(np.random.PCG64(self.seed_sequence(split, consumer, epoch, k)))

    def key(self, split, consumer, epoch=None, k=None):
        """A JAX threefry key for one stream, built from data (jit never retraces on it)."""
        import jax
        words = self.seed_sequence(split, consumer, epoch, k).generate_state(2, dtype=np.uint32)
        return jax.random.wrap_key_data(words, impl="threefry2x32")

    def batch(self, split: str, epoch: Optional[int] = None, k: Optional[int] = None) -> "BatchStreams":
        """Every stream of one batch of a dataset's split, at (split, epoch, k)."""
        return BatchStreams(self, check_split(split), epoch, k)

    def direct(self, k: Optional[int] = None) -> "BatchStreams":
        """The streams of a draw made outside a dataset."""
        return BatchStreams(self, DIRECT_SPLIT, None, k)

    def metadata(self) -> dict:
        return {"root": self.root, "source": self.source, "prefix": self.prefix,
                "tree_version": TREE_VERSION}

    def __eq__(self, other):
        return isinstance(other, SeedTree) and (self.root, self.prefix) == (other.root, other.prefix)

    def __hash__(self):
        return hash((self.root, self.prefix))

    def __repr__(self):
        prefix = f", prefix={self.prefix!r}" if self.prefix else ""
        return f"SeedTree({self.root}{prefix})"

    def __getstate__(self):
        return (self.root, self.source, self.prefix)

    def __setstate__(self, state):
        self.root, self.source, self.prefix = state


class BatchStreams:
    """
    The streams of one batch, (tree, split, epoch, k), seen from a consumer prefix.

    Passed down to the code that draws: ``streams.sub("wave/0:CBCGenerator")``
    narrows the view, and ``stream(name)`` or ``key(name)`` draw from the
    stream at the view's prefix joined to ``name``.
    """

    __slots__ = ("tree", "split", "epoch", "k", "prefix")

    def __init__(self, tree: SeedTree, split: str, epoch: Optional[int], k: Optional[int], prefix: str = ""):
        check_name(split, "split")
        self.tree, self.split, self.epoch, self.k, self.prefix = tree, split, epoch, k, prefix

    def sub(self, prefix: str) -> "BatchStreams":
        return BatchStreams(self.tree, self.split, self.epoch, self.k, _join(self.prefix, prefix))

    def stream(self, consumer: str) -> np.random.Generator:
        return self.tree.stream(self.split, _join(self.prefix, consumer), self.epoch, self.k)

    def key(self, consumer: str):
        return self.tree.key(self.split, _join(self.prefix, consumer), self.epoch, self.k)

    def __repr__(self):
        return (f"BatchStreams({self.tree!r}, split={self.split!r}, epoch={self.epoch}, "
                f"k={self.k}, prefix={self.prefix!r})")


def _join(prefix: str, name: str) -> str:
    if not prefix:
        return name
    if not name:
        return prefix
    return f"{prefix}/{name}"


def resolve_streams(streams, seed, consumer_prefix: str) -> BatchStreams:
    """
    The streams a direct call draws from.

    ``streams`` (a BatchStreams) is used as given. Without it, a direct call
    draws from the reserved ``direct`` split of ``SeedTree(seed)``, prefixed
    with ``consumer_prefix``: deterministic, so a script is repeatable.
    """
    if streams is not None and not isinstance(streams, BatchStreams):
        raise TypeError(
            f"streams must be a gf.BatchStreams (or None), got {type(streams).__name__}; "
            "pass an integer seed as seed=."
        )
    if streams is not None:
        return streams
    return SeedTree(seed).direct().sub(consumer_prefix)
