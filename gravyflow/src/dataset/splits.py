"""
The split: which of train, validate and test a stretch of GPS time belongs to.

One frozen ``SplitSpec`` decides it for every route (noise, glitches, events,
datasets, composed datasets, the validator), from absolute GPS alone:

- Time is cut into blocks of ``block_seconds`` counted from GPS 0, so block k is
  ``[k B, (k + 1) B)``. A block's split is a hash of the spec's ``version`` and
  k, never of a seed, the configured runs or the order of anything, so a GPS
  time is in one split whatever asks.
- Where two blocks of different splits meet, ``guard_seconds`` of time, half on
  each side of the edge, belongs to no split. Data of two splits is therefore
  always at least the guard apart; joining two blocks of one split costs nothing.
- The hash is placed among the fractions from both ends: test fills [0, f_test)
  from the bottom, validate fills (1 - f_validate, 1] from the top, and train is
  the rest. Raising the test or validate fraction thus takes time from train
  alone, and changing one of them never moves the other.

- Time a benchmark tests on is reserved (``Reservation``): it is in no train or
  validate draw, so no model gravyflow trains or selects has seen it, and test
  keeps it. By default MLGWSC-1's O3a segments are reserved
  (``acquisition.mlgwsc1``); ``reserved=()``, asked for explicitly, is the
  challenge-faithful mode, which trains on them as the challenge's entries did.

A spec's ``fingerprint`` names it, its reservations included, in every cache key,
index file and model's metadata, so data split by one spec is never read as
another's; ``mode`` says in words whether benchmark time is reserved.
"""

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Tuple, Union

import numpy as np

from gravyflow.src.dataset.acquisition.gwosc_data import coalesce_segments, subtract_segments
from gravyflow.src.dataset.reservation import Reservation

#: The splits, in canonical order: test first.
SPLIT_NAMES: Tuple[str, ...] = ("test", "validate", "train")

#: The splits reserved time is held out of: those a model trains and is selected on.
RESERVED_SPLITS: Tuple[str, ...] = ("validate", "train")

#: What ``SplitSpec.assign`` returns for an interval that belongs to no split:
#: it crosses an edge between splits, lies inside a guard, or, in train or
#: validate, meets reserved time.
EXCLUDED = "excluded"

#: ``SplitSpec.mode`` of a spec that reserves nothing.
CHALLENGE_FAITHFUL = "challenge-faithful"


def _default_reservations() -> Tuple[Reservation, ...]:
    from gravyflow.src.dataset.acquisition.mlgwsc1 import MLGWSC1_O3A

    return (MLGWSC1_O3A,)


def hash_uniform(*parts) -> float:
    """
    sha256 of ``parts`` joined by ``":"`` as a uniform number in [0, 1), from its
    first 8 bytes: a draw that depends on nothing but its parts.
    """
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


@lru_cache(maxsize=1 << 16)
def _block_uniform(version: str, block: int) -> float:
    """sha256 of ``"{version}:{block}"`` as a uniform number in [0, 1)."""
    return hash_uniform(version, block)


def _canonical_fractions(fractions) -> Tuple[Tuple[str, float], ...]:
    fractions = dict(fractions)
    if set(fractions) != set(SPLIT_NAMES):
        raise ValueError(
            f"Split fractions must name exactly the splits {SPLIT_NAMES}, got {sorted(fractions)}."
        )
    values = [float(fractions[name]) for name in SPLIT_NAMES]
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"Split fractions must be non-negative, got {fractions}.")
    if abs(sum(values) - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1, got {fractions} (sum {sum(values)}).")
    return tuple(zip(SPLIT_NAMES, values))


@dataclass(frozen=True)
class SplitSpec:
    """
    The one split of GPS time into train, validate and test (see the module).

    Args:
        fractions: {split: fraction} for exactly test, validate and train, summing
            to 1; kept in canonical order.
        block_seconds: The block length; blocks are counted from GPS 0.
        guard_seconds: The time at every edge between blocks of different splits
            that belongs to no split, half on each side.
        version: A salt: specs of different versions split time independently
            (cross-validation folds, say).
        reserved: The ``Reservation``s held out of train and validate; by default
            MLGWSC-1's O3a segments (``acquisition.mlgwsc1.MLGWSC1_O3A``). ``()`` is
            the challenge-faithful mode: only ask for it to reproduce a benchmark's
            published training, never for a model meant to be compared on it.
    """

    fractions: Tuple[Tuple[str, float], ...] = (("test", 0.10), ("validate", 0.10), ("train", 0.80))
    block_seconds: float = 8192.0
    guard_seconds: float = 128.0
    version: str = "gravyflow-split-v1"
    reserved: Tuple[Reservation, ...] = field(default_factory=_default_reservations)

    def __post_init__(self):
        reserved = tuple(self.reserved)
        if not all(isinstance(reservation, Reservation) for reservation in reserved):
            raise TypeError(f"reserved holds gf.Reservation objects, got {reserved!r}.")
        names = [reservation.name for reservation in reserved]
        if len(set(names)) != len(names):
            raise ValueError(f"Reservations must have distinct names, got {names}.")
        object.__setattr__(self, "reserved", reserved)
        object.__setattr__(self, "fractions", _canonical_fractions(self.fractions))
        object.__setattr__(self, "block_seconds", float(self.block_seconds))
        object.__setattr__(self, "guard_seconds", float(self.guard_seconds))
        if not self.block_seconds > 0.0:
            raise ValueError(f"block_seconds must be positive, got {self.block_seconds}.")
        if not 0.0 <= self.guard_seconds < self.block_seconds:
            raise ValueError(
                f"guard_seconds must be in [0, block_seconds), got {self.guard_seconds}."
            )
        if not isinstance(self.version, str) or not self.version:
            raise ValueError(f"version must be a non-empty string, got {self.version!r}.")

    # ------------------------------------------------------------------ names

    @staticmethod
    def check_name(name: str) -> str:
        """``name`` if it is a split; else KeyError naming the splits."""
        if name is None:
            raise KeyError(f"Name the split (group=): a split is one of {list(SPLIT_NAMES)}.")
        if not isinstance(name, str) or name not in SPLIT_NAMES:
            raise KeyError(
                f"Unknown split {name!r}: a split is one of {list(SPLIT_NAMES)}."
            )
        return name

    def fraction(self, name: str) -> float:
        return dict(self.fractions)[self.check_name(name)]

    # ----------------------------------------------------------------- blocks

    def block_split(self, block: int) -> str:
        """The split of block ``block``, ``[k B, (k + 1) B)`` in GPS."""
        u = _block_uniform(self.version, int(block))
        if u < self.fraction("test"):
            return "test"
        if u >= 1.0 - self.fraction("validate"):
            return "validate"
        return "train"

    def block_of(self, gps: float) -> int:
        return int(np.floor(float(gps) / self.block_seconds))

    def _interior(self, block: int) -> Tuple[float, float]:
        """Block ``block`` less half the guard at each edge with another split."""
        name = self.block_split(block)
        half = self.guard_seconds / 2.0
        start = block * self.block_seconds
        end = start + self.block_seconds
        if self.block_split(block - 1) != name:
            start += half
        if self.block_split(block + 1) != name:
            end -= half
        return start, end

    # ------------------------------------------------------------- intervals

    def assign(self, start: float, end: float) -> str:
        """
        The split of the interval [start, end): its split by blocks and guards
        (``block_assign``) when, for train or validate, it meets no detector's
        reserved time (``reserving``); else ``EXCLUDED``.
        """
        name = self.block_assign(start, end)
        if name in RESERVED_SPLITS and self.reserving(start, end) is not None:
            return EXCLUDED
        return name

    def block_assign(self, start: float, end: float) -> str:
        """
        The split of the interval [start, end) by blocks and guards alone: the
        split of every block it touches when that is one split and the interval
        keeps half the guard from every edge with another split; else ``EXCLUDED``.
        Reserved time is ``assign``'s.
        """
        start, end = float(start), float(end)
        if not end >= start:
            raise ValueError(f"An interval must have end >= start, got [{start}, {end}).")
        first = self.block_of(start)
        last = self.block_of(np.nextafter(end, -np.inf)) if end > start else first
        name = self.block_split(first)
        if any(self.block_split(k) != name for k in range(first + 1, last + 1)):
            return EXCLUDED
        lower, _ = self._interior(first)
        _, upper = self._interior(last)
        return name if lower <= start and end <= upper else EXCLUDED

    # ------------------------------------------------------------ reservations

    @property
    def mode(self) -> str:
        """In words, what the spec holds out of train and validate."""
        if not self.reserved:
            return CHALLENGE_FAITHFUL
        return "reserved: " + ", ".join(reservation.name for reservation in self.reserved)

    def reserved_time(self, detector=None) -> np.ndarray:
        """
        The time held out of train and validate on ``detector`` (a ``gf.IFO`` or its
        name): every reservation's, as sorted, coalesced (N, 2) GPS intervals. For
        None, every detector's: the time a record read on any detector must avoid.
        """
        return coalesce_segments(np.concatenate(
            [np.empty((0, 2))] + [reservation.of(detector) for reservation in self.reserved]))

    def reserving(self, start: float, end: float, detector=None) -> Optional[Reservation]:
        """
        The first reservation whose time on ``detector`` (every detector's for None)
        meets [start, end), or None.
        """
        for reservation in self.reserved:
            time = reservation.of(detector)
            k = np.searchsorted(time[:, 1], float(start), side="right")
            if k < len(time) and time[k, 0] < float(end):
                return reservation
        return None

    def admissible(self, segments, name: str, detector=None) -> np.ndarray:
        """
        ``segments`` ((N, 2) GPS intervals) intersected with the time of split
        ``name``: its blocks, less half the guard at each edge with another
        split, and for train or validate less ``detector``'s reserved time
        (``reserved_time``; every detector's for None). Returns sorted (M, 2)
        intervals; same-split blocks stay joined.
        """
        self.check_name(name)
        segments = np.reshape(np.asarray(segments, dtype=np.float64), (-1, 2))
        segments = segments[segments[:, 1] > segments[:, 0]]
        if len(segments) == 0:
            return np.empty((0, 2))
        out = []
        for start, end in segments[np.argsort(segments[:, 0], kind="stable")]:
            for block in range(self.block_of(start), self.block_of(np.nextafter(end, -np.inf)) + 1):
                if self.block_split(block) != name:
                    continue
                lower, upper = self._interior(block)
                lower, upper = max(lower, start), min(upper, end)
                if upper <= lower:
                    continue
                if out and out[-1][1] >= lower:
                    out[-1][1] = max(out[-1][1], upper)
                else:
                    out.append([lower, upper])
        out = np.array(out, dtype=np.float64).reshape(-1, 2)
        if name in RESERVED_SPLITS and self.reserved:
            out = subtract_segments(out, self.reserved_time(detector)).reshape(-1, 2)
        return out

    # ------------------------------------------------------------ provenance

    def fingerprint(self) -> str:
        """
        The spec's identity: sha256 of its canonical JSON, first 16 hex digits. Its
        reservations are in it by name and time, so the challenge-faithful mode
        (none) and the default never share a fingerprint.
        """
        text = json.dumps({
            "fractions": [[name, value] for name, value in self.fractions],
            "block_seconds": self.block_seconds,
            "guard_seconds": self.guard_seconds,
            "version": self.version,
            "reserved": [reservation.specification() for reservation in self.reserved],
        }, sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:16]


#: The split of every route given none.
DEFAULT_SPLIT = SplitSpec()


def as_split_spec(spec: Union[SplitSpec, None]) -> SplitSpec:
    """``spec``, or the default split for None."""
    if spec is None:
        return DEFAULT_SPLIT
    if isinstance(spec, SplitSpec):
        return spec
    raise TypeError(f"A split is a gf.SplitSpec, got {type(spec).__name__}.")


def require_split_fingerprint(recorded, spec: SplitSpec, where: str) -> None:
    """
    Refuse data or a model split by another spec: ``recorded`` (a fingerprint,
    or None when nothing was recorded) must equal ``spec``'s.
    """
    live = spec.fingerprint()
    if recorded is None:
        raise ValueError(
            f"{where} records no split fingerprint, so the split its data came from is "
            f"unknown; this run's split is {live}."
        )
    if str(recorded) != live:
        raise ValueError(
            f"{where} was split by SplitSpec {recorded}, but this run splits by {live}: "
            "its held-out data may be this run's training data."
        )
