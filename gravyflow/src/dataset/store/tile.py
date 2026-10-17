"""
The store's tile format: raw strain as the source publishes it, one source file per
tile, written once.

A tile is keyed by (source, detector, rate, k) and covers GPS [k*T, (k+1)*T), where T
is the source's file length at that rate, so a tile lies on the absolute GPS grid and
one download is one tile. On disk it is two files in
``<root>/v<STORE_FORMAT>/<source_id>/<detector>/<rate>/``:

- ``k.npy``, the samples: float32, one dimension, ``rate * T`` long, memory-mapped by
  readers;
- ``k.json``, the commit record: the key, ``t0``, the unheld sample intervals with
  their reasons, the sha256 of ``k.npy``, the inode ``k.npy`` was committed as, and
  the fetch metadata.

Opening a tile checks its identity, belt and braces: the record must name the
key its path names, and the samples must be the very file committed (one ``fstat``)
or else hash to the record's sha256, so a renamed, moved or swapped file is refused
by name. The full hash is otherwise checked at commit and by ``verify``.

A tile exists only when its record exists. The record is the authority on which
samples are held: unheld samples are NaN in ``k.npy`` as poison only, and no reader
infers anything from a value. The held intervals are the complement of the unheld
ones, so the record carries one truth, not two.
"""

from __future__ import annotations

import hashlib
import json
import operator
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np

#: The on-disk format of tiles and their records. It names the store's root
#: directory, so a new format never reads an old one.
STORE_FORMAT = 1

#: The samples' dtype on disk.
SAMPLE_DTYPE = np.dtype("<f4")

#: Record fields that describe one writer's commit, not the tile's content: two
#: writers of one tile differ in them and agree in every other field.
COMMIT_METADATA = ("fetch", "samples_ino")

#: A source id or detector is one plain path component.
_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")

_HASH_CHUNK = 1 << 23


class StoreError(Exception):
    """The base of every error the store raises by name."""


class TileMissing(StoreError, LookupError):
    """No committed record exists for a tile: it has not been committed."""


class TileRefused(StoreError):
    """A committed tile fails its checks on open and is never served."""


class TileIncomplete(TileRefused):
    """
    A tile's record exists without its samples (a ``forget`` raced a commit). It is
    never served; the next writer of the tile links the samples and repairs it.
    """


class Unheld(str, Enum):
    """Why a committed tile does not hold an interval, judged from the bytes received."""

    NON_FINITE = "NON_FINITE"  # the source returned non-finite samples
    NOT_IN_FILE = "NOT_IN_FILE"  # the source's file did not cover the interval


def _component(name: str, value) -> str:
    if not isinstance(value, str) or not _COMPONENT.fullmatch(value):
        raise ValueError(
            f"A tile's {name} must be one plain path component (letters, digits, '.', "
            f"'_', '+', '-', not starting with a symbol); got {value!r}."
        )
    return value


def _whole(name: str, value, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"A tile's {name} must be a whole number, got {value!r}.")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"A tile's {name} must be a whole number, got {value!r}.") from None
    if value < minimum:
        raise ValueError(f"A tile's {name} must be at least {minimum}, got {value}.")
    return value


@dataclass(frozen=True)
class TileKey:
    """
    One tile: ``index`` k of the grid of ``tile_seconds`` T at ``rate``, for one
    detector of one source. Its samples are the absolute sample indices
    [k*T*rate, (k+1)*T*rate) at that rate, so sample n of the tile is at GPS
    ``t0 + n / rate`` exactly.
    """

    source_id: str
    detector: str
    rate: int
    tile_seconds: int
    index: int

    def __post_init__(self):
        _component("source_id", self.source_id)
        _component("detector", self.detector)
        object.__setattr__(self, "rate", _whole("rate", self.rate, 1))
        object.__setattr__(self, "tile_seconds", _whole("tile_seconds", self.tile_seconds, 1))
        object.__setattr__(self, "index", _whole("index", self.index, 0))

    @property
    def t0(self) -> float:
        """The GPS time of the tile's first sample, exact in float64."""
        return float(self.index * self.tile_seconds)

    @property
    def length(self) -> int:
        """The tile's number of samples."""
        return self.rate * self.tile_seconds

    @property
    def start(self) -> int:
        """The absolute sample index (samples since GPS 0 at this rate) of sample 0."""
        return self.index * self.length

    def directory(self, root: Path) -> Path:
        return Path(root) / f"v{STORE_FORMAT}" / self.source_id / self.detector / str(self.rate)

    def samples_path(self, root: Path) -> Path:
        return self.directory(root) / f"{self.index}.npy"

    def record_path(self, root: Path) -> Path:
        return self.directory(root) / f"{self.index}.json"

    def describe(self) -> str:
        return (
            f"tile {self.index} of {self.source_id} {self.detector} at {self.rate} Hz "
            f"(GPS [{self.t0:.0f}, {self.t0 + self.tile_seconds:.0f}))"
        )


def covering(source_id: str, detector: str, rate: int, tile_seconds: int,
             gps_start: float, gps_stop: float) -> Tuple[TileKey, ...]:
    """The keys of every tile that overlaps GPS [gps_start, gps_stop)."""
    if not gps_stop > gps_start:
        raise ValueError(f"An interval needs stop > start, got [{gps_start}, {gps_stop}).")
    first = int(np.floor(gps_start / tile_seconds))
    last = int(np.ceil(gps_stop / tile_seconds))
    return tuple(TileKey(source_id, detector, rate, tile_seconds, k) for k in range(first, last))


Interval = Tuple[int, int, Unheld]


def _runs(mask: np.ndarray) -> Sequence[Tuple[int, int]]:
    """The [start, stop) runs of True in a boolean mask."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return tuple(zip(edges[::2].tolist(), edges[1::2].tolist()))


def seal(key: TileKey, samples, not_in_file: Iterable[Tuple[int, int]] = ()
         ) -> Tuple[np.ndarray, Tuple[Interval, ...]]:
    """
    The samples as the store keeps them, and the unheld intervals.

    ``samples`` is the whole tile as fetched; ``not_in_file`` lists the [start, stop)
    sample intervals, relative to the tile, that the source's file did not cover.
    Every other non-finite sample is ``NON_FINITE``. Unheld samples are set to NaN, so
    the bytes, and their sha256, depend only on the held samples.
    """
    values = np.asarray(samples)
    if values.shape != (key.length,):
        raise ValueError(
            f"{key.describe()} holds {key.length} samples; got an array of shape {values.shape}."
        )
    if not (np.issubdtype(values.dtype, np.floating) or np.issubdtype(values.dtype, np.integer)):
        raise TypeError(f"Strain samples must be real numbers, got dtype {values.dtype}.")
    absent = np.zeros(key.length, dtype=bool)
    for start, stop in not_in_file:
        if not 0 <= start < stop <= key.length:
            raise ValueError(
                f"A not-in-file interval [{start}, {stop}) lies outside {key.describe()}'s "
                f"{key.length} samples."
            )
        absent[start:stop] = True
    with np.errstate(over="ignore", invalid="ignore"):
        sealed = values.astype(SAMPLE_DTYPE)
    overflowed = np.isfinite(values) & ~np.isfinite(sealed) & ~absent
    if overflowed.any():
        raise ValueError(
            f"{int(overflowed.sum())} finite samples of {key.describe()} overflow float32."
        )
    non_finite = ~np.isfinite(sealed) & ~absent
    sealed[absent | non_finite] = np.nan
    unheld = sorted(
        [(a, b, Unheld.NOT_IN_FILE) for a, b in _runs(absent)]
        + [(a, b, Unheld.NON_FINITE) for a, b in _runs(non_finite)]
    )
    return sealed, tuple(unheld)


def file_sha256(handle) -> str:
    """The sha256 of an open binary file's whole content."""
    handle.seek(0)
    digest = hashlib.sha256()
    while chunk := handle.read(_HASH_CHUNK):
        digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TileRecord:
    """A tile's commit record, ``k.json``."""

    key: TileKey
    sha256: str
    unheld: Tuple[Interval, ...]
    samples_ino: int
    fetch: Mapping

    @property
    def held(self) -> Tuple[Tuple[int, int], ...]:
        """The held [start, stop) sample intervals: the complement of the unheld ones."""
        held, cursor = [], 0
        for start, stop, _ in self.unheld:
            if start > cursor:
                held.append((cursor, start))
            cursor = stop
        if cursor < self.key.length:
            held.append((cursor, self.key.length))
        return tuple(held)

    def to_json(self) -> dict:
        key = self.key
        return {
            "store_format": STORE_FORMAT,
            "source_id": key.source_id,
            "detector": key.detector,
            "rate": key.rate,
            "tile_seconds": key.tile_seconds,
            "index": key.index,
            "t0": key.t0,
            "sha256": self.sha256,
            "unheld": [[start, stop, reason.value] for start, stop, reason in self.unheld],
            "samples_ino": self.samples_ino,
            "fetch": dict(self.fetch),
        }

    def encode(self) -> bytes:
        return (json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")) + "\n").encode()


def content(fields: Mapping) -> dict:
    """A record's fields that describe the tile's content: two writers agree on these."""
    return {name: value for name, value in fields.items() if name not in COMMIT_METADATA}


def parse_record(text: bytes, key: TileKey, path: Path) -> TileRecord:
    """
    The record at ``path``, checked against the key its path names (identity,
    belt and braces): a renamed or moved record is refused by name, never served.
    """
    try:
        fields = json.loads(text)
        unheld = tuple(
            (operator.index(start), operator.index(stop), Unheld(reason))
            for start, stop, reason in fields["unheld"]
        )
        record = TileRecord(
            key=TileKey(fields["source_id"], fields["detector"], fields["rate"],
                        fields["tile_seconds"], fields["index"]),
            sha256=str(fields["sha256"]),
            unheld=unheld,
            samples_ino=operator.index(fields["samples_ino"]),
            fetch=dict(fields["fetch"]),
        )
        stated = {name: fields[name] for name in ("store_format", "t0")}
    except (ValueError, TypeError, KeyError) as error:
        raise TileRefused(f"The record {path} is malformed ({error!r}); it is never served.") from error
    expected = {
        "store_format": STORE_FORMAT, "source_id": key.source_id, "detector": key.detector,
        "rate": key.rate, "tile_seconds": key.tile_seconds, "index": key.index, "t0": key.t0,
    }
    found = dict(stated, **{
        name: getattr(record.key, name)
        for name in ("source_id", "detector", "rate", "tile_seconds", "index")
    })
    mismatched = [
        f"{name} {found[name]!r} (the path says {value!r})"
        for name, value in expected.items() if found[name] != value
    ]
    if mismatched:
        raise TileRefused(
            f"The record {path} is not {key.describe()}'s: it states {', '.join(mismatched)}. "
            "A renamed or moved tile is never served."
        )
    cursor = 0
    for start, stop, _ in unheld:
        if not cursor <= start < stop <= key.length:
            raise TileRefused(
                f"The record {path} lists unheld intervals that are unsorted, overlapping "
                f"or outside the tile's {key.length} samples; it is never served."
            )
        cursor = stop
    return record


def read_record(root: Path, key: TileKey) -> TileRecord:
    """The committed record of ``key``, or ``TileMissing`` when it has none."""
    path = key.record_path(root)
    try:
        text = path.read_bytes()
    except FileNotFoundError:
        raise TileMissing(f"{key.describe()} is not committed under {root}.") from None
    return parse_record(text, key, path)


def key_from_path(root: Path, record_path: Path, tile_seconds: int) -> TileKey:
    """The key a record's path names, with ``tile_seconds`` from the record."""
    relative = Path(record_path).relative_to(Path(root) / f"v{STORE_FORMAT}")
    source_id, detector, rate, name = relative.parts
    return TileKey(source_id, detector, int(rate), tile_seconds, int(name[: -len(".json")]))
