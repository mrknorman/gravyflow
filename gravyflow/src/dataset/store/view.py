"""
Reading the store: open a committed tile with its identity checked, and take raw
window views of it.

Tiles are immutable and memory-mapped, and the OS page cache is their only cache: no
process keeps tile contents, so no reading state can pass between streams. A view is
a slice of the map, so a whole tile is never resident.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .tile import (
    SAMPLE_DTYPE, STORE_FORMAT, StoreError, TileIncomplete, TileKey, TileRecord, TileRefused,
    file_sha256, identity, key_from_path, read_record,
)

_HEADER_READERS = {
    (1, 0): np.lib.format.read_array_header_1_0,
    (2, 0): np.lib.format.read_array_header_2_0,
}


class UnheldSamples(StoreError, LookupError):
    """A window touches samples its tile does not hold."""


@dataclass(frozen=True)
class Tile:
    """A committed tile, open: its record and its samples, memory-mapped read-only."""

    record: TileRecord
    samples: np.ndarray

    @property
    def key(self) -> TileKey:
        return self.record.key

    def view(self, start: int, stop: int) -> np.ndarray:
        """
        The samples at absolute sample indices [start, stop), a read-only view of the
        map. A window that touches an unheld interval is refused by name.
        """
        key = self.key
        if not key.start <= start <= stop <= key.start + key.length:
            raise ValueError(
                f"Samples [{start}, {stop}) are not all inside {key.describe()}, which holds "
                f"[{key.start}, {key.start + key.length})."
            )
        first, last = start - key.start, stop - key.start
        for unheld_start, unheld_stop, reason in self.record.unheld:
            if unheld_start < last and first < unheld_stop:
                raise UnheldSamples(
                    f"Samples [{start}, {stop}) touch {key.describe()}'s unheld samples "
                    f"[{key.start + unheld_start}, {key.start + unheld_stop}) "
                    f"(GPS [{key.t0 + unheld_start / key.rate}, {key.t0 + unheld_stop / key.rate})): "
                    f"{reason.value}."
                )
        return self.samples[first:last]


def open_tile(root: Path, key: TileKey, *, verify: bool = False) -> Tile:
    """
    The committed tile ``key``, checked: its record names this key (a renamed or moved
    record is refused), the samples file the record's sha256 names has the tile's shape
    and dtype, and it is the very file committed (``tile.identity``), or else hashes to
    that sha256 (a file altered or put there from outside is refused). ``verify``
    checks the hash in every case. No other samples file is ever read.

    Raises ``TileMissing`` when the tile is not committed, ``TileIncomplete`` when its
    record's samples file is absent, and ``TileRefused`` when a check fails.
    """
    record = read_record(root, key)
    path = key.samples_path(root, record.sha256)
    try:
        handle = open(path, "rb")
    except FileNotFoundError:
        raise TileIncomplete(
            f"{key.describe()} has a record but no samples at {path}; it is never served, "
            "and its next writer repairs it."
        ) from None
    with handle:
        try:
            reader = _HEADER_READERS[np.lib.format.read_magic(handle)]
            shape, fortran_order, dtype = reader(handle)
        except (KeyError, ValueError, OSError) as error:
            raise TileRefused(f"{path} is not a readable .npy file ({error!r}).") from error
        offset = handle.tell()
        stat = os.fstat(handle.fileno())
        expected_size = offset + key.length * SAMPLE_DTYPE.itemsize
        if (dtype, fortran_order, shape, stat.st_size) != (SAMPLE_DTYPE, False, (key.length,), expected_size):
            raise TileRefused(
                f"{path} holds {dtype} {shape} in {stat.st_size} bytes; {key.describe()} is "
                f"{SAMPLE_DTYPE} ({key.length},) in {expected_size}. It is never served."
            )
        if (verify or identity(stat) != record.samples_identity) and file_sha256(handle) != record.sha256:
            raise TileRefused(
                f"{path} is not {key.describe()}'s samples: its sha256 differs from its "
                "record's. An altered samples file is never served."
            )
        samples = np.memmap(handle, dtype=SAMPLE_DTYPE, mode="r", offset=offset, shape=(key.length,))
    return Tile(record, samples)


def read_window(tiles: Sequence[Tile], start: int, stop: int) -> np.ndarray:
    """
    The samples at absolute sample indices [start, stop) from ``tiles``, which are
    exactly the consecutive tiles of one detector and rate that the window overlaps.
    A window inside one tile is a view of its map; one that crosses tiles is a copy.
    """
    if not stop > start:
        raise ValueError(f"A window needs stop > start, got [{start}, {stop}).")
    if not tiles:
        raise ValueError(f"No tiles given for the window [{start}, {stop}).")
    first = tiles[0].key
    needed = list(range(start // first.length, (stop - 1) // first.length + 1))
    grid = {(tile.key.detector, tile.key.rate, tile.key.tile_seconds) for tile in tiles}
    if grid != {(first.detector, first.rate, first.tile_seconds)} or [
            tile.key.index for tile in tiles] != needed:
        raise ValueError(
            f"The window [{start}, {stop}) needs tiles {needed} of one detector and rate; got "
            f"{[tile.key.describe() for tile in tiles]}."
        )
    parts = [
        tile.view(max(start, tile.key.start), min(stop, tile.key.start + tile.key.length))
        for tile in tiles
    ]
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def verify(root: Path) -> List[str]:
    """
    Check every committed tile under ``root`` in full, its sha256 included: one line
    per tile that would be refused, naming it and why. An empty list is a clean store.
    """
    problems = []
    for record_path in sorted((Path(root) / f"v{STORE_FORMAT}").glob("*/*/*/*/*.json")):
        try:
            open_tile(root, key_from_path(root, record_path), verify=True)
        except (StoreError, ValueError, TypeError, OSError) as error:
            problems.append(f"{record_path}: {error}")
    return problems
