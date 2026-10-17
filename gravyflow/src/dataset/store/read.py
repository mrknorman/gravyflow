"""
Reading real strain through the store: the one read path for a source's strain at a rate.

A read of ``n`` samples at ``rate`` from GPS ``t0`` is, in order:

1. the tiles at the source's native rate (``fetch_rate``) that the read's span covers,
   committed first when absent (``Store.ensure``: each fetched whole, once, and kept);
2. a view of their samples, memory-mapped, so no tile is ever held whole;
3. the one resampler (``utils.resample``);
4. the margin cut away: the span reaches the resampler's margin past each end of the
   window, so every sample served is exact, never an edge approximation.

The span never leaves the interval it reads: ``interiors`` gives the first sample and
the whole seconds of an interval whose span lies inside it, and the sampling plan
measures pieces by the same function, so plan and read agree.

The source's timeline is the authority on which time holds data. A read whose span
meets samples a tile does not hold, or a tile that cannot be fetched after
``FETCH_ATTEMPTS`` tries, contradicts it: ``TimelineContradiction``, which names the
detector, the interval, the source, the cause and the ``exclusions`` entry that would
plan around it. Nothing is skipped or zero-filled, and a failed fetch writes nothing.

Memory: every fetch and every read holds its estimated bytes of one process-wide
accountant (``gf.Defaults.memory_bytes``, 2 GiB by default) while it runs; a tile's
fetch or a read's span larger than the whole cap is refused before any fetch (``check``).
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
import warnings
from pathlib import Path
from typing import Dict, Hashable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from gravyflow.src.dataset.acquisition.gwosc_data import user_cache_dir
from gravyflow.src.dataset.acquisition.sources import (
    DataSource, SegmentStrain, StrainUnavailable, fetch_rate,
)
from gravyflow.src.dataset.config import Defaults
from gravyflow.src.resources import PACKAGE
from gravyflow.src.utils.resample import resample, resample_margin, resample_ratio, whole_hertz

from .budget import ByteBudget
from .commit import Fetched, Store
from .tile import TileKey
from .view import Tile, read_window

logger = logging.getLogger(__name__)

#: How many times a tile's fetch is tried before its failure contradicts the timeline.
FETCH_ATTEMPTS = 3
#: The wait before the n-th retry of a fetch is n times this, in seconds.
FETCH_BACKOFF_SECONDS = 2.0
#: How long a read waits for another caller's fetch of a tile, or for memory, in seconds.
WAIT_SECONDS = 600.0
#: Bytes a fetch holds per sample of its tile, estimated: the source's float64 decode
#: and one float64 copy (a crop, or the placement where the file covers part of the
#: tile), the float32 sealed copy and its masks. ``memory_probe.py`` measures it.
FETCH_BYTES_PER_SAMPLE = 32
#: Bytes a read holds per native sample of its span, estimated: a float32 copy where it
#: crosses tiles, the resampler's float64 working copy and its padded input.
READ_BYTES_PER_SAMPLE = 24


class TimelineContradiction(StrainUnavailable):
    """
    The timeline says this time holds data, and the source does not deliver it: a fetch
    failed after its retries, or the bytes delivered are not held (non-finite, or not in
    the source's file). ``exclusion`` is the ``(detector, (start, end), reason)`` entry
    of an obtainer's ``exclusions`` that plans around it.
    """

    def __init__(self, source: DataSource, detector: str, interval: Sequence[float], cause,
                 exclusion: Tuple[str, Tuple[float, float], str]):
        self.exclusion = exclusion
        reason = f"{type(cause).__name__}: {cause}" if isinstance(cause, BaseException) else str(cause)
        super().__init__(
            source, detector, interval,
            f"{reason}. The timeline says this time holds data; to plan around it, add "
            f"exclusions=({exclusion!r},) to the obtainer.",
        )


class Geometry(NamedTuple):
    """How a read at ``rate`` reaches the source: its native rate, ratio and margin."""

    native: int  # the native rate fetched, Hz
    up: int  # rate / native in lowest terms: up / down
    down: int
    margin: int  # native samples read past each end: the resampler's, a multiple of down

    @property
    def margin_out(self) -> int:
        """The margin in output samples."""
        return self.margin * self.up // self.down


def geometry(source: DataSource, sample_rate_hertz) -> Geometry:
    """A read's geometry at ``sample_rate_hertz`` (refusing a rate the source cannot make)."""
    rate = whole_hertz(sample_rate_hertz)
    native = whole_hertz(fetch_rate(source, rate))
    up, down = resample_ratio(native, rate)
    margin = -(-resample_margin(native, rate) // down) * down
    return Geometry(native, up, down, margin)


def interiors(source: DataSource, sample_rate_hertz, intervals) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each GPS interval [start, end): the absolute index (samples since GPS 0 at
    ``sample_rate_hertz``) of the first sample a read of it serves, and how many it
    serves: whole seconds, as many as fit while the read's span, margin included, stays
    inside the interval. The first sample lies on the grid both rates share, so the
    native samples behind it line up exactly. Zero samples where nothing fits.
    """
    g = geometry(source, sample_rate_hertz)
    rate = g.native * g.up // g.down
    bounds = np.reshape(np.asarray(intervals, dtype=np.float64), (-1, 2))
    lo = np.ceil(bounds[:, 0] * g.native).astype(np.int64)  # the first native sample inside
    hi = np.ceil(bounds[:, 1] * g.native).astype(np.int64)  # one past the last
    first = g.up * -((-(lo + g.margin)) // g.down)
    stop = (hi - g.margin) * g.up // g.down
    count = np.maximum(stop - first, 0) // rate * rate
    return first, count


def read_span(g: Geometry, first: int, num_samples: int) -> Tuple[int, int]:
    """
    The native samples [start, stop) a read of ``num_samples`` from ``first`` reaches
    under geometry ``g``: its window and the resampler's margin past each end.
    """
    start = first * g.down // g.up - g.margin
    stop = -(-(first + num_samples) * g.down // g.up) + g.margin
    return start, stop


def check(source: DataSource, sample_rate_hertz, reads: Sequence[Tuple[int, int]] = ()) -> None:
    """
    Refuse, before any fetch, a read the store can never serve: a rate the source cannot
    make (``fetch_rate``), a tile whose fetch is larger than the whole memory cap, or one
    of ``reads`` ((first, num_samples) pairs, as ``read`` takes them) whose span is.
    """
    g = geometry(source, sample_rate_hertz)
    tile_samples = g.native * whole_hertz(source.tile_seconds(g.native))
    budget = _budget()
    budget.check(tile_samples * FETCH_BYTES_PER_SAMPLE)
    for first, num_samples in reads:
        start, stop = read_span(g, int(first), int(num_samples))
        budget.check((stop - start) * READ_BYTES_PER_SAMPLE)


_lock = threading.Lock()
_budgets: Dict[int, ByteBudget] = {}
_stores: Dict[Tuple[Path, int], Store] = {}
_legacy_warned = False


def _budget() -> ByteBudget:
    """The process's accountant for the current cap (``gf.Defaults.memory_bytes``)."""
    capacity = Defaults.memory_bytes
    with _lock:
        if capacity not in _budgets:
            _budgets[capacity] = ByteBudget(capacity)
        return _budgets[capacity]


def process_store() -> Store:
    """
    The process's store under the user cache (``GRAVYFLOW_CACHE_DIR``), with the
    process's accountant: one per root, so every read in the process shares its
    single-flight table. A change of the cap applies to fetches admitted after it.
    """
    root, budget = user_cache_dir("store"), _budget()
    with _lock:
        key = (root, budget.capacity)
        if key not in _stores:
            _stores[key] = Store(root, budget=budget)
        return _stores[key]


def _human(nbytes: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024


def legacy_caches() -> List[Path]:
    """
    The real-noise segment caches an earlier gravyflow wrote: its ``segments``
    directory under the cache root, and the places it used before that root. They
    hold resampled strain under other numerics, and are never read.
    """
    places = [(user_cache_dir() / "segments", "*.hdf5"), (Path("generator_data"), "segment_data_*.hdf5"),
              (PACKAGE.parent, "segment_data_*.hdf5")]
    return sorted({path.resolve() for directory, pattern in places if directory.is_dir()
                   for path in directory.glob(pattern) if path.is_file()})


def warn_of_legacy_caches() -> None:
    """Warn once per process, naming each legacy segment cache and its size; touch none."""
    global _legacy_warned
    with _lock:
        if _legacy_warned:
            return
        _legacy_warned = True
    found = legacy_caches()
    if found:
        listed = "; ".join(f"{path} ({_human(path.stat().st_size)})" for path in found)
        warnings.warn(
            f"gravyflow no longer reads its old real-noise segment caches; real strain is read "
            f"from the store under {user_cache_dir('store')}. These are never read, changed or "
            f"deleted: {listed}. Delete them to reclaim the space.",
            UserWarning,
            stacklevel=2,
        )


def _detector_name(detector) -> str:
    return getattr(detector, "name", str(detector))


def _fetch_tile(source: DataSource, detector, key: TileKey) -> Fetched:
    """
    Tile ``key`` as ``source`` delivers it, tried ``FETCH_ATTEMPTS`` times. Samples the
    source's file does not cover are ``NOT_IN_FILE``. A delivery that is not this tile's
    (another source's strain, another rate, or samples off the tile's grid or past its
    end) is refused by name.
    """
    interval = (key.t0, key.t0 + key.tile_seconds)
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            strain = source.fetch(detector, float(key.rate), interval)
            break
        except StrainUnavailable as error:
            if attempt == FETCH_ATTEMPTS:
                raise
            logger.info(f"Fetch {attempt} of {key.describe()} failed ({error}); trying again.")
            time.sleep(FETCH_BACKOFF_SECONDS * attempt)
    offset = (float(strain.t0) - key.t0) * key.rate
    first = int(round(offset))
    data = np.asarray(strain.data)
    problem = None
    if strain.source != key.source_id or float(strain.sample_rate_hertz) != key.rate:
        problem = f"it served {strain.source} at {strain.sample_rate_hertz:g} Hz"
    elif abs(offset - first) > 1e-3 or data.ndim != 1 or first < 0 or first + len(data) > key.length:
        problem = f"its {data.shape} samples from GPS {strain.t0} do not lie on the tile's grid"
    if problem is not None:
        raise StrainUnavailable(source, detector, interval, f"not {key.describe()}: {problem}")
    not_in_file = [(a, b) for a, b in ((0, first), (first + len(data), key.length)) if b > a]
    if not_in_file:
        placed = np.full(key.length, np.nan, dtype=np.float32 if data.dtype.itemsize <= 4 else np.float64)
        placed[first:first + len(data)] = data
        data = placed
    import gwpy  # Here: the core imports without gwpy; a fetch is read through it.

    return Fetched(data, tuple(not_in_file), {"source": type(source).__name__, "gwpy": gwpy.__version__})


def _tile(store: Store, source: DataSource, detector, key: TileKey, stream: Hashable) -> Tile:
    """Tile ``key``, committed first when absent; a failed fetch contradicts the timeline."""
    try:
        return store.ensure(
            key, lambda: _fetch_tile(source, detector, key), stream=stream,
            nbytes=key.length * FETCH_BYTES_PER_SAMPLE, timeout=WAIT_SECONDS,
        )
    except StrainUnavailable as error:
        span = (key.t0, key.t0 + key.tile_seconds)
        raise TimelineContradiction(
            source, _detector_name(detector), span, error,
            (_detector_name(detector), span, f"the source could not deliver {key.describe()}"),
        ) from error


def read(source: DataSource, detector, sample_rate_hertz, first: int, num_samples: int,
         *, stream: Hashable = None, store: Optional[Store] = None) -> SegmentStrain:
    """
    ``num_samples`` of ``detector``'s strain at ``sample_rate_hertz`` from absolute
    sample index ``first`` (GPS ``first / rate``), which lies on the grid both rates
    share (``interiors`` gives such a first sample). The span read reaches the margin
    past each end, so every sample is exact.

    Raises:
        TimelineContradiction: a tile cannot be fetched, or the span meets samples it
            does not hold; naming the ``exclusions`` entry that plans around it.
        ValueError: ``first`` not on the shared grid, or no samples asked for.
    """
    g = geometry(source, sample_rate_hertz)
    rate = g.native * g.up // g.down
    first, num_samples = int(first), int(num_samples)
    if first % g.up or num_samples < 1:
        raise ValueError(
            f"A read starts on the grid {g.native} Hz and {rate} Hz share (every {g.up}th sample "
            f"at {rate} Hz) and serves at least one sample; got {num_samples} from sample {first}."
        )
    store = process_store() if store is None else store
    warn_of_legacy_caches()
    name = _detector_name(detector)
    start, stop = read_span(g, first, num_samples)
    budget = store.budget
    if budget:  # A span larger than the whole cap is refused before any fetch.
        budget.check((stop - start) * READ_BYTES_PER_SAMPLE)
    tile_seconds = whole_hertz(source.tile_seconds(g.native))
    length = g.native * tile_seconds
    tiles = []
    for index in range(start // length, (stop - 1) // length + 1):
        key = TileKey(source.source_id(index * tile_seconds, g.native), name, g.native, tile_seconds, index)
        tile = _tile(store, source, detector, key, stream)
        for unheld_start, unheld_stop, reason in tile.record.unheld:
            if key.start + unheld_start < stop and start < key.start + unheld_stop:
                gaps = (key.t0 + unheld_start / g.native, key.t0 + unheld_stop / g.native)
                raise TimelineContradiction(
                    source, name, (start / g.native, stop / g.native),
                    f"{key.describe()} does not hold GPS [{gaps[0]}, {gaps[1]}): {reason.value}",
                    (name, gaps, reason.value),
                )
        tiles.append(tile)
    held = budget.admit((stop - start) * READ_BYTES_PER_SAMPLE, WAIT_SECONDS) if budget else contextlib.nullcontext()
    with held:
        span = read_window(tiles, start, stop)
        data, _ = resample(span, start / g.native, g.native, rate)
        data = np.array(data[g.margin_out:g.margin_out + num_samples])
    return SegmentStrain(data, first / rate, float(rate), tiles[0].key.source_id)
