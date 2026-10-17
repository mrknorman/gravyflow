# Standard library imports:
import logging
import sys

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum, auto
from contextlib import closing
from typing import TYPE_CHECKING, List, Tuple, Union, Dict, Any, Optional, Generator, Sequence, FrozenSet
from pathlib import Path
from .cache import MemoryCache
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Third-party imports:
import numpy as np
from numpy.random import default_rng  
import keras
from keras import ops

if TYPE_CHECKING:
    from gwpy.timeseries import TimeSeries

# Local imports:
import gravyflow as gf
from gravyflow.src.resources import PACKAGE
from gravyflow.src.utils.io import open_hdf5_file
from gravyflow.src.utils.resample import resample, whole_hertz
from gravyflow.src.utils.tensor import check_tensor_integrity
from gravyflow.src.utils.numerics import ensure_even, ensure_list
from gravyflow.src.utils.gps import as_gps_times
from gravyflow.src.utils.shapes import ShapeEnforcer
from gravyflow.src.dataset.store.pools import thread_pool
from .gwosc_data import (
    GWOSC_RUNS, DEFAULT_GWOSC_FLAGS, UpsamplingRefused, coalesce_segments,
    subtract_segments, user_cache_dir, warn_of_moved_cache,
)
from .sources import DataSource, SegmentStrain, fetch_rate
from .versions import DATA_VERSIONS


# =============================================================================
# CONSTANTS
# =============================================================================

# Imported from TransientDefaults for consistency (canonical source)
from gravyflow.src.dataset.config import TransientDefaults, ignore_scale_factor
from gravyflow.src.dataset.splits import SplitSpec, as_split_spec
from gravyflow.src.dataset.features.event import EventConfidence

# GPS time precision is 10ms (0.01s) - defined in gravyflow.src.utils.gps
# GPS_PRECISION_DECIMALS was 3 (1ms) which was stale - now using GPS_KEY_PRECISION
GPS_PRECISION_DECIMALS = 2  # 10ms precision (matches gps.py)

# Tolerance for GPS time matching in seconds (used in cache lookups)
GPS_TOLERANCE_SECONDS = TransientDefaults.GPS_TOLERANCE_SECONDS

# Real-noise segments are cut on an absolute grid of this many GPS seconds:
# block k is [2048 k, 2048 (k + 1)).
SEGMENT_GRID_ORIGIN_GPS = 0.0

# Version of the stored real-noise segment layout (vetoes, absolute grid, whole
# segments resampled by gf.resample, each stored with the GPS time of its first sample,
# its rate and its source). It is part of the segment file's hash, so files written
# under an earlier layout are never read back: layout 3 was resampled by FFT, and
# layout 2 stored bare samples under their nominal times.
SEGMENT_LAYOUT_VERSION = "segment-layout-4"


def segment_key(ifo, start, end) -> str:
    """The segment file's key for ``ifo``'s segment [start, end)."""
    return f"segments/segment_{ifo.name}_{start}_{end}"


def ifo_canonical_key(ifos) -> tuple:
    """
    Generate a canonical key for a list of IFOs.
    
    Creates a consistent, hashable representation for caching and comparison.
    
    Args:
        ifos: List of IFO objects or IFO enum values
        
    Returns:
        Tuple of sorted IFO names, e.g., ('H1', 'L1')
    """
    return tuple(sorted(ifo.name for ifo in ifos))


# =============================================================================
# ENUMS
# =============================================================================

class DataQuality(Enum):
    RAW = auto()
    BEST = auto()

class DataLabel(IntEnum):
    """Data type labels with explicit integer values for direct .value access."""
    NOISE = 0
    GLITCHES = 1
    EVENTS = 2


def label_kinds(data_labels) -> FrozenSet[DataLabel]:
    """
    The kinds of data ``data_labels`` (one label or a list) ask for, each read
    from its label's type: a DataLabel is its own kind, a GlitchType asks for
    glitches and an EventConfidence for real events.

    Every question of what an obtainer serves (real events above all, since they
    are evaluation data) is asked of this, never of the labels' numbers: the three
    are IntEnums whose values overlap (GlitchType.CHIRP and DataLabel.EVENTS are
    both 2), so ``==`` across them, and ``in`` a list of them, cannot tell one
    kind from another. Anything else, a bare number included, names no kind and
    is refused.
    """
    from gravyflow.src.dataset.features.glitch import GlitchType  # Here: glitch imports this module.

    kinds = set()
    for label in ensure_list(data_labels):
        if isinstance(label, DataLabel):
            kinds.add(label)
        elif isinstance(label, GlitchType):
            kinds.add(DataLabel.GLITCHES)
        elif isinstance(label, EventConfidence):
            kinds.add(DataLabel.EVENTS)
        else:
            raise ValueError(
                f"{label!r} is not a data label: give gf.DataLabel, gf.GlitchType or gf.EventConfidence "
                "members. A number names no kind of data: 2 is DataLabel.EVENTS and GlitchType.CHIRP alike."
            )
    return frozenset(kinds)


def check_data_labels(data_labels) -> list:
    """``data_labels`` (one label or a list) as a list of label members; ``label_kinds`` refuses any other."""
    labels = ensure_list(data_labels)
    label_kinds(labels)
    return labels


class SegmentOrder(Enum):
    RANDOM = auto()
    SHORTEST_FIRST = auto()
    CHRONOLOGICAL = auto()
    
class AcquisitionMode(Enum):
    NOISE = auto()
    TRANSIENT = auto()  # For point-in-time events (GW events, glitches)


class SamplingMode(Enum):
    """Noise sampling strategy."""
    RANDOM = auto()  # Random subsections with overlap (training) 
    GRID = auto()    # Non-overlapping sequential chunks (validation/FAR)


# =============================================================================
# OBSERVING RUN DATA
# =============================================================================

@dataclass
class ObservingRunData:
    """
    One observing run: its GPS bounds and segment flags.

    The built-in runs take their bounds from the GWOSC run API through
    ``gwosc_runs``: O3, for example, is GWOSC's O3a plus O3b. A run defined by hand
    can give UTC datetimes instead, which are converted to GPS counting leap seconds.

    ``state_flags`` holds, per data quality, the GWOSC timeline flags whose
    intersection defines the usable times (see ``gwosc_data``). Where the run's strain
    is read from is the obtainer's data source (``sources``), not the run's.
    """
    name: str
    start_date_time: Optional[datetime]
    end_date_time: Optional[datetime]
    state_flags: Dict
    gwosc_runs: Tuple[str, ...] = ()

    def __post_init__(self):
        self.gwosc_runs = tuple(self.gwosc_runs)
        if self.gwosc_runs:
            unknown = [run for run in self.gwosc_runs if run not in GWOSC_RUNS]
            if unknown:
                raise ValueError(
                    f"Unknown GWOSC runs {unknown}; known runs: {sorted(GWOSC_RUNS)}."
                )
            self.start_gps_time = float(
                min(GWOSC_RUNS[run]["gps_start"] for run in self.gwosc_runs)
            )
            self.end_gps_time = float(
                max(GWOSC_RUNS[run]["gps_end"] for run in self.gwosc_runs)
            )
            if self.start_date_time is None:
                self.start_date_time = self._to_date_time(self.start_gps_time)
            if self.end_date_time is None:
                self.end_date_time = self._to_date_time(self.end_gps_time)
        else:
            if self.start_date_time is None or self.end_date_time is None:
                raise ValueError(
                    "ObservingRunData needs either gwosc_runs or both datetimes."
                )
            self.start_gps_time = self._to_gps_time(self.start_date_time)
            self.end_gps_time = self._to_gps_time(self.end_date_time)

    @staticmethod
    def _to_gps_time(date_time: datetime) -> float:
        """UTC datetime to GPS seconds, counting the leap seconds in between."""
        from astropy.time import Time
        return float(Time(date_time, scale="utc").gps)

    @staticmethod
    def _to_date_time(gps_time: float) -> datetime:
        """GPS seconds to a naive UTC datetime."""
        from astropy.time import Time
        return Time(gps_time, format="gps").utc.to_datetime()

observing_run_data : Dict = {
    "O1" : ("O1", None, None, {DataQuality.BEST: DEFAULT_GWOSC_FLAGS}, ("O1",)),
    "O2" : ("O2", None, None, {DataQuality.BEST: DEFAULT_GWOSC_FLAGS}, ("O2",)),
    "O3" : ("O3", None, None, {DataQuality.BEST: DEFAULT_GWOSC_FLAGS}, ("O3a", "O3b")),
    "O4" : ("O4", None, None, {DataQuality.BEST: DEFAULT_GWOSC_FLAGS}, ("O4a", "O4b")),
}

class ObservingRun(Enum):
    O1 = ObservingRunData(*observing_run_data["O1"])
    O2 = ObservingRunData(*observing_run_data["O2"])
    O3 = ObservingRunData(*observing_run_data["O3"])
    O4 = ObservingRunData(*observing_run_data["O4"])
    
    @property
    def index(self) -> int:
        """Get integer index for this run (O1=0, O2=1, O3=2, O4=3)."""
        return list(ObservingRun).index(self)



# =============================================================================
# RANDOM SUBSECTION FUNCTIONS (JIT-compiled)
# =============================================================================

#: Samples left spare after the last window of a piece, against edge artifacts from
#: FFT and filtering operations.
WINDOW_EDGE_BUFFER_SAMPLES = 16

#: Tolerance on a piece's duration before it is cut to whole seconds: GPS times near
#: 1e9 s carry float64 rounding of about 1e-7 s, far below one sample.
_KEPT_SECONDS_TOLERANCE = 1e-6


def kept_seconds(duration_seconds):
    """
    The whole seconds of a piece that are loaded: its duration, cut to whole seconds
    at its end (``get_segment``), so that the number of distinct piece lengths, and
    so of compilations of the window sampler, stays bounded. The sampling plan
    measures a piece's window starts on this same length: the plan from the piece's
    GPS span, the loader from the samples it fetched, which agree unless a fetch
    returns less than its span.
    """
    return np.floor(np.asarray(duration_seconds, dtype=np.float64) + _KEPT_SECONDS_TOLERANCE)


def window_start_bounds(
        num_samples,
        num_onsource_samples: int,
        num_offsource_samples: int,
    ) -> Tuple[int, int]:
    """
    The [lo, hi) onsource start indices whose windows fit in a piece of ``num_samples``:
    the offsource before the onsource, and ``WINDOW_EDGE_BUFFER_SAMPLES`` spare after it.
    """
    return num_offsource_samples, num_samples - num_onsource_samples - WINDOW_EDGE_BUFFER_SAMPLES


def window_offsets_seconds(positions, time_interval_seconds) -> np.ndarray:
    """Seconds from a piece's first sample to the samples at ``positions``, in float64."""
    return np.asarray(positions).astype(np.float64) * float(time_interval_seconds)


#: Rejection draws per detector, for each example, before its start is drawn
#: constructively from the starts far enough from the others' (``separated_starts``).
SEPARATION_DRAWS = 16


def separable(bounds: Sequence[Tuple[float, float]], separation: float) -> bool:
    """
    Whether one point per interval (``bounds``: [lo, hi] on one clock) can be placed
    with every two points at least ``separation`` apart. Exact: for each order of the
    points, placing each as early as its interval and the one before allow is optimal
    for that order, so the intervals are separable when some order fits.
    """
    from itertools import permutations

    for order in permutations(range(len(bounds))):
        last = -np.inf
        for i in order:
            lo, hi = bounds[i]
            point = max(lo, last + separation)
            if point > hi:
                break
            last = point
        else:
            return True
    return False


def _allowed_draw(rng: np.random.Generator, lo: int, hi: int, taken: np.ndarray, separation: float):
    """
    A start uniform over the integers of [lo, hi) at least ``separation`` from every
    start in ``taken`` (on the same clock), or None when there is none.
    """
    pieces = [(lo, hi)]
    for centre in np.sort(taken):
        cut_lo, cut_hi = int(np.floor(centre - separation)) + 1, int(np.ceil(centre + separation))
        pieces = [part for a, b in pieces
                  for part in ((a, min(b, cut_lo)), (max(a, cut_hi), b)) if part[1] > part[0]]
    ends = np.cumsum([b - a for a, b in pieces], dtype=np.int64)
    if not len(ends) or ends[-1] == 0:
        return None
    k = int(rng.integers(ends[-1]))
    index = int(np.searchsorted(ends, k, side="right"))
    return pieces[index][0] + k - (int(ends[index - 1]) if index else 0)


def separated_starts(
        rng: np.random.Generator,
        bounds: Sequence[Tuple[int, int]],
        origins_samples: Sequence[float],
        separation_samples: float,
        num_examples: int,
    ) -> np.ndarray:
    """
    Window start indices for up to ``num_examples`` examples, one per detector, no two
    of an example's detectors closer than ``separation_samples`` in absolute time.

    Detector i's start is uniform on ``[lo_i, hi_i)`` (``bounds``) given the earlier
    detectors' starts, over the part of it at least ``separation_samples`` from each
    of them: it is redrawn while it conflicts, up to ``SEPARATION_DRAWS`` times, then
    drawn from that part directly. An example whose detector has no such start is
    dropped, never emitted. ``origins_samples[i]`` is detector i's first sample on a
    common clock, in samples.

    Returns an (M, detectors) int64 array, M <= num_examples.
    """
    origins = np.asarray(origins_samples, dtype=np.float64)
    starts = np.empty((num_examples, len(bounds)), dtype=np.int64)
    alive = np.ones(num_examples, dtype=bool)
    for i, (lo, hi) in enumerate(bounds):
        if hi <= lo:
            return np.empty((0, len(bounds)), dtype=np.int64)
        starts[:, i] = rng.integers(lo, hi, size=num_examples)

        def conflicts():
            gaps = np.abs((starts[:, i] + origins[i])[:, None] - (starts[:, :i] + origins[:i]))
            return alive & np.any(gaps < separation_samples, axis=1)

        for _ in range(SEPARATION_DRAWS):
            conflict = conflicts()
            if not conflict.any():
                break
            starts[conflict, i] = rng.integers(lo, hi, size=int(conflict.sum()))
        for example in np.flatnonzero(conflicts()):
            taken = starts[example, :i] + origins[:i] - origins[i]
            start = _allowed_draw(rng, lo, hi, taken, separation_samples)
            if start is None:
                alive[example] = False
            else:
                starts[example, i] = start
    return starts[alive]


def grid_slots(bounds: Tuple[int, int], stride: int) -> int:
    """The number of GRID slots, ``stride`` apart from ``lo``, among the window starts [lo, hi)."""
    lo, hi = bounds
    return max(-(-(hi - lo) // stride), 0)


def grid_starts(
        bounds: Sequence[Tuple[int, int]],
        origins_samples: Sequence[float],
        stride: int,
        separation_samples: float,
        used: Sequence[np.ndarray],
    ) -> np.ndarray:
    """
    GRID window starts, one per detector, for every example a row can still serve.

    Detector i's starts are its slots ``lo_i + stride * s`` (``grid_slots``), each
    served at most once: ``used[i]`` marks the slots already served, and is updated
    in place. Detector 0's free slots are taken in order; for each, detector k takes
    the first free slot, scanning cyclically from half its slots on, that lies at
    least ``separation_samples`` from the example's other starts in absolute time
    (``origins_samples``, each detector's first sample on a common clock). A slot of
    detector 0 with no such partner is left free. Nothing is drawn: the pass is a
    pure function of the row and what was served before it.

    Returns an (M, detectors) int64 array of start indices.
    """
    times = [origin + lo + stride * np.arange(len(free), dtype=np.float64)
             for origin, (lo, _), free in zip(np.asarray(origins_samples, np.float64), bounds, used)]
    starts = []
    for slot in np.flatnonzero(~used[0]):
        chosen, picks = [times[0][slot]], [slot]
        for k in range(1, len(bounds)):
            n = len(used[k])
            if n == 0:
                break
            order = (slot + -(-n // 2) + np.arange(n)) % n
            fits = ~used[k][order]
            for t in chosen:
                fits &= np.abs(times[k][order] - t) >= separation_samples
            if not fits.any():
                break
            pick = int(order[np.argmax(fits)])
            chosen.append(times[k][pick])
            picks.append(pick)
        else:
            for k, pick in enumerate(picks):
                used[k][pick] = True
            starts.append([lo + stride * pick for (lo, _), pick in zip(bounds, picks)])
    return np.asarray(starts, dtype=np.int64).reshape(-1, len(bounds))


def gather_windows(tensor_data, starts, num_onsource_samples: int, num_offsource_samples: int):
    """The onsource ``[s, s + on)`` and offsource ``[s - off, s)`` of ``tensor_data`` at each start s."""
    starts = np.asarray(starts, dtype=np.int64)[:, None]
    onsource = ops.take(tensor_data, starts + np.arange(num_onsource_samples)[None, :], axis=0)
    offsource = ops.take(tensor_data, starts - num_offsource_samples + np.arange(num_offsource_samples)[None, :],
                         axis=0)
    return onsource, offsource


def concatenate_batches(subarrays, background_chunks, start_times):
    """
    Concatenates batches of subarrays, background chunks, and start times.
    """
    stacked_subarrays = ops.concatenate(subarrays, axis=1)
    stacked_background_chunks = ops.concatenate(background_chunks, axis=1)
    stacked_start_times = np.concatenate([as_gps_times(times) for times in start_times], axis=1)

    return stacked_subarrays, stacked_background_chunks, stacked_start_times


def random_subsection(
        data,
        start_gps_time,
        time_interval_seconds,
        num_onsource_samples: int,
        num_offsource_samples: int,
        num_examples_per_batch: int,
        seed: int,
        min_separation_seconds: float = 0.0,
    ):
    """
    ``num_examples_per_batch`` random windows from each detector's piece (``data[i]``,
    its first sample at GPS ``start_gps_time[i]``), no two detectors' window starts
    closer than ``min_separation_seconds`` (``separated_starts``).

    Returns (onsource, offsource, start GPS times): (B, I, on), (B, I, off) and (B, I)
    float64, the GPS time of each padded onsource's first sample; or three Nones
    when the pieces cannot hold a full batch of separated windows.
    """
    rng = default_rng(seed)
    interval = float(time_interval_seconds)
    bounds = [window_start_bounds(int(ops.shape(tensor)[0]), num_onsource_samples, num_offsource_samples)
              for tensor in data]
    first_gps = as_gps_times(np.asarray(start_gps_time, dtype=np.float64))
    origins = (first_gps - first_gps[0]) / interval
    starts = np.empty((0, len(data)), dtype=np.int64)
    for _ in range(4):
        needed = num_examples_per_batch - len(starts)
        if needed == 0:
            break
        starts = np.concatenate([starts, separated_starts(
            rng, bounds, origins, min_separation_seconds / interval, needed)])
    if len(starts) < num_examples_per_batch:
        return None, None, None

    onsource, offsource = zip(*(gather_windows(tensor, starts[:, i], num_onsource_samples, num_offsource_samples)
                                for i, tensor in enumerate(data)))
    gps_times = first_gps[None, :] + window_offsets_seconds(starts, interval)
    return (ops.stack(onsource, axis=1), ops.stack(offsource, axis=1), gps_times)


# =============================================================================
# IFO DATA CONTAINER
# =============================================================================

@dataclass
class IFOData:
    """
    One row of detector pieces: ``data[i]`` at ``sample_rate_hertz``, its first sample
    at GPS ``start_gps_time[i]``.
    """
    data: Union[List["TimeSeries"], Any, np.ndarray]
    sample_rate_hertz: float
    start_gps_time: List[float]

    def __post_init__(self):
        # Handle different types of input data for uniformity
        if isinstance(self.data, list):
            self.data = [gf.replace_nan_and_inf_with_zero(data)
                         for data in self.data]
        elif isinstance(self.data, np.ndarray):
            self.data = [ops.convert_to_tensor(self.data, dtype="float32")]

        # Calculate the duration in seconds for each data item
        self.duration_seconds = [
            ops.cast(ops.shape(ifo_data)[0], "float32") / self.sample_rate_hertz
            for ifo_data in self.data
        ]

        # The time between samples, in float64: GPS arithmetic never runs in float32.
        self.time_interval_seconds = 1.0 / float(self.sample_rate_hertz)

    def numpy(self):
        # Converts the data to a numpy array, handling different data types
        return [ops.convert_to_numpy(data) for data in self.data]

    def random_subsection(
        self,
        num_onsource_samples: int,
        num_offsource_samples: int,
        num_examples_per_batch: int,
        seed: int,
        min_separation_seconds: float = 0.0,
    ):
        """Random windows of this row's pieces (``random_subsection``)."""
        return random_subsection(
            self.data,
            self.start_gps_time,
            self.time_interval_seconds,
            num_onsource_samples,
            num_offsource_samples,
            num_examples_per_batch,
            seed,
            min_separation_seconds,
        )
    
    def grid_windows(
        self,
        num_onsource_samples: int,
        num_offsource_samples: int,
        min_separation_seconds: float,
        used: Sequence[Optional[np.ndarray]],
    ):
        """
        Every GRID window of this row's pieces not yet served this pass (``grid_starts``),
        one padded onsource apart.

        ``used[i]`` marks detector i's slots already served this pass, or is None for a
        piece not walked before. Returns (onsource, offsource, start GPS times) as
        ``random_subsection`` does, with as many examples as the row has separated
        slots left (perhaps none), and the updated marks.
        """
        first_gps = as_gps_times(np.asarray(self.start_gps_time, dtype=np.float64))
        interval = self.time_interval_seconds
        bounds = [window_start_bounds(int(ops.shape(tensor)[0]), num_onsource_samples, num_offsource_samples)
                  for tensor in self.data]
        used = [np.zeros(grid_slots(b, num_onsource_samples), dtype=bool) if marks is None else marks
                for b, marks in zip(bounds, used)]
        starts = grid_starts(
            bounds,
            (first_gps - first_gps[0]) / interval,
            num_onsource_samples,
            min_separation_seconds / interval,
            used,
        )
        onsource, offsource = zip(*(gather_windows(tensor, starts[:, i], num_onsource_samples, num_offsource_samples)
                                    for i, tensor in enumerate(self.data)))
        gps_times = first_gps[None, :] + window_offsets_seconds(starts, interval)
        return ops.stack(onsource, axis=1), ops.stack(offsource, axis=1), gps_times, used


# =============================================================================
# BASE DATA OBTAINER (Abstract Base Class)
# =============================================================================

@dataclass(init=False)
class BaseDataObtainer(ABC):
    """
    Abstract base class for IFO data obtainers.
    
    Provides shared functionality for segment acquisition, caching, and processing.
    Subclasses implement mode-specific logic for NOISE vs TRANSIENT acquisition.
    """

    _segment_cache_maxsize = 8  # In-memory LRU cache of acquired segments.

    def __init__(
            self, 
            data_quality : DataQuality,
            data_labels : Union[DataLabel, List[DataLabel]],
            observing_runs : Union[ObservingRun, List[ObservingRun]] = None,
            segment_order : SegmentOrder = SegmentOrder.RANDOM,
            max_segment_duration_seconds : float = 2048.0,
            saturation : float = 8.0,
            force_acquisition : bool = False,
            cache_segments : bool = True,
            overrides : dict = None,
            logging_level : int = logging.WARNING,
            augmentations: List = None,  # Training only; None is none
            prefetch_segments : int = 16,
            gwosc_flags : Optional[Tuple[str, ...]] = None,
            split_spec : Optional[SplitSpec] = None,
            source : Optional[DataSource] = None,
        ):
        """
        Args (selection):
            augmentations: Augmentations a dataset applies to its training
                split's raw strain, per example (``gf.SignReversal``,
                ``gf.TimeReversal``, ``gf.RandomShift``, ``gf.AddNoise``); none by
                default. Evaluation data, and the obtainer's own batches, are
                never augmented. Real events refuse TimeReversal and
                RandomShift, and noise refuses RandomShift (see
                ``acquisition.augmentations``).
            source: Where real strain and its segments come from (``gf.DataSource``):
                ``gf.GWOSC()``, or ``gf.LocalFrames(...)`` by name. None takes
                ``gf.Defaults.source`` (``gf.GWOSC()``) as the obtainer is built. It is
                part of the obtainer's specification, so of every fingerprint and
                cache key, and nothing falls back from it to another source.
            split_spec: How GPS time is split into train, validate and test
                (``gf.SplitSpec``); None is the default split.
            gwosc_flags: GWOSC timeline flags whose intersection defines usable
                times, e.g. ``gf.GWOSC_DATA_FLAGS`` or ``gf.GWOSC_CBC_FLAGS``. Each
                flag lists the times that pass it. None uses each run's default:
                ``DATA`` with ``BURST_CAT1`` and ``BURST_CAT2`` applied, and no CBC,
                burst, stochastic or detchar hardware injections
                (``gf.GWOSC_NO_HW_INJECTION_FLAGS``).
        """

        # Initiate logging for ifo_data:
        # Legacy logger setup removed in favor of module-level logger
        
        self.acquisition_mode = None

        # Default to all observing runs (O1, O2, O3) if not specified
        observing_runs = ensure_list(observing_runs) or [ObservingRun.O1, ObservingRun.O2, ObservingRun.O3]

        # Set class attributes with parameters:
        self.data_quality = data_quality
        self.data_labels = check_data_labels(data_labels)
        self.segment_order = segment_order
        self.max_segment_duration_seconds = max_segment_duration_seconds
        self.saturation = saturation
        self.force_acquisition = force_acquisition
        self.cache_segments = cache_segments
        self.split_spec = as_split_spec(split_spec)
        self.source = gf.Defaults.source if source is None else source
        if not isinstance(self.source, DataSource):
            raise TypeError(
                f"source must be a gf.DataSource, such as gf.GWOSC() or gf.LocalFrames(...); "
                f"got {self.source!r}."
            )

        from gravyflow.src.dataset.acquisition.augmentations import check_augmentations
        self.augmentations = check_augmentations(augmentations, self.data_labels)
            
        # Unpack parameters from input observing runs:
        self.unpack_observing_runs(observing_runs, data_quality)
        if gwosc_flags is not None:
            if isinstance(gwosc_flags, str):
                gwosc_flags = (gwosc_flags,)
            self.state_flags = [tuple(gwosc_flags) for _ in self.state_flags]

        # Override observing run attributes if present:
        if overrides:
            self.override_attributes(overrides)
        
        # Prefetching configuration
        self.prefetch_segments = prefetch_segments
        self._prefetch_executor = thread_pool("prefetch") if prefetch_segments > 0 else None

        self.__dict__.update(self._served_state())

    def _served_state(self) -> Dict[str, Any]:
        """
        Every attribute that serving data builds, at its starting value: the
        split's segments, cursors, caches, statistics and rng. The rest of the
        obtainer is its specification.

        Served state belongs to the split and the process that built it, so a
        copy or a pickle of an obtainer never carries it: each split reads
        from a fresh copy, and no split can see another's segments.
        """
        return {
            # The split the served state was built for, and the order it was
            # given (see _serve_split).
            "_served_split": None,
            "_served_order": None,
            # This build of the served state: a generator serves only while it
            # is unchanged (see _while_served).
            "_served_build": object(),
            "rng": None,
            "ifos": None,
            "file_path": None,
            "valid_segments": None,
            "sampling_plan": None,
            "_current_segment_index": 0,
            "_current_batch_index": 0,
            "_segment_exhausted": True,
            "_num_batches_in_current_segment": 0,
            "_segment_cache": MemoryCache(maxsize=self._segment_cache_maxsize),
            "_prefetch_futures": {},
            "_dropped_segments_count": 0,
            "_zero_filled_segments_count": 0,
        }

    #: Attributes that never change the data served, only how it is fetched: whether
    #: the disk cache is read or written, and how far ahead segments are fetched.
    _NOT_SPECIFICATION = ("force_acquisition", "cache_segments", "prefetch_segments", "_prefetch_executor")

    def specification(self) -> Dict[str, Any]:
        """
        Everything that determines the data this obtainer serves: every attribute but
        its served state (``_served_state``) and ``_NOT_SPECIFICATION``. Content
        fingerprints (``validate.provenance.canonical``) read it, so changing any
        parameter that changes the data changes the fingerprint.
        """
        excluded = set(self._served_state()) | set(self._NOT_SPECIFICATION)
        specification = {name: value for name, value in sorted(vars(self).items()) if name not in excluded}
        # The numerics that make the served samples from fetched strain.
        specification["data_versions"] = dict(DATA_VERSIONS)
        return specification

    def _split(self, ifos: List[gf.IFO], group_name: str) -> Tuple:
        """
        The split a call names, as a key: its detectors, in order, its name
        (refused unless it is a split) and the spec's fingerprint. Every call
        that serves a split, through get_valid_segments or _serve_group, names
        it by this one key.
        """
        return (tuple(ensure_list(ifos)), self.split_spec.check_name(group_name),
                self.split_spec.fingerprint())

    def _serve_split(self, split: Tuple, order: Tuple = ()) -> bool:
        """
        Make the served state ``split``'s (see _split), in ``order``: a
        hashable description of what orders it. State built for another split
        or order is dropped first, so an obtainer called for one split, or one
        set of detectors, and then another never serves the first's records as
        the second's. Returns True when the split's state is still to be built.
        """
        if (self._served_split, self._served_order) == (split, order):
            return False
        self.__dict__.update(self._served_state())
        self._served_split, self._served_order = split, order
        return True

    @property
    def _served_group(self) -> Optional[str]:
        """The name of the split served, or None before one is built."""
        return self._served_split[1] if self._served_split is not None else None

    def _serve_group(self, ifos: List[gf.IFO], seed: int, group: str) -> None:
        """
        Serve split ``group`` on ``ifos`` to a call: the split already served
        on those detectors is kept, with the order get_valid_segments gave it
        (``seed`` here orders only a split built afresh, since a call's seed
        draws its windows); any other split, or the same split on other
        detectors, is built afresh, so a call never serves another split's
        records or another detector's.
        """
        if self._split(ifos, group) != self._served_split:
            self.get_valid_segments(ifos=ifos, seed=seed, group_name=group)

    def _while_served(self, batches: Generator) -> Generator:
        """
        ``batches`` while the obtainer still serves the state they started
        from. A generator's position (its cursors, its order over the split's
        records) lives in that state, so once the obtainer is called for
        another split, resuming it would read the new split's records by the
        old split's position: it refuses instead, before touching anything.
        """
        built = None
        while True:
            if built is not None and self._served_build is not built:
                raise RuntimeError(
                    "This generator's obtainer now serves another split; call the "
                    "obtainer again for a generator of the split it serves."
                )
            try:
                batch = next(batches)
            except StopIteration:
                return
            # The first batch builds the split's state, or reads the state built.
            built = self._served_build
            yield batch

    def override_attributes(
        self,
        overrides : Dict
    ) -> None:
        for key, value in overrides.items():    
            if hasattr(self, key):
                setattr(self, key, [value])
            else:
                raise ValueError(
                    f"Invalide override value {key} not attribute of "
                    "IFODataObtainer"
                )

    def unpack_observing_runs(
        self,
        observing_runs : List[ObservingRun],
        data_quality : DataQuality
        ) -> None:
        
        observing_runs = [run.value for run in observing_runs]
                
        self.start_gps_times = [run.start_gps_time for run in observing_runs]
        self.end_gps_times = [run.end_gps_time for run in observing_runs]

        self.state_flags = \
            [run.state_flags[data_quality] for run in observing_runs]

    def __getstate__(self):
        """The specification only: served state and the executor stay behind."""
        served = self._served_state()
        return {
            name: value for name, value in self.__dict__.items()
            if name not in served and name != "_prefetch_executor"
        }

    def __setstate__(self, state):
        """A fresh obtainer of the same specification, with its own executor."""
        self.__dict__.update(state)
        self.__dict__.update(self._served_state())
        if self.prefetch_segments > 0:
            self._prefetch_executor = thread_pool("prefetch")
        else:
            self._prefetch_executor = None
        
    def __del__(self):
        if getattr(self, "_prefetch_executor", None) is not None:
            self._prefetch_executor.shutdown(wait=False)
            
    def close(self):
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=True)
            self._prefetch_executor = None
    
    def _post_process_batch(
        self,
        batch: dict,
        sample_rate_hertz: float,
        onsource_duration_seconds: float,
        whiten: bool = False,
        crop: bool = False
    ) -> dict:
        """
        Apply post-processing to a batch dictionary.
        
        Shared implementation for both NOISE and TRANSIENT modes.
        Handles whitening, cropping, and NaN cleanup.
        
        Args:
            batch: Dict containing at least ONSOURCE and OFFSOURCE tensors
            sample_rate_hertz: Sample rate for cropping/whitening
            onsource_duration_seconds: Target onsource duration for cropping
            whiten: Whether to apply whitening
            crop: Whether to crop to onsource_duration_seconds
            
        Returns:
            The batch dict, its entries replaced; the arrays it held are never written.
            Whitening is scale free (see gf.whiten), so the strain is whitened as it is.
        """
        RV = gf.ReturnVariables
        
        if batch is None:
            return None
            
        if whiten:
            batch[RV.ONSOURCE] = gf.whiten(
                batch[RV.ONSOURCE],
                batch[RV.OFFSOURCE],
                sample_rate_hertz
            )

        if crop:
            batch[RV.ONSOURCE] = gf.crop_samples(
                batch[RV.ONSOURCE], 
                onsource_duration_seconds, 
                sample_rate_hertz
            )
            
        batch[RV.ONSOURCE] = gf.replace_nan_and_inf_with_zero(batch[RV.ONSOURCE])
        batch[RV.OFFSOURCE] = gf.replace_nan_and_inf_with_zero(batch[RV.OFFSOURCE])
        
        return batch
    
    def _prefetch_segment_for_index(
        self,
        segment_index: int,
        valid_segments: np.ndarray,
        sample_rate_hertz: float,
        ifos: List
    ) -> Optional[List[Optional["SegmentStrain"]]]:
        """
        Fetch row ``segment_index``'s segments, one per detector (None where one cannot
        be had), in a background thread. Nothing is written: the main thread stores them.
        """
        if segment_index >= len(valid_segments):
            return None
        return [self._fetch_segment(ifo, start, end, sample_rate_hertz, "Prefetch")
                for ifo, (start, end) in zip(ifos, valid_segments[segment_index])]

    def generate_file_path(
        self,
        sample_rate_hertz : float,
        group : str,
        data_directory_path : Optional[Path] = None
        ) -> Path:
        
        # TRANSIENT mode: Skip segment file creation - transients go to TransientCache only
        if self.acquisition_mode == AcquisitionMode.TRANSIENT:
            self.file_path = None
            return None

        default_directory = data_directory_path is None
        if default_directory:
            # The one user cache root; never beside the installed package.
            data_directory_path = user_cache_dir("segments")

        # Everything a stored segment depends on, by name: its source (the data source's
        # specification, and the run bounds), its selection (flags, labels, the longest
        # segment and the split), its rate, the numerics that resampled it and the layout
        # it is stored in. Each stored segment also carries its own first-sample time,
        # rate and source (see _cache_segment).
        from gravyflow.src.validate.provenance import config_fingerprint
        segment_hash = config_fingerprint({
            "source": self.source,
            "run_bounds": [list(map(float, bounds))
                           for bounds in zip(self.start_gps_times, self.end_gps_times)],
            "state_flags": self.state_flags,
            "data_labels": self.data_labels,
            "max_segment_duration_seconds": self.max_segment_duration_seconds,
            "sample_rate_hertz": float(sample_rate_hertz),
            "group": group,
            "data_versions": dict(DATA_VERSIONS),
            "layout": SEGMENT_LAYOUT_VERSION,
        })[:32]
        
        # Determine prefix based on content
        if DataLabel.NOISE in self.data_labels:
            prefix = "segment_data"
        else:
            prefix = "transient_data"
            
        self.file_path = Path(data_directory_path) / f"{prefix}_{segment_hash}.hdf5"
        if default_directory:
            # Where the default put segment caches before the one cache root.
            warn_of_moved_cache(
                self.file_path,
                PACKAGE.parent / self.file_path.name,
                Path("generator_data") / self.file_path.name,
            )

        return self.file_path
    
    def get_segment_times(
        self,
        start: float,
        stop: float,
        ifo: gf.IFO,
        state_flag: Union[str, Tuple[str, ...]]
    ) -> np.ndarray:
        """
        Times in [start, stop) when ``ifo`` passes every GWOSC flag in ``state_flag``,
        from the obtainer's source (``DataSource.segments``): the public GWOSC timeline,
        where the source holds strain. Returns a sorted (N, 2) array.
        """
        return self.source.segments(ifo, state_flag, (start, stop))

    def get_all_segment_times(
        self,
        ifo : gf.IFO
    ) -> np.ndarray:
        """
        Usable times of ``ifo`` over every configured run, as one sorted, coalesced
        (N, 2) array, whatever order the runs were given in.
        """
        valid_segments = []
        for index, start_gps_time in enumerate(self.start_gps_times):
            valid_segments.append(
                np.reshape(
                    self.get_segment_times(
                        self.start_gps_times[index],
                        self.end_gps_times[index],
                        ifo,
                        self.state_flags[index]
                    ),
                    (-1, 2),
                )
            )

        # Each run yields a different number of segments: concatenate, never stack.
        # Then sort and coalesce, so that runs given out of order still give a
        # chronological list for the vetoes and the binning.
        return coalesce_segments(np.concatenate(valid_segments, axis=0))
    
    def get_event_veto_segments(self) -> np.ndarray:
        """
        Veto segments for every event GWOSC publishes: the signal's time in band from
        20 Hz plus padding (``gravyflow.get_event_veto_segments``), and the event's
        footprint, the time an example of it may read (``event_footprints``), widened
        by the split's guard. Events are test data, so no other data comes within the
        guard of them.
        """
        from gravyflow.src.dataset.features.event import get_event_catalogue, get_event_veto_segments
        from gravyflow.src.dataset.features.transient_index import event_footprints

        catalogue = get_event_catalogue()
        guard = self.split_spec.guard_seconds
        footprints = event_footprints(catalogue["gps"], catalogue["chirp_mass_source"])
        return np.concatenate([
            np.reshape(get_event_veto_segments(catalogue=catalogue), (-1, 2)),
            footprints + np.array([-guard, guard]),
        ])

    def _glitch_catalogues(self, ifo: gf.IFO, **kwargs) -> List[Dict]:
        """
        The Gravity Spy catalogue of ``ifo`` over every configured run. Runs that the
        public release does not cover (O4) give a warning and no glitches.
        """
        from gravyflow.src.dataset.features.glitch import get_glitch_catalogue

        catalogues = []
        for start, end in zip(self.start_gps_times, self.end_gps_times):
            try:
                catalogues.append(get_glitch_catalogue(
                    ifo, start_gps_time=start, end_gps_time=end, **kwargs
                ))
            except ValueError as error:
                logger.warning(f"No glitch veto for {ifo.name} over GPS {start}-{end}: {error}")
        return catalogues

    def get_glitch_veto_segments(self, ifo: gf.IFO) -> np.ndarray:
        """
        Veto segments for the Gravity Spy entries of ``ifo`` over every configured run:
        each entry's whole extent plus its class's padding on each side
        (``glitch.glitch_veto_padding``). The entries vetoed are
        ``NOISE_VETO_GLITCH_TYPES`` at ``NOISE_VETO_MIN_ML_CONFIDENCE`` or above.
        """
        from gravyflow.src.dataset.features.glitch import (
            NOISE_VETO_GLITCH_TYPES, glitch_veto_padding,
            NOISE_VETO_MIN_ML_CONFIDENCE, glitch_extents,
        )
        catalogues = self._glitch_catalogues(
            ifo,
            glitch_types=NOISE_VETO_GLITCH_TYPES,
            min_ml_confidence=NOISE_VETO_MIN_ML_CONFIDENCE,
            exclude_events=False,
            exclude_hardware_injections=False,
        )
        if not catalogues:
            return np.empty((0, 2))
        return np.concatenate(
            [glitch_extents(catalogue, glitch_veto_padding(catalogue)) for catalogue in catalogues]
        )

    def veto_time_segments(
        self,
        segments: np.ndarray,
        veto_segments: np.ndarray
    ) -> np.ndarray:
        """
        Remove every vetoed time from ``segments``.

        Both lists may be in any order and may overlap: they are sorted and coalesced
        before the subtraction. Returns a sorted, coalesced (N, 2) array, with no piece
        of zero length.
        """
        return subtract_segments(
            np.reshape(np.asarray(segments, dtype=np.float64), (-1, 2)),
            np.reshape(np.asarray(veto_segments, dtype=np.float64), (-1, 2)),
        )

    def compress_segments(self, segments: np.ndarray) -> np.ndarray:
        if segments.size == 0:
            return segments

        segments = segments[segments[:, 0].argsort()]
        compressed = segments.copy()

        overlaps = compressed[1:, 0] <= compressed[:-1, 1]
        compressed[:-1][overlaps, 1] = np.maximum(compressed[:-1][overlaps, 1], compressed[1:][overlaps, 1])
        keep_segments = np.append([True], ~overlaps)

        return compressed[keep_segments]
    
    def find_segment_intersections(self, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """
        Find intersections between two sets of segments.
        
        For each segment in arr1, finds the best overlapping segment in arr2
        and returns the intersection bounds.
        
        Args:
            arr1: Array of shape (N, 2) with [start, end] for each segment
            arr2: Array of shape (M, 2) with [start, end] for each segment
            
        Returns:
            Array of shape (K, 2) with intersection [start, end] bounds,
            where K <= N (segments with no overlap are excluded)
        """
        if len(arr1) == 0 or len(arr2) == 0:
            return np.empty((0, 2))
            
        latest_starts = np.maximum(arr1[:, None, 0], arr2[None, :, 0])
        earliest_ends = np.minimum(arr1[:, None, 1], arr2[None, :, 1])

        overlap_durations = np.clip(earliest_ends - latest_starts, 0, None)
        overlap_mask = overlap_durations > 0
        best_overlap_indices = np.argmax(overlap_durations, axis=-1)

        starts = latest_starts[np.arange(latest_starts.shape[0]), best_overlap_indices]
        ends = earliest_ends[np.arange(earliest_ends.shape[0]), best_overlap_indices]

        valid_mask = overlap_mask[np.arange(overlap_mask.shape[0]), best_overlap_indices]
        starts = starts[valid_mask]
        ends = ends[valid_mask]

        return np.vstack((starts, ends)).T

    def order_segments(
        self,
        valid_segments : np.ndarray,
        segment_order : SegmentOrder,
        seed : int
    ):
        rng = default_rng(seed)

        match segment_order:
            case SegmentOrder.RANDOM:
                rng.shuffle(valid_segments)

            case SegmentOrder.SHORTEST_FIRST:
                sort_by_duration = lambda segments: segments[
                    np.argsort(segments[:, 0, 1] - segments[:, 0, 0])
                ]
                valid_segments = sort_by_duration(valid_segments)
            case SegmentOrder.CHRONOLOGICAL:
                pass
            case _:
                raise ValueError(
                    f"""
                    Order {segment_order.name} not recognised, please choose 
                    from SegmentOrder.RANDOM, SegmentOrder.SHORTEST_FIRST, or
                    SegmentOrder.CHRONOLOGICAL
                    """
                )
        
        return valid_segments

    def cut_segments(
        self,
        segments: np.ndarray,
        chunk_size: float,
        start_time: float
    ) -> np.ndarray:
        """
        Cut segments at ``start_time + k * chunk_size`` for every integer k.

        Real-noise binning uses ``start_time = SEGMENT_GRID_ORIGIN_GPS`` (GPS 0), so
        the cut is absolute and does not depend on which runs are configured.
        Returns an (N, 2) array.
        """
        new_segments = []
        for start, end in segments:
            curr = start
            while curr < end:
                chunk_idx = int(np.floor((curr - start_time) / chunk_size))
                chunk_end = start_time + (chunk_idx + 1) * chunk_size
                
                cut_point = min(end, chunk_end)
                
                if cut_point > curr:
                    new_segments.append([curr, cut_point])
                
                curr = cut_point
                
        return np.reshape(np.array(new_segments, dtype=np.float64), (-1, 2))

    def _cache_segment(self, key: str, strain: "SegmentStrain") -> None:
        """
        Store ``strain`` under ``key`` with its provenance (first-sample GPS time, rate,
        source and layout). A key already stored holds the same data (the file's hash
        and the key fix every input to it), so it is written again only when
        ``force_acquisition`` asks for fresh data.
        """
        if self.file_path is None:
            return

        with closing(open_hdf5_file(self.file_path, logger, mode="a")) as f:
            if key in f:
                if not self.force_acquisition:
                    return
                del f[key]
            dataset = f.create_dataset(key, data=strain.data)
            dataset.attrs["t0"] = float(strain.t0)
            dataset.attrs["sample_rate_hertz"] = float(strain.sample_rate_hertz)
            dataset.attrs["source"] = str(strain.source)
            dataset.attrs["layout"] = SEGMENT_LAYOUT_VERSION

    def get_segment(
        self,
        segment_start_gps_time: float,
        segment_end_gps_time: float,
        sample_rate_hertz: float,
        ifo: gf.IFO,
        segment_key: str
    ) -> Optional["SegmentStrain"]:
        """
        The strain of one segment at ``sample_rate_hertz``, or None if it cannot be had.

        The whole segment is fetched, with no margin trimmed at either end. Only its
        tail is cut, to whole seconds (``kept_seconds``). No sample before the
        segment's start is lost, and less than one second at its end. The strain
        carries the GPS time of its first sample as the source returned it, which
        every window's GPS time is counted from, never the nominal segment start.
        """
        memory_key = f"{segment_key}_{ifo.name}_{sample_rate_hertz}"
        cached = self._segment_cache.get(memory_key)
        if cached is not None:
            return cached

        segment = self._read_cached_segment(segment_key, sample_rate_hertz)
        if segment is None:
            segment = self._download_segment(
                segment_start_gps_time, segment_end_gps_time, sample_rate_hertz, ifo
            )
        if segment is None:
            return None
        if not check_tensor_integrity(segment.data, 1, 10):
            logging.debug(f"Segment integrity check failed for {ifo.name}, will zero-fill")
            return None
        self._segment_cache.put(memory_key, segment)
        return segment

    def _read_cached_segment(self, segment_key: str, sample_rate_hertz: float) -> Optional["SegmentStrain"]:
        """
        ``segment_key`` from the segment file, or None when it is not there, fresh data
        is forced, or its stored provenance is not this request's (it is then fetched
        again).
        """
        if (self.acquisition_mode == AcquisitionMode.TRANSIENT or self.file_path is None
                or self.force_acquisition or not Path(self.file_path).exists()):
            return None
        with closing(open_hdf5_file(self.file_path, logger, mode="r")) as segment_file:
            if segment_key not in segment_file:
                return None
            dataset = segment_file[segment_key]
            attrs = dict(dataset.attrs)
            if (attrs.get("layout") != SEGMENT_LAYOUT_VERSION or "t0" not in attrs
                    or float(attrs.get("sample_rate_hertz", -1.0)) != float(sample_rate_hertz)):
                logging.warning(
                    f"Stored segment {segment_key} in {self.file_path} has provenance "
                    f"{attrs}, not layout {SEGMENT_LAYOUT_VERSION} at {sample_rate_hertz} Hz; "
                    "fetching it again."
                )
                return None
            return SegmentStrain(
                data=ops.convert_to_tensor(dataset[()], dtype="float32"),
                t0=float(attrs["t0"]),
                sample_rate_hertz=float(sample_rate_hertz),
                source=str(attrs.get("source", "")),
            )

    def _download_segment(
        self,
        segment_start_gps_time: float,
        segment_end_gps_time: float,
        sample_rate_hertz: float,
        ifo: gf.IFO,
    ) -> Optional["SegmentStrain"]:
        """
        Fetch one segment from the obtainer's source at the native rate the request
        needs (``fetch_rate``) and resample it (see get_segment), or None if it cannot
        be had.
        """
        # Refuse impossible rates loudly, before the zero-fill fallback.
        native_rate = fetch_rate(self.source, sample_rate_hertz)
        try:
            raw_segment = self.source.fetch(
                ifo, native_rate, (segment_start_gps_time, segment_end_gps_time)
            )

            # Keep every sample from the segment's start; cut only the tail, to
            # whole seconds (see get_segment).
            num_samples = len(raw_segment.data)
            whole_seconds = int(kept_seconds(num_samples / raw_segment.sample_rate_hertz))
            if whole_seconds < 1:
                logging.debug(
                    f"Segment shorter than one second ({num_samples} samples); skipped."
                )
                return None
            samples = raw_segment.data[:whole_seconds * whole_hertz(raw_segment.sample_rate_hertz)]

            # Resampling keeps the first sample's time: sample k is at t0 + k / rate.
            # The segment's own samples alone are resampled, so within the resampler's
            # margin of either end the samples are approximate (``gf.resample``).
            data, t0 = resample(samples, raw_segment.t0, raw_segment.sample_rate_hertz, sample_rate_hertz)
            return SegmentStrain(
                data=ops.convert_to_tensor(data, dtype="float32"),
                t0=t0,
                sample_rate_hertz=float(sample_rate_hertz),
                source=raw_segment.source,
            )
        except Exception as e:
            logging.debug(f"Segment acquisition failed (will zero-fill): {type(e).__name__}, {str(e)}")
            return None

    def _fetch_segment(self, ifo, start, end, sample_rate_hertz, what: str = "Segment") -> Optional["SegmentStrain"]:
        """get_segment for one detector's segment of a row, or None (logged) if it fails."""
        try:
            return self.get_segment(start, end, sample_rate_hertz, ifo, segment_key(ifo, start, end))
        except UpsamplingRefused:
            raise
        except Exception as e:
            logging.warning(f"{what} failed for {ifo.name} at {start}: {e}")
            return None

    def _row_data(
        self,
        segment_times,
        ifos: List[gf.IFO],
        strains: List[Optional["SegmentStrain"]],
        sample_rate_hertz: float,
        quiet_zero_fill: bool,
    ) -> IFOData:
        """
        One row's IFOData from its detectors' strains: each stored to the segment file
        when caching is on, and each detector's GPS time that of its first sample. A
        segment that could not be had is zeros from its nominal start, and counted.
        """
        data, first_sample_gps = [], []
        for ifo, (start, end), strain in zip(ifos, segment_times, strains):
            if strain is not None:
                if self.cache_segments:
                    self._cache_segment(segment_key(ifo, start, end), strain)
                data.append(strain.data)
                first_sample_gps.append(strain.t0)
                continue
            self._zero_filled_segments_count += 1
            if not quiet_zero_fill:
                logging.warning(
                    f"Segment missing for {ifo.name} at {start}. Filling with zeros "
                    f"(total zero-filled: {self._zero_filled_segments_count})."
                )
            num_samples = int(kept_seconds(end - start) * sample_rate_hertz)
            data.append(ops.zeros((num_samples,), dtype="float32"))
            first_sample_gps.append(float(start))
        return IFOData(data, sample_rate_hertz, first_sample_gps)

    def acquire(
        self,
        sample_rate_hertz: Optional[float] = None,
        valid_segments: Optional[np.ndarray] = None,
        ifos: List[gf.IFO] = [gf.IFO.L1],
        quiet_zero_fill: bool = False  # Suppress zero-fill warnings (for batch cache downloads)
    ) -> Generator[IFOData, None, None]:
        """
        One IFOData per row of ``valid_segments`` (default: the served split's), each
        detector's piece labelled with the GPS time of its first sample.

        A different array than the served segments is read statelessly (a one-off
        download, no prefetch); the served segments are streamed from the obtainer's
        position, the next row prefetched in the background.
        """
        if sample_rate_hertz is None:
            sample_rate_hertz = gf.Defaults.sample_rate_hertz
        # Refuse rates the source does not hold, or the resampler cannot make, before
        # any fetch and its zero-fill fallback.
        fetch_rate(self.source, sample_rate_hertz)

        if self.file_path is not None and self.cache_segments:
            gf.ensure_directory_exists(self.file_path.parent)

        if valid_segments is None:
            valid_segments = self.valid_segments

        def row(segment_times, strains):
            return self._row_data(
                segment_times, ifos, strains, sample_rate_hertz, quiet_zero_fill
            )

        if valid_segments is not self.valid_segments:
            # A one-off download (e.g. cache misses): no prefetch, no state changed.
            for segment_times in valid_segments:
                yield row(segment_times, [
                    self._fetch_segment(ifo, start, end, sample_rate_hertz)
                    for ifo, (start, end) in zip(ifos, segment_times)
                ])
            return

        # Streaming the served segments, with prefetch.
        self._prefetch_futures.clear()

        while self._current_segment_index < len(valid_segments):
            current_idx = self._current_segment_index
            self._current_segment_index += 1

            strains = None
            if current_idx in self._prefetch_futures:
                future = self._prefetch_futures.pop(current_idx)
                try:
                    strains = future.result(timeout=300)
                except Exception as e:
                    logging.warning(f"Prefetch result failed: {e}")
            segment_times = valid_segments[current_idx]
            if strains is None:
                strains = [self._fetch_segment(ifo, start, end, sample_rate_hertz)
                           for ifo, (start, end) in zip(ifos, segment_times)]

            multi_segment = row(segment_times, strains)

            if self.prefetch_segments > 0 and self._prefetch_executor is not None:
                next_idx = self._current_segment_index
                if next_idx < len(valid_segments) and next_idx not in self._prefetch_futures:
                    self._prefetch_futures[next_idx] = self._prefetch_executor.submit(
                        self._prefetch_segment_for_index,
                        next_idx,
                        valid_segments,
                        sample_rate_hertz,
                        ifos
                    )

            yield multi_segment

        self._current_segment_index = 0
        self._prefetch_futures.clear()

    def get_acquisition_stats(self) -> Dict[str, int]:
        """
        Get statistics about segment acquisition issues.
        
        Returns:
            Dict with 'dropped_segments' and 'zero_filled_segments' counts.
        """
        return {
            'dropped_segments': self._dropped_segments_count,
            'zero_filled_segments': self._zero_filled_segments_count,
        }
    
    def reset_acquisition_stats(self) -> None:
        """Reset acquisition statistics counters."""
        self._dropped_segments_count = 0
        self._zero_filled_segments_count = 0

    def clear_valid_segments(self) -> None:
        """Drop every split's served state (see _served_state)."""
        self.__dict__.update(self._served_state())

    @abstractmethod
    def get_valid_segments(
            self,
            ifos: List[gf.IFO],
            seed: int,
            group_name: str,
            segment_order: SegmentOrder = None,
        ) -> List:
        """
        The segments of split ``group_name`` (under ``self.split_spec``) on
        ``ifos``, ordered by ``seed``. Must be implemented by subclasses.
        """
        pass

    def __call__(
            self,
            sample_rate_hertz: float = None,
            onsource_duration_seconds: float = None,
            crop_duration_seconds: float = None,
            offsource_duration_seconds: float = None,
            num_examples_per_batch: int = None,
            ifos: List[gf.IFO] = None,
            scale_factor: None = None,
            seed: int = None,
            sampling_mode: SamplingMode = None,
            whiten: bool = False,
            crop: bool = False,
            group: str = None
        ) -> Generator[Dict, None, None]:
        """
        Unified entry point for data acquisition.
        
        Acts as a wrapper for get_onsource_offsource_chunks.
        """
        return self.get_onsource_offsource_chunks(
            sample_rate_hertz=sample_rate_hertz,
            onsource_duration_seconds=onsource_duration_seconds,
            crop_duration_seconds=crop_duration_seconds,
            offsource_duration_seconds=offsource_duration_seconds,
            num_examples_per_batch=num_examples_per_batch,
            ifos=ifos,
            scale_factor=scale_factor,
            seed=seed,
            sampling_mode=sampling_mode,
            whiten=whiten,
            crop=crop,
            group=group
        )

    def get_onsource_offsource_chunks(
            self,
            sample_rate_hertz: float = None,
            onsource_duration_seconds: float = None,
            crop_duration_seconds: float = None,
            offsource_duration_seconds: float = None,
            num_examples_per_batch: int = None,
            ifos: List[gf.IFO] = None,
            scale_factor: None = None,
            seed: int = None,
            sampling_mode: SamplingMode = SamplingMode.RANDOM,
            whiten: bool = False,
            crop: bool = False,
            group: str = None
        ) -> Generator[Dict, None, None]:
        """
        Batches of onsource and offsource strain from split ``group``, each
        checked against the shape contracts. The generator serves only while
        the obtainer serves its split (see _while_served).

        Args:
            sample_rate_hertz: Sample rate in Hz
            onsource_duration_seconds: Final onsource window duration
            crop_duration_seconds: Cropping duration on each side (for edge effects)
            offsource_duration_seconds: Background window duration
            num_examples_per_batch: Batch size
            ifos: List of interferometers
            scale_factor: Deprecated and ignored: the strain is served as it is.
            seed: Random seed
            sampling_mode: RANDOM or GRID sampling
            whiten: Whether to whiten each batch
            crop: Whether to crop each batch to its onsource duration
            group: The split served; None serves the split already served
        """
        ignore_scale_factor(scale_factor)
        batches = self._yield_onsource_offsource_chunks(
            sample_rate_hertz,
            onsource_duration_seconds,
            crop_duration_seconds,
            offsource_duration_seconds,
            num_examples_per_batch,
            ifos,
            seed,
            sampling_mode,
            whiten,
            crop,
            group
        )
        num_ifos = len(ensure_list(ifos) or [gf.IFO.L1])
        return ShapeEnforcer.wrap_generator(self._while_served(batches), num_ifos=num_ifos)

    @abstractmethod
    def _yield_onsource_offsource_chunks(
            self,
            sample_rate_hertz: float,
            onsource_duration_seconds: float,
            crop_duration_seconds: float,
            offsource_duration_seconds: float,
            num_examples_per_batch: int,
            ifos: List[gf.IFO],
            seed: int,
            sampling_mode: SamplingMode,
            whiten: bool,
            crop: bool,
            group: str
        ) -> Generator[Dict, None, None]:
        """The unchecked batches of get_onsource_offsource_chunks, for this acquisition mode."""
