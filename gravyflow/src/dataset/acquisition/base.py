# Standard library imports:
import logging
import sys

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, List, Tuple, Union, Dict, Any, Optional, Generator, Sequence, FrozenSet
from pathlib import Path
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
from gravyflow.src.utils.numerics import ensure_even, ensure_list
from gravyflow.src.utils.gps import as_gps_times
from gravyflow.src.utils.shapes import ShapeEnforcer
from gravyflow.src.dataset.store import read as store_read
from .gwosc_data import (
    GWOSC_RUNS, DEFAULT_GWOSC_FLAGS, coalesce_segments, subtract_segments,
)
from .sources import DataSource
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


def _exclusions(entries) -> Tuple[Tuple[str, Tuple[float, float], str], ...]:
    """
    ``exclusions`` as the obtainer keeps them: ``(detector name, (start, end), reason)``
    entries, each checked and in one order, so equal exclusions fingerprint equally.
    """
    checked = []
    for entry in entries:
        try:
            detector, (start, end), reason = entry
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            raise ValueError(
                "An exclusion is (detector, (start, end), reason), such as "
                f"('H1', (1238166018.0, 1238170114.0), 'outage'); got {entry!r}."
            ) from None
        name = getattr(detector, "name", detector)
        if not (isinstance(name, str) and isinstance(reason, str) and reason
                and np.isfinite(start) and np.isfinite(end) and end > start):
            raise ValueError(
                "An exclusion names a detector, a GPS interval with end > start and a "
                f"reason; got {entry!r}."
            )
        checked.append((name, (start, end), reason))
    return tuple(sorted(checked))


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


class Coincidence(Enum):
    """
    Whether an example's detectors are read at one time.

    - ``SEPARATED`` (the default): never, but for a real event seen in several
      detectors, the stated exception. Noise windows start at least two footprints
      apart, and a glitch's other detectors are read at partner times.
    - ``COINCIDENT``: the opt-in. Noise is drawn from the time every detector holds,
      each detector's window starting at one GPS time, and a transient is read on
      every detector at its own time.

    Every real-noise batch says which examples are coincident
    (``ReturnVariables.COINCIDENT``).
    """
    SEPARATED = auto()
    COINCIDENT = auto()


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


def _common_clock(origins_samples: Sequence[float]) -> np.ndarray:
    """
    Each detector's first sample on the common clock, in whole samples: coincident
    pieces share one sample grid, so their detectors can start at one time.
    """
    origins = np.asarray(origins_samples, dtype=np.float64)
    shift = np.rint(origins).astype(np.int64)
    if not np.allclose(origins, shift, rtol=0.0, atol=1e-6):
        raise ValueError(f"Coincident pieces must share one sample grid; their origins are {origins} samples.")
    return shift


def coincident_starts(
        rng: np.random.Generator,
        bounds: Sequence[Tuple[int, int]],
        origins_samples: Sequence[float],
        num_examples: int,
    ) -> np.ndarray:
    """
    Window start indices for ``num_examples`` examples whose detectors all start at one
    time (the ``COINCIDENT`` opt-in): uniform over the times every detector's
    ``[lo_i, hi_i)`` allows on the common clock (``origins_samples``).

    Returns an (M, detectors) int64 array: M is ``num_examples``, or 0 when the
    pieces share no start.
    """
    shift = _common_clock(origins_samples)
    lo = max(lo_i + s for (lo_i, _), s in zip(bounds, shift))
    hi = min(hi_i + s for (_, hi_i), s in zip(bounds, shift))
    if hi <= lo:
        return np.empty((0, len(bounds)), dtype=np.int64)
    return rng.integers(lo, hi, size=num_examples)[:, None] - shift[None, :]


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
        coincident: bool = False,
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

    ``coincident`` (the ``COINCIDENT`` opt-in): every detector takes the slot at
    detector 0's time instead, so a row's pieces must start at one sample; each free
    slot of them all is one example.

    Returns an (M, detectors) int64 array of start indices.
    """
    if coincident:
        shift = _common_clock(origins_samples)
        if len({lo + s for (lo, _), s in zip(bounds, shift)}) != 1:
            raise ValueError("A coincident GRID row's windows must start on one clock: its pieces start apart.")
        n = min(len(free) for free in used)
        slots = np.flatnonzero(~np.any([free[:n] for free in used], axis=0))
        for free in used:
            free[slots] = True
        return np.stack([lo + stride * slots for lo, _ in bounds], axis=1).astype(np.int64).reshape(-1, len(bounds))
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
        coincident: bool = False,
    ):
    """
    ``num_examples_per_batch`` random windows from each detector's piece (``data[i]``,
    its first sample at GPS ``start_gps_time[i]``), no two detectors' window starts
    closer than ``min_separation_seconds`` (``separated_starts``); or, ``coincident``,
    every detector's starting at one time (``coincident_starts``).

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
    if coincident:
        starts = coincident_starts(rng, bounds, origins, num_examples_per_batch)
    else:
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
        # Strain read from the store is finite by construction: a read that meets samples
        # its tile does not hold is refused there. A non-finite sample here is a defect,
        # refused by name, never zeroed (GF:15 C1).
        pieces = self.data if isinstance(self.data, list) else [self.data]
        for index, piece in enumerate(pieces):
            if not np.all(np.isfinite(ops.convert_to_numpy(piece))):
                raise ValueError(
                    f"Piece {index} of a row holds non-finite strain; held strain is finite, "
                    "so the row is refused."
                )
        self.data = [ops.convert_to_tensor(piece, dtype="float32") for piece in pieces]

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
        coincident: bool = False,
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
            coincident,
        )
    
    def grid_windows(
        self,
        num_onsource_samples: int,
        num_offsource_samples: int,
        min_separation_seconds: float,
        used: Sequence[Optional[np.ndarray]],
        coincident: bool = False,
    ):
        """
        Every GRID window of this row's pieces not yet served this pass (``grid_starts``,
        ``coincident`` or not), one padded onsource apart.

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
            coincident,
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

    def __init__(
            self, 
            data_quality : DataQuality,
            data_labels : Union[DataLabel, List[DataLabel]],
            observing_runs : Union[ObservingRun, List[ObservingRun]] = None,
            segment_order : SegmentOrder = SegmentOrder.RANDOM,
            max_segment_duration_seconds : float = 2048.0,
            saturation : float = 8.0,
            force_acquisition : bool = False,
            overrides : dict = None,
            logging_level : int = logging.WARNING,
            augmentations: List = None,  # Training only; None is none
            gwosc_flags : Optional[Tuple[str, ...]] = None,
            split_spec : Optional[SplitSpec] = None,
            source : Optional[DataSource] = None,
            coincidence : Coincidence = Coincidence.SEPARATED,
            exclusions : Sequence[Tuple[Any, Tuple[float, float], str]] = (),
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
            exclusions: Times the source's timeline says hold data, but which are not to
                be read: ``(detector, (start, end), reason)`` entries, such as the one a
                ``TimelineContradiction`` names. Each is taken from that detector's
                readable time (``get_all_segment_times``) before anything is planned.
                They are part of the specification, so of every fingerprint.
            force_acquisition: The transient cache's: fetch transients afresh. Real
                noise is read through the store, which is never bypassed.
            source: Where real strain and its segments come from (``gf.DataSource``):
                ``gf.GWOSC()``, or ``gf.LocalFrames(...)`` by name. None takes
                ``gf.Defaults.source`` (``gf.GWOSC()``) as the obtainer is built. It is
                part of the obtainer's specification, so of every fingerprint and
                cache key, and nothing falls back from it to another source.
            split_spec: How GPS time is split into train, validate and test
                (``gf.SplitSpec``); None is the default split.
            coincidence: Whether an example's detectors are read at one time
                (``gf.Coincidence``): never by default, or under the opt-in
                ``Coincidence.COINCIDENT``.
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
        self.exclusions = _exclusions(exclusions)
        self.split_spec = as_split_spec(split_spec)
        self.source = gf.Defaults.source if source is None else source
        if not isinstance(self.source, DataSource):
            raise TypeError(
                f"source must be a gf.DataSource, such as gf.GWOSC() or gf.LocalFrames(...); "
                f"got {self.source!r}."
            )

        self.coincidence = Coincidence(coincidence)

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
            "valid_segments": None,
            "sampling_plan": None,
            "_current_batch_index": 0,
            "_segment_exhausted": True,
            "_num_batches_in_current_segment": 0,
        }

    #: Attributes that never change the data served, only how it is fetched: whether
    #: the transient cache is read.
    _NOT_SPECIFICATION = ("force_acquisition",)

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
        """The specification only: served state stays behind."""
        served = self._served_state()
        return {name: value for name, value in self.__dict__.items() if name not in served}

    def __setstate__(self, state):
        """A fresh obtainer of the same specification."""
        self.__dict__.update(state)
        self.__dict__.update(self._served_state())
    
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
            
        # Held strain is finite, so a non-finite sample here was made by the processing
        # above: a defect, refused by name, never zeroed (GF:15 C1).
        for name in (RV.ONSOURCE, RV.OFFSOURCE):
            if not np.all(np.isfinite(ops.convert_to_numpy(batch[name]))):
                raise ValueError(
                    f"The batch's {name.name} holds non-finite samples after "
                    f"{'whitening' if whiten else 'post-processing'}; it is refused."
                )

        return batch
    
    
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
        The readable times of ``ifo``: every configured run's times that pass its
        flags, less the obtainer's ``exclusions`` for ``ifo``, as one sorted, coalesced
        (N, 2) array, whatever order the runs were given in. Strain is read only inside
        them, and a read that meets samples the source does not hold is refused by name.
        Noise draws from these times less the vetoes (``remove_unwanted_segments``).
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
        readable = coalesce_segments(np.concatenate(valid_segments, axis=0))
        excluded = [interval for name, interval, _ in self.exclusions if name == ifo.name]
        return subtract_segments(readable, np.array(excluded)) if excluded else readable
    
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

    def _row_data(self, segment_times, ifos: List[gf.IFO], sample_rate_hertz: float) -> IFOData:
        """
        One row's IFOData: each detector's piece read through the store
        (``store.read``), as its interior: the whole seconds whose read lies inside the
        piece, each detector's GPS time that of its first sample. A piece the source does
        not deliver is refused by name (``TimelineContradiction``), never zero-filled.
        """
        firsts, counts = store_read.interiors(self.source, sample_rate_hertz, segment_times)
        # Every detector's read, refused before any fetch if it could never fit the cap.
        store_read.check(self.source, sample_rate_hertz, zip(firsts, counts))
        data, first_sample_gps = [], []
        for ifo, (start, end), first, count in zip(ifos, segment_times, firsts, counts):
            if count < 1:
                raise ValueError(
                    f"The {ifo.name} piece GPS [{start}, {end}) holds no whole second at "
                    f"{sample_rate_hertz:g} Hz inside the resampler's margin."
                )
            strain = store_read.read(self.source, ifo, sample_rate_hertz, first, count,
                                     stream=self._served_split)
            data.append(strain.data)
            first_sample_gps.append(strain.t0)
        return IFOData(data, sample_rate_hertz, first_sample_gps)

    def acquire(
        self,
        sample_rate_hertz: Optional[float] = None,
        valid_segments: Optional[np.ndarray] = None,
        ifos: List[gf.IFO] = [gf.IFO.L1],
    ) -> Generator[IFOData, None, None]:
        """
        One IFOData per row of ``valid_segments`` (default: the served split's), each
        detector's piece read through the store and labelled with the GPS time of its
        first sample (``_row_data``). Reads hold no state between rows.
        """
        if sample_rate_hertz is None:
            sample_rate_hertz = gf.Defaults.sample_rate_hertz
        # Refuse a rate the source cannot make, or a fetch larger than the memory cap,
        # before any fetch.
        store_read.check(self.source, sample_rate_hertz)
        if valid_segments is None:
            valid_segments = self.valid_segments
        for segment_times in valid_segments:
            yield self._row_data(segment_times, ifos, sample_rate_hertz)

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
