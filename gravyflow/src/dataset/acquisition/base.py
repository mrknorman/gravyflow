# Standard library imports:
import logging
import sys

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum, auto
from contextlib import closing
from typing import List, Tuple, Union, Dict, Any, Optional, Generator, NamedTuple, Sequence
from pathlib import Path
from .cache import MemoryCache
from concurrent.futures import ThreadPoolExecutor, Future
from abc import ABC, abstractmethod

# Suppress verbose logging from external libraries (gwpy, gwosc)
logging.getLogger('gwpy').setLevel(logging.WARNING)
logging.getLogger('gwosc').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Third-party imports:
import numpy as np
from numpy.random import default_rng  
import keras
from keras import ops
import jax
import jax.numpy as jnp

from gwdatafind import find_urls
from gwpy.timeseries import TimeSeries

# Local imports:
import gravyflow as gf
from gravyflow.src.utils.tensor import resample_fft
from gravyflow.src.utils.numerics import ensure_even, ensure_list
from gravyflow.src.utils.gps import as_gps_times
from gravyflow.src.utils.shapes import ShapeEnforcer
from .gwosc_data import (
    GWOSC_RUNS, DEFAULT_GWOSC_FLAGS, UpsamplingRefused, coalesce_segments,
    get_gwosc_segments, gwosc_sample_rate_hertz, strain_download_cache, subtract_segments,
    user_cache_dir, warn_of_moved_cache,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Imported from TransientDefaults for consistency (canonical source)
from gravyflow.src.dataset.config import TransientDefaults
from gravyflow.src.dataset.splits import split_fractions

# GPS time precision is 10ms (0.01s) - defined in gravyflow.src.utils.gps
# GPS_PRECISION_DECIMALS was 3 (1ms) which was stale - now using GPS_KEY_PRECISION
GPS_PRECISION_DECIMALS = 2  # 10ms precision (matches gps.py)

# Tolerance for GPS time matching in seconds (used in cache lookups)
GPS_TOLERANCE_SECONDS = TransientDefaults.GPS_TOLERANCE_SECONDS

# Real-noise segments are cut on an absolute grid of this many GPS seconds:
# block k is [2048 k, 2048 (k + 1)).
SEGMENT_GRID_ORIGIN_GPS = 0.0

# Version of the stored real-noise segment layout (vetoes, absolute grid, whole
# segments, each stored with the GPS time of its first sample, its rate and its
# source). It is part of the segment file's hash, so files written under an earlier
# layout (layout 2 stored bare samples under their nominal times) are never read back.
SEGMENT_LAYOUT_VERSION = "segment-layout-3"


class SegmentStrain(NamedTuple):
    """
    One detector's strain over one segment: its samples, the GPS time of the first
    sample as the source returned it, the rate and where the samples came from.
    """
    data: Any
    t0: float
    sample_rate_hertz: float
    source: str


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
    One observing run: its GPS bounds, channels, frame types and segment flags.

    The built-in runs take their bounds from the GWOSC run API through
    ``gwosc_runs``: O3, for example, is GWOSC's O3a plus O3b. A run defined by hand
    can give UTC datetimes instead, which are converted to GPS counting leap seconds.

    ``state_flags`` holds, per data quality, the GWOSC timeline flags whose
    intersection defines the usable times (see ``gwosc_data``).
    """
    name: str
    start_date_time: Optional[datetime]
    end_date_time: Optional[datetime]
    channels: Dict
    frame_types: Dict
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
        from gwpy.time import to_gps
        return float(to_gps(date_time))

    @staticmethod
    def _to_date_time(gps_time: float) -> datetime:
        """GPS seconds to a naive UTC datetime."""
        from gwpy.time import from_gps
        return from_gps(gps_time).replace(tzinfo=None)

# Channels and frame types name GWOSC's public frames for O1 and O2 (gwdatafind type
# "{ifo}_{frame_type}", channel "{ifo}:{channel}") and LVK's calibrated frames for O3
# and O4. Without local frames, strain comes from GWOSC through fetch_open_data.
observing_run_data : Dict = {
    "O1" : (
        "O1",
        None,
        None,
        {DataQuality.BEST: "LOSC-STRAIN"},
        {DataQuality.BEST: "LOSC_4_V1"},
        {DataQuality.BEST: DEFAULT_GWOSC_FLAGS},
        ("O1",),
    ),
    "O2" : (
        "O2",
        None,
        None,
        {DataQuality.BEST: "GWOSC-4KHZ_R1_STRAIN"},
        {DataQuality.BEST: "GWOSC_O2_4KHZ_R1"},
        {DataQuality.BEST: DEFAULT_GWOSC_FLAGS},
        ("O2",),
    ),
    "O3" : (
        "O3",
        None,
        None,
        {DataQuality.BEST: "DCS-CALIB_STRAIN_CLEAN_C01"},
        {DataQuality.BEST: "HOFT_C01"},
        {DataQuality.BEST: DEFAULT_GWOSC_FLAGS},
        ("O3a", "O3b"),
    ),
    "O4" : (
        "O4",
        None,
        None,
        {DataQuality.BEST: "DCS-CALIB_STRAIN_CLEAN_C01"},
        {DataQuality.BEST: "HOFT_C01"},
        {DataQuality.BEST: DEFAULT_GWOSC_FLAGS},
        ("O4a", "O4b"),
    )
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


def _random_subsection(
        tensor_data,
        num_examples_per_batch: int,
        num_onsource_samples: int,
        num_offsource_samples: int,
        time_interval_seconds,
        start_gps_times : float,
        seed
    ):
    """
    Generate random subsections from a tensor.

    Returns the onsource and offsource subsections and the GPS time of each
    (padded) onsource subsection's first sample, as NumPy float64.
    """
    batch_subarrays, batch_background_chunks, random_starts = _random_subsection_jit(
        tensor_data,
        num_examples_per_batch,
        num_onsource_samples,
        num_offsource_samples,
        seed
    )

    # The start times are computed outside JAX, in float64.
    start_times = as_gps_times(start_gps_times) + window_offsets_seconds(random_starts, time_interval_seconds)

    return batch_subarrays, batch_background_chunks, start_times


@jax.jit(static_argnames=["num_examples_per_batch", "num_onsource_samples", "num_offsource_samples"])
def _random_subsection_jit(
        tensor_data,
        num_examples_per_batch: int,
        num_onsource_samples: int,
        num_offsource_samples: int,
        seed
    ):
    """
    Random subsections of a tensor and their start indices. JIT compiled for efficiency.
    """
    # Determine the size of the input tensor.
    num_samples = ops.shape(tensor_data)[0]

    # The range of starts whose windows fit (see window_start_bounds).
    minval, maxval = window_start_bounds(num_samples, num_onsource_samples, num_offsource_samples)

    # Generate uniformly distributed random start indices for each batch.
    # Use JAX random
    key = jax.random.PRNGKey(seed)
    random_starts = jax.random.randint(
        key,
        shape=(num_examples_per_batch,),
        minval=minval,
        maxval=maxval
    )
    random_starts = ops.cast(random_starts, "int32")

    # Create a tensor representing a sequence from 0 to num_onsource_samples - 1.
    range_tensor = ops.arange(num_onsource_samples)

    # Reshape range_tensor for broadcasting.
    range_tensor = range_tensor[None, :]

    # Reshape random_starts for broadcasting.
    random_starts_column = random_starts[:, None]

    # Calculate indices for onsource subarrays.
    indices_for_subarrays = random_starts_column + range_tensor

    # Gather subsections of onsource data using the calculated indices.
    batch_subarrays = ops.take(tensor_data, indices_for_subarrays, axis=0)

    # Create and reshape a tensor for offsource range.
    range_tensor_offsource = ops.arange(num_offsource_samples)
    range_tensor_offsource = range_tensor_offsource[None, :]

    # Calculate and reshape starting indices for offsource data.
    background_chunk_starts = random_starts - num_offsource_samples
    background_chunk_starts_column = background_chunk_starts[:, None]

    # Calculate indices for offsource background chunks.
    background_chunk_indices = background_chunk_starts_column + range_tensor_offsource

    # Gather subsections of offsource data using the calculated indices.
    batch_background_chunks = ops.take(
        tensor_data, background_chunk_indices, axis=0
    )

    return batch_subarrays, batch_background_chunks, random_starts


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
        seed : int
    ):
        all_batch_subarrays, all_batch_background_chunks, all_subsections_start_gps_time = (
            [], [], []
        )
        
        rng = default_rng(seed)

        for tensor_data, start_gps_time in zip(data, start_gps_time):
            
            s = rng.integers(1000000000)
            
            batch_subarrays, batch_background_chunks, subsections_start_gps_time = (
                _random_subsection(
                    tensor_data,
                    num_examples_per_batch,
                    num_onsource_samples,
                    num_offsource_samples,
                    time_interval_seconds,
                    start_gps_time,
                    s
                )
            )

            # Append expanded results
            all_batch_subarrays.append(
                ops.expand_dims(batch_subarrays, 1)
            )
            all_batch_background_chunks.append(
                ops.expand_dims(batch_background_chunks, 1)
            )
            all_subsections_start_gps_time.append(
                np.expand_dims(subsections_start_gps_time, 1)
            )

        # Concatenate the batches
        return concatenate_batches(
            all_batch_subarrays, 
            all_batch_background_chunks, 
            all_subsections_start_gps_time
        )
    

# =============================================================================
# IFO DATA CONTAINER
# =============================================================================

@dataclass
class IFOData:
    """
    One row of detector pieces: ``data[i]`` at ``sample_rate_hertz``, its first sample
    at GPS ``start_gps_time[i]``.
    """
    data: Union[List[TimeSeries], Any, np.ndarray]
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

    def downsample(self, new_sample_rate_hertz: Union[int, float]):
        # To implement
        return self

    def scale(self, scale_factor: Union[int, float]):
        # Scale the data by the given factor
        self.data = [data * scale_factor for data in self.data]
        return self

    def numpy(self):
        # Converts the data to a numpy array, handling different data types
        return [ops.convert_to_numpy(data) for data in self.data]

    def random_subsection(
        self, 
        num_onsource_samples: int, 
        num_offsource_samples: int, 
        num_examples_per_batch: int,
        seed : int
    ):
        # Create random number generator from seed:
        rng = default_rng(seed)

        generated_seed = int(rng.integers(1E10))

        return random_subsection(
            self.data,
            self.start_gps_time,
            self.time_interval_seconds,
            num_onsource_samples, 
            num_offsource_samples, 
            num_examples_per_batch,
            generated_seed   
        )
    
    def grid_subsection(
        self,
        num_onsource_samples: int,
        num_offsource_samples: int,
        num_examples_per_batch: int,
        cursors: Sequence[Optional[int]],
    ):
        """
        Sequential windows, one onsource apart, from each detector's piece.

        Detector i walks its piece's window starts (``window_start_bounds``) from
        ``cursors[i]``, an offset into them, wrapping at their end. A cursor of None
        starts a piece not walked before, at i / n of its starts, so that detectors
        on one stretch of time take different windows.

        Returns: (subarrays, background_chunks, gps_times, new_cursors), or Nones
        when a piece is too short for one window.
        """
        all_subarrays = []
        all_background_chunks = []
        all_gps_times = []
        new_cursors = []

        num_detectors = max(len(self.data), 1)
        steps = num_onsource_samples * np.arange(num_examples_per_batch)

        for ifo_idx, (tensor_data, start_gps_time) in enumerate(zip(self.data, self.start_gps_time)):
            lo, hi = window_start_bounds(
                int(ops.shape(tensor_data)[0]), num_onsource_samples, num_offsource_samples
            )
            span = hi - lo
            if span <= 0:
                return None, None, None, cursors
            cursor = cursors[ifo_idx]
            if cursor is None:
                cursor = ifo_idx * (span // num_detectors)
            positions = lo + (cursor + steps) % span
            new_cursors.append(int((cursor + num_examples_per_batch * num_onsource_samples) % span))

            onsource = ops.stack([tensor_data[pos:pos + num_onsource_samples] for pos in positions])
            offsource = ops.stack(
                [tensor_data[pos - num_offsource_samples:pos] for pos in positions]
            )
            gps_times = float(start_gps_time) + window_offsets_seconds(positions, self.time_interval_seconds)

            all_subarrays.append(ops.expand_dims(onsource, axis=1))
            all_background_chunks.append(ops.expand_dims(offsource, axis=1))
            all_gps_times.append(np.expand_dims(as_gps_times(gps_times), axis=1))

        stacked_subarrays = ops.concatenate(all_subarrays, axis=1)
        stacked_background = ops.concatenate(all_background_chunks, axis=1)
        stacked_gps = np.concatenate(all_gps_times, axis=1)

        return stacked_subarrays, stacked_background, stacked_gps, new_cursors


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
            augmentations: List = None,  # List of augmentation instances
            prefetch_segments : int = 16,
            gwosc_flags : Optional[Tuple[str, ...]] = None,
        ):
        """
        Args (selection):
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
        data_labels = ensure_list(data_labels)
        
        # Set class attributes with parameters:
        self.data_quality = data_quality
        self.data_labels = data_labels
        self.segment_order = segment_order
        self.max_segment_duration_seconds = max_segment_duration_seconds
        self.saturation = saturation
        self.force_acquisition = force_acquisition
        self.cache_segments = cache_segments
            
        # Data augmentation settings - list of augmentation instances
        from gravyflow.src.dataset.acquisition.augmentations import default_augmentations
        self.augmentations = augmentations if augmentations is not None else default_augmentations()
            
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
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1) if prefetch_segments > 0 else None

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
            "_grid_cursors": {},  # GRID's position in each piece it walks
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
        return {name: value for name, value in sorted(vars(self).items()) if name not in excluded}

    @staticmethod
    def _split(
            ifos: List[gf.IFO],
            group_name: str,
            groups: Optional[Dict[str, float]] = None,
        ) -> Tuple:
        """
        The split a call names, as a key: its detectors, in order, its name and
        the fractions it is drawn from. Every call that serves a split, through
        get_valid_segments or _serve_group, names it by this one key.
        """
        return (tuple(ensure_list(ifos)), group_name,
                tuple(split_fractions(groups, group_name).items()))

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
        
        self.frame_types = \
            [run.frame_types[data_quality] for run in observing_runs]
        self.channels = \
            [run.channels[data_quality] for run in observing_runs]
        self.state_flags = \
            [run.state_flags[data_quality] for run in observing_runs]
    
    def _get_frame_channel_for_gps(self, gps_time: float) -> tuple:
        """
        Get the correct frame_type and channel for a given GPS time.
        
        Returns:
            (frame_type, channel) tuple for the observing run containing this GPS.
            Falls back to first configured run if no match found.
        """
        for i, (start, end) in enumerate(zip(self.start_gps_times, self.end_gps_times)):
            if start <= gps_time <= end:
                return self.frame_types[i], self.channels[i]
        # Fall back to first if no match (shouldn't happen normally)
        return self.frame_types[0], self.channels[0]
    
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
            self._prefetch_executor = ThreadPoolExecutor(max_workers=1)
        else:
            self._prefetch_executor = None
        
    def __del__(self):
        if getattr(self, "_prefetch_executor", None) is not None:
            self._prefetch_executor.shutdown(wait=False)
            
    def close(self):
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=True)
            self._prefetch_executor = None
    
    def _apply_augmentation(self, data):
        """
        Apply augmentations to data based on configured augmentation list.
        
        Each augmentation instance carries its own probability and parameters.
        
        Args:
            data: Tensor of shape (Batch, IFOs, Samples)
            
        Returns:
            Augmented tensor with same shape
        """
        from gravyflow.src.dataset.acquisition.augmentations import (
            SignReversal, TimeReversal
        )
        
        if self.rng is None or not self.augmentations:
            return data
        
        for aug in self.augmentations:
            if self.rng.random() >= aug.probability:
                continue
                
            if isinstance(aug, SignReversal):
                data = -data
            elif isinstance(aug, TimeReversal):
                data = ops.flip(data, axis=-1)
            # Subclasses handle their specific augmentation types
            
        return data
    
    def _post_process_batch(
        self,
        batch: dict,
        sample_rate_hertz: float,
        onsource_duration_seconds: float,
        scale_factor: float = 1.0,
        whiten: bool = False,
        crop: bool = False
    ) -> dict:
        """
        Apply post-processing to a batch dictionary.
        
        Shared implementation for both NOISE and TRANSIENT modes.
        Handles whitening (with correct scale management), cropping, and NaN cleanup.
        
        Args:
            batch: Dict containing at least ONSOURCE and OFFSOURCE tensors
            sample_rate_hertz: Sample rate for cropping/whitening
            onsource_duration_seconds: Target onsource duration for cropping
            scale_factor: Current amplitude scale factor applied to data
            whiten: Whether to apply whitening
            crop: Whether to crop to onsource_duration_seconds
            
        Returns:
            The batch dict, its entries replaced; the arrays it held are never written.

        Note:
            Whitening works on copies scaled to O(1e21), for numerical stability; the
            offsource is served unscaled. Whitened onsource stays at unit variance -
            this is intentional for model training.
        """
        RV = gf.ReturnVariables
        
        if batch is None:
            return None
            
        if whiten:
            # Whiten at O(1e21) amplitude, to avoid underflow with raw strain.
            # The scaled arrays are new values, never the batch's buffers scaled
            # in place: on CPU, JAX may read a numpy array without copying it,
            # and whitens asynchronously, so a write after the call can change
            # what the whitening reads. The offsource is served as it came.
            target_scale = gf.Defaults.scale_factor
            temp_scale = target_scale / (scale_factor if scale_factor > 0 else 1.0)
            batch[RV.ONSOURCE] = gf.whiten(
                batch[RV.ONSOURCE] * temp_scale,
                batch[RV.OFFSOURCE] * temp_scale,
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
    
    def _get_effective_saturation(self):
        """
        Calculate effective saturation accounting for augmentation.
        """
        effective = self.saturation
        for aug in (self.augmentations or []):
            effective *= (1 + aug.probability)
        return effective
    
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

        # Everything a stored segment depends on, by name: its source (frame types,
        # channels and run bounds), its selection (flags, labels, the longest segment and
        # the split), its rate and the layout it is stored in. Each stored segment also
        # carries its own first-sample time, rate and source (see _cache_segment).
        from gravyflow.src.validate.provenance import config_fingerprint
        segment_hash = config_fingerprint({
            "frame_types": self.frame_types,
            "channels": self.channels,
            "run_bounds": [list(map(float, bounds))
                           for bounds in zip(self.start_gps_times, self.end_gps_times)],
            "state_flags": self.state_flags,
            "data_labels": self.data_labels,
            "max_segment_duration_seconds": self.max_segment_duration_seconds,
            "sample_rate_hertz": float(sample_rate_hertz),
            "group": group,
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
                gf.PATH.parent / self.file_path.name,
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
        Times in [start, stop) when ``ifo`` passes every GWOSC flag in ``state_flag``.

        Segments come from the public GWOSC timeline (see ``gwosc_data``), over the
        GWOSC runs that overlap [start, stop). Returns a sorted (N, 2) array.
        """
        flags = (state_flag,) if isinstance(state_flag, str) else tuple(state_flag)
        for flag in flags:
            if ":" in flag or "-" in flag:
                raise ValueError(
                    f"Flag {flag!r} looks like a DQSegDB flag. Real-noise segments come "
                    "from the public GWOSC timeline: use GWOSC flags such as 'DATA', "
                    "'BURST_CAT1', 'BURST_CAT2', 'CBC_CAT1' or 'CBC_CAT2'."
                )
        gwosc_runs = [
            name for name, run in GWOSC_RUNS.items()
            if run["gps_start"] < stop and run["gps_end"] > start
        ]
        return get_gwosc_segments(ifo.name, gwosc_runs, flags, start=start, end=stop)

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
        Veto segments for every event GWOSC publishes, each covering the signal's time
        in band from 20 Hz plus padding. See ``gravyflow.get_event_veto_segments``.
        """
        from gravyflow.src.dataset.features.event import get_event_veto_segments
        return get_event_veto_segments()

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

        with closing(gf.open_hdf5_file(self.file_path, logger, mode="a")) as f:
            if key in f:
                if not self.force_acquisition:
                    return
                del f[key]
            dataset = f.create_dataset(key, data=strain.data)
            dataset.attrs["t0"] = float(strain.t0)
            dataset.attrs["sample_rate_hertz"] = float(strain.sample_rate_hertz)
            dataset.attrs["source"] = str(strain.source)
            dataset.attrs["layout"] = SEGMENT_LAYOUT_VERSION

    def get_segment_data(
            self,
            segment_start_gps_time: float,
            segment_end_gps_time: float,
            ifo: gf.IFO,
            frame_type: str,
            channel: str,
            sample_rate_hertz: Optional[float] = None,
        ) -> TimeSeries:

        """
        Fetches new segment data from local frames, or else from GWOSC, and reads
        it into a TimeSeries object. See ``get_segment_data_static``.
        """

        for run in ObservingRun:
            if run.value.start_gps_time <= segment_start_gps_time <= run.value.end_gps_time:
                new_frame_type = run.value.frame_types[DataQuality.BEST]
                if new_frame_type != frame_type:
                    logger.warning(
                        f"Observing Run Mismatch: Requested GPS {segment_start_gps_time} "
                        f"falls in {run.value.name}, but configured for different run. "
                        f"Auto-switching frame type {frame_type}->{new_frame_type}."
                    )
                break

        return self.get_segment_data_static(
            segment_start_gps_time,
            segment_end_gps_time,
            ifo,
            frame_type,
            channel,
            sample_rate_hertz=sample_rate_hertz,
        )

    @staticmethod
    def get_segment_data_static(
        segment_start_gps_time: float,
        segment_end_gps_time: float,
        ifo: "gf.IFO",
        frame_type: str,
        channel: str,
        sample_rate_hertz: Optional[float] = None,
    ) -> TimeSeries:
        """
        Thread-safe static version of get_segment_data.

        Fetches segment data without using any instance state.
        Can be safely called from multiple threads concurrently.

        Args:
            segment_start_gps_time: Start GPS time of segment
            segment_end_gps_time: End GPS time of segment
            ifo: Interferometer to fetch data for
            frame_type: Data frame type (e.g., 'HOFT_C00')
            channel: Channel name (e.g., 'DCS-CALIB_STRAIN_CLEAN_C00')
            sample_rate_hertz: The rate the caller needs. GWOSC strain is fetched at
                4096 Hz when that suffices and at 16384 Hz otherwise. None fetches
                4096 Hz.

        Returns:
            TimeSeries containing the requested data

        Raises:
            UpsamplingRefused: If ``sample_rate_hertz`` exceeds 16384 Hz.
            ValueError: If data cannot be fetched from local or remote sources
        """
        open_data_rate = gwosc_sample_rate_hertz(
            sample_rate_hertz if sample_rate_hertz is not None else 4096.0
        )

        # Auto-detect observing run and adjust frame_type/channel if needed
        for run in ObservingRun:
            if run.value.start_gps_time <= segment_start_gps_time <= run.value.end_gps_time:
                run_data = run.value
                new_frame_type = run_data.frame_types[DataQuality.BEST]
                new_channel = run_data.channels[DataQuality.BEST]
                if new_frame_type != frame_type:
                    frame_type = new_frame_type
                    channel = new_channel
                break

        try:
            files = find_urls(
                site=ifo.name.strip("1"),
                frametype=f"{ifo.name}_{frame_type}",
                gpsstart=segment_start_gps_time,
                gpsend=segment_end_gps_time,
                urltype="file",
            )

            if not files:
                raise ValueError("No local files found.")

            full_segment_data = TimeSeries.read(
                files,
                channel=f"{ifo.name}:{channel}",
                start=segment_start_gps_time,
                end=segment_end_gps_time,
                nproc=1  # Single-threaded to avoid fork issues with JAX
            )

        except Exception as e:
            try:
                with strain_download_cache():
                    full_segment_data = TimeSeries.fetch_open_data(
                        ifo.name,
                        segment_start_gps_time,
                        segment_end_gps_time,
                        sample_rate=int(open_data_rate),
                        cache=True,
                        timeout=30  # 30 second timeout
                    )
            except Exception as e_remote:
                raise ValueError(f"Failed to acquire data from local ({e}) or remote ({e_remote}).")

        return full_segment_data

    @staticmethod
    def get_frame_channel_for_gps_static(
        gps_time: float,
        frame_types: List[str],
        channels: List[str],
        start_gps_times: List[float],
        end_gps_times: List[float]
    ) -> Tuple[str, str]:
        """
        Thread-safe static lookup for frame type and channel.
        
        Determines the correct frame_type and channel for a given GPS time
        based on the observing run boundaries.
        
        Args:
            gps_time: GPS time to look up
            frame_types: List of frame types per observing run
            channels: List of channels per observing run
            start_gps_times: Start GPS times of each run
            end_gps_times: End GPS times of each run
            
        Returns:
            Tuple of (frame_type, channel) for the matching run,
            or first entry if no match found
        """
        for i, (start, end) in enumerate(zip(start_gps_times, end_gps_times)):
            if start <= gps_time <= end:
                return frame_types[i], channels[i]
        return frame_types[0], channels[0]  # Fallback to first run

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
        if not gf.check_tensor_integrity(segment.data, 1, 10):
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
        with closing(gf.open_hdf5_file(self.file_path, logger, mode="r")) as segment_file:
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
        """Fetch and resample one segment (see get_segment), or None if it cannot be had."""
        # Refuse impossible rates loudly, before the zero-fill fallback.
        gwosc_sample_rate_hertz(sample_rate_hertz)
        try:
            # Get correct frame_type/channel for this GPS time's observing run
            frame_type, channel = self._get_frame_channel_for_gps(segment_start_gps_time)
            raw_segment = self.get_segment_data(
                segment_start_gps_time,
                segment_end_gps_time,
                ifo,
                frame_type,
                channel,
                sample_rate_hertz=sample_rate_hertz,
            )

            original_sample_rate = float(raw_segment.sample_rate.value)
            if original_sample_rate < sample_rate_hertz:
                raise UpsamplingRefused(
                    f"{ifo.name} data at {segment_start_gps_time} is sampled at "
                    f"{original_sample_rate} Hz, below the requested "
                    f"{sample_rate_hertz} Hz; gravyflow does not upsample real data."
                )

            segment_data = jnp.array(raw_segment.value, dtype=jnp.float32)

            # Keep every sample from the segment's start; cut only the tail, to
            # whole seconds (see get_segment).
            samples_per_second = int(round(original_sample_rate))
            num_samples = segment_data.shape[0]
            whole_seconds = int(kept_seconds(num_samples / original_sample_rate))
            if whole_seconds < 1:
                logging.debug(
                    f"Segment shorter than one second ({num_samples} samples); skipped."
                )
                return None
            segment_data = segment_data[:whole_seconds * samples_per_second]

            # Resampling keeps the first sample's time: sample k is at t0 + k / rate.
            return SegmentStrain(
                data=ops.convert_to_tensor(
                    resample_fft(segment_data, original_sample_rate, sample_rate_hertz),
                    dtype="float32",
                ),
                t0=float(raw_segment.t0.value),
                sample_rate_hertz=float(sample_rate_hertz),
                source=f"{raw_segment.channel} at {original_sample_rate} Hz",
            )
        except UpsamplingRefused:
            raise
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
        scale_factor: float,
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
        return IFOData(data, sample_rate_hertz, first_sample_gps).scale(scale_factor)

    def acquire(
        self,
        sample_rate_hertz: Optional[float] = None,
        valid_segments: Optional[np.ndarray] = None,
        ifos: List[gf.IFO] = [gf.IFO.L1],
        scale_factor: float = 1.0,
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
        # Refuse rates no public source holds, before any zero-fill fallback.
        gwosc_sample_rate_hertz(sample_rate_hertz)

        if self.file_path is not None and self.cache_segments:
            gf.ensure_directory_exists(self.file_path.parent)

        if valid_segments is None:
            valid_segments = self.valid_segments

        def row(segment_times, strains):
            return self._row_data(
                segment_times, ifos, strains, sample_rate_hertz, scale_factor, quiet_zero_fill
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
            groups: Dict[str, float] = None,
            group_name: str = "train",
            segment_order: SegmentOrder = None,
        ) -> List:
        """
        Get valid segments for this acquisition mode.
        Must be implemented by subclasses.
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
            scale_factor: float = None,
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
            scale_factor: float = None,
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
            scale_factor: Amplitude scaling factor
            seed: Random seed
            sampling_mode: RANDOM or GRID sampling
            whiten: Whether to whiten each batch
            crop: Whether to crop each batch to its onsource duration
            group: The split served; None serves the split already served
        """
        batches = self._yield_onsource_offsource_chunks(
            sample_rate_hertz,
            onsource_duration_seconds,
            crop_duration_seconds,
            offsource_duration_seconds,
            num_examples_per_batch,
            ifos,
            scale_factor,
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
            scale_factor: float,
            seed: int,
            sampling_mode: SamplingMode,
            whiten: bool,
            crop: bool,
            group: str
        ) -> Generator[Dict, None, None]:
        """The unchecked batches of get_onsource_offsource_chunks, for this acquisition mode."""
