# Standard library imports:
import hashlib
import logging
import sys

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum, auto
from contextlib import closing
from typing import List, Tuple, Union, Dict, Any, Optional, Generator
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
from gwpy.segments import DataQualityDict, SegmentList
from gwpy.table import EventTable
from gwpy.timeseries import TimeSeries

# Local imports:
import gravyflow as gf
from gravyflow.src.utils.tensor import resample_fft
from gravyflow.src.utils.numerics import ensure_even, ensure_list


# =============================================================================
# CONSTANTS
# =============================================================================

# Imported from TransientDefaults for consistency (canonical source)
from gravyflow.src.dataset.config import TransientDefaults

# GPS time precision is 10ms (0.01s) - defined in gravyflow.src.utils.gps
# GPS_PRECISION_DECIMALS was 3 (1ms) which was stale - now using GPS_KEY_PRECISION
GPS_PRECISION_DECIMALS = 2  # 10ms precision (matches gps.py)

# Tolerance for GPS time matching in seconds (used in cache lookups)
GPS_TOLERANCE_SECONDS = TransientDefaults.GPS_TOLERANCE_SECONDS

# Epsilon buffer for segment boundary trimming (avoids edge artifacts)
SEGMENT_EPSILON_SECONDS = TransientDefaults.SEGMENT_EPSILON_SECONDS


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
    name: str
    start_date_time: datetime
    end_date_time: datetime
    channels: Dict
    frame_types: Dict
    state_flags: Dict

    def __post_init__(self):
        self.start_gps_time = self._to_gps_time(self.start_date_time)
        self.end_gps_time = self._to_gps_time(self.end_date_time)

    @staticmethod
    def _to_gps_time(date_time: datetime) -> float:
        gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
        time_diff = date_time - gps_epoch
        # Current number of leap seconds as of 2021 (change if needed):
        leap_seconds = 18  
        total_seconds = time_diff.total_seconds() - leap_seconds
        return total_seconds

observing_run_data : Dict = {
    "O1" : (
        "O1", 
        datetime(2015, 9, 12, 0, 0, 0), 
        datetime(2016, 1, 19, 0, 0, 0),
        {DataQuality.BEST: "LOSC_4_V1"},
        {DataQuality.BEST: "GWOSC-4KHZ_R1_STRAIN"},
        {DataQuality.BEST: "DCS-ANALYSIS_READY_C01:1"}
    ),
    "O2" : (
        "O2", 
        datetime(2016, 11, 30, 0, 0, 0), 
        datetime(2017, 8, 25, 0, 0, 0),
        {DataQuality.BEST: "LOSC_4_V1"},
        {DataQuality.BEST: "GWOSC-4KHZ_R1_STRAIN"},
        {DataQuality.BEST: "DCS-ANALYSIS_READY_C01:1"}
    ),
    "O3" : (
        "O3", 
        datetime(2019, 4, 1, 0, 0, 0), 
        datetime(2020, 3, 27, 0, 0, 0),
        {DataQuality.BEST: "DCS-CALIB_STRAIN_CLEAN_C01"},
        {DataQuality.BEST: "HOFT_C01"},
        {DataQuality.BEST: "DCS-ANALYSIS_READY_C01:1"}
    ),
    "O4" : (
        "O4", 
        datetime(2023, 5, 24, 15, 0, 0),  # O4 Start (Start of O4a)
        datetime(2025, 11, 18, 16, 0, 0),  # O4 End (End of O4c)
        {DataQuality.BEST: "DCS-CALIB_STRAIN_CLEAN_C01"},
        {DataQuality.BEST: "HOFT_C01"},
        {DataQuality.BEST: "DCS-ANALYSIS_READY_C01:1"}
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

@jax.jit(static_argnames=["num_examples_per_batch", "num_onsource_samples", "num_offsource_samples"])
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
    Generate random subsections from a tensor. JIT compiled for efficiency.
    """
    # Cast input parameters
    start_gps_times = ops.cast(start_gps_times, "float64")

    # Determine the size of the input tensor.
    num_samples = ops.shape(tensor_data)[0]

    # Calculate the range within which to generate random starts.
    # The -16 provides a safety buffer to avoid edge artifacts from FFT/filtering operations.
    maxval = num_samples - num_onsource_samples - 16
    minval = num_offsource_samples

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

    # Calculate the start times for each subsection.
    start_times = ops.cast(random_starts, "float64") * ops.cast(time_interval_seconds, "float64")

    start_times += start_gps_times

    return batch_subarrays, batch_background_chunks, start_times


def concatenate_batches(subarrays, background_chunks, start_times):
    """
    Concatenates batches of subarrays, background chunks, and start times.
    """
    stacked_subarrays = ops.concatenate(subarrays, axis=1)
    stacked_background_chunks = ops.concatenate(background_chunks, axis=1)
    stacked_start_times = ops.concatenate(start_times, axis=1)
    
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
                ops.expand_dims(subsections_start_gps_time, 1)
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

        # Set the time interval between samples
        self.time_interval_seconds = 1.0 / self.sample_rate_hertz
        self.time_interval_seconds = ops.cast(self.time_interval_seconds, "float32")

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
        grid_position: int
    ):
        """
        Extract sequential non-overlapping chunks from segment.
        Each detector gets an offset to ensure different GPS times.
        
        Returns: (subarrays, background_chunks, gps_times, new_grid_position)
        """
        all_subarrays = []
        all_background_chunks = []
        all_gps_times = []
        
        num_detectors = len(self.data)
        
        for ifo_idx, (tensor_data, start_gps_time) in enumerate(zip(self.data, self.start_gps_time)):
            segment_length = int(ops.shape(tensor_data)[0])
            
            # Offset each detector by segment_length // num_detectors to ensure different GPS times
            detector_offset = ifo_idx * (segment_length // max(num_detectors, 1))
            
            subarrays_batch = []
            background_batch = []
            gps_times_batch = []
            
            current_pos = grid_position
            
            for _ in range(num_examples_per_batch):
                # Position with detector offset, wrapped if needed
                pos = (current_pos + detector_offset) % (segment_length - num_onsource_samples - num_offsource_samples)
                
                # Ensure we have room for offsource before onsource
                if pos < num_offsource_samples:
                    pos = num_offsource_samples
                
                # Extract onsource chunk
                onsource_start = pos
                onsource_end = pos + num_onsource_samples
                
                if onsource_end > segment_length:
                    # Segment exhausted for this batch
                    break
                    
                onsource_chunk = tensor_data[onsource_start:onsource_end]
                
                # Extract offsource (background before onsource)
                offsource_start = pos - num_offsource_samples
                offsource_end = pos
                offsource_chunk = tensor_data[offsource_start:offsource_end]
                
                # Calculate GPS time
                gps_time = float(start_gps_time) + (pos * float(self.time_interval_seconds))
                
                subarrays_batch.append(onsource_chunk)
                background_batch.append(offsource_chunk)
                gps_times_batch.append(gps_time)
                
                # Advance position for next sample (non-overlapping)
                current_pos += num_onsource_samples
            
            if len(subarrays_batch) < num_examples_per_batch:
                # Couldn't fill batch - segment exhausted
                return None, None, None, current_pos
            
            # Stack into batch
            all_subarrays.append(ops.expand_dims(ops.stack(subarrays_batch), axis=1))
            all_background_chunks.append(ops.expand_dims(ops.stack(background_batch), axis=1))
            all_gps_times.append(ops.expand_dims(ops.convert_to_tensor(gps_times_batch, dtype="float64"), axis=1))
        
        # Concatenate across detectors
        stacked_subarrays = ops.concatenate(all_subarrays, axis=1)
        stacked_background = ops.concatenate(all_background_chunks, axis=1)
        stacked_gps = ops.concatenate(all_gps_times, axis=1)
        
        # New grid position (advanced by batch)
        new_grid_position = grid_position + (num_examples_per_batch * num_onsource_samples)
        
        return stacked_subarrays, stacked_background, stacked_gps, new_grid_position


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
            cache_segments : bool = True,
            overrides : dict = None,
            logging_level : int = logging.WARNING,
            augmentations: List = None,  # List of augmentation instances
            prefetch_segments : int = 16,
        ):
        
        # Initiate logging for ifo_data:
        # Legacy logger setup removed in favor of module-level logger
        
        self.acquisition_mode = None

        self._current_segment_index = 0
        self._current_batch_index = 0
        self._segment_exhausted = True
        self._num_batches_in_current_segment = 0
        self._grid_position = 0
        self.rng = None
        self.ifos = None
        
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
        
        # Override observing run attributes if present:
        if overrides:
            self.override_attributes(overrides)
        
        # Set file name to none, will be set up if caching is requested
        self.file_path = None

        self.valid_segments = None
        self.valid_segments_adjusted = None
        
        # In-memory LRU cache for segments (unified cache interface)
        self._segment_cache_maxsize = 8
        self._segment_cache = MemoryCache(maxsize=self._segment_cache_maxsize)
        
        # Prefetching configuration
        self.prefetch_segments = prefetch_segments
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1) if prefetch_segments > 0 else None
        self._prefetch_futures: Dict[int, Future] = {}
        
        # Segment acquisition tracking (for debugging silent failures)
        self._dropped_segments_count = 0
        self._zero_filled_segments_count = 0
                
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
        """Exclude ThreadPoolExecutor from pickling (can't be serialized)."""
        state = self.__dict__.copy()
        state['_prefetch_executor'] = None
        state['_prefetch_futures'] = {}
        return state
    
    def __setstate__(self, state):
        """Restore object and recreate ThreadPoolExecutor if needed."""
        self.__dict__.update(state)
        if self.prefetch_segments > 0:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=1)
        else:
            self._prefetch_executor = None
        self._prefetch_futures = {}
        
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
            Processed batch dict (modified in place)
            
        Note:
            Whitening scales data up before processing (to O(1e21) range for numerical
            stability) and scales offsource back down afterward. Whitened onsource 
            stays at unit variance - this is intentional for model training.
        """
        RV = gf.ReturnVariables
        
        if batch is None:
            return None
            
        if whiten:
            # Scale up before whitening to avoid numerical underflow with raw data
            # Adjust scaling based on existing scale_factor to ensure data is O(1e21) range
            target_scale = gf.Defaults.scale_factor
            temp_scale = target_scale / (scale_factor if scale_factor > 0 else 1.0)
            
            batch[RV.ONSOURCE] *= temp_scale
            batch[RV.OFFSOURCE] *= temp_scale
            
            batch[RV.ONSOURCE] = gf.whiten(
                batch[RV.ONSOURCE], 
                batch[RV.OFFSOURCE], 
                sample_rate_hertz
            )

            # Whitened onsource stays at unit variance - don't scale back
            # Scale offsource back down for consistency
            batch[RV.OFFSOURCE] /= temp_scale
            
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
    ) -> List:
        """
        Prefetch a segment by index - runs in background thread.
        """
        if segment_index >= len(valid_segments):
            return None
            
        segment_times = valid_segments[segment_index]
        results = []
        
        for ifo, (segment_start_gps_time, segment_end_gps_time) in zip(ifos, segment_times):
            # Include IFO name in key to prevent cache collision for multi-IFO transient events
            segment_key = f"segments/segment_{ifo.name}_{segment_start_gps_time}_{segment_end_gps_time}"
            try:
                segment_data = self.get_segment(
                    segment_start_gps_time,
                    segment_end_gps_time,
                    sample_rate_hertz,
                    ifo,
                    segment_key
                )
            except Exception as e:
                logging.warning(f"Prefetch failed for {ifo.name} at {segment_start_gps_time}: {e}")
                segment_data = None
            
            results.append((segment_data, segment_start_gps_time, segment_end_gps_time, segment_key))
        
        return results
        
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

        if data_directory_path is None:
            data_directory_path = gf.PATH.parent
        
        segment_parameters = [
                self.frame_types, 
                self.channels, 
                self.state_flags, 
                self.data_labels, 
                self.max_segment_duration_seconds,
                sample_rate_hertz,
                group
            ]  
        
        segment_parameters = [
                str(parameter) for parameter in segment_parameters
            ]
        
        def generate_hash_from_list(params):
            return hashlib.md5("".join(params).encode()).hexdigest()

        segment_hash = generate_hash_from_list(segment_parameters)
        
        # Determine prefix based on content
        if DataLabel.NOISE in self.data_labels:
            prefix = "segment_data"
        else:
            prefix = "transient_data"
            
        self.file_path = Path(data_directory_path) / f"{prefix}_{segment_hash}.hdf5"
        
        return self.file_path
    
    def _get_cached_valid_segments(self, ifos: List, group_name: str) -> np.ndarray:
        """
        Try to load cached valid segments from HDF5 file.
        """
        if self.file_path is None or not Path(self.file_path).exists():
            return None
        
        cache_key = f"valid_segments/{group_name}"
        try:
            with closing(gf.open_hdf5_file(self.file_path, logger, mode="r")) as f:
                if cache_key in f:
                    cached_segments = f[cache_key][()]
                    logger.info(f"Loaded cached valid segments for group '{group_name}'")
                    return cached_segments
        except Exception as e:
            logger.warning(f"Failed to load cached segments: {e}")
        return None
    
    def _cache_valid_segments(self, valid_segments: np.ndarray, group_name: str) -> None:
        """
        Cache valid segments to HDF5 file for future use.
        """
        if self.file_path is None:
            return
        
        cache_key = f"valid_segments/{group_name}"
        try:
            gf.ensure_directory_exists(self.file_path.parent)
            with closing(gf.open_hdf5_file(self.file_path, logger, mode="a")) as f:
                if cache_key in f:
                    del f[cache_key]
                f.create_dataset(cache_key, data=valid_segments)
                logger.info(f"Cached valid segments for group '{group_name}'")
        except Exception as e:
            logger.warning(f"Failed to cache segments: {e}")

    def get_segment_times(
        self,
        start: float,
        stop: float,
        ifo: gf.IFO,
        state_flag: str
    ) -> np.ndarray:
        
        segments = \
            DataQualityDict.query_dqsegdb(
                [f"{ifo.name}:{state_flag}"],
                start,
                stop,
            )

        intersection = segments[f"{ifo.name}:{state_flag}"].active.copy()
                    
        return np.array(intersection)
    
    def get_all_segment_times(
        self,
        ifo : gf.IFO
    ) -> np.ndarray:
        
        valid_segments = []
        for index, start_gps_time in enumerate(self.start_gps_times):         
            valid_segments.append(
                self.get_segment_times(
                    self.start_gps_times[index],
                    self.end_gps_times[index],
                    ifo,
                    self.state_flags[index]
                )
            )
        
        valid_segments = np.array(valid_segments)
        return np.concatenate(valid_segments)
    
    def get_all_event_times(self) -> np.ndarray:
        cache_file_path : Path = gf.PATH / "res/cached_event_times.npy"
        
        if cache_file_path.exists():
            return np.load(cache_file_path)
        
        logging.info("Cache not found, fetching event times.")
        catalogues = [
            "GWTC", 
            "GWTC-1-confident", 
            "GWTC-1-marginal", 
            "GWTC-2", 
            "GWTC-2.1-auxiliary", 
            "GWTC-2.1-confident", 
            "GWTC-2.1-marginal", 
            "GWTC-3-confident", 
            "GWTC-3-marginal",
            "GWTC-4",
        ]

        gps_times = np.array([])
        for catalogue in catalogues:
            events = EventTable.fetch_open_data(catalogue)
            gps_times = np.append(gps_times, events["GPS"].data.compressed())
        
        cache_file_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file_path, gps_times)
        return gps_times

    def pad_gps_times_with_veto_window(
        self,
        gps_times: np.ndarray,
        start_padding_seconds: float = 2.0,
        end_padding_seconds: float = 2.0
    ) -> np.ndarray:
        
        if len(gps_times) == 0:
            return np.empty((0, 2))
            
        starts = gps_times - start_padding_seconds
        ends = gps_times + end_padding_seconds
        
        return np.stack((starts, ends), axis=1)

    def veto_time_segments(
        self,
        segments: np.ndarray,
        veto_segments: np.ndarray
    ) -> np.ndarray:
        
        if len(veto_segments) == 0:
            return segments
            
        seg_list = SegmentList([tuple(s) for s in segments])
        veto_list = SegmentList([tuple(v) for v in veto_segments])
        
        result = seg_list - veto_list
        return np.array(result)

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
                
        return np.array(new_segments)

    def _cache_segment(self, key: str, segment: Any) -> None:
        if self.file_path is None:
            return
            
        with closing(gf.open_hdf5_file(self.file_path, logger, mode="a")) as f:
            if key in f:
                del f[key]
            f.create_dataset(key, data=segment)

    def get_segment_data(
            self,
            segment_start_gps_time: float, 
            segment_end_gps_time: float, 
            ifo: gf.IFO, 
            frame_type: str, 
            channel: str
        ) -> TimeSeries:

        """
        Fetches new segment data from specific URLs and reads it into a 
        TimeSeries object.
        """
        
        for run in ObservingRun:
            if run.value.start_gps_time <= segment_start_gps_time <= run.value.end_gps_time:
                run_data = run.value
                
                new_frame_type = run_data.frame_types[DataQuality.BEST]
                new_channel = run_data.channels[DataQuality.BEST]
                
                if new_frame_type != frame_type:
                    logger.warning(
                        f"Observing Run Mismatch: Requested GPS {segment_start_gps_time} "
                        f"falls in {run_data.name}, but configured for different run. "
                        f"Auto-switching frame type {frame_type}->{new_frame_type}."
                    )
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
                nproc=TransientDefaults.TIMESERIES_NPROC if hasattr(TransientDefaults, 'TIMESERIES_NPROC') else 100
            )
            
        except Exception as e:
            try:
                full_segment_data = TimeSeries.fetch_open_data(
                    ifo.name, 
                    segment_start_gps_time, 
                    segment_end_gps_time,
                    cache=True
                )
            except Exception as e_remote:
                raise ValueError(f"Failed to acquire data from local ({e}) or remote ({e_remote}).")
                
        return full_segment_data

    def get_segment(
        self,
        segment_start_gps_time: float,
        segment_end_gps_time: float,
        sample_rate_hertz: float,
        ifo: gf.IFO,
        segment_key: str
    ):
        epsilon = SEGMENT_EPSILON_SECONDS
        segment = None
        
        cache_key = f"{segment_key}_{ifo.name}_{sample_rate_hertz}"
        
        # Check memory cache first (using unified cache interface)
        cached = self._segment_cache.get(cache_key)
        if cached is not None:
            return cached
    
        # TRANSIENT mode: Skip disk caching entirely - transients go to TransientCache
        use_disk_cache = (
            self.acquisition_mode != AcquisitionMode.TRANSIENT and
            self.file_path is not None and 
            Path(self.file_path).exists()
        )
        
        if use_disk_cache or (self.cache_segments and self.acquisition_mode != AcquisitionMode.TRANSIENT):
            # Only try to read from file if it actually exists
            if self.file_path is None:
                pass
            elif not Path(self.file_path).exists():
                # File doesn't exist yet - skip cache read attempt
                pass
            else:
                with closing(
                        gf.open_hdf5_file(self.file_path, logger, mode="r")
                    ) as segment_file:    

                    if (segment_key in segment_file) and not self.force_acquisition:

                        segment = segment_file[segment_key][()]
                        segment = ops.convert_to_tensor(segment, dtype="float32")
                        
                        if gf.check_tensor_integrity(segment, 1, 10):
                            self._segment_cache.put(cache_key, segment)
                            return segment
                        else:
                            logging.error(
                                "Segment integrity compromised, skipping"
                            )
                            return None
                    else: 
                        logging.debug(
                            "Cached segment not found or force acquisition is set"
                        )
        
        if segment is None: 
            try:
                # Get correct frame_type/channel for this GPS time's observing run
                frame_type, channel = self._get_frame_channel_for_gps(segment_start_gps_time)
                raw_segment = self.get_segment_data(
                    segment_start_gps_time + epsilon,
                    segment_end_gps_time - epsilon, 
                    ifo, 
                    frame_type, 
                    channel
                )
                
                original_sample_rate = float(raw_segment.sample_rate.value)
                
                segment_data = jnp.array(raw_segment.value, dtype=jnp.float32)
                
                downsample_ratio = original_sample_rate / sample_rate_hertz
                std_sizes_at_original = [int(s * downsample_ratio) for s in gf.Defaults.STANDARD_SEGMENT_SAMPLES]
                num_samples = segment_data.shape[0]
                
                std_size = None
                for s in reversed(std_sizes_at_original):
                    if s <= num_samples:
                        std_size = s
                        break
                
                if std_size is not None:
                    # Use center-truncation to preserve event centering (important for TRANSIENT mode)
                    # Start-truncation would shift centered events toward the end or cut them off
                    excess = num_samples - std_size
                    start_offset = excess // 2
                    segment_data = segment_data[start_offset:start_offset + std_size]
                else:
                    logging.warning(f"Segment too small ({num_samples}) for standardization")
                
                segment = resample_fft(
                    segment_data, 
                    original_sample_rate, 
                    sample_rate_hertz
                )

            except Exception as e:
                logging.error(f"Error acquiring segment: {type(e).__name__}, {str(e)}")
                return None
            
            if segment is not None:
                segment = ops.convert_to_tensor(segment, dtype="float32")
                
                if gf.check_tensor_integrity(segment, 1, 10):
                    self._segment_cache.put(cache_key, segment)
                    return segment
                else:
                    logging.error("Segment integrity compromised, skipping")
                    return None
            else:
                logging.error("Segment is None for some reason, skipping")
                return None

    def acquire(
        self, 
        sample_rate_hertz: Optional[float] = None,
        valid_segments: Optional[np.ndarray] = None,
        ifos: List[gf.IFO] = [gf.IFO.L1],
        scale_factor: float = 1.0
    ) -> Generator[IFOData, None, None]:
        
        if sample_rate_hertz is None:
            sample_rate_hertz = gf.Defaults.sample_rate_hertz

        if self.file_path is not None and self.cache_segments:
            gf.ensure_directory_exists(self.file_path.parent)

        if valid_segments is None:
            valid_segments = self.valid_segments

        # Auto-detect: if segments is a different array (subset), use local index
        # This prevents state corruption when acquire() is used for one-off downloads
        use_local_index = (
            valid_segments is not None and 
            valid_segments is not self.valid_segments
        )
        
        if use_local_index:
            # STATELESS MODE: One-off download (e.g., cache misses)
            # Use local index, no prefetch, no state modification
            for idx in range(len(valid_segments)):
                segments = []
                gps_start_times = []
                
                segment_times = valid_segments[idx]
                for ifo, (segment_start_gps_time, segment_end_gps_time) in zip(ifos, segment_times):
                    segment_key = f"segments/segment_{ifo.name}_{segment_start_gps_time}_{segment_end_gps_time}"
                    try:
                        segment_data = self.get_segment(
                            segment_start_gps_time,
                            segment_end_gps_time,
                            sample_rate_hertz,
                            ifo,
                            segment_key
                        )
                    except Exception as e:
                        logging.warning(f"Failed to get segment for {ifo.name} at {segment_start_gps_time}: {e}")
                        segment_data = None
                    
                    if segment_data is not None:
                        segments.append(segment_data)
                        gps_start_times.append(segment_start_gps_time)
                        if self.cache_segments:
                            self._cache_segment(segment_key, segment_data)
                    else:
                        self._zero_filled_segments_count += 1
                        logging.warning(
                            f"Segment missing for {ifo.name} at {segment_start_gps_time}. "
                            f"Filling with zeros (total zero-filled: {self._zero_filled_segments_count})."
                        )
                        duration = segment_end_gps_time - segment_start_gps_time
                        num_samples = int(duration * sample_rate_hertz)
                        segments.append(ops.zeros((num_samples,), dtype="float32"))
                        gps_start_times.append(segment_start_gps_time)
                
                if not segments:
                    logging.error("No segments acquired, skipping to next iteration.")
                    continue
                
                try:
                    multi_segment = IFOData(segments, sample_rate_hertz, gps_start_times)
                    if not multi_segment.data:
                        raise ValueError("Input data should not be empty.")
                    multi_segment = multi_segment.scale(scale_factor)
                    yield multi_segment
                except Exception as e:
                    logging.error(f"Error processing segment: {e}")
                    continue
            return  # Exit generator after stateless download

        # STATEFUL MODE: Streaming with prefetch (existing behavior)
        self._prefetch_futures.clear()

        while self._current_segment_index < len(valid_segments):
            current_idx = self._current_segment_index
            self._current_segment_index += 1
            
            segments = []
            gps_start_times = []
            
            prefetch_result = None
            if current_idx in self._prefetch_futures:
                future = self._prefetch_futures.pop(current_idx)
                try:
                    prefetch_result = future.result(timeout=300)
                except Exception as e:
                    logging.warning(f"Prefetch result failed: {e}")
                    prefetch_result = None
            
            if prefetch_result is not None:
                for segment_data, segment_start_gps_time, segment_end_gps_time, segment_key in prefetch_result:
                    if segment_data is not None:
                        segments.append(segment_data)
                        gps_start_times.append(segment_start_gps_time)
                        if self.cache_segments:
                            self._cache_segment(segment_key, segment_data)
                    else:
                        self._zero_filled_segments_count += 1
                        logging.warning(
                            f"Prefetched segment missing. Filling with zeros "
                            f"(total zero-filled: {self._zero_filled_segments_count})."
                        )
                        duration = segment_end_gps_time - segment_start_gps_time
                        num_samples = int(duration * sample_rate_hertz)
                        segments.append(ops.zeros((num_samples,), dtype="float32"))
                        gps_start_times.append(segment_start_gps_time)
            else:
                segment_times = valid_segments[current_idx]
                for ifo, (segment_start_gps_time, segment_end_gps_time) in zip(ifos, segment_times):
                    # Include IFO name in key to prevent cache collision for multi-IFO transient events
                    segment_key = f"segments/segment_{ifo.name}_{segment_start_gps_time}_{segment_end_gps_time}"
                    try:
                        segment_data = self.get_segment(
                            segment_start_gps_time, 
                            segment_end_gps_time,
                            sample_rate_hertz, 
                            ifo, 
                            segment_key
                        )
                    except Exception as e:
                        logging.warning(f"Failed to get segment for {ifo.name} at {segment_start_gps_time}: {e}")
                        segment_data = None
                        
                    if segment_data is not None:
                        segments.append(segment_data)
                        gps_start_times.append(segment_start_gps_time)
                        if self.cache_segments:
                            self._cache_segment(segment_key, segment_data)
                    else:
                        self._zero_filled_segments_count += 1
                        logging.warning(
                            f"Segment missing for {ifo.name} at {segment_start_gps_time}. "
                            f"Filling with zeros (total zero-filled: {self._zero_filled_segments_count})."
                        )
                        duration = segment_end_gps_time - segment_start_gps_time
                        num_samples = int(duration * sample_rate_hertz)
                        segments.append(ops.zeros((num_samples,), dtype="float32"))
                        gps_start_times.append(segment_start_gps_time)

            if not segments:
                logging.error("No segments acquired, skipping to next iteration.")
                continue

            try:
                multi_segment = IFOData(segments, sample_rate_hertz, gps_start_times)

                if not multi_segment.data:
                    raise ValueError("Input data should not be empty.")
                
                multi_segment = multi_segment.scale(scale_factor)
                
                if self.prefetch_segments > 0 and self._prefetch_executor is not None:
                    next_idx = self._current_segment_index
                    if next_idx < len(valid_segments) and next_idx not in self._prefetch_futures:
                        future = self._prefetch_executor.submit(
                            self._prefetch_segment_for_index,
                            next_idx,
                            valid_segments,
                            sample_rate_hertz,
                            ifos
                        )
                        self._prefetch_futures[next_idx] = future

                yield multi_segment
            except Exception as e:
                logging.error(f"Error processing segment: {e}")
                continue

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
        self.valid_segments = None
        self.valid_segments_adjusted = None
        self.ifos = None
        self._current_segment_index = 0
        self._current_batch_index = 0
        self._segment_exhausted = True
        self.reset_acquisition_stats()

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
            crop: bool = False
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
            crop=crop
        )

    @abstractmethod
    def get_onsource_offsource_chunks(
            self,
            sample_rate_hertz: float,
            onsource_duration_seconds: float,
            crop_duration_seconds: float,
            offsource_duration_seconds: float,
            num_examples_per_batch: int = None,
            ifos: List[gf.IFO] = None,
            scale_factor: float = None,
            seed: int = None,
            sampling_mode: SamplingMode = SamplingMode.RANDOM,
            whiten: bool = False,
            crop: bool = False
        ):
        """
        Generate onsource/offsource data chunks.
        
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
            
        Must be implemented by subclasses.
        """
        pass
