# Standard library imports:
import hashlib
import logging
import sys

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from contextlib import closing
from typing import List, Tuple, Union, Dict, Any, Optional, Generator
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future

# Suppress verbose logging from external libraries (gwpy, gwosc)
logging.getLogger('gwpy').setLevel(logging.WARNING)
logging.getLogger('gwosc').setLevel(logging.WARNING)

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
from gravyflow.src.dataset.features.event import EventType


def ensure_even(number):
    if number % 2 != 0:
        number -= 1
    return number

# Enums
class DataQuality(Enum):
    RAW = auto()
    BEST = auto()

class DataLabel(Enum):
    NOISE = auto()
    GLITCHES = auto()
    EVENTS = auto()
    
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
    # ops.take expects indices to be flattened or we use advanced indexing?
    # ops.take(x, indices, axis=0)
    # If indices is (Batch, Samples), output is (Batch, Samples, ...).
    # tensor_data is (TotalSamples,).
    # Output should be (Batch, Samples).
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
        
        # Seed handling: we need unique seeds for each data stream if iterating?
        # Or just one seed passed in?
        # The wrapper passes one seed.
        # We should split it?
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
        # ops.convert_to_numpy
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
    
@dataclass
class IFODataObtainer:
    
    def __init__(
            self, 
            data_quality : DataQuality,
            data_labels : Union[DataLabel, List[DataLabel]],
            observing_runs : Union[ObservingRun, List[ObservingRun]] = None,
            segment_order : SegmentOrder = SegmentOrder.RANDOM,
            max_segment_duration_seconds : float = 2048.0,
            saturation : float = 8.0,  # Higher = more samples (8x oversampling)
            force_acquisition : bool = False,
            cache_segments : bool = True,
            overrides : dict = None,
            event_types : List[EventType] = [EventType.CONFIDENT],
            logging_level : int = logging.WARNING,
            random_sign_reversal : bool = True,
            random_time_reversal : bool = True,
            augmentation_probability : float = 0.5,
            prefetch_segments : int = 16,  # Higher default for TRANSIENT (64s segments)
            # TRANSIENT/FEATURE mode augmentations
            random_shift : bool = False,  # Shift event off-center
            shift_fraction : float = 0.25,  # Max shift as fraction of onsource
            add_noise : bool = False,  # Add random noise perturbations
            noise_amplitude : float = 0.1,  # Noise amplitude (relative to std)
            # Class balancing for glitch classification
            balanced_glitch_types : bool = False  # Equal sampling from each glitch type
        ):
        
        # Initiate logging for ifo_data:
        self.logger = logging.getLogger("ifo_data_aquisition")
        stream_handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(stream_handler)
        self.logger.setLevel(logging_level)
        
        self.acquisition_mode = None

        self._current_segment_index = 0
        self._current_batch_index = 0
        self._segment_exausted = True
        self._num_batches_in_current_segment = 0
        self._grid_position = 0  # For GRID mode: current position in segment
        self.rng = None
        self.ifos = None
        
        # Default to all observing runs (O1, O2, O3) if not specified
        if observing_runs is None:
            observing_runs = [ObservingRun.O1, ObservingRun.O2, ObservingRun.O3]

        # Ensure parameters are lists for consistency:
        if not isinstance(observing_runs, list):
            observing_runs = [observing_runs]
        if not isinstance(data_labels, list):
            data_labels = [data_labels]
        
        #Set class atributed with parameters:
        self.data_quality = data_quality
        self.data_labels = data_labels
        self.segment_order = segment_order
        self.max_segment_duration_seconds = max_segment_duration_seconds
        self.saturation = saturation
        self.force_acquisition = force_acquisition
        self.cache_segments = cache_segments
        self.segment_file = None
        self.event_types = event_types
        
        # Data augmentation settings (for real noise)
        self.random_sign_reversal = random_sign_reversal
        self.random_time_reversal = random_time_reversal
        self.augmentation_probability = augmentation_probability
        
        # TRANSIENT/FEATURE mode augmentations
        self.random_shift = random_shift
        self.shift_fraction = shift_fraction
        self.add_noise = add_noise
        self.noise_amplitude = noise_amplitude
        
        # Class balancing
        self.balanced_glitch_types = balanced_glitch_types
            
        # Unpack parameters from input observing runs:
        self.unpack_observing_runs(observing_runs, data_quality)
        
        # Override observing run attributes if present:
        if overrides:
            self.override_attributes(overrides)
        
        # Set file name to none, will be set up if caching is requested
        self.file_path = None

        self.valid_segments = None
        self.valid_segments_adjusted = None
        
        # In-memory LRU cache for segments to reduce HDF5 disk reads
        # Size adjusted when acquisition_mode is set:
        #   NOISE: 8 segments (~8 GB for 30+ min segments)
        #   TRANSIENT: 128 segments (~128 MB for 64s segments)
        self._segment_cache = OrderedDict()
        self._segment_cache_maxsize = 8  # Default for NOISE mode
        
        # Prefetching configuration
        self.prefetch_segments = prefetch_segments
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1) if prefetch_segments > 0 else None
        self._prefetch_futures: Dict[int, Future] = {}  # segment_index -> Future
                
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
    
    def __getstate__(self):
        """Exclude ThreadPoolExecutor from pickling (can't be serialized)."""
        state = self.__dict__.copy()
        # Remove unpicklable objects
        state['_prefetch_executor'] = None
        state['_prefetch_futures'] = {}
        return state
    
    def __setstate__(self, state):
        """Restore object and recreate ThreadPoolExecutor if needed."""
        self.__dict__.update(state)
        # Recreate executor if prefetching was enabled
        if self.prefetch_segments > 0:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=1)
        else:
            self._prefetch_executor = None
        self._prefetch_futures = {}
        
    def __del__(self):
        if getattr(self, "segment_file", None) is not None:
            self.segment_file.close()
        if getattr(self, "_prefetch_executor", None) is not None:
            self._prefetch_executor.shutdown(wait=False)
            
    def close(self):
        if self.segment_file is not None:
            self.segment_file.close()
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=True)
            self._prefetch_executor = None
    
    def _apply_augmentation(self, data, is_transient: bool = False):
        """
        Apply random augmentations to data.
        
        Args:
            data: Tensor of shape (Batch, IFOs, Samples)
            is_transient: If True, apply TRANSIENT-specific augmentations (shift, noise)
            
        Returns:
            Augmented tensor with same shape
        """
        if self.rng is None:
            return data
            
        # Sign reversal (y-axis flip)
        if self.random_sign_reversal and self.rng.random() < self.augmentation_probability:
            data = -data
            
        # Time reversal (x-axis flip)
        if self.random_time_reversal and self.rng.random() < self.augmentation_probability:
            data = ops.flip(data, axis=-1)
        
        # TRANSIENT-specific augmentations
        if is_transient:
            # Random shift (move event off-center)
            if self.random_shift and self.rng.random() < self.augmentation_probability:
                num_samples = int(ops.shape(data)[-1])
                max_shift = int(self.shift_fraction * num_samples)
                shift = self.rng.integers(-max_shift, max_shift + 1)
                data = ops.roll(data, shift, axis=-1)
            
            # Add noise perturbations
            if self.add_noise and self.rng.random() < self.augmentation_probability:
                # Calculate std per sample for proper scaling
                data_np = ops.convert_to_numpy(data)
                noise = self.rng.normal(0, self.noise_amplitude, data_np.shape)
                # Scale noise by data standard deviation
                std = np.std(data_np)
                noise = noise * std if std > 0 else noise
                data = ops.convert_to_tensor(data_np + noise, dtype=data.dtype)
            
        return data
    
    def _lookup_labels(self, gps_times):
        """
        Look up glitch type labels for given GPS times.
        
        Args:
            gps_times: List of GPS times to look up
            
        Returns:
            NumPy array of integer labels (glitch type indices)
        """
        if not hasattr(self, '_feature_labels') or not self._feature_labels:
            # No labels available, return -1 (unknown)
            return np.full(len(gps_times), -1, dtype=np.int32)
        
        labels = []
        for gps in gps_times:
            label = -1  # Default: unknown
            
            # Check glitch labels
            if gf.DataLabel.GLITCHES in self._feature_labels:
                times_arr, labels_arr = self._feature_labels[gf.DataLabel.GLITCHES]
                if len(times_arr) > 0:
                    # Find closest match within tolerance
                    diffs = np.abs(times_arr - gps)
                    min_idx = np.argmin(diffs)
                    if diffs[min_idx] < 1.0:  # Within 1 second
                        label = int(labels_arr[min_idx])
            
            labels.append(label)
        
        return np.array(labels, dtype=np.int32)
    
    def _get_effective_saturation(self):
        """
        Calculate effective saturation accounting for augmentation.
        
        Higher saturation = more samples from same data. Since augmentation
        increases unique samples by (1 + probability) per enabled augmentation,
        we multiply saturation to sample more densely.
        """
        effective = self.saturation
        if self.random_sign_reversal:
            effective *= (1 + self.augmentation_probability)
        if self.random_time_reversal:
            effective *= (1 + self.augmentation_probability)
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
        Returns list of (segment_data, gps_start_time) tuples for each IFO.
        """
        if segment_index >= len(valid_segments):
            return None
            
        segment_times = valid_segments[segment_index]
        results = []
        
        for ifo, (segment_start_gps_time, segment_end_gps_time) in zip(ifos, segment_times):
            segment_key = f"segments/segment_{segment_start_gps_time}_{segment_end_gps_time}"
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
        if gf.DataLabel.NOISE in self.data_labels:
            prefix = "segment_data"
        else:
            prefix = "transient_data"
            
        self.file_path = Path(data_directory_path) / f"{prefix}_{segment_hash}.hdf5"
        
        return self.file_path
    
    def _get_cached_valid_segments(self, ifos: List, group_name: str) -> np.ndarray:
        """
        Try to load cached valid segments from HDF5 file.
        Returns None if not cached.
        """
        if self.file_path is None or not Path(self.file_path).exists():
            return None
        
        cache_key = f"valid_segments/{group_name}"
        try:
            with closing(gf.open_hdf5_file(self.file_path, self.logger, mode="r")) as f:
                if cache_key in f:
                    cached_segments = f[cache_key][()]
                    self.logger.info(f"Loaded cached valid segments for group '{group_name}'")
                    return cached_segments
        except Exception as e:
            self.logger.warning(f"Failed to load cached segments: {e}")
        return None
    
    def _cache_valid_segments(self, valid_segments: np.ndarray, group_name: str) -> None:
        """
        Cache valid segments to HDF5 file for future use without auth.
        """
        if self.file_path is None:
            return
        
        cache_key = f"valid_segments/{group_name}"
        try:
            gf.ensure_directory_exists(self.file_path.parent)
            with closing(gf.open_hdf5_file(self.file_path, self.logger, mode="a")) as f:
                if cache_key in f:
                    del f[cache_key]
                f.create_dataset(cache_key, data=valid_segments)
                self.logger.info(f"Cached valid segments for group '{group_name}'")
        except Exception as e:
            self.logger.warning(f"Failed to cache segments: {e}")
    
    def _cluster_transients(
        self, 
        segments: np.ndarray, 
        request_overhead_seconds: float = 15.0,
        data_download_rate: float = 0.01
    ) -> np.ndarray:
        """
        Cluster nearby transients using a greedy cost-optimized algorithm.
        
        Makes per-gap decisions: merge if downloading the gap data is cheaper
        than making a separate request. The breakeven point is:
            gap_threshold = request_overhead / data_download_rate
        
        Defaults based on GWOSC benchmark (15s overhead, 0.01s/s rate) -> 1500s threshold.
        
        Args:
            segments: Array of (start, end) GPS times, shape (N, 2)
            request_overhead_seconds: Fixed cost per network request (seconds)
            data_download_rate: Time to download 1 second of data (seconds)
                               e.g., 0.005 means 100s of data takes 0.5s
                               
        Returns:
            Merged segments array of shape (M, 2) where M <= N
        """
        if len(segments) == 0:
            return segments
        
        if len(segments) == 1:
            return segments
        
        # Breakeven gap: merge if gap * data_rate < request_overhead
        gap_threshold = request_overhead_seconds / data_download_rate
            
        # Sort by start time
        sorted_idx = np.argsort(segments[:, 0])
        sorted_segs = segments[sorted_idx]
        
        # Greedy merge: for each gap, decide whether to merge or keep separate
        merged = []
        current_start, current_end = sorted_segs[0]
        
        for start, end in sorted_segs[1:]:
            gap = start - current_end
            
            # Merge if gap is negative (overlap) or within cost threshold
            if gap <= 0 or gap <= gap_threshold:
                current_end = max(current_end, end)
            else:
                merged.append([current_start, current_end])
                current_start, current_end = start, end
        
        # Don't forget the last segment
        merged.append([current_start, current_end])
        
        merged_array = np.array(merged)
        
        # Log the reduction
        original_count = len(segments)
        merged_count = len(merged_array)
        if merged_count < original_count:
            logging.info(
                f"Greedy clustering: {original_count} transients -> "
                f"{merged_count} download segments (gap threshold: {gap_threshold:.0f}s)"
            )
        
        return merged_array

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
            logging.info("Loading event times from cache.")
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
        
        # Ensure parent directory exists before saving
        cache_file_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file_path, gps_times)
        return gps_times
    
    def remove_unwanted_segments(
            self,
            ifo: gf.IFO,
            valid_segments: np.ndarray,
            get_times: List = None
        ):
                
        if get_times is None:
            get_times = []

        veto_segments = []

        if DataLabel.EVENTS in get_times or DataLabel.EVENTS not in self.data_labels:
            event_times = self.get_all_event_times()
        else:
            event_times = []

        if DataLabel.GLITCHES in get_times or DataLabel.GLITCHES not in self.data_labels:
            glitch_times = gf.get_glitch_times(
                ifo,
                start_gps_time=self.start_gps_times[0],
                end_gps_time=self.end_gps_times[0]
            )
        else:
            glitch_times = []

        if DataLabel.EVENTS not in self.data_labels:
            veto_segments.append(
                self.pad_gps_times_with_veto_window(event_times)
            )

        if DataLabel.GLITCHES not in self.data_labels:
            veto_segments.append(
                gf.get_glitch_segments(
                    ifo,
                    start_gps_time=self.start_gps_times[0],
                    end_gps_time=self.end_gps_times[0]
                )
            )

        if veto_segments:
            veto_segments = np.concatenate(veto_segments)
            valid_segments = \
                self.veto_time_segments(valid_segments, veto_segments)

        feature_times = {
            gf.DataLabel.EVENTS: event_times,
            gf.DataLabel.GLITCHES: glitch_times    
        }

        return valid_segments, feature_times

    def find_segment_intersections(self, arr1, arr2):
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
    
    def return_wanted_segments(
            self,
            ifo : gf.IFO,
            valid_segments : np.ndarray,        
            start_padding_seconds : float = 32.0,
            end_padding_seconds : float = 32.0,
        ):
        """
        Get segments containing desired features (events/glitches).
        
        Supersede logic:
        - EVENTS supersedes individual EventType members (CONFIDENT, MARGINAL)
        - GLITCHES supersedes individual GlitchType members
        """
        from gravyflow.src.dataset.features.event import EventType
        from gravyflow.src.dataset.features.glitch import GlitchType
        
        wanted_segments = []
        feature_times = {}
        
        # --- Event handling with supersede ---
        # Check for EVENTS (supersedes) or individual EventTypes
        has_events_supersede = DataLabel.EVENTS in self.data_labels
        event_types_in_labels = [
            label for label in self.data_labels 
            if isinstance(label, EventType)
        ]
        
        if has_events_supersede or event_types_in_labels:
            # If EVENTS present, get all; otherwise filter by individual types
            if has_events_supersede:
                event_times = gf.get_all_event_times()
            else:
                event_times = gf.get_event_times_by_type(event_types_in_labels)
            
            if len(event_times) > 0:
                wanted_segments.append(
                    self.pad_gps_times_with_veto_window(
                        event_times,
                        start_padding_seconds=start_padding_seconds,
                        end_padding_seconds=end_padding_seconds
                    )
                )
                feature_times[gf.DataLabel.EVENTS] = event_times
        
        # --- Glitch handling with supersede ---
        # Check for GLITCHES (supersedes) or individual GlitchTypes
        has_glitches_supersede = DataLabel.GLITCHES in self.data_labels
        glitch_types_in_labels = [
            label for label in self.data_labels 
            if isinstance(label, GlitchType)
        ]
        
        if has_glitches_supersede or glitch_types_in_labels:
            # If GLITCHES present, get all; otherwise filter by individual types
            if has_glitches_supersede:
                glitch_types_to_fetch = None  # None = all types
            else:
                glitch_types_to_fetch = glitch_types_in_labels
            
            glitch_segments = gf.get_glitch_segments(
                ifo,
                start_gps_time=self.start_gps_times[0],
                end_gps_time=self.end_gps_times[0],
                glitch_types=glitch_types_to_fetch
            )
            
            # Apply padding to glitch segments
            if len(glitch_segments) > 0:
                padded_glitches = glitch_segments.copy()
                padded_glitches[:, 0] -= start_padding_seconds
                padded_glitches[:, 1] += end_padding_seconds
                wanted_segments.append(padded_glitches)
                
                # Get glitch times with labels for classification
                glitch_times, glitch_labels = gf.get_glitch_times_with_labels(
                    ifo,
                    start_gps_time=self.start_gps_times[0],
                    end_gps_time=self.end_gps_times[0],
                    glitch_types=glitch_types_to_fetch,
                    balanced=self.balanced_glitch_types
                )
                feature_times[gf.DataLabel.GLITCHES] = glitch_times
                # Store labels for lookup during batch assembly
                self._feature_labels = {
                    gf.DataLabel.GLITCHES: (glitch_times, glitch_labels)
                }
            
        if wanted_segments:
            wanted_segments = np.concatenate(wanted_segments)
            
            valid_segments = self.find_segment_intersections(
                valid_segments,
                wanted_segments
            )
        else:
            raise ValueError("Cannot find any features which suit requirement!")

        if not any(np.any(segment) for segment in valid_segments):
            raise ValueError(
                "Cannot find any features which overlap required times!"
            )
        
        return valid_segments, feature_times

    def remove_short_segments(
            self,
            segments: np.ndarray, 
            minimum_duration_seconds: float
        ) -> np.ndarray:
        
        """
        Removes columns where at least one of the durations in the column is 
        less than the specified minimum duration.

        Parameters:
        segments (np.ndarray): Input array of shape [N, X, 2].
        minimum_duration_seconds (float): Minimum allowed duration.

        Returns:
        np.ndarray: Array with columns removed.
        """
        # Calculate durations for each pair in each column:
        durations = segments[:, :, 1] - segments[:, :, 0] 
        
        # Check if all durations in a column are valid:
        valid_columns = np.all(durations >= minimum_duration_seconds, axis=1) 
        
         # Select only columns where all durations are valid:
        filtered_segments = segments[valid_columns, :, :] 
        return filtered_segments

    def compress_segments(self, segments: np.ndarray) -> np.ndarray:
        if segments.size == 0:
            return segments

        # Sorting and compressing using NumPy operations
        segments = segments[segments[:, 0].argsort()]  # Ensures the segments are sorted by start time
        compressed = segments.copy()

        # Use a vectorized comparison to find segments that overlap with their predecessor
        overlaps = compressed[1:, 0] <= compressed[:-1, 1]
        # Update the end times of segments that overlap with their predecessor
        compressed[:-1][overlaps, 1] = np.maximum(compressed[:-1][overlaps, 1], compressed[1:][overlaps, 1])
        # Keep segments that do not overlap with their successor
        keep_segments = np.append([True], ~overlaps)

        return compressed[keep_segments]
    
    def calculate_bin_indices(self, segments, interval, start_period):
        # Calculate the relative position of each segment start and end to the start_period
        relative_starts = (segments[:, 0] - start_period) / interval
        relative_ends = (segments[:, 1] - start_period) / interval

        # Calculate the bin indices for the starts and ends
        start_bins = np.floor(relative_starts).astype(int)
        end_bins = np.ceil(relative_ends).astype(int)

        return start_bins, end_bins

    def merge_bins(self, new_segments_list, interval):
        """
        Merges split segments into bins of shape [G, N].

        Parameters:
        new_segments_list (list of np.ndarray): 
            List of 2D arrays with split segments.
        interval (float): 
            Time interval of bins.

        Returns:
        list of list of np.ndarray: List of lists of arrays, with shape [G, N].
        """
        # Determine the global minimum and maximum times
        min_time = min(segment[0, 0] for segment in new_segments_list)
        max_time = max(segment[-1, 1] for segment in new_segments_list)

        # Create bins covering the global time range
        bins = np.arange(
            np.floor(min_time / interval) * interval, 
            np.ceil(max_time / interval) * interval, interval
        )

        # Initialize the result data structure
        result = [[] for _ in bins]

        # Iterate over each bin and each array, collecting segments within each bin
        for i, bin_start in enumerate(bins):
            for segments in new_segments_list:
                # Find segments that fall within this bin
                in_bin = segments[
                    (segments[:, 0] < bin_start + interval) 
                    & (segments[:, 1] > bin_start)
                ]
                if in_bin.size > 0:
                    # Adjust the segments to fit within the bin
                    in_bin[:, 0] = np.maximum(in_bin[:, 0], bin_start)
                    in_bin[:, 1] = np.minimum(
                        in_bin[:, 1], bin_start + interval
                    )
                    result[i].append(in_bin)
                else:
                    result[i].append(np.empty((0, 2)))
                    
        # Filter out bins where one of the lists is empty
        filtered_result = [
            bin for bin in result if all(len(arr) > 0 for arr in bin)
        ]

        return filtered_result
    
    def largest_segments_per_bin(self, filtered_bins_result):
        """
        Extracts the largest segment in each bin from each list and converts 
        back to a list of 2 arrays.

        Parameters:
        filtered_bins_result (list of list of np.ndarray): 
            Filtered result from merge_bins function.

        Returns:
        list of np.ndarray: 
            List of 2 arrays of largest segments, with equal length.
        """
        # Initialize lists to store largest segments from each original array
        largest_segments_list = [
            [] for _ in range(len(filtered_bins_result[0]))
        ]

        # Iterate over each bin
        for bin in filtered_bins_result:
            # For each list in a bin, find the segment with the largest duration
            for j, segments in enumerate(bin):
                
                # Calculate durations of segments:
                durations = segments[:, 1] - segments[:, 0]  
                
                # Find index of largest segment:
                largest_segment_index = np.argmax(durations)  
                
                # Add largest segment to list:
                largest_segments_list[j].append(segments[largest_segment_index])  
        
        # Convert lists of largest segments into arrays
        result_arrays = [
            np.array(segments) for segments in largest_segments_list
        ]

        return np.array(result_arrays)
    
    def order_segments(
        self,
        valid_segments : np.ndarray,
        segment_order : SegmentOrder,
        seed : int
    ):
        # Create random number generator from seed:
        # Create a random number generator with the provided seed
        rng = default_rng(seed)

        # Order segments by requested order:
        match segment_order:
            case SegmentOrder.RANDOM:
                # Shuffle data sements randomly.
                rng.shuffle(valid_segments)

            case SegmentOrder.SHORTEST_FIRST:
                # Sort by shortest first (useful for debugging).
                sort_by_duration = lambda segments: segments[
                    np.argsort(segments[:, 0, 1] - segments[:, 0, 0])
                ]
                valid_segments = sort_by_duration(valid_segments)
            case SegmentOrder.CHRONOLOGICAL:
                # Do nothing as default order should be chronological.
                pass
            case _:
                # Raise error in the default case.
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
                # Identify which chunk 'curr' falls into
                chunk_idx = int(np.floor((curr - start_time) / chunk_size))
                chunk_end = start_time + (chunk_idx + 1) * chunk_size
                
                # The cut point is the end of this chunk
                cut_point = min(end, chunk_end)
                
                if cut_point > curr:
                    new_segments.append([curr, cut_point])
                
                curr = cut_point
                
        return np.array(new_segments)

    def get_segments_for_group(
        self,
        segments: np.ndarray,
        chunk_size: float,
        group_name: str,
        groups: Dict[str, float],
        start_time: float
    ) -> np.ndarray:
        
        selected_segments = []
        
        group_names = list(groups.keys())
        probs = list(groups.values())
        total = sum(probs)
        probs = [p/total for p in probs]
        
        for start, end in segments:
            # Determine which grid cell this segment belongs to.
            idx = int(np.floor((start - start_time) / chunk_size))
            
            # Deterministic RNG based on index
            h = hashlib.md5(str(idx).encode()).hexdigest()
            val = int(h, 16) / (2**128)
            
            cumulative = 0.0
            assigned_group = group_names[-1]
            for name, prob in zip(group_names, probs):
                cumulative += prob
                if val <= cumulative:
                    assigned_group = name
                    break
            
            if assigned_group == group_name:
                selected_segments.append([start, end])
                
        return np.array(selected_segments)

    def _extract_feature_event(
        self,
        event_segment: 'IFOData',
        num_onsource_samples: int,
        num_offsource_samples: int,
        sample_rate_hertz: float
    ) -> tuple:
        """
        Extract onsource/offsource windows from a feature segment.
        
        Centers extraction on the middle of the segment (where the event is).
        
        Args:
            event_segment: IFOData containing strain data for all IFOs
            num_onsource_samples: Number of samples for onsource window
            num_offsource_samples: Number of samples for offsource window
            sample_rate_hertz: Sample rate
            
        Returns:
            Tuple of (onsource_stacked, offsource_stacked, event_gps_time)
            or (None, None, None) if extraction fails
        """
        data_len = ops.shape(event_segment.data[0])[0]
        
        if data_len < num_onsource_samples:
            logging.warning(f"Feature segment too short ({data_len} < {num_onsource_samples})")
            return None, None, None
        
        # Center extraction on segment middle (where event is)
        center_idx = data_len // 2
        half_onsource = num_onsource_samples // 2
        start_onsource = center_idx - half_onsource
        end_onsource = start_onsource + num_onsource_samples
        
        # Calculate event GPS time
        feature_gps_start = event_segment.start_gps_time[0]
        event_gps_time = feature_gps_start + (center_idx / sample_rate_hertz)
        
        temp_onsource = []
        temp_offsource = []
        
        for channel_data in event_segment.data:
            # Extract onsource
            chunk = channel_data[start_onsource:end_onsource]
            if ops.shape(chunk)[0] != num_onsource_samples:
                return None, None, None
            temp_onsource.append(chunk)
            
            # Extract offsource (ends at start of onsource)
            off_start = start_onsource - num_offsource_samples
            if off_start >= 0:
                off_chunk = channel_data[off_start:start_onsource]
            elif end_onsource + num_offsource_samples <= data_len:
                off_chunk = channel_data[end_onsource:end_onsource + num_offsource_samples]
            else:
                off_chunk = ops.zeros((num_offsource_samples,), dtype="float32")
            temp_offsource.append(off_chunk)
        
        # Stack IFOs: list of (Samples,) -> (IFO, Samples)
        onsource_stacked = ops.stack(temp_onsource)
        offsource_stacked = ops.stack(temp_offsource)
        
        return onsource_stacked, offsource_stacked, event_gps_time

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

    def _cache_segment(self, key: str, segment: Any) -> None:
        if self.file_path is None:
            return
            
        with closing(gf.open_hdf5_file(self.file_path, self.logger, mode="a")) as f:
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
        
        # Helper to find correct observing run definition for this time
        for run in ObservingRun:
            if run.value.start_gps_time <= segment_start_gps_time <= run.value.end_gps_time:
                # Found the correct run for this time
                run_data = run.value
                
                # Assuming BEST quality as a fallback/default for events
                # This fixes the issue where asking for O1 event with O3 config uses wrong frame type
                new_frame_type = run_data.frame_types[DataQuality.BEST]
                new_channel = run_data.channels[DataQuality.BEST]
                
                if new_frame_type != frame_type:
                    self.logger.warning(
                        f"Observing Run Mismatch: Requested GPS {segment_start_gps_time} "
                        f"falls in {run_data.name}, but configured for different run. "
                        f"Auto-switching frame type {frame_type}->{new_frame_type}."
                    )
                    frame_type = new_frame_type
                    channel = new_channel
                break

        # Try to find local files first (preferred for cluster usage)
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
                nproc=100
            )
            
        except Exception as e:
            # Fallback to Open Data (GWOSC) if local fetch fails
            # This is critical for O1/O2 public data or when running off-cluster
            # logging.info(f"Local fetch failed for {ifo.name} at {segment_start_gps_time}: {e}. Trying GWOSC...")
            try:
                full_segment_data = TimeSeries.fetch_open_data(
                    ifo.name, 
                    segment_start_gps_time, 
                    segment_end_gps_time,
                    cache=True
                )
            except Exception as e_remote:
                # If both fail, raise the original error or the new one
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
        epsilon = 0.1         
        segment = None
        
        # Create a unique cache key for this segment
        cache_key = f"{segment_key}_{ifo.name}_{sample_rate_hertz}"
        
        # Check in-memory LRU cache first (fastest path)
        if cache_key in self._segment_cache:
            # Move to end (most recently used) and return
            self._segment_cache.move_to_end(cache_key)
            return self._segment_cache[cache_key]
        
        if (self.file_path is not None and Path(self.file_path).exists()) or self.cache_segments:
            # Ensure file path is generated if caching is on
            if self.file_path is None:
                 # We can't cache if we don't have a path. 
                 # But generate_file_path should have been called.
                 pass
            else:
                with closing(
                        gf.open_hdf5_file(self.file_path, self.logger, mode="r")
                    ) as segment_file:    

                    if (segment_key in segment_file) and not self.force_acquisition:

                        segment = segment_file[segment_key][()]
                        # Convert to JAX/Keras tensor
                        segment = ops.convert_to_tensor(segment, dtype="float32")
                        
                        if gf.check_tensor_integrity(segment, 1, 10):
                            # Add to in-memory cache
                            self._add_to_segment_cache(cache_key, segment)
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
                raw_segment = self.get_segment_data(
                    segment_start_gps_time + epsilon,
                    segment_end_gps_time - epsilon, 
                    ifo, 
                    self.frame_types[0], 
                    self.channels[0]
                )
                
                # Get original sample rate from GwPy TimeSeries
                original_sample_rate = float(raw_segment.sample_rate.value)
                
                # Convert to JAX tensor
                segment_data = jnp.array(raw_segment.value, dtype=jnp.float32)
                
                # Truncate BEFORE resample for efficiency - FFT operates on less data
                # Calculate equivalent standard sizes at original sample rate
                downsample_ratio = original_sample_rate / sample_rate_hertz
                # Scale standard sizes to original sample rate
                std_sizes_at_original = [int(s * downsample_ratio) for s in gf.Defaults.STANDARD_SEGMENT_SAMPLES]
                num_samples = segment_data.shape[0]
                
                # Find largest standard size that fits
                std_size = None
                for s in reversed(std_sizes_at_original):
                    if s <= num_samples:
                        std_size = s
                        break
                
                if std_size is not None:
                    segment_data = segment_data[:std_size]
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
                    # Add to in-memory cache
                    self._add_to_segment_cache(cache_key, segment)
                    return segment
                else:
                    logging.error("Segment integrity compromised, skipping")
                    return None
            else:
                logging.error("Segment is None for some reason, skipping")
                return None
    
    def _add_to_segment_cache(self, cache_key: str, segment):
        """Add segment to in-memory LRU cache, evicting oldest if needed."""
        self._segment_cache[cache_key] = segment
        
        # Evict oldest entries if cache exceeds max size
        while len(self._segment_cache) > self._segment_cache_maxsize:
            self._segment_cache.popitem(last=False)

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

        # Clear any stale prefetch futures
        self._prefetch_futures.clear()

        while self._current_segment_index < len(valid_segments):
            current_idx = self._current_segment_index
            self._current_segment_index += 1
            
            segments = []
            gps_start_times = []
            
            # Check if we have prefetched results for this segment
            prefetch_result = None
            if current_idx in self._prefetch_futures:
                future = self._prefetch_futures.pop(current_idx)
                try:
                    prefetch_result = future.result(timeout=300)  # 5 min timeout
                except Exception as e:
                    logging.warning(f"Prefetch result failed: {e}")
                    prefetch_result = None
            
            if prefetch_result is not None:
                # Use prefetched data
                for segment_data, segment_start_gps_time, segment_end_gps_time, segment_key in prefetch_result:
                    if segment_data is not None:
                        segments.append(segment_data)
                        gps_start_times.append(segment_start_gps_time)
                        if self.cache_segments:
                            self._cache_segment(segment_key, segment_data)
                    else:
                        logging.warning(f"Prefetched segment missing. Filling with zeros.")
                        duration = segment_end_gps_time - segment_start_gps_time
                        num_samples = int(duration * sample_rate_hertz)
                        segments.append(ops.zeros((num_samples,), dtype="float32"))
                        gps_start_times.append(segment_start_gps_time)
            else:
                # No prefetch, acquire synchronously
                segment_times = valid_segments[current_idx]
                for ifo, (segment_start_gps_time, segment_end_gps_time) in zip(ifos, segment_times):
                    segment_key = f"segments/segment_{segment_start_gps_time}_{segment_end_gps_time}"
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
                        logging.warning(f"Segment missing for {ifo.name} at {segment_start_gps_time}. Filling with zeros.")
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
                
                # Start prefetching next segment before yielding
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

    def get_onsource_offsource_chunks(
            self,
            sample_rate_hertz : float,
            onsource_duration_seconds : float,
            padding_duration_seconds : float,
            offsource_duration_seconds : float,
            num_examples_per_batch : int = None,
            ifos : List[gf.IFO] = gf.IFO.L1,
            scale_factor : float = None,
            seed : int = None,
            sampling_mode : SamplingMode = SamplingMode.RANDOM
        ):

        if num_examples_per_batch is None:
            num_examples_per_batch = gf.Defaults.num_examples_per_batch
        if scale_factor is None:
            scale_factor = gf.Defaults.scale_factor
        if seed is None:
            seed = gf.Defaults.seed
        if self.rng is None:
            self.rng = default_rng(seed)
        # Ensure ifos are list:
        if not isinstance(ifos, list) and not isinstance(ifos, tuple):
            ifos = [ifos]
        
        # Padding is multiplied by 2 because it's two sided:
        total_padding_duration_seconds : float = padding_duration_seconds * 2.0
        
        # Total onsource duration includes padding:
        total_onsource_duration_seconds : float = \
            onsource_duration_seconds + total_padding_duration_seconds 
        
        # Calculate number of samples required to fullfill onsource and 
        # offsource durations:
        num_onsource_samples : int = ensure_even(int(total_onsource_duration_seconds * sample_rate_hertz))
        num_offsource_samples : int = ensure_even(int(offsource_duration_seconds * sample_rate_hertz))

        if self.acquisition_mode == AcquisitionMode.NOISE:
            # --- STANDARD NOISE MODE (RANDOM / GRID) ---
            
            # Early validation: check if valid_segments exists and has correct shape
            if self.valid_segments is None or len(self.valid_segments) == 0:
                logging.warning("No valid segments available in NOISE mode. Generator will be empty.")
                return
            
            # Check valid_segments has expected 3D shape [N, IFOs, 2]
            if not hasattr(self.valid_segments, 'ndim') or self.valid_segments.ndim != 3:
                logging.warning(f"valid_segments has unexpected shape: {getattr(self.valid_segments, 'shape', 'N/A')}. Expected 3D array. Generator will be empty.")
                return
            
            if not self._current_batch_index and not self._current_segment_index:
                min_segment_duration_seconds : int = \
                    total_onsource_duration_seconds + offsource_duration_seconds
                
                # Multiply by 2 for saftey odd things were happening
                min_segment_duration_seconds *= 2.0
                
                self.valid_segments_adjusted = self.remove_short_segments(
                        self.valid_segments, 
                        min_segment_duration_seconds
                    )
            
            # Check if we have any valid segments
            if self.valid_segments_adjusted is None or len(self.valid_segments_adjusted) == 0:
                logging.warning("No valid segments available in NOISE mode. Generator will be empty.")
                return

            # Track consecutive StopIteration to avoid infinite loop on empty segments
            _consecutive_stop_iterations = 0
            
            # Infinite loop to cycle through all segments
            while True:
                # Get a new segment when current one is exhausted
                while self._segment_exausted:
                    try:
                        self.current_segment = next(self.acquire(
                                sample_rate_hertz, 
                                self.valid_segments_adjusted, 
                                ifos,
                                scale_factor
                            ))
                        _consecutive_stop_iterations = 0  # Reset on success
                    except StopIteration:
                        _consecutive_stop_iterations += 1
                        # If we've been through twice with no data, segments are truly empty
                        if _consecutive_stop_iterations > 1:
                            logging.warning("No valid segments available. Generator exhausted.")
                            return
                        # Reset segment index to loop infinitely through segments
                        self._current_segment_index = 0
                        self._segment_exausted = True
                        continue
                
                    min_num_samples = min([ops.shape(tensor)[0] for tensor in self.current_segment.data])

                    if min_num_samples < (num_onsource_samples + num_offsource_samples):
                        logging.warning("Segment too short!")
                        self._segment_exausted = True
                    else: 
                        self._segment_exausted = False

                    min_num_samples = ops.cast(min_num_samples, "float32")

                    # Calculate number of batches current segment can produce
                    segment_duration : float = min_num_samples / sample_rate_hertz

                    self._num_batches_in_current_segment : int = max(1, int(
                              segment_duration * self._get_effective_saturation()
                            / (num_examples_per_batch * onsource_duration_seconds)
                        ))
                
                # Yield batches until current segment is exhausted
                while self._current_batch_index < self._num_batches_in_current_segment and not self._segment_exausted:

                    if sampling_mode == SamplingMode.RANDOM:
                        # Random subsections (training) - may overlap
                        subarrays, background_chunks, start_gps_times = self.current_segment.random_subsection(
                                num_onsource_samples, 
                                num_offsource_samples, 
                                num_examples_per_batch,
                                self.rng.integers(1E10)
                            )
                    else:
                        # GRID mode: sequential non-overlapping chunks (validation/FAR)
                        result = self.current_segment.grid_subsection(
                                num_onsource_samples, 
                                num_offsource_samples, 
                                num_examples_per_batch,
                                self._grid_position
                            )
                        subarrays, background_chunks, start_gps_times, new_grid_pos = result
                        if subarrays is not None:
                            self._grid_position = new_grid_pos

                    if subarrays is None or background_chunks is None or start_gps_times is None:
                        if subarrays is None:
                            logging.error("Subarrays returned None!")
                        if background_chunks is None:
                            logging.error("Background Chunks returned None!")
                        if start_gps_times is None:
                            logging.error("start_gps_times returned None!")
                        # For GRID mode, segment is exhausted if we can't fill a batch
                        if sampling_mode == SamplingMode.GRID:
                            self._segment_exausted = True
                            self._grid_position = 0
                        continue
                    
                    self._current_batch_index += 1
                    if not self._current_batch_index < self._num_batches_in_current_segment:
                        self._segment_exausted = True
                        self._current_batch_index = 0
                        if sampling_mode == SamplingMode.GRID:
                            self._grid_position = 0

                    yield self._apply_augmentation(subarrays), self._apply_augmentation(background_chunks), start_gps_times, None
                
                # Inner loop exited - segment exhausted, outer while True loops back to get new segment

        else:
            # --- TRANSIENT MODE ---
            # Each valid_segment is a padded event window. Aggregate into batches.
            self.valid_segments_adjusted = self.valid_segments
            
            segment_generator = self.acquire(
                sample_rate_hertz,
                self.valid_segments_adjusted,
                ifos,
                scale_factor
            )

            while True:
                batch_subarrays = []
                batch_backgrounds = []
                batch_gps_times = []
                
                while len(batch_subarrays) < num_examples_per_batch:
                    try:
                        event_segment = next(segment_generator)
                        
                    except StopIteration:
                         # Reset index to loop infinitely and recreate generator
                        self._current_segment_index = 0
                        segment_generator = self.acquire(
                            sample_rate_hertz,
                            self.valid_segments_adjusted,
                            ifos,
                            scale_factor
                        )
                        continue
                        
                    except Exception as e:
                        logging.error(f"Error acquiring feature segment: {e}")
                        continue
                    
                    # Extract onsource/offsource using helper
                    onsource, offsource, gps_time = self._extract_feature_event(
                        event_segment,
                        num_onsource_samples,
                        num_offsource_samples,
                        sample_rate_hertz
                    )
                    
                    if onsource is None:
                        continue
                    
                    batch_subarrays.append(onsource)
                    batch_backgrounds.append(offsource)
                    batch_gps_times.append(gps_time)
                
                # Stack batch: list of (IFO, Samples) -> (Batch, IFO, Samples)
                final_subarrays = ops.cast(ops.stack(batch_subarrays), "float32")
                final_background = ops.cast(ops.stack(batch_backgrounds), "float32")
                final_gps = ops.expand_dims(
                    ops.convert_to_tensor(batch_gps_times, dtype="float64"), 
                    axis=-1
                )
                
                # Look up labels for batch GPS times
                batch_labels = self._lookup_labels(batch_gps_times)
                
                yield self._apply_augmentation(final_subarrays, is_transient=True), self._apply_augmentation(final_background, is_transient=True), final_gps, batch_labels


    def clear_valid_segments(self) -> None:
        self.valid_segments = None
        self.valid_segments_adjusted = None
        self.ifos = None
        self._current_segment_index = 0
        self._current_batch_index = 0
        self._segment_exausted = True
        
    def get_valid_segments(
            self,
            ifos: List[gf.IFO],
            seed: int,
            groups: Dict[str, float] = None,
            group_name: str = "train",
            segment_order: SegmentOrder = None,
        ) -> List:

        # Ensure parameters are lists for consistency:
        if not isinstance(ifos, list) and not isinstance(ifos, tuple):
            ifos = [ifos]

        # If no segment_order requested use class attribute as default, defaults
        # to SegmentOrder.RANDOM:
        if not segment_order:
            segment_order = self.segment_order

        # If not groups dictionary input, resort to default test, train,
        # validate split:
        if not groups:
            groups = {
                    "train" : 0.98,
                    "validate" : 0.01,
                    "test" : 0.01
                }

        # Check to ensure group name is key in group dictionary:
        # Special case: 'all' bypasses group splitting and uses all segments
        if group_name == "all":
            groups = {"all": 1.0}  # Use 100% of segments
        elif group_name not in groups:
            raise KeyError(
                f"Group {group_name} not present in groups dictionary check "
                "input."
            )

        if self.valid_segments is None or len(self.valid_segments) != len(ifos):
            self.valid_segments = []

            # Check to see if noise with no features is desired data product, if
            # not extracting features is a very different process to randomly 
            # sampling from large noise vectors:
            if DataLabel.NOISE in self.data_labels:
                self.acquisition_mode = AcquisitionMode.NOISE
            else:
                self.acquisition_mode = AcquisitionMode.TRANSIENT
                self._segment_cache_maxsize = 128  # TRANSIENT uses smaller segments

            if self.acquisition_mode == AcquisitionMode.NOISE:
                for ifo in ifos:
                    # Get segments which fall within gps time boundaries and have the 
                    # requested ifo and state flag:
                    valid_segments = self.get_all_segment_times(ifo)

                    # First split by a constant duration so that groups always contain 
                    # the same times no matter what max duration is:
                    group_split_seconds: float = 8196.0

                    valid_segments: np.ndarray = \
                        self.cut_segments(
                            valid_segments, 
                            group_split_seconds,
                            self.start_gps_times[0]
                        )

                    # Distribute segments deterministically amongst groups, those can
                    # be used to separate validation and testing data from training 
                    # data:
                    valid_segments: np.ndarray = self.get_segments_for_group(
                        valid_segments, 
                        group_split_seconds, 
                        group_name, 
                        groups,
                        self.start_gps_times[0]
                    )

                    valid_segments, feature_times = self.remove_unwanted_segments(
                        ifo,
                        valid_segments
                    )

                    # Finally, split seconds so that max duration is no greater than 
                    # max:
                    valid_segments: np.ndarray = \
                        self.cut_segments(
                            valid_segments, 
                            self.max_segment_duration_seconds,
                            self.start_gps_times[0]
                        )

                    # If there are no valid segments raise an error:
                    if (len(valid_segments) == 0):
                        raise ValueError(f"IFO {ifo} has found no valid segments (NOISE mode)!")

                    self.valid_segments.append(valid_segments)
            
            else:
                # TRANSIENT MODE: Collect UNION of all events/features
                all_feature_segments_list = []
                
                # Check if feature_segments are already pre-populated (e.g. by TransientObtainer)
                if hasattr(self, 'feature_segments') and self.feature_segments is not None and len(self.feature_segments) > 0:
                    # Use overridden segments
                    all_feature_segments_list.append(self.feature_segments)
                else: 
                    padding = 32.0 # Default padding
                    
                    # Define global search window
                    global_start_gps = min(self.start_gps_times)
                    global_end_gps = max(self.end_gps_times)
    
                    # 1. EVENTS (Global)
                    if DataLabel.EVENTS in self.data_labels:
                        event_times = gf.get_event_times_by_type(self.event_types)
                        # Filter events within observing runs
                        event_times = [t for t in event_times if global_start_gps <= t <= global_end_gps]
                        
                        if len(event_times) > 0:
                            evt_segs = self.pad_gps_times_with_veto_window(
                                np.array(event_times), 
                                start_padding_seconds=padding, 
                                end_padding_seconds=padding
                            )
                            all_feature_segments_list.append(evt_segs)

                    # 2. GLITCHES (Per IFO)
                    if DataLabel.GLITCHES in self.data_labels:
                        for ifo in ifos:
                            try:
                                glitch_times = gf.get_glitch_times(
                                    ifo,
                                    start_gps_time=global_start_gps,
                                    end_gps_time=global_end_gps
                                )
                                if len(glitch_times) > 0:
                                    gl_segs = self.pad_gps_times_with_veto_window(
                                        np.array(glitch_times),
                                        start_padding_seconds=padding,
                                        end_padding_seconds=padding
                                    )
                                    all_feature_segments_list.append(gl_segs)
                            except Exception as e:
                                logging.warning(f"Failed to fetch glitches for {ifo}: {e}")
                                continue

                if not all_feature_segments_list:
                     raise ValueError("No features (Events/Glitches) found in requested range.")
                
                # Union and Unique
                combined = np.concatenate(all_feature_segments_list)
                unique_segments = np.unique(combined, axis=0) # Unique events
                
                # Order segments deterministically
                unique_segments = self.order_segments(unique_segments, segment_order, seed)
                
                # Store original transients for reference (before clustering)
                self.feature_segments = unique_segments # Store 2D version
                
                # Apply Grouping (Index-based) - BEFORE clustering
                # This ensures each group gets their fair share of original transients
                if group_name == 'all':
                    target_segments = unique_segments
                else:
                    n_segments = len(unique_segments)
                    total_weight = sum(groups.values())
                    acc_weight = 0.0
                    start_idx = 0
                    end_idx = 0
                    found = False
                    
                    # Iterate to find group range
                    for g_name, weight in sorted(groups.items()):
                        w = weight / total_weight
                        if g_name == group_name:
                            start_idx = int(acc_weight * n_segments)
                            end_idx = int((acc_weight + w) * n_segments)
                            found = True
                            break
                        acc_weight += w
                        
                    if found:
                        target_segments = unique_segments[start_idx:end_idx]
                    else:
                        target_segments = np.empty((0, 2))
                
                if len(target_segments) == 0:
                     logging.warning(f"No feature segments found for group {group_name}")
                
                # Apply temporal clustering AFTER group split
                # This merges nearby transients for efficient downloading
                target_segments = self._cluster_transients(target_segments)
                
                # Expand for Multi-IFO: (N, IFOs, 2)
                num_ifos = len(ifos)
                expanded = np.expand_dims(target_segments, axis=1) # (N, 1, 2)
                self.valid_segments = np.repeat(expanded, num_ifos, axis=1) # (N, IFOs, 2)

            match self.acquisition_mode:
                case AcquisitionMode.NOISE:
                    self.valid_segments = self.merge_bins(
                            self.valid_segments, 
                            self.max_segment_duration_seconds
                        )
                            
                    self.valid_segments = self.largest_segments_per_bin(
                            self.valid_segments
                        )
                            
                    self.valid_segments = np.swapaxes(self.valid_segments, 1, 0)
                    
                    # Order segments by requested order:
                    self.valid_segments = self.order_segments(
                        self.valid_segments, 
                        segment_order,
                        seed
                    )
                    
                    # Cache for future use without auth
                    self._cache_valid_segments(self.valid_segments, group_name)

                    return self.valid_segments
            
                case AcquisitionMode.TRANSIENT:
                    # For TRANSIENT mode, set up valid segments for iteration
                    if hasattr(self, 'feature_segments') and self.feature_segments is not None:
                        # Check if valid_segments is already correctly set up (3D, matching IFO count)
                        # If so, do not overwrite it with simple expansion of feature_segments
                        # My new logic above sets valid_segments to (N, IFOs, 2).
                        # Old logic below sets it to (N, 1, 2) which is bad for multi-IFO.
                        
                        # If valid_segments is not set correctly, then fallback to expansion:
                        needs_expansion = True
                        if (self.valid_segments is not None 
                            and isinstance(self.valid_segments, np.ndarray) 
                            and self.valid_segments.ndim == 3 
                            and self.valid_segments.shape[1] == len(ifos)):
                            needs_expansion = False
                            
                        if needs_expansion:
                            # Expand dims to match expected 3D format [N, num_ifos, 2]
                            # feature_segments is 2D [N, 2], need [N, 1, 2]
                            if self.feature_segments.ndim == 2:
                                self.valid_segments = np.expand_dims(self.feature_segments, axis=1)
                            else:
                                self.valid_segments = self.feature_segments
                        
                        num_features = len(self.feature_segments)
                        if num_features > 0:
                            logging.info(f"TRANSIENT MODE: {num_features} feature segments ready")
                        
                        self._cache_valid_segments(self.valid_segments, group_name)
                        return self.valid_segments
                    else:
                        raise ValueError(
                            "Feature segments not found. Ensure EVENTS or GLITCHES "
                            "are in data_labels for TRANSIENT mode."
                        )
