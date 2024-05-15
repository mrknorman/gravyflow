# Standard library imports:
import hashlib
import logging
import sys
import gc

from itertools import cycle
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from contextlib import closing
from typing import List, Tuple, Union, Dict, Any, Optional
from pathlib import Path

# Third-party imports:
import numpy as np
from numpy.random import default_rng  
import tensorflow as tf
import tensorflow_io as tfio

from gwdatafind import find_urls
from gwpy.segments import DataQualityDict
from gwpy.table import EventTable
from gwpy.timeseries import TimeSeries

# Local imports:
import gravyflow as gf

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
    FEATURES = auto()
    
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
        {DataQuality.BEST: "DCS-CALIB_STRAIN_CLEAN_C01"},
        {DataQuality.BEST: "HOFT_C01"},
        {DataQuality.BEST: "DCS-ANALYSIS_READY_C01:1"}
    ),
    "O2" : (
        "O2", 
        datetime(2016, 11, 30, 0, 0, 0), 
        datetime(2017, 8, 25, 0, 0, 0),
        {DataQuality.BEST: "DCS-CALIB_STRAIN_CLEAN_C01"},
        {DataQuality.BEST: "HOFT_C01"},
        {DataQuality.BEST: "DCS-ANALYSIS_READY_C01:1"}
    ),
    "O3" : (
        "O3", 
        datetime(2019, 4, 1, 0, 0, 0), 
        datetime(2020, 3, 27, 0, 0, 0),
        {DataQuality.BEST: "DCS-CALIB_STRAIN_CLEAN_C01"},
        {DataQuality.BEST: "HOFT_C01"},
        {DataQuality.BEST: "DCS-ANALYSIS_READY_C01:1"}
    )
}

class ObservingRun(Enum):
    O1 = ObservingRunData(*observing_run_data["O1"])
    O2 = ObservingRunData(*observing_run_data["O2"])
    O3 = ObservingRunData(*observing_run_data["O3"])

@tf.function(jit_compile=True)
def random_subsection_(
        tensor_data: tf.Tensor,
        num_examples_per_batch: int,
        num_onsource_samples: int,
        num_offsource_samples: int,
        time_interval_seconds: float,
        start_gps_time: float,
        seed: int
    ):

    """
    Generate random subsections from a tensor, along with corresponding background 
    chunks and start times. This function is designed for efficient extraction of 
    random tensor slices, suitable for batch processing in machine learning models.

    The function first calculates random start points within the tensor. It then 
    extracts subsections of data ('on-source') and background data ('off-source') 
    based on these start points. Additionally, it calculates the start times for 
    each subsection, considering the provided start GPS time and time interval.

    Parameters
    ----------
    tensor_data : tf.Tensor
        A TensorFlow tensor from which the subsections and background chunks will 
        be extracted.
    
    num_examples_per_batch : int
        The number of random subsections to extract per batch.

    num_onsource_samples : int
        The number of samples in each on-source subsection.

    num_offsource_samples : int
        The number of samples in each off-source (background) subsection.

    time_interval_seconds : float
        The time interval in seconds between samples in the tensor.

    start_gps_time : float
        The starting GPS time for the first sample in the tensor.

    seed : int
        An integer seed for random number generation to ensure reproducibility.

    Returns
    -------
    tuple
        A tuple containing three elements:
        1. batch_subarrays (tf.Tensor): A tensor of on-source subsections.
        2. batch_background_chunks (tf.Tensor): A tensor of off-source background 
        chunks.
        3. subsections_start_gps_time (tf.Tensor): A tensor of start GPS times for 
        each subsection.

    Notes
    -----
    - The function ensures that the random subsections and background chunks do 
    not exceed the bounds of the input tensor.
    - The random starts are calculated using a stateless uniform distribution for 
    reproducibility.
    - The function is decorated with @tf.function and jit_compile=True for performance 
    optimization, allowing TensorFlow to compile it into a high-performance graph.
    """

    # Cast input parameters to appropriate TensorFlow data types.
    seed_tensor = tf.cast(seed, tf.int32)
    time_interval_seconds = tf.cast(time_interval_seconds, tf.float32)
    start_gps_time = tf.cast(start_gps_time, tf.float32)

    # Determine the size of the input tensor.
    num_samples = tf.shape(tensor_data)[0]

    # Calculate the range within which to generate random starts.
    maxval = num_samples - num_onsource_samples - 16
    minval = num_offsource_samples

    # Generate uniformly distributed random start indices for each batch.
    random_starts_shape = (num_examples_per_batch,)
    random_starts = tf.random.stateless_uniform(
        shape=random_starts_shape,
        seed=seed_tensor,
        minval=minval,
        maxval=maxval,
        dtype=tf.int32
    )

    # Create a tensor representing a sequence from 0 to num_onsource_samples - 1.
    range_tensor = tf.range(num_onsource_samples)

    # Reshape range_tensor for broadcasting.
    range_tensor = range_tensor[tf.newaxis, :]

    # Reshape random_starts for broadcasting.
    random_starts_column = random_starts[:, tf.newaxis]

    # Calculate indices for onsource subarrays.
    indices_for_subarrays = random_starts_column + range_tensor

    # Gather subsections of onsource data using the calculated indices.
    batch_subarrays = tf.gather(tensor_data, indices_for_subarrays, axis=0)

    # Create and reshape a tensor for offsource range.
    range_tensor_offsource = tf.range(num_offsource_samples)
    range_tensor_offsource = range_tensor_offsource[tf.newaxis, :]

    # Calculate and reshape starting indices for offsource data.
    background_chunk_starts = random_starts - num_offsource_samples
    background_chunk_starts_column = background_chunk_starts[:, tf.newaxis]

    # Calculate indices for offsource background chunks.
    background_chunk_indices = background_chunk_starts_column + range_tensor_offsource

    # Gather subsections of offsource data using the calculated indices.
    batch_background_chunks = tf.gather(
        tensor_data, background_chunk_indices, axis=0
    )

    # Calculate the start times for each subsection.
    start_times = tf.cast(random_starts, tf.float32) * time_interval_seconds
    subsections_start_gps_time = start_gps_time + start_times

    return batch_subarrays, batch_background_chunks, subsections_start_gps_time

@dataclass
class IFOData:
    data: Union[List[TimeSeries], tf.Tensor, np.ndarray]
    sample_rate_hertz: float
    start_gps_time: List[float]

    def __post_init__(self):
        # Handle different types of input data for uniformity
        if isinstance(self.data, list):
            self.data = [gf.replace_nan_and_inf_with_zero(data)
                         for data in self.data]
        elif isinstance(self.data, np.ndarray):
            self.data = [tf.convert_to_tensor(self.data, dtype=tf.float32)]

        # Calculate the duration in seconds for each data item
        self.duration_seconds = [
            tf.cast(tf.shape(ifo_data)[0], tf.float32) / self.sample_rate_hertz
            for ifo_data in self.data
        ]

        # Set the time interval between samples
        self.time_interval_seconds = 1.0 / self.sample_rate_hertz

    def downsample(self, new_sample_rate_hertz: Union[int, float]):
        # To implement
        return self

    def scale(self, scale_factor: Union[int, float]):
        # Scale the data by the given factor
        self.data = [data * scale_factor for data in self.data]
        return self

    def numpy(self):
        # Converts the data to a numpy array, handling different data types
        return [data.numpy() if isinstance(data, tf.Tensor) else data
                for data in self.data]

    def random_subsection(
        self, 
        num_onsource_samples: int, 
        num_offsource_samples: int, 
        num_examples_per_batch: int,
        seed : int
    ):
        # Create random number generator from seed:
        # Create a random number generator with the provided seed
        rng = default_rng(seed)

        # Calculate parameters for random subsection extraction
        minval = num_offsource_samples
        min_tensor_size = num_onsource_samples + num_offsource_samples + 16

        try:
            all_batch_subarrays, all_batch_background_chunks, all_subsections_start_gps_time = (
                [], [], []
            )

            for tensor_data, start_gps_time in zip(self.data, self.start_gps_time):
                # Verify tensor dimensions and size
                num_samples = tf.shape(tensor_data)[0].numpy()
                maxval = num_samples - num_onsource_samples - 16
                self._validate_tensor_data(
                    tensor_data, 
                    num_samples, 
                    maxval, 
                    minval, 
                    min_tensor_size
                )
                
                # Extract random subsections
                time_interval_seconds = self.time_interval_seconds
                generated_seed = rng.integers(1E10, size=2)
                batch_subarrays, batch_background_chunks, subsections_start_gps_time = (
                    random_subsection_(
                        tensor_data,
                        num_examples_per_batch,
                        num_onsource_samples,
                        num_offsource_samples,
                        time_interval_seconds,
                        start_gps_time,
                        generated_seed
                    )
                )

                # Append expanded results
                all_batch_subarrays.append(
                    tf.expand_dims(batch_subarrays, 1)
                    )
                all_batch_background_chunks.append(
                    tf.expand_dims(batch_background_chunks, 1)
                )
                all_subsections_start_gps_time.append(
                    tf.expand_dims(subsections_start_gps_time, 1)
                )

            # Concatenate the batches
            return self._concatenate_batches(
                all_batch_subarrays, 
                all_batch_background_chunks, 
                all_subsections_start_gps_time
            )

        except Exception as e:
            print("Failed to get data because:", e)
            return None, None, None

    def _validate_tensor_data(
            self, 
            tensor_data : tf.Tensor, 
            num_samples : int, 
            maxval : int, 
            minval : int, 
            min_tensor_size : int
        ):
        """
        Validates the tensor data for suitability in random subsection extraction.

        Parameters
        ----------
        tensor_data : tf.Tensor
            The tensor data to be validated.
        N : int
            The size of the tensor data.
        maxval : int
            The maximum value for random start index generation.
        minval : int
            The minimum value for random start index generation.
        min_tensor_size : int
            The minimum required size of the tensor data.

        Raises
        ------
        ValueError
            If tensor data does not meet the requirements for processing.
        """
        if len(tensor_data.shape) != 1:
            raise ValueError(
                f"Input tensor must be 1D, got shape {tensor_data.shape}."
            )
        if num_samples < min_tensor_size:
            raise ValueError(
                (f"Input tensor too small ({num_samples}) for the requested samples"
                 f" and buffer {min_tensor_size}.")
            )
        if maxval <= minval:
            raise ValueError(
                (f"Invalid combination of onsource/offsource samples and buffer"
                 f" for the given data. {maxval} <= {minval}!")
            )

    def _concatenate_batches(self, subarrays, background_chunks, start_times):
        """
        Concatenates batches of subarrays, background chunks, and start times.

        Parameters
        ----------
        subarrays : List[tf.Tensor]
            List of batch subarrays.
        background_chunks : List[tf.Tensor]
            List of batch background chunks.
        start_times : List[tf.Tensor]
            List of start times for each batch.

        Returns
        -------
        tuple
            Concatenated batches of subarrays, background chunks, and start times.

        Raises
        ------
        ValueError
            If concatenation fails.
        """
        try:
            stacked_subarrays = tf.concat(subarrays, axis=1)
            stacked_background_chunks = tf.concat(background_chunks, axis=1)
            stacked_start_times = tf.concat(start_times, axis=1)
            return stacked_subarrays, stacked_background_chunks, stacked_start_times
        except:
            raise ValueError("Failed to stack arrays!")

    
@dataclass
class IFODataObtainer:
    
    def __init__(
            self, 
            observing_runs : Union[ObservingRun, List[ObservingRun]],
            data_quality : DataQuality,
            data_labels : Union[DataLabel, List[DataLabel]],
            segment_order : SegmentOrder = SegmentOrder.RANDOM,
            max_segment_duration_seconds : float = 2048.0,
            saturation : float = 1.0,
            force_acquisition : bool = False,
            cache_segments : bool = True,
            overrides : dict = None,
            logging_level : int = logging.WARNING
        ):
        
        # Initiate logging for ifo_data:
        self.logger = logging.getLogger("ifo_data_aquisition")
        stream_handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(stream_handler)
        self.logger.setLevel(logging_level)
        
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
            
        # Unpack parameters from input observing runs:
        self.unpack_observing_runs(observing_runs, data_quality)
        
        # Override observing run attributes if present:
        if overrides:
            self.override_attributes(overrides)
        
        # Set file name to none, will be set up if caching is requested
        self.file_path = None

        self.valid_segments = None
                
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
        
    def __del__(self):
        if self.segment_file is not None:
            self.segment_file.close()
            
    def close(self):
        if self.segment_file is not None:
            self.segment_file.close()
        
    def generate_file_path(
        self,
        sample_rate_hertz : float,
        group : str,
        data_directory_path : Optional[Path] = None
        ) -> Path:

        if data_directory_path is None:
            data_directory_path = gf.PATH.parent
        
        # Generate unique segment filename from list of independent 
        # segment parameters:
        segment_parameters = [
                self.frame_types, 
                self.channels, 
                self.state_flags, 
                self.data_labels, 
                self.max_segment_duration_seconds,
                sample_rate_hertz,
                group
            ]  
        
        # Ensure parameters are all strings so they can be hashed:
        segment_parameters = [
                str(parameter) for parameter in segment_parameters
            ]
        
        # Generate the hash for the segment parameters:
        segment_hash = generate_hash_from_list(segment_parameters)
        
        # Construct the segment filename using the hash
        self.file_path = Path(data_directory_path) / f"segment_data_{segment_hash}.hdf5"
        
        return self.file_path
    
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
        
        catalogues = \
            [
                "GWTC", 
                "GWTC-1-confident", 
                "GWTC-1-marginal", 
                "GWTC-2", 
                "GWTC-2.1-auxiliary", 
                "GWTC-2.1-confident", 
                "GWTC-2.1-marginal", 
                "GWTC-3-confident", 
                "GWTC-3-marginal"
            ]

        gps_times = np.array([])
        for catalogue in catalogues:
            events = EventTable.fetch_open_data(catalogue)
            gps_times = np.append(gps_times, events["GPS"].data.compressed())

        return gps_times
    
    def remove_unwanted_segments(
        self,
        ifo : gf.IFO,
        valid_segments : np.ndarray,
        get_times : List = None
        ):

        if get_times is None:
            get_times = []
        
        # Collect veto segment times from excluded data labels: 
        veto_segments = []
        
        if DataLabel.EVENTS in get_times or DataLabel.EVENTS not in self.data_labels:
            event_times = self.get_all_event_times()
        else:
            event_times = []

        if DataLabel.GLITCHES in get_times or DataLabel.GLITCHES not in self.data_labels:
            glitch_times = gf.get_glitch_times(
                ifo,
                start_gps_time = self.start_gps_times[0],
                end_gps_time = self.end_gps_times[0]
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
                    start_gps_time = self.start_gps_times[0],
                    end_gps_time = self.end_gps_times[0]
                )
            )

        # Remove veto segment segments from valid segments list:
        if veto_segments:
            veto_segments = np.concatenate(veto_segments)
            valid_segments = \
                self.veto_time_segments(valid_segments, veto_segments)
            
        feature_times = {
            gf.DataLabel.EVENTS : event_times,
            gf.DataLabel.GLITCHES : glitch_times    
        }
            
        return valid_segments, feature_times
    
    def find_segment_intersections(self, arr1, arr2):
        # Calculate the latest starts and earliest ends
        latest_starts = np.maximum(arr1[:, None, 0], arr2[None, :, 0])
        earliest_ends = np.minimum(arr1[:, None, 1], arr2[None, :, 1])

        # Compute the overlaps and their durations
        overlap_durations = np.clip(earliest_ends - latest_starts, 0, None)

        # Mask for actual overlaps
        overlap_mask = overlap_durations > 0

        # Since we want the best match, we select the maximum duration
        best_overlap_indices = np.argmax(overlap_durations, axis=-1)

        # Select the best overlaps for each interval in arr1
        starts = latest_starts[np.arange(latest_starts.shape[0]), best_overlap_indices]
        ends = earliest_ends[np.arange(earliest_ends.shape[0]), best_overlap_indices]

        # Filter out non-overlapping intervals
        valid_mask = overlap_mask[np.arange(overlap_mask.shape[0]), best_overlap_indices]
        starts = starts[valid_mask]
        ends = ends[valid_mask]

        # Combine starts and ends into a single array
        return np.vstack((starts, ends)).T
    
    def return_wanted_segments(
            self,
            ifo : gf.IFO,
            valid_segments : np.ndarray,        
            start_padding_seconds : float = 64.0,
            end_padding_seconds : float = 64.0,
        ):
        
        # Get feature times:
        event_times = self.get_all_event_times()          
        glitch_times = gf.get_glitch_times(
            ifo,
            start_gps_time = self.start_gps_times[0],
            end_gps_time = self.end_gps_times[0]
        )

        
        # Collect veto segment times from excluded data labels: 
        wanted_segments = []
        if DataLabel.EVENTS in self.data_labels:
            wanted_segments.append(
                self.pad_gps_times_with_veto_window(
                    event_times,
                    start_padding_seconds=start_padding_seconds,
                    end_padding_seconds=end_padding_seconds
                )
            )

        if DataLabel.GLITCHES in self.data_labels:
            wanted_segments.append(
                gf.get_glitch_segments(
                    ifo,
                    start_gps_time = self.start_gps_times[0],
                    end_gps_time = self.end_gps_times[0],
                    start_padding_seconds=start_padding_seconds,
                    end_padding_seconds=end_padding_seconds
                )
            )
            
        # Remove veto segment segments from valid segments list:
        if wanted_segments:
            wanted_segments = np.concatenate(wanted_segments)
            
            valid_segments = self.find_segment_intersections(
                valid_segments,
                wanted_segments
            )
        else:
            raise ValueError("Cannot find any features which suit requirement!")
            
        feature_times = {
            gf.DataLabel.EVENTS : event_times,
            gf.DataLabel.GLITCHES : glitch_times    
        }
        
        return cycle(valid_segments), feature_times
        
    def get_valid_segments(
        self,
        ifos : List[gf.IFO],
        seed : int,
        groups : Dict[str, float] = None,
        group_name : str = "train",
        segment_order : SegmentOrder = None,
    ) -> List:
                
        # Ensure parameters are lists for consistency:
        if not isinstance(ifos, list) and not isinstance(ifos, tuple):
            ifos = [ifos]
                
        # If no segment_order requested use class atribute as default, defaults
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
        if group_name not in groups:
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
                self.acquisition_mode = AcquisitionMode.FEATURES
            
            for ifo in ifos:
                
                # Get segments which fall within gps time boundaries and have the 
                # requested ifo and state flag:
                valid_segments = self.get_all_segment_times(ifo)
                
                # First split by a constant duration so that groups always contain 
                # the same times no matter what max duration is:
                group_split_seconds : float = 8196.0

                valid_segments : np.ndarray = \
                    self.cut_segments(
                        valid_segments, 
                        group_split_seconds,
                        self.start_gps_times[0]
                    )

                # Distibute segments deterministically amongst groups, thos can
                # be used to separate validation and testing data from training 
                # data:
                valid_segments : np.ndarray = self.get_segments_for_group(
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

                match self.acquisition_mode:
                    
                    case AcquisitionMode.NOISE:
                        
                        # Finally, split seconds so that max duration is no greater than 
                        # max:
                        valid_segments : np.ndarray = \
                            self.cut_segments(
                                valid_segments, 
                                self.max_segment_duration_seconds,
                                self.start_gps_times[0]
                            )
                        
                    case AcquisitionMode.FEATURES:
                        
                        # If in feature aquisition mode, get the times of feature
                        # segments:
                        feature_segments, feature_times = self.return_wanted_segments(
                            ifo,
                            valid_segments
                        )
                        
                        self.feature_segments = self.order_segments(
                            feature_segments,
                            segment_order,
                            seed
                        )
                
                # If there are no valid segments raise and error:
                if (len(valid_segments) == 0):
                    raise ValueError(f"IFO {ifo} has found no valid segments!")

                self.valid_segments.append(valid_segments)
            
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
        
        return self.valid_segments
    
    def pad_gps_times_with_veto_window(
        self,
        gps_times: np.ndarray, 
        start_padding_seconds: int = 60, 
        end_padding_seconds: int = 10
    ) -> np.ndarray:
        
        left = gps_times - start_padding_seconds
        right = gps_times + end_padding_seconds
        result = np.stack((left, right), axis=1)
        
        return result
    
    def cut_segments(self, segments, interval, start_period):
        """
        Splits time segments in each array at fixed intervals, aligned to a start period.

        Parameters:
        segments (np.ndarray): 
            2D array where each row contains a time segment (start, end).
        interval (float): 
            Time interval at which to make cuts.
        start_period (float):
            The reference start time for the first bin.

        Returns:
        np.ndarray: 2D array with split segments.
        """

        # Ensure all start points are calculated relative to the start_period
        relative_starts = segments[:, 0] - start_period
        start_points = np.ceil(relative_starts / interval) * interval + start_period
        end_points = segments[:, 1]
        
        cut_points = [
            np.arange(max(start, start_period), end, interval) 
            for start, end in zip(start_points, end_points)
        ]

        # Split segments based on calculated cut_points
        new_segments = []
        for (start, end), cuts in zip(segments, cut_points):
            if cuts.size > 0:
                # Ensure that the start of the first segment is not before start_period
                start = max(start, start_period)
                new_segs = np.column_stack(
                    (np.concatenate(([start], cuts)),
                    np.concatenate((cuts, [end])))
                )
                new_segments.extend(new_segs)
            else:
                # Handle the case where the segment does not need cutting
                new_segments.append([max(start, start_period), end])

        return np.array(new_segments)

    def remove_short_segments(
            self,
            segments: np.ndarray, 
            minimum_duration_seconds: float
        ) -> np.ndarray:
        
        """
        Removes columns where at least one of the durations in the column is 
        less than the specified minimum duration.

        Parameters:
        segments (np.ndarray): Input array of shape [X, N, 2].
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
    
    def find_overlaps(self, valid, veto):
        start = valid[0]
        end = valid[1]
        # Using broadcasting to find overlaps
        overlaps = (veto[:, 0] < end) & (veto[:, 1] > start)
        return overlaps

    def process_overlaps(self, valid_period, veto_periods):
        overlaps = self.find_overlaps(valid_period, veto_periods)
        overlapping_vetos = veto_periods[overlaps]

        if not overlapping_vetos.size:
            return [valid_period]
        
        periods = []
        start, end = valid_period

        for veto_start, veto_end in overlapping_vetos:
            if start >= veto_end or end <= veto_start:
                continue

            if start < veto_start:
                periods.append([start, veto_start])
            
            if end > veto_end:
                start = veto_end
            else:
                start = end
                break

        if start < end:
            periods.append([start, end])

        return periods

    def veto_time_segments(self, valid_periods, veto_periods):
        valid_periods = valid_periods[np.argsort(valid_periods[:, 0])]
        veto_periods = veto_periods[np.argsort(veto_periods[:, 0])]

        updated_valid_periods = []
        for valid_period in valid_periods:
            processed_periods = self.process_overlaps(valid_period, veto_periods)
            updated_valid_periods.extend(processed_periods)
        
        return np.array(updated_valid_periods)

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

    def get_segments_for_group(self, segments, interval, group_name, groups, start_period):
        # Compute the total number of bins
        total_time = max(segments[:,1]) - start_period
        total_bins = int(np.ceil(total_time / interval))

        # Create a mapping of bins to groups
        group_names = np.array(list(groups.keys()))
        group_proportions = np.array(list(groups.values()))
        group_thresholds = np.cumsum(group_proportions) * total_bins

        #Ensure group thresholds are integers
        group_thresholds = np.cumsum(group_proportions * total_bins).astype(int)

        # Make sure the last threshold equals the total number of bins to avoid creating extra bins
        group_thresholds[-1] = total_bins

        # Create a vector with group assignments for each bin
        bins_to_groups = np.repeat(group_names, np.diff(np.pad(group_thresholds, (1, 0), constant_values=0))).astype(str)

        # Shuffle bins deterministically
        rng = default_rng(0)  # Seed for reproducibility
        rng.shuffle(bins_to_groups)

        # Calculate the bin indices for each segment
        start_bins, end_bins = self.calculate_bin_indices(segments, interval, start_period)

        # Find segments whose bins fall into the specified group
        mask = (bins_to_groups[start_bins] == group_name) | (bins_to_groups[end_bins-1] == group_name)
        
        return segments[mask]
    
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
                # Sort by shortest first (usefull for debugging).
                sort_by_duration = lambda segments: segments[
                    np.argsort(segments[0][:, 1] - segments[0][:, 0])
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

        return np.array(largest_segments_list)
                
    def acquire(
        self,
        sample_rate_hertz : float = None,
        valid_segments : np.ndarray = None,
        ifos : List[gf.IFO] = gf.IFO.L1,
        scale_factor : float = 1.0
    ): 
        if sample_rate_hertz is None:
            sample_rate_hertz = gf.Defaults.sample_rate_hertz

        # Check if self.file_path is intitiated:
        if self.file_path is None:
            raise ValueError("""
            Segment file path not initulised. Ensure to run generate_file_path
            before attempting to load
            """)

        if self.cache_segments:
            # Ensure parent directory exists 
            gf.ensure_directory_exists(self.file_path)
        
        # If no valid segments inputted revert to default list:
        if valid_segments is None:
            valid_segments = self.valid_segments
        
        assert valid_segments.shape[1] == len(ifos), \
            "Num ifos should equal num segment lists"
        
        gps_start_times = []
                                
        for segment_times in valid_segments:
            
            segments = []
            for ifo, (segment_start_gps_time, segment_end_gps_time) in \
                zip(ifos, segment_times):

                # Generate segment key to use to locate or save segment data  
                # within the associated hdf5 file:
                segment_key = (f"segments/segment_{segment_start_gps_time}_"
                     "{segment_end_gps_time}")
                
                # Acquire segment data, either from local stored file or remote:
                segment = self.get_segment(
                        segment_start_gps_time,
                        segment_end_gps_time,
                        sample_rate_hertz,
                        ifo,
                        segment_key
                    )
                
                if segment is not None:

                    segments.append(segment)
                    gps_start_times.append(segment_start_gps_time)

                    # Save acquired segment if it does not already exist in the  
                    # local file:
                    if self.cache_segments:  

                        with closing(
                            gf.open_hdf5_file(
                                self.file_path, 
                                self.logger,
                                mode = "r+"
                            )
                        ) as segment_file:    

                            # Ensure hdf5 file has group "segments":
                            segment_file.require_group("segments")

                            if (segment_key not in segment_file) or \
                                self.force_acquisition:
                                segment_file.create_dataset(
                                    segment_key, 
                                    data = segment.numpy()
                                )
                else:
                    self.logger.error(
                        "No segment acquired, skipping to next iteration."
                    )

                    # If no segment was retrieved move to next loop iteration:
                    segments = None
                    break

            if segments is None:
                self.logger.error(
                    "No segments acquired, skipping to next iteration."
                )

                continue

            # Convert to IFOData class which uses tf.Tensors
            multi_segment : IFOData = IFOData(
                    segments, 
                    sample_rate_hertz,
                    gps_start_times
                )

            # Check segment integrity:
            # Check for empty input data:
            try:
                if not multi_segment.data:
                    raise ValueError("Input data should not be empty.")
                # Check for positive sample sizes and batch size
            except:
                continue
            
            # Scale to reduce precision errors:
            multi_segment = multi_segment.scale(scale_factor)  

            yield multi_segment
    
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

        The URLs are found using the provided segment start and end times, ifo, 
        and frame type. The TimeSeries data is then read from these files with 
        the given channel.

        Parameters
        ----------
        segment_start : int
            The start time of the segment.
        segment_end : int
            The end time of the segment.
        ifo : gf.IFO
            The Interferometric Gravitational-Wave Observatory (IFO) to use.
        frame_type : str
            The frame type to use.
        channel : str
            The channel to use.

        Returns
        -------
        TimeSeries
            The segment data read into a TimeSeries object.
        """

        files = find_urls(
            site=ifo.name.strip("1"),
            frametype=f"{ifo.name}_{frame_type}",
            gpsstart=segment_start_gps_time,
            gpsend=segment_end_gps_time,
            urltype="file",
        )
        data = TimeSeries.read(
            files, 
            channel=f"{ifo.name}:{channel}", 
            start=segment_start_gps_time, 
            end=segment_end_gps_time, 
            nproc=10
        )
        
        return data
    
    def get_segment(
            self,
            segment_start_gps_time : float,
            segment_end_gps_time : float,
            sample_rate_hertz : float,
            ifo : gf.IFO,
            segment_key : str
        ) -> tf.Tensor:

        epsilon = 0.1 
        # Shrink data collection window by epsilon because precision error occouring
        
        # Default segment data to None in case of very possible acquisition 
        # error:
        segment = None
        expected_duration_seconds : float = \
                segment_end_gps_time - segment_start_gps_time - 2*epsilon
        
        if Path(self.file_path).exists() or self.cache_segments:
            with closing(
                gf.open_hdf5_file(
                    self.file_path, 
                    self.logger,
                    mode = "r"
                )
            ) as segment_file:    

                # Check if segment_key is present in segment file, and load if it
                # else acquire segment from database

                if (segment_key in segment_file) and not self.force_acquisition:
                    self.logger.info(
                        f"Reading segments of duration "
                        f"{expected_duration_seconds}..."
                    )
                    segment = segment_file[segment_key][()]
                    
                    segment : tf.Tensor = tf.convert_to_tensor(
                        segment, 
                        dtype=tf.float32
                    )
                    
                    if gf.check_tensor_integrity(segment, 1, 10):
                        self.logger.info("Complete!")
                        return segment
                    
                    else:
                        self.logger.error("Segment integrity comprimised, skipping")
                        return None
                    
                else: 
                    segment = None
        
        if segment is None: 
            self.logger.info(
                    "Acquiring segments of duration "
                    f"{expected_duration_seconds}..."
            )
            try:

                # Added epsilon value to solve precision error:
                segment : TimeSeries = self.get_segment_data(
                    segment_start_gps_time + epsilon,
                    segment_end_gps_time - epsilon, 
                    ifo, 
                    self.frame_types[0], 
                    self.channels[0]
                )
                
                segment.resample(sample_rate_hertz)

            except Exception as e:
                # If any exception raised, skip segment
                self.logger.error(
                    f"Unexpected error: {type(e).__name__}, {str(e)}"
                )

                segment = None
            
            if segment is not None:

                segment : tf.Tensor = tf.convert_to_tensor(
                    segment.value, 
                    dtype=tf.float32
                )
                
                if gf.check_tensor_integrity(segment, 1, 10):
                    self.logger.info("Complete!")
                    return segment
                
                else:
                    self.logger.error("Segment integrity comprimised, skipping")
                    return None
            
            else:
                self.logger.error(
                    f"Segment is none for some reason, skipping."
                )         
                return None
    
    def get_onsource_offsource_chunks(
            self,
            sample_rate_hertz : float,
            onsource_duration_seconds : float,
            padding_duration_seconds : float,
            offsource_duration_seconds : float,
            num_examples_per_batch : int = None,
            ifos : List[gf.IFO] = gf.IFO.L1,
            scale_factor : float = None,
            seed : int = None
        ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, int]:

        if num_examples_per_batch is None:
            num_examples_per_batch = gf.Defaults.num_examples_per_batch
        if scale_factor is None:
            scale_factor = gf.Defaults.scale_factor
        if seed is None:
            seed = gf.Defaults.seed
        
        # Ensure ifos are list:
        if not isinstance(ifos, list) and not isinstance(ifos, tuple):
            ifos = [ifos]
        
        # Padding is multiplied by 2 because it's two sided:
        total_padding_duration_seconds : float = padding_duration_seconds * 2.0
        
        # Total onsource duration includes padding:
        total_onsource_duration_seconds : float = \
            onsource_duration_seconds + total_padding_duration_seconds 
        
        # Remove segments which are shorter than than
        # (onsource_duration_seconds + padding_duration_seconds * 2.0) *
        # num_examples_per_batch + offsource_duration_seconds
        # This ensures that at least one batch with enough room for offsource
        # can be gathered:
        min_segment_duration_seconds : int = \
            (total_onsource_duration_seconds) \
            * num_examples_per_batch + offsource_duration_seconds
        
        # Multiply by 2 for saftey odd things were happening
        min_segment_duration_seconds *= 2.0
        
        valid_segments = self.remove_short_segments(
                self.valid_segments, 
                min_segment_duration_seconds
            )
        
        # Calculate number of samples required to fullfill onsource and 
        # offsource durations:
        num_onsource_samples : int = ensure_even(int(total_onsource_duration_seconds * sample_rate_hertz))
        num_offsource_samples : int = ensure_even(int(offsource_duration_seconds * sample_rate_hertz))
        
        for segment in self.acquire(
                sample_rate_hertz, 
                valid_segments, 
                ifos,
                scale_factor
            ):

            min_num_samples = min([tf.shape(tensor)[0] for tensor in segment.data])

            if min_num_samples < (num_onsource_samples + num_offsource_samples):
                print("Segment too short!")
                continue
                
            min_num_samples = tf.cast(min_num_samples, tf.float32)

            # Calculate number of batches current segment can produce, this
            # is dependant on the segment duration and the onsource duration:

            segment_duration : float = min_num_samples / sample_rate_hertz

            num_batches_in_segment : int = int(
                      segment_duration 
                    / (
                        self.saturation * 
                        num_examples_per_batch * onsource_duration_seconds
                    )
                )
            
            # Yield offsource, onsource, and gps_times for unique batches untill
            # current segment is exausted:
            for batch_index in range(num_batches_in_segment):

                subarrays, background_chunks, start_gps_times = segment.random_subsection(
                        num_onsource_samples, 
                        num_offsource_samples, 
                        num_examples_per_batch,
                        seed
                    )

                if subarrays is None or background_chunks is None or start_gps_times is None:
                    
                    if subarrays is None:
                        print("Subarrays returned None!")
                    if background_chunks is None:
                        print("Background Chunks returned None!")
                    if start_gps_times is None:
                        print("start_gps_times returned None!")
                    continue
                
                yield subarrays, background_chunks, start_gps_times

def generate_hash_from_list(input_list: List[Any]) -> str:
    """
    Generate a unique hash based on the input list.

    The function creates a SHA-1 hash from the string representation of the 
    input list.

    Parameters
    ----------
    input_list : List[Any]
        The input list to be hashed.

    Returns
    -------
    str
        The SHA-1 hash of the input list.

    """
    
    # Convert the list to a string:
    input_string = str(input_list)  
    # Generate a SHA-1 hash from the string
    input_hash = hashlib.sha1(input_string.encode()).hexdigest()  

    return input_hash