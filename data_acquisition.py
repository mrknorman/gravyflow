# Standard library imports
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
import hashlib
import random
from typing import List, Tuple, Union, Dict, Any
from pathlib import Path

# Third-party imports
import numpy as np
import tensorflow as tf
from tensorflow.data.experimental import AutoShardPolicy
from gwdatafind import find_urls
from gwpy.segments import DataQualityDict
from gwpy.table import EventTable
from gwpy.timeseries import TimeSeries

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
    
class IFO(Enum):
    L1 = auto()
    H1 = auto()
    V1 = auto()
    
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

observing_run_data = {
    "O1" : ("O1", datetime(2015, 9, 12, 0, 0, 0), datetime(2016, 1, 19, 0, 0, 0),
     {DataQuality.BEST: "DCS-CALIB_STRAIN_CLEAN_C01"},
     {DataQuality.BEST: "HOFT_C01"},
     {DataQuality.BEST: "DCS-ANALYSIS_READY_C01:1"}),
    "O2" : ("O2", datetime(2016, 11, 30, 0, 0, 0), datetime(2017, 8, 25, 0, 0, 0),
     {DataQuality.BEST: "DCS-CALIB_STRAIN_CLEAN_C01"},
     {DataQuality.BEST: "HOFT_C01"},
     {DataQuality.BEST: "DCS-ANALYSIS_READY_C01:1"}),
    "O3" : ("O3", datetime(2019, 4, 1, 0, 0, 0), datetime(2020, 3, 27, 0, 0, 0),
     {DataQuality.BEST: "DCS-CALIB_STRAIN_CLEAN_C01"},
     {DataQuality.BEST: "HOFT_C01"},
     {DataQuality.BEST: "DCS-ANALYSIS_READY_C01:1"})
}

class ObservingRun(Enum):
    O1 = ObservingRunData(*observing_run_data["O1"])
    O2 = ObservingRunData(*observing_run_data["O2"])
    O3 = ObservingRunData(*observing_run_data["O3"])

@dataclass
class IFOData:
    data        : Union[TimeSeries, tf.Tensor, np.ndarray]
    t0          : float
    sample_rate : float
    
    def __post_init__(self):
        if (type(self.data) == TimeSeries):
            self.data = tf.convert_to_tensor(self.data.value, dtype=tf.float32)
        elif (type(self.data) == np.ndarray):
            self.data = tf.convert_to_tensor(self.data, dtype=tf.float32)
        
        self.data = replace_nan_and_inf_with_zero(self.data)
                    
        self.duration = \
            tf.cast(tf.shape(self.data)[0], tf.float32) / self.sample_rate
        self.dt = 1.0 / self.sample_rate
            
    def downsample(self, new_sample_rate: Union[int, float]):    
        #to impliment
        return self
    
    def scale(self, scale_factor:  Union[int, float]):
        self.data *= scale_factor
        return self
    
    def numpy(self):
        """Converts the data to a numpy array."""
        return self.data.numpy()

@dataclass
class IFODataConfig:
    start_gps_times : List[float]
    end_gps_times : List[float]
    frame_types : List[str]
    channels : List[str]
    state_flags : List[str]
    data_quality : DataQuality
    data_labels : List[DataLabel]
    ifos  : List[IFO]
    file_path : Path
    valid_segments : np.ndarray
    
    def __init__(
        self, 
        observing_runs : Union[ObservingRun, List[ObservingRun]],
        data_quality : DataQuality,
        data_labels : Union[DataLabel, List[DataLabel]],
        ifos : Union[IFO, List[IFO]],
        overrides : dict = None
        ):
        
        # Ensure parameters are lists for consistency:
        if not isinstance(observing_runs, list):
            observing_runs = [observing_runs]
        if not isinstance(data_labels, list):
            data_labels = [data_labels]
        if not isinstance(ifos, list):
            ifos = [ifos]
        
        #Set data quality:
        self.data_quality = data_quality
        
        # Set data labels:
        self.data_labels = data_labels
        
        # Set interferometers:
        self.ifos = ifos
            
        # Unpack parameters from input observing runs:
        self.unpack_observing_runs(observing_runs, data_quality)
        
        # Override observing run attributes if present:
        if overrides:
            self.override_attributes(overrides)
        
        # Set file name to none, will be set up if caching is requested
        self.file_path = None
                
    def override_attributes(
        self,
        overrides : Dict
    ) -> None:
        for key, value in overrides.items():    
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Invalide override value {key} not attribute of "
                    "IFODataConfig"
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
        
    def generate_file_path(
        self,
        max_segment_duration_seconds : float,
        sample_rate_hertz : float,
        data_directory : Path = Path("./")
        ) -> Path:
        
        # Generate unique segment filename from list of independent 
        # segment parameters:
        segment_parameters = \
            [
                self.frame_types, 
                self.channels, 
                self.state_flags, 
                self.data_labels, 
                max_segment_duration_seconds,
                sample_rate_hertz
            ]  
        
        # Ensure parameters are all strings so they can be hashed:
        segment_parameters = \
            [
                str(parameter) for parameter in segment_parameters
            ]
        
        # Generate the hash for the segment parameters:
        segment_hash = generate_hash_from_list(segment_parameters)
        
        # Construct the segment filename using the hash
        self.file_path = \
            Path(data_directory) / f"segment_data_{segment_hash}.hdf5"
        
        return self.file_path
    
    def get_segment_times(
        self,
        start: float,
        stop: float,
        ifo: IFO,
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
    
    def get_all_segment_times(self) -> np.ndarray:
        
        valid_segments = []
        for index, start_gps_time in enumerate(self.start_gps_times):         
            valid_segments.append(
                self.get_segment_times(
                    self.start_gps_times[index],
                    self.end_gps_times[index],
                    self.ifos[0],
                    self.state_flags[index]
                )
            )
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
        
    def get_valid_segments(
        self,
        max_segment_duration_seconds : float,
        min_segment_duration_seconds : float,
        groups : Dict[str, float] = \
        {
            "train" : 0.98,
            "validate" : 0.01,
            "test" : 0.01
        },
        group_name : str = "train",
        segment_order : SegmentOrder = SegmentOrder.RANDOM
    ):
        
        # Get segments which fall within gps time boundaries and have the 
        # requested ifo and state flag:
        valid_segments = self.get_all_segment_times()
        
        # Collect veto segment times from excluded data labels: 
        veto_segments = []
        if DataLabel.EVENTS not in self.data_labels:
            event_times = self.get_all_event_times()
            veto_segments.append(
                self.pad_gps_times_with_veto_window(event_times)
            )
        if DataLabel.GLITCHES not in self.data_labels:
            pass
            #veto_segments.append(get_all_glitch_segments(ifo))
        
        # Remove veto segment segments from valid segments list:
        if veto_segments:
            veto_segments = np.concatenate(veto_segments)
            valid_segments = \
                self.veto_time_segments(valid_segments, veto_segments)
        
        # Split seconds so that max duration is no greateer than max
        valid_segments = \
            self.split_segments(
                valid_segments, 
                max_segment_duration_seconds
            )
        
        # Remove segments which are shorter than than
        # (onsource_duration_seconds + window_duration_seconds) *
        # num_examples_per_batch + offsource duration seconds
        # This ensures that at least one batch with enough room for offsource
        # can be gathered:
        valid_segments = \
            self.remove_short_segments(
                valid_segments, 
                min_segment_duration_seconds
            )
        
        # Finally distibute segments deterministically amongst groups, thos can
        # be used to separate validation and testing data from training data:
        valid_segments =\
            self.distribute_segments_by_ratio(
                valid_segments, 
                groups,
                group_name
            )
        
        # If there are no valid segments raise and error:
        if (len(valid_segments) == 0):
            raise ValueError("No valid segments!")
            
        # Set class atribute:
        self.valid_segments = valid_segments
        
        # Order segments by requested order:
        self.order_segments(segment_order)
    
    def pad_gps_times_with_veto_window(
        self,
        gps_times: np.ndarray, 
        offset: int = 60, 
        increment: int = 10
    ) -> np.ndarray:
        
        left = gps_times - offset
        right = gps_times + increment
        result = np.stack((left, right), axis=1)
        
        return result
    
    def veto_time_segments(
        self,
        valid_segments: np.ndarray, 
        veto_segments: np.ndarray
        ) -> np.ndarray:

        valid_segments = self.compress_segments(valid_segments)
        veto_segments = self.compress_segments(veto_segments)
        result = \
            np.vstack([
                self.remove_overlap(valid_start, valid_end, veto_segments) 
                for valid_start, valid_end in valid_segments
            ])
        
        return result
    
    def split_segments(
        self,
        segments: np.ndarray, 
        maximum_duration_seconds: float
    ) -> np.ndarray:
        
        result = []
        for start, end in segments:
            n_splits = int(np.ceil((end - start) / maximum_duration_seconds))
            starts = np.linspace(
                start, 
                start + maximum_duration_seconds * (n_splits - 1), 
                n_splits
            )
            ends = np.minimum(starts + maximum_duration_seconds, end)
            result.append(np.vstack((starts, ends)).T)
        
        return np.vstack(result)

    def remove_short_segments(
        self,
        segments: np.ndarray, 
        minimum_duration_seconds: float
    ) -> np.ndarray:
        
        return segments[
            np.where(segments[:, 1] - segments[:, 0] >= minimum_duration_seconds)
        ]
    
    def compress_segments(
        self,
        segments: np.ndarray
    ) -> np.ndarray:
        
        segments = segments[segments[:,0].argsort()]
        compressed = []

        for segment in segments:
            if not compressed or compressed[-1][1] < segment[0]:
                compressed.append(segment)
            else:
                compressed[-1] = (
                    compressed[-1][0], max(compressed[-1][1], segment[1])
                )

        return np.array(compressed)
    
    def remove_overlap(
        self,
        start: float,
        end: float, 
        veto_segments: np.ndarray
        ) -> np.ndarray:

        result = np.array([[start, end]])
        for veto_start, veto_end in veto_segments:
            new_result = []
            for segment_start, segment_end in result:
                if segment_start < veto_start < segment_end \
                and segment_start < veto_end < segment_end:
                    new_result.append([segment_start, veto_start])
                    new_result.append([veto_end, segment_end])
                elif veto_start <= segment_start < veto_end < segment_end:
                    new_result.append([veto_end, segment_end])
                elif segment_start < veto_start < segment_end <= veto_end:
                    new_result.append([segment_start, veto_start])
                elif veto_end <= segment_start or segment_end <= veto_start:
                    new_result.append([segment_start, segment_end])
            result = np.array(new_result)
        return result

    def distribute_segments_by_ratio(
        self,
        segments: np.ndarray, 
        group_ratios: Dict[str, float],
        group_name : str
    ) -> np.ndarray:

        """
        Distribute segments into groups based on specified ratios.

        Parameters
        ----------
        segments : np.ndarray
            2D NumPy array of shape (N, 2) where N is the number of segments.
            Each row represents a segment with the first and second columns 
            being the start and end times, respectively.
        group_ratios : Dict[str, float]
            Dictionary with group names as keys and their corresponding ratios 
            as values.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with group names as keys and 2D NumPy arrays of segments 
            as values.

        """
        # Calculate total duration from the 2D NumPy array:
        total_duration = np.sum(segments[:, 1] - segments[:, 0])
        target_durations = \
            {
                group: total_duration * ratio \
                    for group, ratio in group_ratios.items()
            }

        # Initialize dictionaries to hold result and accumulated durations:
        result = defaultdict(list)
        accumulated_durations = {group: 0.0 for group in group_ratios.keys()}

        # Sort segments by start_time for representative sampling:
        sorted_segments = segments[np.argsort(segments[:, 0])]

        for segment in sorted_segments:
            start, end = segment
            segment_duration = end - start
            min_group = \
                min(
                    accumulated_durations, 
                    key=lambda k: accumulated_durations[k]/target_durations[k]
                )

            # Append this segment to the group with the least proportion of its 
            # target duration filled:
            result[min_group].append(segment)
            accumulated_durations[min_group] += segment_duration

        # Convert lists to 2D NumPy arrays before returning:
        return np.array(result[group_name])
    
    def order_segments(
        self,
        segment_order : SegmentOrder
    ):
        # Order segments by requested order:
        match segment_order:
            case SegmentOrder.RANDOM:
                # Shuffle data sements randomly.
                np.random.shuffle(self.valid_segments)
            case SegmentOrder.SHORTEST_FIRST:
                # Sort by shortest first (usefull for debugging).
                sort_by_duration = \
                    lambda segments: \
                        segments[np.argsort(segments[:, 1] - segments[:, 0])]
                valid_segments = sort_by_duration(self.valid_segments)
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