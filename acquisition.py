# Standard library imports
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
import hashlib
import random
from typing import List, Tuple, Union, Dict, Any
from pathlib import Path
import logging
import sys
from contextlib import closing

# Third-party imports
import numpy as np
import tensorflow as tf
from tensorflow.data.experimental import AutoShardPolicy
from gwdatafind import find_urls
from gwpy.segments import DataQualityDict
from gwpy.table import EventTable
from gwpy.timeseries import TimeSeries

from .setup import open_hdf5_file, ensure_directory_exists
from .maths import replace_nan_and_inf_with_zero
from .detector import IFO

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
    data               : Union[TimeSeries, tf.Tensor, np.ndarray]
    start_gps_time     : float
    sample_rate_hertz  : float
        
    def __post_init__(self):
        if (type(self.data) == TimeSeries):
            self.data = tf.convert_to_tensor(self.data.value, dtype=tf.float32)
        elif (type(self.data) == np.ndarray):
            self.data = tf.convert_to_tensor(self.data, dtype=tf.float32)
        
        self.data = replace_nan_and_inf_with_zero(self.data)
                    
        self.duration_seconds = \
            tf.cast(tf.shape(self.data)[0], tf.float32) / self.sample_rate_hertz
        self.time_interval_seconds = 1.0 / self.sample_rate_hertz
            
    def downsample(self, new_sample_rate_hertz: Union[int, float]):    
        #to impliment
        return self
    
    def scale(self, scale_factor:  Union[int, float]):
        self.data *= scale_factor
        return self
    
    def numpy(self):
        """Converts the data to a numpy array."""
        return self.data.numpy()
    
    def random_subsection(
        self,
        num_onsource_samples: int, 
        num_offsource_samples: int, 
        num_examples_per_batch: int
    ):      
        assert len(self.data.shape) == 1, "Input array must be 1D"

        N = tf.shape(self.data)[0]
        
        maxval = N.numpy() - num_onsource_samples - num_offsource_samples + 1
        
        random_starts = tf.random.uniform(
            shape=(num_examples_per_batch,), 
            minval=num_offsource_samples, 
            maxval=maxval, 
            dtype=tf.int32
        )

        def slice_data(start, num_samples):
            return tf.slice(self.data, [start], [num_samples])

        batch_subarrays = tf.map_fn(
            lambda start: slice_data(start, num_onsource_samples), 
            random_starts, 
            fn_output_signature=tf.TensorSpec(
                shape=[num_onsource_samples], dtype=tf.float32
            )
        )

        batch_background_chunks = tf.map_fn(
            lambda start: \
                slice_data(start - num_offsource_samples, num_offsource_samples), 
            random_starts, 
            fn_output_signature=tf.TensorSpec(
                shape=[num_offsource_samples], dtype=tf.float32)
        )

        subsections_start_gps_time = tf.cast(self.start_gps_time, tf.float32) + \
            tf.cast(random_starts, tf.float32) \
            * tf.cast(self.time_interval_seconds, tf.float32)

        return batch_subarrays, batch_background_chunks, subsections_start_gps_time
    
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
        data_directory_path : Path = Path("./")
        ) -> Path:
        
        # Generate unique segment filename from list of independent 
        # segment parameters:
        segment_parameters = \
            [
                self.frame_types, 
                self.channels, 
                self.state_flags, 
                self.data_labels, 
                self.max_segment_duration_seconds,
                sample_rate_hertz,
                group
            ]  
        
        # Ensure parameters are all strings so they can be hashed:
        segment_parameters = \
            [
                str(parameter) for parameter in segment_parameters
            ]
        
        # Generate the hash for the segment parameters:
        segment_hash = generate_hash_from_list(segment_parameters)
        
        # Ensure parent directory exists 
        ensure_directory_exists(data_directory_path)
        
        # Construct the segment filename using the hash
        self.file_path = \
            Path(data_directory_path) / f"segment_data_{segment_hash}.hdf5"
        
        return self.file_path
    
    def get_segment_times(
        self,
        start: float,
        stop: float,
        ifos: List[IFO],
        state_flag: str
    ) -> np.ndarray:
        
        if not isinstance(ifos, list):
            ifos = [ifos]
        
        segment_times = []
    
        for ifo in ifos:
            segments = \
                DataQualityDict.query_dqsegdb(
                    [f"{ifo.name}:{state_flag}"],
                    start,
                    stop,
                )

            intersection = segments[f"{ifo.name}:{state_flag}"].active.copy()
            
            segment_times.append(intersection)
        
        return np.array(intersection)
    
    def get_all_segment_times(
        self,
        ifos : List[IFO]
    ) -> np.ndarray:
        
        valid_segments = []
        for index, start_gps_time in enumerate(self.start_gps_times):         
            valid_segments.append(
                self.get_segment_times(
                    self.start_gps_times[index],
                    self.end_gps_times[index],
                    ifos,
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
        
    def get_valid_segments(
        self,
        ifos : List[IFO],
        groups : Dict[str, float] = None,
        group_name : str = "train",
        segment_order : SegmentOrder = None
    ):
        # Ensure parameters are lists for consistency:
        if not isinstance(ifos, list):
            ifos = [ifos]
        
        # If no segment_order requested use class atribute as default, defaults
        # to SegmentOrder.RANDOM:
        if not segment_order:
            segment_order = self.segment_order
        
        # If not groups dictionary input, resort to default test, train,
        # validate split: 
        if not groups:
            groups = \
                {
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
        
        # Get segments which fall within gps time boundaries and have the 
        # requested ifo and state flag:
        valid_segments = self.get_all_segment_times(ifos)
        
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
        
        # First split by a constant duration so that groups always contain the
        # same times no matter what max duration is:
        group_split_seconds : float = 8196.0
        valid_segments : np.ndarray = \
            self.split_segments(
                valid_segments, 
                group_split_seconds
            )
                
        # Distibute segments deterministically amongst groups, thos can
        # be used to separate validation and testing data from training data:
        valid_segments : np.ndarray = \
            self.distribute_segments_by_ratio(
                valid_segments, 
                groups,
                group_name
            )
        
        # Finally, split seconds so that max duration is no greateer than max:
        valid_segments : np.ndarray = \
            self.split_segments(
                valid_segments, 
                self.max_segment_duration_seconds
            )
        
        # If there are no valid segments raise and error:
        if (len(valid_segments) == 0):
            raise ValueError("No valid segments!")
            
        # Set class atribute:
        self.valid_segments = valid_segments
        
        # Order segments by requested order:
        self.order_segments(segment_order)
        
        return self.valid_segments
    
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
                
    def acquire(
        self,
        sample_rate_hertz : float,
        valid_segments : np.ndarray = None,
        ifos : List[IFO] = IFO.L1,
        scale_factor : float = 1.0
    ): 
        # Check if self.file_path is intitiated:
        if self.file_path is None:
            raise ValueError("""
            Segment file path not initulised. Ensure to run generate_file_path
            before attempting to load
            """)
        
        # If no valid segments inputted revert to default list:
        if valid_segments is None:
            valid_segments = self.valid_segments
                        
        for segment_start_gps_time, segment_end_gps_time in valid_segments:

            # Generate segment key to use to locate or save segment data within the
            # associated hdf5 file:
            segment_key = \
                f"segments/segment_{segment_start_gps_time}_{segment_end_gps_time}"

            # Acquire segment data, either from local stored file or remote:
            segment = \
                self.get_segment(
                    segment_start_gps_time,
                    segment_end_gps_time,
                    sample_rate_hertz,
                    ifos,
                    segment_key
                )

            if segment is not None:

                # Save aquired segment if it does not alread exist in the local file:
                if self.cache_segments:  
                    
                    with closing(open_hdf5_file(self.file_path, mode = "r+")) as segment_file:    

                        # Ensure hdf5 file has group "segments":
                        segment_file.require_group("segments")

                        if (segment_key not in segment_file) or self.force_acquisition:
                            segment_file.create_dataset(
                                segment_key, 
                                data = segment.data.numpy()
                            )

                # Scale to reduce precision errors:
                segment = segment.scale(scale_factor)  

                yield segment

            else:
                # If no segment was retrieved move to next loop iteration:
                continue
                
    def get_segment_data(
            self,
            segment_start_gps_time: float, 
            segment_end_gps_time: float, 
            ifos: List[IFO], 
            frame_type: str, 
            channel: str
        ) -> TimeSeries:

        """
        Fetches new segment data from specific URLs and reads it into a TimeSeries 
        object.

        The URLs are found using the provided segment start and end times, ifo, and 
        frame type. The TimeSeries data is then read from these files with the given 
        channel.

        Parameters
        ----------
        segment_start : int
            The start time of the segment.
        segment_end : int
            The end time of the segment.
        ifo : IFO
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
        
        for ifo in ifos:
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
                nproc=4
            )

            return data
    
    def get_segment(
            self,
            segment_start_gps_time : float,
            segment_end_gps_time : float,
            sample_rate_hertz : float,
            ifos : List[IFO],
            segment_key : str
        ) -> IFOData:
        
        # Default segment data to None in case of very possible aquisition error:
        segment = None
        expected_duration_seconds : float = \
                segment_end_gps_time - segment_start_gps_time
        
        with closing(open_hdf5_file(self.file_path, mode = "r")) as segment_file:    

            # Check if segment_key is present in segment file, and load if it
            # else acquire segment from database
            if (segment_key in segment_file) and not self.force_acquisition:
                self.logger.info(
                    f"Reading segments of duration "
                    f"{expected_duration_seconds}..."
                )
                segment : IFOData = \
                    IFOData(
                        segment_file[segment_key][()], 
                        segment_start_gps_time, 
                        sample_rate_hertz
                    )
            else: 
                self.logger.info(
                      "Acquiring segments of duration "
                      f"{expected_duration_seconds}..."
                )
                try:
                    segment : Timseries = \
                        self.get_segment_data(
                            segment_start_gps_time, 
                            segment_end_gps_time, 
                            ifos, 
                            self.frame_types[0], 
                            self.channels[0]
                    )
                except Exception as e:

                    # If any exception raised, skip segment
                    self.logger.error(
                        f"Unexpected error: {type(e).__name__}, {str(e)}"
                    )

                    segment = None

                if segment is not None:
                    # Resample segment using GwPy function
                    #Would be nice to do this on the gpu:
                    segment : TimeSeries = \
                        segment.resample(sample_rate_hertz)

                    # Convert to IFOData class which uses tf.Tensors
                    segment : IFOData = \
                        IFOData(
                            segment, 
                            segment.t0.value, 
                            segment.sample_rate.value
                        )

            self.logger.info("Complete!")

        return segment
    
    def get_onsource_offsource_chunks(
            self,
            sample_rate_hertz : float,
            onsource_duration_seconds : float,
            padding_duration_seconds : float,
            offsource_duration_seconds : float,
            num_examples_per_batch : int = 32,
            ifos : List[IFO] = IFO.L1,
            scale_factor : float = 1.0
        ) -> (tf.Tensor, tf.Tensor, tf.Tensor, int):
        
        # Ensure ifos are list:
        if not isinstance(ifos, list):
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
        
        # Remove short segments:
        valid_segments : np.ndarray = \
            self.remove_short_segments(
                self.valid_segments, 
                min_segment_duration_seconds
            )
        
        # Calculate number of samples required to fullfill onsource and offsource
        # durations:
        num_onsource_samples : int = \
            int(total_onsource_duration_seconds * sample_rate_hertz)
        num_offsource_samples : int = \
            int(offsource_duration_seconds * sample_rate_hertz)
        
        for segment in self.acquire(
                sample_rate_hertz, 
                valid_segments, 
                ifos,
                scale_factor
            ):
            
            if tf.shape(segment.data)[0] < (num_onsource_samples + num_offsource_samples):
                continue
                        
            # Calculate number of batches current segment can produce, this
            # is dependant on the segment duration and the onsource duration:
            num_batches_in_segment : int = \
                int(
                      segment.duration_seconds
                    / (
                        self.saturation * 
                        num_examples_per_batch * total_onsource_duration_seconds
                    )
                )
            
            # Yeild offsource, onsource, and gps_times for unique batches untill
            # current segment is exausted:
            for batch_index in range(num_batches_in_segment):
                
                yield segment.random_subsection(
                        num_onsource_samples, 
                        num_offsource_samples, 
                        num_examples_per_batch
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