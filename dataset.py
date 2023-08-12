from dataclasses import dataclass
from typing import List, Tuple, Union, Dict, Any
from datetime import datetime

import numpy as np
import hashlib
import tensorflow as tf

from gwdatafind import find_urls
from gwpy.segments import DataQualityDict
from gwpy.timeseries import TimeSeries
from gwpy.table import EventTable

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from itertools import cycle

from .cuphenom.py.cuphenom import generate_phenom_d
from .whiten import whiten
from .snr    import scale_to_snr
from .setup  import randomise_arguments, randomise_dict

import h5py

@dataclass
class ObservingRun:
    name: str
    start_date_time: datetime
    end_date_time: datetime
    channels: dict
    frame_types: dict
    state_flags: dict

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

IFOS = ("H1", "L1", "V1")

observing_run_data = (
    ("O1", datetime(2015, 9, 12, 0, 0, 0), datetime(2016, 1, 19, 0, 0, 0),
     {"best": "DCS-CALIB_STRAIN_CLEAN_C01"},
     {"best": "HOFT_C01"},
     {"best": "DCS-ANALYSIS_READY_C01:1"}),
    ("O2", datetime(2016, 11, 30, 0, 0, 0), datetime(2017, 8, 25, 0, 0, 0),
     {"best": "DCS-CALIB_STRAIN_CLEAN_C01"},
     {"best": "HOFT_C01"},
     {"best": "DCS-ANALYSIS_READY_C01:1"}),
    ("O3", datetime(2019, 4, 1, 0, 0, 0), datetime(2020, 3, 27, 0, 0, 0),
     {"best": "DCS-CALIB_STRAIN_CLEAN_C01"},
     {"best": "HOFT_C01"},
     {"best": "DCS-ANALYSIS_READY_C01:1"})
)

OBSERVING_RUNS = {
    name: ObservingRun(
        name, start_date_time, end_date_time, channels, frame_types, state_flags
    ) for name, start_date_time, end_date_time, channels, frame_types, state_flags in observing_run_data}

O1 = OBSERVING_RUNS["O1"]
O2 = OBSERVING_RUNS["O2"]
O3 = OBSERVING_RUNS["O3"]  

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

@tf.function
def random_subsection(
        data: tf.Tensor,
        dt  : float,
        t0  : float,
        num_onsource_samples: int, 
        num_offsource_samples: int, 
        num_examples_per_batch: int
    ):      
        assert len(data.shape) == 1, "Input array must be 1D"

        N = tf.shape(data)[0]

        maxval = N - num_onsource_samples - num_offsource_samples + 1
        random_starts = tf.random.uniform(
            shape=(num_examples_per_batch,), 
            minval=num_offsource_samples, 
            maxval=maxval, 
            dtype=tf.int32
        )

        def slice_data(start, num_samples):
            return tf.slice(data, [start], [num_samples])

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

        t0_subsections = tf.cast(t0, tf.float32) + \
            tf.cast(random_starts, tf.float32) * tf.cast(dt, tf.float32)

        return batch_subarrays, batch_background_chunks, t0_subsections

@tf.function
def replace_nan_and_inf_with_zero(tensor):
    tensor = tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)
    tensor = tf.where(tf.math.is_inf(tensor), tf.zeros_like(tensor), tensor)
    return tensor    

def get_segment_times(
    start: float,
    stop: float,
    ifo: str,
    state_flag: str
    ) -> list:
    
    segments = DataQualityDict.query_dqsegdb(
        [f"{ifo}:{state_flag}"],
        start,
        stop,
    )
    
    intersection = segments[f"{ifo}:{state_flag}"].active.copy()
    
    return np.array(intersection)
    
def get_all_event_times() -> np.ndarray:
    
    catalogues = [
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

def pad_gps_times_with_veto_window(
        arr: np.ndarray, 
        offset: int = 60, 
        increment: int = 10
    ) -> np.ndarray:
    left = arr - offset
    right = arr + increment
    result = np.stack((left, right), axis=1)
    return result

def compress_periods(periods: np.ndarray) -> np.ndarray:
    periods = periods[periods[:,0].argsort()]
    compressed = []

    for period in periods:
        if not compressed or compressed[-1][1] < period[0]:
            compressed.append(period)
        else:
            compressed[-1] = (
                compressed[-1][0], max(compressed[-1][1], period[1])
            )

    return np.array(compressed)

def remove_overlap(
    start: float,
    end: float, 
    veto_periods: np.ndarray
    ) -> np.ndarray:
    
    result = np.array([[start, end]])
    for veto_start, veto_end in veto_periods:
        new_result = []
        for period_start, period_end in result:
            if period_start < veto_start < period_end \
            and period_start < veto_end < period_end:
                new_result.append([period_start, veto_start])
                new_result.append([veto_end, period_end])
            elif veto_start <= period_start < veto_end < period_end:
                new_result.append([veto_end, period_end])
            elif period_start < veto_start < period_end <= veto_end:
                new_result.append([period_start, veto_start])
            elif veto_end <= period_start or period_end <= veto_start:
                new_result.append([period_start, period_end])
        result = np.array(new_result)
    return result

def veto_time_periods(
    valid_periods: np.ndarray, 
    veto_periods: np.ndarray
    ) -> np.ndarray:
    
    valid_periods = compress_periods(valid_periods)
    veto_periods = compress_periods(veto_periods)
    result = np.vstack([
        remove_overlap(valid_start, valid_end, veto_periods) 
        for valid_start, valid_end in valid_periods
    ])
    return result

def split_periods(periods: np.ndarray, max_length: float) -> np.ndarray:
    result = []
    for start, end in periods:
        n_splits = int(np.ceil((end - start) / max_length))
        starts = np.linspace(
            start, 
            start + max_length * (n_splits - 1), 
            n_splits
        )
        ends = np.minimum(starts + max_length, end)
        result.append(np.vstack((starts, ends)).T)
    return np.vstack(result)

def remove_short_periods(periods: np.ndarray, min_length: float) -> np.ndarray:
    return periods[np.where(periods[:, 1] - periods[:, 0] >= min_length)]

def open_hdf5_file(
    file_path : Union[str, Path], 
    mode : str ='r+'
    ) -> h5py.File:
    
    file_path = Path(file_path)
    try:
        # Try to open the HDF5 file in the specified mode
        f = h5py.File(file_path, mode)
        f.close()
    except OSError:
        # The file does not exist, so create it in write mode
        f = h5py.File(file_path, 'w')
        f.close()
        print(f'The file {file_path} was created in write mode.')
    else:
        print(f'The file {file_path} was opened in {mode} mode.')
    return h5py.File(file_path, mode)

def ensure_directory_exists(
    directory: Union[str, Path]
    ):
    
    directory = Path(directory)  # Convert to Path if not already
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

@tf.function
def roll_vector_zero_padding_(vector, min_roll, max_roll):
    roll_amount = tf.random.uniform(
        shape=(), minval=min_roll, maxval=max_roll, dtype=tf.int32
    )

    # Create a zero vector of the same size as the input
    zeros = tf.zeros_like(vector)

    # Create the rolled vector by concatenating the sliced vector and zeros
    rolled_vector = tf.concat(
        [vector[roll_amount:], zeros[:roll_amount]], axis=-1
    )

    return rolled_vector

@tf.function
def roll_vector_zero_padding(tensor, min_roll, max_roll):
    return tf.map_fn(
        lambda vec: roll_vector_zero_padding_(vec, min_roll, max_roll), tensor
    )

def generate_injections(
    config: Dict[str, Any],
    num_injections: int, 
    common_args: dict,
    fduration: float,
    sample_rate_hertz: float,
    onsource_duration_seconds: float,
    num_examples_per_batch: int
    ):

    config["args"].update(common_args)
    injection_chance = config["injection_chance"]

    injections = []
    injection_masks = []
    snrs = []
    
    injection_masks = np.random.choice(
        [False, True], 
        size=num_injections, 
        p=[1-injection_chance, injection_chance]
    )
        
    num_waveforms = np.sum(injection_masks)
        
    config["args"].update({
        "num_waveforms":
            {
            "value": num_waveforms, 
            "distribution_type": "constant",
            "dtype" : "int"
            }
        })
    
    length_one_args = ['num_waveforms', 'sample_rate_hertz', 'duration_seconds']
    length_three_args = ['spin_1_in', 'spin_2_in']
    
    # Update dictionary to create random variables for each waveform:
    for key, value in config["args"].items():
        if key not in length_one_args + length_three_args:    
            value.update({"num_values": num_waveforms})
            config["args"].update({key: value})
    
    num_spin_components = 3
    # Spins have three times more variables:
    for key, value in config["args"].items():
        if key in length_three_args:
            value.update({"num_values": num_waveforms*num_spin_components})
            config["args"].update({key: value})
                    
    injections, injection_parameters = \
        randomise_arguments(
            config["args"], generate_phenom_d
        )
    
    # At the moment take only one polarization: 
    injections = injections[:, 0]
        
    # Reshape to split injections:
    num_samples = int(sample_rate_hertz*(onsource_duration_seconds + fduration))
    injections = injections.reshape(-1, num_samples)
    
    # Arbitrary scale factor:
    injections *= 10.0E20 
        
    for i in range(np.sum(injection_masks)):
        if type(config["snr"]) == np.ndarray:  
            snr = config["snr"][-1]
            if i < len(config["snr"]):
                snr = config["snr"][i]
        elif type(config["snr"]) == dict:
            snr = randomise_dict(config["snr"])
        else:
            raise ValueError("Unsupported SNR type!") 

        snrs.append(snr)         

    crop_duration = fduration / 2.0

    min_roll_num_samples = \
        int((crop_duration \
             + config["padding_seconds"]["back"])*sample_rate_hertz) 
    max_roll_num_samples = \
        int((onsource_duration_seconds + fduration) *sample_rate_hertz) - \
        int((crop_duration \
             + config["padding_seconds"]["front"])*sample_rate_hertz) 
        
    injections = tf.convert_to_tensor(injections, dtype = tf.float32)
    injection_masks = tf.convert_to_tensor(injection_masks, dtype = tf.bool)
    snrs = tf.convert_to_tensor(snrs, dtype = tf.float32)
            
    injections = \
        roll_vector_zero_padding( 
            injections, 
            min_roll_num_samples, 
            max_roll_num_samples
        )
    
    injections = \
        expand_tensor(
            injections, 
            injection_masks
        )
    
    snrs = \
        expand_tensor(
            snrs,
            injection_masks
        )
    
    injections = batch_tensor(injections, num_examples_per_batch)
    injection_masks = batch_tensor(injection_masks, num_examples_per_batch)
    snrs = batch_tensor(snrs, num_examples_per_batch)
    
    return injections, injection_masks, snrs

@tf.function
def expand_tensor(signal: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """
    This function expands a tensor along the X axis by inserting zeros wherever a 
    corresponding boolean in a 1D tensor is False, and elements from the original 
    tensor where the boolean is True. It works for both 1D and 2D tensors.

    Parameters
    ----------
    signal : tf.Tensor
        A 1D or 2D tensor representing signal injections, where the length of 
        the tensor's first dimension equals the number of True values in the mask.
    mask : tf.Tensor
        A 1D boolean tensor. Its length will determine the length of the expanded tensor.

    Returns
    -------
    tf.Tensor
        The expanded tensor.

    """
    # Ensure that the signal tensor is 1D or 2D
    assert signal.ndim in (1, 2), 'Signal must be a 1D or 2D tensor'
    
    # Ensure that the mask is 1D
    assert mask.ndim == 1, 'Mask must be a 1D tensor'
    
    # Ensure that the length of the signal tensor matches the number of True 
    # values in the mask
    assert tf.reduce_sum(tf.cast(mask, tf.int32)) == signal.shape[0], \
        'Signal tensor length must match number of True values in mask'
    
    # Create a tensor full of zeros with the final shape
    if signal.ndim == 1:
        expanded_signal = tf.zeros(mask.shape[0], dtype=signal.dtype)
    else: # signal.ndim == 2
        N = signal.shape[1]
        expanded_signal = tf.zeros((mask.shape[0], N), dtype=signal.dtype)

    # Get the indices where mask is True
    indices = tf.where(mask)

    # Scatter the signal tensor elements/rows into the expanded_signal tensor at the 
    # positions where mask is True
    expanded_signal = tf.tensor_scatter_nd_update(expanded_signal, indices, signal)

    return expanded_signal

def expand_tensor(signal: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """
    This function expands a tensor along the X axis by inserting zeros wherever a 
    corresponding boolean in a 1D tensor is False, and elements from the original 
    tensor where the boolean is True. It works for both 1D and 2D tensors.

    Parameters
    ----------
    signal : tf.Tensor
        A 1D or 2D tensor representing signal injections, where the length of 
        the tensor's first dimension equals the number of True values in the mask.
    mask : tf.Tensor
        A 1D boolean tensor. Its length will determine the length of the expanded tensor.

    Returns
    -------
    tf.Tensor
        The expanded tensor.

    """
    # Ensure that the signal tensor is 1D or 2D
    assert signal.ndim in (1, 2), 'Signal must be a 1D or 2D tensor'
    
    # Ensure that the mask is 1D
    assert mask.ndim == 1, 'Mask must be a 1D tensor'
    
    # Ensure that the length of the signal tensor matches the number of True 
    # values in the mask
    assert tf.reduce_sum(tf.cast(mask, tf.int32)) == signal.shape[0], \
        'Signal tensor length must match number of True values in mask'
    
    # Create a tensor full of zeros with the final shape
    if signal.ndim == 1:
        expanded_signal = tf.zeros(mask.shape[0], dtype=signal.dtype)
    else: # signal.ndim == 2
        N = signal.shape[1]
        expanded_signal = tf.zeros((mask.shape[0], N), dtype=signal.dtype)

    # Get the indices where mask is True
    indices = tf.where(mask)

    # Scatter the signal tensor elements/rows into the expanded_signal tensor at the 
    # positions where mask is True
    expanded_signal = tf.tensor_scatter_nd_update(expanded_signal, indices, signal)

    return expanded_signal

def process_time_interval(
    time_interval: Union[Tuple, 'ObservingRun'], 
    frame_type: str = None, 
    channel: str = None, 
    state_flag: str = None, 
    data_quality: str = None
    ) -> Tuple[str, str, str]:
    """
    Processes the given time interval, and return the appropriate start, 
    stop time, frame_type, channel, and state_flag.

    If time_interval is an instance of ObservingRun, then it uses the properties 
    of the ObservingRun object. If not, it expects time_interval to be a tuple 
    representing the start and stop times.

    Parameters
    ----------
    time_interval : Union[Tuple, 'ObservingRun']
        The time interval to be processed. It can be either a tuple or an 
        ObservingRun object.
    frame_type : str, optional
        The frame type to use. If None, it will be derived from the 
        time_interval if it's an ObservingRun object.
    channel : str, optional
        The channel to use. If None, it will be derived from the time_interval 
        if it's an ObservingRun object.
    state_flag : str, optional
        The state flag to use. If None, it will be derived from the 
        time_interval if it's an ObservingRun object.
    data_quality : str, optional
        The data quality to use to derive frame_type, channel, and state_flag 
        if they are None and time_interval is an ObservingRun object.

    Returns
    -------
    Tuple[str, str, str]
        The final values of frame_type, channel, and state_flag.

    Raises
    ------
    TypeError
        If the time_interval is not a tuple or an ObservingRun object.
    """
    
    # Check the type of time_interval
    if isinstance(time_interval, tuple):
        start, stop = time_interval
    elif isinstance(time_interval, ObservingRun):
        start, stop = time_interval.start_gps_time, time_interval.end_gps_time    

        # Set the frame_type, channel, and state_flag if not provided, based on 
        # the ObservingRun object
        if frame_type is None:
            frame_type = time_interval.frame_types[data_quality]
        if channel is None:
            channel = time_interval.channels[data_quality]
        if state_flag is None:
            state_flag = time_interval.state_flags[data_quality]
    else:
        raise TypeError(
            "time_interval must be either a tuple or an ObservingRun object"
        )
    
    return start, stop, frame_type, channel, state_flag

def process_valid_segments(
    start: str, 
    stop: str, 
    ifo: str, 
    state_flag: str, 
    data_labels: List[str], 
    max_segment_size: int, 
    onsource_duration_seconds: int, 
    fduration: int, 
    num_examples_per_batch: int, 
    offsource_duarion_seconds: int,
    order: str
    ) -> Union[np.ndarray, cycle]:
    """
    Process the valid segments given the start, stop times, ifo, and state flag. 
    This involves multiple steps including getting segment times, vetoing 
    certain segments, and adjusting segment times.

    Parameters
    ----------
    start : str
        The start time.
    stop : str
        The stop time.
    ifo : str
        The IFO (Interferometric Gravitational-Wave Observatory) to use.
    state_flag : str
        The state flag to use.
    data_labels : List[str]
        The list of data labels.
    max_segment_size : int
        The maximum segment size.
    onsource_duration_seconds : int
        The example duration in seconds.
    fduration : int
        The fduration to use.
    num_examples_per_batch : int
        The number of examples per batch.
    offsource_duarion_seconds : int
        The background duration in seconds.
    order: str
        How to order the segments.
        
    Returns
    -------
    Union[np.ndarray, cycle]
        The processed valid segments.
        
    Raises
    ------
    ValueError
        If no valid segments were found.
    ValueError
        If inputted order value not recognised.
    """
    valid_segments = get_segment_times(
        start,
        stop,
        ifo,
        state_flag
    )
    
    veto_segments = []
    if "events" not in data_labels:
        event_times = get_all_event_times()
        veto_segments.append(pad_gps_times_with_veto_window(event_times))
    if "glitches" not in data_labels:
        # glitches = EventTable.fetch('gravityspy', 'glitches')
        # print(glitches)
        print("Glitch vetos not implemented!")
        pass
    
    if veto_segments != []:
        veto_segments = np.concatenate(veto_segments)
        valid_segments = veto_time_periods(valid_segments, veto_segments)
    
    valid_segments = split_periods(valid_segments, max_segment_size)
    valid_segments = remove_short_periods(
        valid_segments, 
        (onsource_duration_seconds + fduration)*num_examples_per_batch 
        + offsource_duarion_seconds
    )
    
    if (len(valid_segments) == 0):
        raise ValueError("No valid segments!")
        
    if order == "random":
        np.random.shuffle(valid_segments)
    elif order == "shortest_first":
        sort_by_duration = \
            lambda segments: \
                segments[np.argsort(segments[:, 1] - segments[:, 0])]
        valid_segments = sort_by_duration(valid_segments)
    elif order == "chronological":
        pass
    else:
        raise ValueError(
            f"Order {order} not recognised, please choose from \"random\","
            "\"shortest_firsr\", or \"chronological\"."
        )
    
    valid_segments = cycle(valid_segments)
    
    return valid_segments

def get_new_segment_data(
    segment_start: int, 
    segment_end: int, 
    ifo: str, 
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
    ifo : str
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
        site=ifo.strip("1"),
        frametype=f"{ifo}_{frame_type}",
        gpsstart=segment_start,
        gpsend=segment_end,
        urltype="file",
    )
    data = TimeSeries.read(
        files, 
        channel=f"{ifo}:{channel}", 
        start=segment_start, 
        end=segment_end, 
        nproc=4
    )

    return data

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

def get_sorted_values(dictionary):
    sorted_keys = sorted(dictionary.keys())
    sorted_values = [dictionary[key] for key in sorted_keys]
    return sorted_values

def generate_filenames(
    data_directory: Union[str, Path],         
    segment_parameters: List[Any], 
    seed: int
    ) -> (Path, List[Path]):
    """
    Generate unique filenames based on a hash of the input parameters and 
    injection configurations.

    Parameters
    ----------
    data_directory : Union[str, Path]
        The directory where the data is stored.
    segment_parameters : List[Any]
        The list of parameters used for segment generation.

    Returns
    -------
    Path
        The segment filename

    """
    # Generate the hash for the segment parameters
    segment_hash = generate_hash_from_list(segment_parameters)
    # Construct the segment filename using the hash
    segment_filename = \
        Path(data_directory) / f"segment_data_{segment_hash}.hdf5"

    return segment_filename


@tf.function
def crop_samples(
    batched_onsource: tf.Tensor, 
    onsource_duration_seconds: float, 
    sample_rate_hertz: float
    ) -> tf.Tensor:
    
    """
    Crop to remove edge effects and ensure same data is retrieved in all cases.
    
    This function calculates the desired number of samples based on the duration 
    of examples in seconds and the sample rate, then it finds the start and end 
    index for cropping. It then crops the batched_onsource using these indices.
    
    Parameters
    ----------
    batched_onsource : tf.Tensor
        The batch of examples to be cropped.
    onsource_duration_seconds : float
        The duration of an example in seconds.
    sample_rate_hertz : float
        The sample rate in hertz.
    
    Returns
    -------
    tf.Tensor
        The cropped batched_onsource.
    """
    
    if len(batched_onsource.shape) == 1:
        batched_onsource = tf.expand_dims(batched_onsource, 0)
    
    # Calculate the desired number of samples based on example duration and 
    # sample rate:
    desired_num_samples = int(onsource_duration_seconds * sample_rate_hertz)
    
    # Calculate the start and end index for cropping
    start = (batched_onsource.shape[-1] - desired_num_samples) // 2
    end = start + desired_num_samples
    
    # Crop the batched_onsource
    batched_onsource = batched_onsource[..., start:end]
    
    return batched_onsource

@tf.function
def batch_tensor(tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
    """
    Batches a tensor into batches of a specified size. If the first dimension
    of the tensor is not exactly divisible by the batch size, remaining elements
    are discarded.

    Parameters
    ----------
    tensor : tf.Tensor
        The tensor to be batched.
    batch_size : int
        The size of each batch.

    Returns
    -------
    tf.Tensor
        The reshaped tensor in batches.
    """
    # Ensure that the tensor is 1D or 2D
    assert len(tensor.shape) in (1, 2), 'Tensor must be 1D or 2D'

    # Calculate the number of full batches that can be created
    num_batches = tensor.shape[0] // batch_size

    # Slice the tensor to only include enough elements for the full batches
    tensor = tensor[:num_batches * batch_size]

    if len(tensor.shape) == 1:
        # Reshape the 1D tensor into batches
        batched_tensor = tf.reshape(tensor, (num_batches, batch_size))
    else: # tensor.ndim == 2
        # Reshape the 2D tensor into batches
        batched_tensor = tf.reshape(tensor, (num_batches, batch_size, -1))

    return batched_tensor

def get_ifo_data(
    time_interval: Union[tuple, ObservingRun], 
    data_labels: List[str], 
    ifo: str,
    sample_rate_hertz: float,    
    data_quality: str = "best",
    channel: str = None,
    frame_type: str = None,
    state_flag: str = None,
    injection_configs: list = [], 
    saturation: float = 1.0,
    onsource_duration_seconds: float = 1.0,
    offsource_duarion_seconds: float = 16.0,
    apply_whitening: bool = False,
    num_examples_per_batch: int = 1,
    scale_factor: float = 1.0e20,
    max_segment_size = 2000,
    order: str = "random",
    seed: int = 1000,
    force_generation: bool = False,
    data_directory: Union[str, Path] = "./generator_data",
    save_segment_data: bool = True,
    input_keys = ["onsource", "offsource", "gps_time", "injectons", "snr"],
    output_keys = ["onsource", "offsource", "gps_time", "injectons", "snr"],
    fduration = 1.0
):
    data_directory = Path(data_directory)
    ensure_directory_exists(data_directory)
    
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    # Pull information from observing run object:
    start, stop, frame_type, channel, state_flag = \
        process_time_interval(
            time_interval,
            frame_type,
            channel,
            state_flag,
            data_quality
        )
    
    max_segments_per_file = 1.0E6 
    
    segment_parameters = \
        [
            frame_type, 
            channel, 
            state_flag, 
            str(max_segment_size),
            str(data_labels), 
            sample_rate_hertz
        ]  
    
    segment_filename = \
        generate_filenames(
            data_directory,         
            segment_parameters, 
            seed
        )
        
    # Get segment start and stop times given input parameters
    valid_segments = \
        process_valid_segments(
            start, 
            stop, 
            ifo, 
            state_flag, 
            data_labels, 
            max_segment_size, 
            onsource_duration_seconds, 
            fduration, 
            num_examples_per_batch, 
            offsource_duarion_seconds,
            order
        )
    
    batch_index = 0
    injection_indicies = [0] * len(injection_configs)
    
    with open_hdf5_file(segment_filename) as segment_file:
        
        segment_file.require_group("segments")
        for segment_index, (current_segment_start, current_segment_end) in enumerate(valid_segments):
            
            segment_key = \
                f"segments/segment_{current_segment_start}_{current_segment_end}"
                            
            current_segment_data = None
            
            expected_duration_seconds = \
                    current_segment_end - current_segment_start
            if (segment_key in segment_file) and save_segment_data:
                
                print(
                    f"Reading segments of duration "
                    f"{expected_duration_seconds}..."
                )
                current_segment_data = \
                    IFOData(
                        segment_file[segment_key][()], 
                        current_segment_start, 
                        sample_rate_hertz)
            else: 
                print(f"Acquiring segments of duration "
                      f"{expected_duration_seconds}..."
                     )
                try:
                    current_segment_data = \
                        get_new_segment_data(
                            current_segment_start, 
                            current_segment_end, 
                            ifo, 
                            frame_type, 
                            channel
                        )
                    
                    #Would be nice to do this on the gpu:
                    current_segment_data = \
                        current_segment_data.resample(sample_rate_hertz)
                
                except Exception as e:
                    print(f"Unexpected error: {type(e).__name__}, {str(e)}")
                    continue
                
                current_segment_data = \
                    IFOData(
                        current_segment_data, 
                        current_segment_data.t0.value, 
                        current_segment_data.sample_rate.value
                    )
                #current_segment_data = 
                    #current_segment_data.downsample(sample_rate_hertz) 
                
                if save_segment_data:
                    segment_file.create_dataset(
                        segment_key, 
                        data = current_segment_data.numpy()
                    )
            print("Complete!")
            
            current_segment_data = current_segment_data.scale(scale_factor)  
            max_batch_count = \
                int(
                    max_segment_size
                    / (saturation * num_examples_per_batch)
                )
            
            current_max_batch_count = \
                int(
                      (current_segment_end - current_segment_start) 
                    / (saturation * num_examples_per_batch)
                )
            
            # Set common parameters before entering the loop
            common_args = {
                "sample_rate_hertz": \
                    {
                        "value": sample_rate_hertz, 
                         "distribution_type": "constant"
                    },
                "duration_seconds": \
                    {
                        "value": onsource_duration_seconds + fduration, 
                         "distribution_type": "constant"
                    }
            }
            
            #Generate injections:
            injections = []
            injection_masks = []
            snrs = []
            for config in injection_configs:    
    
                injections_, injection_masks_, snrs_ = \
                    generate_injections(
                        config,
                        current_max_batch_count*num_examples_per_batch, 
                        common_args,
                        fduration,
                        sample_rate_hertz,
                        onsource_duration_seconds,
                        num_examples_per_batch
                    )

                injections.append(
                    injections_
                )    
            
                injection_masks.append(
                    injection_masks_
                )
                
                snrs.append(
                    snrs_
                )
                
            injection_masks = tf.stack(injection_masks)
            snrs = tf.stack(snrs)
            
            for batch_index in range(current_max_batch_count):
                
                num_onsource_samples = \
                    int(
                        (onsource_duration_seconds + fduration)
                        *sample_rate_hertz
                    )
                num_offsource_samples = \
                    int(offsource_duarion_seconds * sample_rate_hertz)
                
                batched_onsource, batched_offsource, batched_gps_times = \
                    random_subsection(
                        current_segment_data.data,
                        current_segment_data.dt,
                        current_segment_data.t0,
                        num_onsource_samples, 
                        num_offsource_samples, 
                        num_examples_per_batch
                    )
                
                cropped_injections = []
                amplitudes = []
                for injection_index, config in enumerate(injection_configs):
                                        
                    scaled_injections = \
                        scale_to_snr(
                            injections[injection_index][batch_index], 
                            batched_onsource, 
                            snrs[injection_index][batch_index],
                            window_duration_seconds=1.0,
                            sample_rate_hertz=sample_rate_hertz,
                            fft_duration_seconds=1.0,
                            overlap_duration_seconds=0.5
                        )
                    
                    scaled_injections = \
                        replace_nan_and_inf_with_zero(scaled_injections)
                    
                    amplitudes.append(
                        tf.reduce_max(tf.abs(scaled_injections), axis=1)*100
                    )
                    
                    batched_onsource += scaled_injections

                    cropped_injections.append(
                        crop_samples(
                            scaled_injections, 
                            onsource_duration_seconds, 
                            sample_rate_hertz
                        )
                    )
                
                # Whiten data: 
                if apply_whitening:
                    batched_onsource = \
                        whiten(
                            batched_onsource, 
                            batched_offsource, 
                            sample_rate_hertz, 
                            fftlength = 1.0,
                            overlap = 0.5,
                            fduration = fduration
                        )
                    
                # Crop to remove edge effects, crop with or without whitening to
                # ensure same data is retrieve in both cases
                batched_onsource = crop_samples(
                    batched_onsource, 
                    onsource_duration_seconds, 
                    sample_rate_hertz
                )
                                
                def create_dict(keys, batch_index=None):
                    operations = {
                        'onsource': lambda: \
                            tf.cast(
                                batched_onsource, 
                                tf.float16
                            ),
                        'offsource': lambda: \
                            tf.cast(
                                batched_offsource, 
                                tf.float16
                            ),
                        'gps_time': lambda: \
                            tf.convert_to_tensor(
                                batched_gps_times,
                                dtype=tf.int64
                            ),
                        'injections': lambda: \
                            cropped_injections,
                        'injection_masks': lambda: \
                            injection_masks[:, batch_index],
                        'snr': lambda: \
                            snrs[:, batch_index],
                        'amplitude': lambda: \
                            amplitudes
                    }

                    return {
                        key: operations[key]() \
                        for key in keys if key in operations
                    }

                input_dict = create_dict(input_keys, batch_index)
                output_dict = create_dict(output_keys, batch_index)
                                
                yield (input_dict, output_dict)

def get_ifo_data_generator(
    time_interval: Union[tuple, ObservingRun], 
    data_labels: List[str], 
    ifo: str,
    sample_rate_hertz: float,        
    input_keys = [
        "onsource", 
        "offsource", 
        "gps_time", 
        "injections", 
        "snr"],
    output_keys = [
        "onsource", 
        "offsource", 
        "gps_time", 
        "injections", 
        "snr"],
    **kwargs  # Capture all other arguments
    ):
    
    num_examples_per_batch = kwargs.get('num_examples_per_batch', 1)
    num_onsource_samples = \
        int(kwargs.get('onsource_duration_seconds', 1.0)*sample_rate_hertz)
    num_offsource_samples = \
        int(kwargs.get('offsource_duration_seconds', 16.0)*sample_rate_hertz)
    num_injection_configs = len(kwargs.get('injection_configs', {}))
    
    output_signature_dict = {
        'onsource'       : \
            tf.TensorSpec(
                shape=(num_examples_per_batch, num_onsource_samples), 
                dtype=tf.float16
            ),
        'offsource' : \
            tf.TensorSpec(
                shape=(num_examples_per_batch, num_offsource_samples), 
                dtype=tf.float16
            ),
        'gps_time' : 
            tf.TensorSpec(
                shape=(num_examples_per_batch), 
                dtype=tf.int64
            ),
        'injections' : 
            tf.TensorSpec(
                shape=(
                    num_injection_configs, 
                    num_examples_per_batch, 
                    num_onsource_samples
                ), 
                dtype=tf.float16
            ),
        'injection_masks' : 
            tf.TensorSpec(
                shape=(
                    num_injection_configs, 
                    num_examples_per_batch
                ), 
                dtype=tf.bool
            ),
        'snr' : 
            tf.TensorSpec(
                shape=(
                    num_injection_configs,
                    num_examples_per_batch
                ), 
                dtype=tf.float64
            ),
        'amplitude' : 
            tf.TensorSpec(
                shape=(
                    num_injection_configs,
                    num_examples_per_batch
                ), 
                dtype=tf.float64
            ),
    }
    
    output_signature = (
        {k: output_signature_dict[k] for k in input_keys},
        {k: output_signature_dict[k] for k in output_keys}
    )
    
    generator = lambda: \
        get_ifo_data(
            time_interval, 
            data_labels, 
            ifo, 
            sample_rate_hertz, 
            input_keys = input_keys, 
            output_keys = output_keys,
            **kwargs
        )
    
    return tf.data.Dataset.from_generator(
            generator = generator,
            output_signature = output_signature
        )