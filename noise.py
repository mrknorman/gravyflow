from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto
from typing import Union, Iterator, List

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from .acquisition import (IFODataObtainer, ObservingRun, DataQuality, DataLabel, 
                          SegmentOrder, IFO)
from.psd import calculate_psd

class NoiseType(Enum):
    WHITE = auto()
    COLORED = auto()
    PSEUDO_REAL = auto()
    REAL = auto()
    
@tf.function
def _generate_white_noise(
    num_examples_per_batch: int,
    num_samples: int
) -> tf.Tensor:
    """
    Optimized function to generate white Gaussian noise.

    Parameters:
    ----------
    num_examples_per_batch : int
        Number of examples per batch.
    num_samples : int
        Number of samples per example.
        
    Returns:
    -------
    tf.Tensor
        A tensor containing white Gaussian noise.
    """
    return tf.random.normal(
        shape=[num_examples_per_batch, num_samples],
        mean=0.0,
        stddev=1.0,
        dtype=tf.float32
    )

def white_noise_generator(
    num_examples_per_batch: int,
    onsource_duration_seconds: float,
    crop_duration_seconds : float,
    offsource_duration_seconds: float,
    sample_rate_hertz: float
) -> Iterator[tf.Tensor]:
    """
    Generator function that yields white Gaussian noise.

    Parameters:
    ----------
    num_examples_per_batch : int
        Number of examples per batch.
    onsource_duration_seconds : float
        Duration of the onsource segment in seconds.
    sample_rate_hertz : float
        Sample rate in Hz.

    Yields:
    -------
    tf.Tensor
        A tensor containing white Gaussian noise.
    """
    total_onsource_duration_seconds : float = \
        onsource_duration_seconds + (crop_duration_seconds * 2.0)
    
    num_onsource_samples : int = \
        int(total_onsource_duration_seconds * sample_rate_hertz)
    
    num_offsource_samples : int = \
        int(offsource_duration_seconds * sample_rate_hertz)

    while True:
        yield _generate_white_noise(
            num_examples_per_batch, 
            num_onsource_samples
        ), 
        _generate_white_noise(
            num_examples_per_batch, 
            num_offsource_samples
        ),
        tf.fill([num_examples_per_batch], -1.0)
        
@tf.function
def _generate_colored_noise(
    num_examples_per_batch: int,
    num_samples: int,
    interpolated_psd: tf.Tensor
) -> tf.Tensor:
    """
    Function to generate colored Gaussian noise.

    Parameters:
    ----------
    num_examples_per_batch : int
        Number of examples per batch.
    num_samples : int
        Number of samples per example.
    interpolated_psd : tf.Tensor
        Interpolated power spectral density.

    Returns:
    -------
    tf.Tensor
        A tensor containing colored Gaussian noise.
    """
    white_noise = tf.random.normal(
        shape=[num_examples_per_batch, num_samples],
        mean=0.0,
        stddev=1.0,
        dtype=tf.float32
    )

    white_noise_fft = tf.signal.rfft(white_noise)
    colored_noise_fft = tf.sqrt(interpolated_psd) * white_noise_fft
    colored_noise = tf.signal.irfft(colored_noise_fft)
    
    return colored_noise

def interpolate_onsource_offsource_psd(
    num_samples_list : List,
    psd_freq : tf.Tensor,
    psd_val : tf.Tensor
    ) -> List[tf.Tensor]:

    interpolated_psd_onsource, interpolated_psd_offsource = [
        tf.cast(
            tfp.math.interp_regular_1d_grid(
                tf.cast(
                    tf.linspace(
                        psd_freq[0], 
                        psd_freq[-1], 
                        num // 2 + 1
                    ), tf.float32
                ), 
                psd_freq[0], 
                psd_freq[-1], 
                psd_val, 
                axis=-1
            ), 
            tf.complex64
        )
        for num in num_samples_list
    ]
    
    return interpolated_psd_onsource, interpolated_psd_offsource

def colored_noise_generator(
    num_examples_per_batch: int,
    onsource_duration_seconds: float,
    crop_duration_seconds : float,
    offsource_duration_seconds: float,
    ifo : IFO,
    sample_rate_hertz: float,
    scale_factor : float
) -> Iterator[tf.Tensor]:
    """
    Generator function that yields colored Gaussian noise.

    Parameters:
    ----------
    num_examples_per_batch : int
        Number of examples per batch.
    onsource_duration_seconds : float
        Duration of the onsource segment in seconds.
    sample_rate_hertz : float
        Sample rate in Hz.
    psd_freq : tf.Tensor
        Frequencies for the power spectral density.
    psd_val : tf.Tensor
        Power spectral density values.

    Yields:
    -------
    tf.Tensor
        A tensor containing colored Gaussian noise.
    """
    
    total_onsource_duration_seconds : float = \
        onsource_duration_seconds + (crop_duration_seconds * 2.0)
    
    durations_seconds = [
        total_onsource_duration_seconds, 
        offsource_duration_seconds
    ]
    
    frequencies, psd = np.loadtxt(
        ifo.value.optimal_psd_path, delimiter=","
    ).T

    frequencies = tf.convert_to_tensor(frequencies, dtype=tf.float32)
    psd = tf.convert_to_tensor(psd, dtype=tf.float32)
    
    psd *= scale_factor
    
    num_samples_list = [
        int(duration * sample_rate_hertz) for duration in durations_seconds
    ]

    interpolated_psd_onsource, interpolated_psd_offsource = \
        interpolate_onsource_offsource_psd(
            num_samples_list,
            frequencies,
            psd
        )

    while True:
        yield _generate_colored_noise(
                num_examples_per_batch, 
                num_samples_list[0], 
                interpolated_psd_onsource
            ), \
              _generate_colored_noise(
                num_examples_per_batch, 
                num_samples_list[1], 
                interpolated_psd_offsource
            ), \
              tf.fill([num_examples_per_batch], -1.0)
    

@dataclass
class NoiseObtainer:
    data_directory_path : Path = Path("./generator_data")
    ifo_data_obtainer : Union[None, IFODataObtainer] = None
    ifos : List[IFO] = IFO.L1
    noise_type : NoiseType = NoiseType.REAL
    groups : dict = None
    
    def __post_init__(self):
        
        if not isinstance(self.ifos, list):
            self.ifos = [self.ifos]
        
        # Set default groups here as dataclass will not allow mutable defaults:
        if not self.groups:
            self.groups = \
            {
                "train" : 0.98,
                "validate" : 0.01,
                "test" : 0.01
            }
    
    def init_generator(
            self,
            sample_rate_hertz : float,
            onsource_duration_seconds : float,
            crop_duration_seconds : float,
            offsource_duration_seconds : float,
            num_examples_per_batch : float,
            scale_factor : float = 1.0,
            group : str = "train"
        ) -> Iterator:
        
        # Configure noise based on type
        
        self.generator = None
        
        match self.noise_type:
            case NoiseType.WHITE:
                self.generator = \
                    white_noise_generator(
                        num_examples_per_batch,
                        onsource_duration_seconds,
                        crop_duration_seconds,
                        offsource_duration_seconds,
                        sample_rate_hertz
                    )
                
            case NoiseType.COLORED:
                self.generator = colored_noise_generator(
                    num_examples_per_batch,
                    onsource_duration_seconds,
                    crop_duration_seconds,
                    offsource_duration_seconds,
                    self.ifos[0],
                    sample_rate_hertz,
                    scale_factor
                )
            
            case NoiseType.PSEUDO_REAL:
                
                self.generator = self.pseudo_real_noise_generator(
                    num_examples_per_batch,
                    onsource_duration_seconds,
                    crop_duration_seconds,
                    offsource_duration_seconds,
                    sample_rate_hertz,
                    group,
                    scale_factor
                )
            
            case NoiseType.REAL:
                # Get real ifo data
                
                # If noise type is real, get real noise time segments that fit 
                # criteria, segments will be stored as a 2D numpy array as pairs 
                # of start and end times:
                
                if not self.ifo_data_obtainer:
                    # Check to see if obtatainer object has been set up, raise
                    # error if not
                    raise ValueError("""
                        No IFO obtainer object present. In order to acquire real 
                        noise please parse a IFOObtainer object to NoiseObtainer
                        either during initlisation or through setting
                        NoiseObtainer.ifo_data_obtainer
                    """)
                else:
                    
                    self.ifo_data_obtainer.get_valid_segments(
                        self.ifos,
                        self.groups,
                        group
                    )
                    
                    # Setup noise_file_path, file path is created from
                    # hash of unique parameters
                    self.ifo_data_obtainer.generate_file_path(
                        sample_rate_hertz,
                        group,
                        self.data_directory_path
                    )
                
                # Initilise generator function:
                self.generator = \
                    self.ifo_data_obtainer.get_onsource_offsource_chunks(
                        sample_rate_hertz,
                        onsource_duration_seconds,
                        crop_duration_seconds,
                        offsource_duration_seconds,
                        num_examples_per_batch,
                        self.ifos,
                        scale_factor
                    )
                
            case _:
                # Raise error if noisetyp not recognised.
                raise ValueError(
                    f"NoiseType {self.noise_type} not recognised, please choose"
                    "from NoiseType.WHITE, NoiseType.COLORED, "
                    "NoiseType.PSEUDO_REAL, or NoiseType.REAL. "
                )
                
        if self.generator is None:
            raise ValueError(
                "Noise generator failed to initilise.."
            )
                
        return self.generator
    
    def pseudo_real_noise_generator(
        self,
        num_examples_per_batch: int,
        onsource_duration_seconds: float,
        crop_duration_seconds : float,
        offsource_duration_seconds: float,
        sample_rate_hertz: float,
        group : str,
        scale_factor : float
    ):
        
        total_onsource_duration_seconds : float = \
            onsource_duration_seconds + (crop_duration_seconds * 2.0)  
        
        durations_seconds = [
            total_onsource_duration_seconds, 
            offsource_duration_seconds
        ]
        
        num_samples_list = [
            int(duration * sample_rate_hertz) for duration in durations_seconds
        ]

        if not self.ifo_data_obtainer:
            # Check to see if obtatainer object has been set up, raise
            # error if not
            raise ValueError("""
                No IFO obtainer object present. In order to acquire real 
                noise please parse a IFOObtainer object to NoiseObtainer
                either during initlisation or through setting
                NoiseObtainer.ifo_data_obtainer
            """)
        else:

            valid_segments = self.ifo_data_obtainer.get_valid_segments(
                self.ifos,
                self.groups,
                group
            )

            # In the case of pseduo real noise we only want to acquire
            # offsource_duration amount of data:
            valid_segments : np.ndarray = \
                self.ifo_data_obtainer.split_segments(
                    valid_segments, 
                    offsource_duration_seconds
                )

            # Ensure all segments are at least length:
            valid_segments : np.ndarray = \
                self.ifo_data_obtainer.remove_short_segments(
                    self.ifo_data_obtainer.valid_segments, 
                    offsource_duration_seconds
                )

            # Setup noise_file_path, file path is created from
            # hash of unique parameters:
            self.ifo_data_obtainer.generate_file_path(
                sample_rate_hertz,
                group,
                self.data_directory_path
            )

        for segment in self.ifo_data_obtainer.acquire(
            sample_rate_hertz, 
            valid_segments, 
            self.ifos,
            scale_factor
        ):

            frequencies, psd = calculate_psd(
                segment.data, 
                nperseg=1024, 
                sample_rate_hertz=sample_rate_hertz
            )
            
            interpolated_psd_onsource, interpolated_psd_offsource = \
                interpolate_onsource_offsource_psd(
                    num_samples_list,
                    frequencies,
                    psd
                )
            
            # Calculate number of batches current segment can produce, this
            # is dependant on the segment duration and the onsource duration:
            num_batches_in_segment : int = \
                int(
                      self.ifo_data_obtainer.max_segment_duration_seconds
                    / (
                        self.ifo_data_obtainer.saturation * 
                        num_examples_per_batch * onsource_duration_seconds
                    )
                )
            
            for _ in range(num_batches_in_segment):
                yield _generate_colored_noise(
                        num_examples_per_batch, 
                        num_samples_list[0], 
                        interpolated_psd_onsource
                    ), \
                      _generate_colored_noise(
                        num_examples_per_batch, 
                        num_samples_list[1], 
                        interpolated_psd_offsource
                    ), \
                      tf.fill([num_examples_per_batch], segment.start_gps_time)
            
            