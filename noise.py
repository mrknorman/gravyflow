from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto
from typing import Union, Iterator, List

import tensorflow as tf

from .acquisition import (IFODataObtainer, ObservingRun, DataQuality, DataLabel, 
                          SegmentOrder, IFO)

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
        dtype=tf.float16
    )

def white_noise_generator(
    num_examples_per_batch: int,
    onsource_duration_seconds: float,
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
    num_onsource_samples : int = \
        int(onsource_duration_seconds * sample_rate_hertz)
    
    num_offsource_samples : int = \
        int(offsource_duration_seconds * sample_rate_hertz)

    while True:
        yield _generate_white_noise(num_examples_per_batch, num_onsource_samples), \
            _generate_white_noise(num_examples_per_batch, num_offsource_samples), \
            tf.fill([num_examples_per_batch], -1.0)

@dataclass
class NoiseObtainer:
    data_directory_path : Path = Path("./generator_data")
    ifo_data_obtainer : Union[None, IFODataObtainer] = None
    ifos : List[IFO] = [IFO.L1]
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
                        offsource_duration_seconds,
                        sample_rate_hertz
                    )
                
            case NoiseType.COLORED:
                
                np.load_text()
                
                raise ValueError("Not implemented")
            
            case NoiseType.PSEUDO_REAL:
                raise ValueError("Not implemented")
            
            case NoiseType.REAL:
                # Get real ifo data
                
                # If noise type is real, get real noise time segments that fit 
                # criteria, segments will be stored as a 2D numpy array as pairs 
                # of start and end times:
                
                if not self.ifo_data_obtainer:
                    # Check to see if obtatainer object has been set up, raise
                    # error if not
                    raise ValueError("""
                        No IFO obtainer object present. In order to aquire real 
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