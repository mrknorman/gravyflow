from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto
from typing import Union, Iterator

import tensorflow as tf

from .acquisition import (IFODataObtainer, ObservingRun, DataQuality, DataLabel, 
                          SegmentOrder, IFO)

class NoiseType(Enum):
    WHITE = auto()
    COLORED = auto()
    PSEUDO_REAL = auto()
    REAL = auto()

@dataclass
class NoiseObtainer:
    data_directory_path : Path = Path("./generator_data")
    ifo_data_obtainer : Union[None, IFODataObtainer] = None
    noise_type : NoiseType = NoiseType.REAL
    groups : dict = None
    
    def __post_init__(self):
        
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
            padding_duration_seconds : float,
            offsource_duration_seconds : float,
            num_examples_per_batch : float,
            scale_factor : float = 1.0
        ) -> Iterator:
        
        # Configure noise based on type
        
        self.generator = None
        
        match self.noise_type:
            case NoiseType.WHITE:
                print("Not implemented")
                
            case NoiseType.COLORED:
                print("Not implemented")
            
            case NoiseType.PSEUDO_REAL:
                print("Not implemented")
            
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
                        self.groups,
                        "train"
                    )
                
                    # Setup noise_file_path, file path is created from
                    # hash of unique parameters
                    self.ifo_data_obtainer.generate_file_path(
                        sample_rate_hertz,
                        self.data_directory_path
                    )
                
                # Initilise generator function:
                self.generator = \
                    self.ifo_data_obtainer.get_onsource_offsource_chunks(
                        sample_rate_hertz,
                        onsource_duration_seconds,
                        padding_duration_seconds,
                        offsource_duration_seconds,
                        num_examples_per_batch,
                        scale_factor
                    )
                
            case _:
                # Raise error if noisetyp not recognised.
                raise ValueError(
                    f"""
                    NoiseType {self.noise_type} not recognised, please choose 
                    from NoiseType.WHITE, NoiseType.COLORED, NoiseType.PSEUDO_REAL,
                    or NoiseType.REAL.
                    """
                )
                
        if self.generator is None:
            raise ValueError("Noise generator failed to initlise, for some reason...")
                
        return self.generator