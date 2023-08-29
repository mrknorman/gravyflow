from .data_acquisition import (IFODataConfig, ObservingRun, DataQuality,
                               DataLabel, SegmentOrder, IFO)
from pathlib import Path
from enum import Enum, auto

class NoiseType(Enum):
    WHITE = auto()
    COLORED = auto()
    PSEUDO_REAL = auto()
    REAL = auto()

@dataclass
class NoiseConfig:
    noise_data_path : Path = Path("./generator_data")
    force_segment_aquisition : bool = False
    save_segment_data: bool = True
    ifo_data_config : Union[None, IFODataConfig] = None
    noise_type : NoiseType = NoiseType.REAL
    order: str = SegmentOrder.RANDOM
    saturation: float = 1.0
    groups : dict = \
    {
        "train" : 0.98,
        "validate" : 0.01,
        "test" : 0.01
    }
    group_name = "train"
    
    def setup_noise(self):
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
                self.data_config.get_valid_segments(
                    max_segment_duration_seconds,
                    min_segment_duration_seconds,
                    groups,
                    "train",
                    SegmentOrder.RANDOM
                )
                
                # Setup noise_file_path if required, file path is created from
                # hash of unique parameters
                if save_segment_data or not force_segment_aquisition:
                    
                    # Generate file for user cache:
                    data_config.generate_file_path(
                        max_segment_duration_seconds,
                        sample_rate_hertz
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
    
                    
    