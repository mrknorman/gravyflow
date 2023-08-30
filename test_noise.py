# Built-In imports:
from pathlib import Path
from itertools import islice
import logging

# Local imports:
from .setup import find_available_GPUs, setup_cuda
from .acquisition import (IFODataObtainer, SegmentOrder, ObservingRun, 
                          DataQuality, DataLabel, IFO)
from .noise import NoiseObtainer, NoiseType

def test_real_noise(num_tests : int = 10):
    
    # Test parameters:
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 1.0    
    padding_duration_seconds : float = 0.5
    offsource_duration_seconds : float = 16.0
    num_examples_per_batch : float = 32
    scale_factor : float = 1.0E20
    
    # Setup ifo data acquisition object:
    ifo_data_obtainer = \
        IFODataObtainer(
            ObservingRun.O3, 
            DataQuality.BEST, 
            [
                DataLabel.NOISE, 
                DataLabel.GLITCHES
            ],
            IFO.L1,
            SegmentOrder.RANDOM,
            force_acquisition = True,
            cache_segments = False
        )
    
    # Initilise noise generator wrapper:
    noise = \
        NoiseObtainer(
            ifo_data_obtainer = ifo_data_obtainer,
            noise_type = NoiseType.REAL
        )
    
    # Create generator:
    generator = \
        noise.init_generator(
            sample_rate_hertz,
            onsource_duration_seconds,
            padding_duration_seconds,
            offsource_duration_seconds,
            num_examples_per_batch,
            scale_factor
        )
        
    # Iterate through num_tests batches to check correct operation:
    for onsource, offsource, gps_times in islice(generator, num_tests):
        
        


if __name__ == "__main__":
    
    # ---- User parameters ---- #
    
    # GPU setup:
    min_gpu_memory_mb : int = 4000
    num_gpus_to_request : int = 1
    memory_to_allocate_tf : int = 2000
    
    # Setup CUDA
    gpus = find_available_GPUs(min_gpu_memory_mb, num_gpus_to_request)
    setup_cuda(gpus, max_memory_limit = memory_to_allocate_tf, verbose = True)    
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    # Test IFO noise generator:
    test_real_noise()
    
    