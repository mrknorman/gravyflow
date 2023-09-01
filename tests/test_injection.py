# Built-In imports:
import logging

# Library imports:
import numpy as np

# Local imports:
from ..maths import Distribution, DistributionType
from ..setup import find_available_GPUs, setup_cuda, ensure_directory_exists
from ..injection import cuPhenomDGenerator, InjectionGenerator

def test_phenom_d_injection(num_tests : int = 10):
    
    # Test Parameters:
    num_examples_per_generation_batch : int = 2048
    num_examples_per_batch : int = 32
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 1.0
    crop_duration_seconds : float = 0.5
    scale_factor : float = 1.0E21
    
    phenom_d_generator : cuPhenomDGenerator = cuPhenomDGenerator(
            injection_chance=1.0,
            front_padding_duration_seconds=0.3,
            back_padding_duration_seconds=0.0,
            mass_1_msun=\
                Distribution(min_=5.0, max_=95.0, type_=DistributionType.UNIFORM),
            mass_2_msun=\
                Distribution(min_=5.0, max_=95.0, type_=DistributionType.UNIFORM),
            inclination_radians=\
                Distribution(min_=0.0, max_=np.pi, type_=DistributionType.UNIFORM),
            distance_mpc=\
                Distribution(value=1000.0, type_=DistributionType.CONSTANT),
            reference_orbital_phase_in= \
                Distribution(value=0.0, type_=DistributionType.CONSTANT),
            ascending_node_longitude= \
                Distribution(value=0.0, type_=DistributionType.CONSTANT),
            eccentricity= \
                Distribution(value=0.0, type_=DistributionType.CONSTANT),
            mean_periastron_anomaly= \
                Distribution(value=0.0, type_=DistributionType.CONSTANT),
            spin_1_in= \
                Distribution(value=0.0, type_=DistributionType.CONSTANT),
            spin_2_in= \
                Distribution(value=0.0, type_=DistributionType.CONSTANT),
        )
    
    injection_generator : InjectionGenerator = InjectionGenerator(phenom_d_generator)
        
    iterator = \
        injection_generator.get_generator(
            sample_rate_hertz,
            onsource_duration_seconds,
            crop_duration_seconds,
            num_examples_per_generation_batch,
            num_examples_per_batch,
            scale_factor
        )
    
    

if __name__ == "__main__":
    
    # ---- User parameters ---- #
    
    # GPU setup:
    min_gpu_memory_mb : int = 4000
    num_gpus_to_request : int = 1
    memory_to_allocate_tf : int = 2000
    
    # Setup CUDA
    gpus = find_available_GPUs(min_gpu_memory_mb, num_gpus_to_request)
    stratergy = setup_cuda(
        gpus, 
        max_memory_limit = memory_to_allocate_tf, 
        logging_level=logging.WARNING
    )    
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    # Test IFO noise generator:
    test_phenom_d_injection()