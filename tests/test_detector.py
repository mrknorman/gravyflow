# Built-In imports:
import logging
from pathlib import Path

# Library imports:
import tensorflow as tf
import numpy as np
from pycbc.detector import (add_detector_on_earth, _ground_detectors, 
                            get_available_detectors, Detector)

# Local imports:
from ..setup import find_available_GPUs, setup_cuda
from ..detector import Network, IFO

def test_detector():
    
    # Create from dictionary:
    latitude = 0.3
    longitude = 0.5
    
    network = Network({
        "longitude_radians" : latitude, 
        "latitude_radians" : longitude,
        "y_angle_radians" : 0.0,  # Batched tensor
        "x_angle_radians" : None,  # Batched tensor or None
        "height_meters" : 0.0,  # Batched tensor
        "x_length_meters" : 4000.0,  # Batched tensor
        "y_length_meters" : 4000.0   # Batched tensor
    })
    
    add_detector_on_earth(
        "test", 
        latitude, 
        longitude,              
        )
    
    np.testing.assert_allclose(
        network.xvec.numpy()[0], 
        _ground_detectors["test"]["xvec"],
        rtol=0, 
        atol=1e-07, 
        equal_nan=True, 
        err_msg="Tensorflow Network construction does not equal pycbc method.", 
        verbose=True
    )
    
    # Create from Enum:
    network = Network(
        [IFO.L1, IFO.H1, IFO.V1]
    )
    
    # From loading:
    
    example_network_directory : Path = Path(
        "./py_ml_tools/tests/example_network_parameters/example_network.json"
    )
    
    network = Network.load(example_network_directory)

if __name__ == "__main__":
        
    # ---- User parameters ---- #
    
    # GPU setup:
    min_gpu_memory_mb : int = 4000
    num_gpus_to_request : int = 1
    memory_to_allocate_tf : int = 2000
    
    # Setup CUDA
    gpus = find_available_GPUs(
        min_gpu_memory_mb, 
        num_gpus_to_request
    )
    strategy = setup_cuda(
        gpus, 
        max_memory_limit = memory_to_allocate_tf, 
        logging_level=logging.WARNING
    )    
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    # Test psd calculation:
    with strategy.scope():
        test_detector()