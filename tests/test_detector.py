import logging

import tensorflow as tf

from ..setup import find_available_GPUs, setup_cuda
from ..detector import add_detectors_on_earth

from pycbc.detector import add_detector_on_earth, _ground_detectors


def test_detector():
    
    latitude = 0.3
    longitude = 0.5
    
    detectors = add_detectors_on_earth(
        tf.constant([latitude]), 
        tf.constant([longitude]),
        yangle=tf.constant([0.0]),  # Batched tensor
        xangle=None,  # Batched tensor or None
        height=tf.constant([0.0]),  # Batched tensor
        xlength=tf.constant([4000.0]),  # Batched tensor
        ylength=tf.constant([4000.0])   # Batched tensor
    )
    
    add_detector_on_earth(
        "test", 
        latitude, 
        longitude,              
        )
    
    print(detectors)
    
    print("\n--------\n")
    
    print(_ground_detectors["test"])
    

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