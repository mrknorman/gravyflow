# Built-In imports:
import logging
from pathlib import Path
from itertools import islice
import sys

# Library imports:
import numpy as np
import tensorflow as tf
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from tqdm import tqdm

# Local imports:
import gravyflow as gf

def test_noise_memory(
        num_tests : int = int(1.0E2)
    ):
    
    # Test parameters:
    sample_rate_hertz : float = 1024.0
    onsource_duration_seconds : float = 1.0    
    crop_duration_seconds : float = 0.5
    offsource_duration_seconds : float = 4.0
    num_examples_per_batch : int = 32
    scale_factor : float = 1.0E20
    
    num_batches : int = num_tests//num_examples_per_batch
    
    # Setup ifo data acquisition object:
    ifo_data_obtainer : gf.IFODataObtainer = \
        gf.IFODataObtainer(
            gf.ObservingRun.O3, 
            gf.DataQuality.BEST, 
            [
                gf.DataLabel.NOISE, 
                gf.DataLabel.GLITCHES
            ],
            gf.SegmentOrder.RANDOM,
            force_acquisition=True,
            cache_segments=False,
            logging_level=logging.INFO
        )
    
    # Initilise noise generator wrapper:
    noise : gf.NoiseObtainer = \
        gf.NoiseObtainer(
            ifo_data_obtainer = ifo_data_obtainer,
            noise_type = gf.NoiseType.REAL,
            ifos = gf.IFO.L1
        )
    
    # Set logging level:
    logger = logging.getLogger("memory_logger")
    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    for _ in range(10):
        # Create generator:
        generator : Iterator = \
            noise.init_generator(
                sample_rate_hertz,
                onsource_duration_seconds,
                crop_duration_seconds,
                offsource_duration_seconds,
                num_examples_per_batch,
                scale_factor
            )

        # Measure memory before loop
        memory_before_loop_mb = gf.get_tf_memory_usage()

        logging.info("Start iteration tests...")
        for index, _ in tqdm(enumerate(islice(generator, num_batches))):
            pass

        # Measure memory after loop
        memory_after_loop_mb = gf.get_tf_memory_usage()

        # Calculate the difference
        memory_difference_mb = memory_after_loop_mb - memory_before_loop_mb

        logger.info(f"Memory before loop: {memory_before_loop_mb} MB")
        logger.info(f"Memory after loop: {memory_after_loop_mb} MB")
        logger.info(f"Memory consumed by loop: {memory_difference_mb} MB")

if __name__ == "__main__":
    
    # ---- User parameters ---- #
    
    # GPU setup:
    min_gpu_memory_mb : int = 4000
    num_gpus_to_request : int = 1
    memory_to_allocate_tf : int = 2000
    
    # Setup CUDA
    gpus = gf.find_available_GPUs(min_gpu_memory_mb, num_gpus_to_request)
    strategy = gf.setup_cuda(
        gpus, 
        max_memory_limit = memory_to_allocate_tf, 
        logging_level=logging.WARNING
    )    
    
    # Test gf.IFO noise dataset:
    with strategy.scope():
        test_noise_memory()