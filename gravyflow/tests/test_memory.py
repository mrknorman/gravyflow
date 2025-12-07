# Built-In imports:
import logging
from pathlib import Path
from typing import Dict
from itertools import islice
import sys
import gc

# Library imports:
import gravyflow as gf
import numpy as np
import jax
import pytest
from tqdm import tqdm
from _pytest.config import Config

# Local imports:
import gravyflow as gf

def get_jax_memory_usage():
    """Get approximate memory usage in MB."""
    # Force garbage collection
    gc.collect()
    # JAX doesn't have a direct memory query like TF
    # Use device memory stats if available
    try:
        devices = jax.devices()
        if devices and hasattr(devices[0], 'memory_stats'):
            stats = devices[0].memory_stats()
            if stats:
                return stats.get('bytes_in_use', 0) / (1024 * 1024)
    except:
        pass
    return 0.0

@pytest.mark.slow
def test_memory_noise(
        pytestconfig : Config
    ) -> None:
    
    num_tests = gf.tests.num_tests_from_config(pytestconfig)

    with gf.env():
        # Setup ifo data acquisition object:
        ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
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
        noise : gf.NoiseObtainer = gf.NoiseObtainer(
                ifo_data_obtainer = ifo_data_obtainer,
                noise_type = gf.NoiseType.REAL,
                ifos = gf.IFO.L1
            )
        
        # Set logging level:
        logger = logging.getLogger("memory_logger")
        logger.setLevel(logging.INFO)

        for i in range(10):
            # Measure memory before loop
            memory_before_loop_mb = get_jax_memory_usage()

            logging.info("Start iteration tests...")
            for index, _ in tqdm(enumerate(islice(noise(), num_tests))):
                pass

            # Measure memory after loop
            memory_after_loop_mb = get_jax_memory_usage()

            # Calculate the difference
            memory_difference_mb = memory_after_loop_mb - memory_before_loop_mb

            if i > 0 and memory_before_loop_mb > 0: 
                np.testing.assert_allclose(
                    memory_before_loop_mb, 
                    memory_after_loop_mb,             
                    atol=2, 
                    err_msg=f"Memory leak possible! Lost {memory_difference_mb} MB."
                )

            logger.info(f"Memory before loop: {memory_before_loop_mb} MB")
            logger.info(f"Memory after loop: {memory_after_loop_mb} MB")
            logger.info(f"Memory consumed by loop: {memory_difference_mb} MB")