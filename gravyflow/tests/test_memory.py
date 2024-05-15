# Built-In imports:
import logging
from pathlib import Path
from typing import Dict
from itertools import islice
import sys

# Library imports:
import numpy as np
import tensorflow as tf
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from tqdm import tqdm
from _pytest.config import Config

# Local imports:
import gravyflow as gf

def _test_memory_noise(
        num_tests : int
    ) -> None:

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
            memory_before_loop_mb = gf.get_tf_memory_usage()

            logging.info("Start iteration tests...")
            for index, _ in tqdm(enumerate(islice(noise(), num_tests))):
                pass

            # Measure memory after loop
            memory_after_loop_mb = gf.get_tf_memory_usage()

            # Calculate the difference
            memory_difference_mb = memory_after_loop_mb - memory_before_loop_mb

            if i > 0: 
                np.testing.assert_allclose(
                    memory_before_loop_mb, 
                    memory_after_loop_mb,             
                    atol=2, 
                    err_msg=f"Memory leak possible! Lost {memory_difference_mb} MB."
                )

            logger.info(f"Memory before loop: {memory_before_loop_mb} MB")
            logger.info(f"Memory after loop: {memory_after_loop_mb} MB")
            logger.info(f"Memory consumed by loop: {memory_difference_mb} MB")

def test_memory_noise(
        pytestconfig : Config
    ) -> None:
    
    _test_memory_noise(
        num_tests=gf.tests.num_tests_from_config(pytestconfig)
    )