from pathlib import Path
import logging
import os

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import mixed_precision, layers
from tensorflow.keras import backend as K
from tensorflow.data.experimental import AutoShardPolicy

# Local imports:
import gravyflow as gf

def test_model(
        num_tests : int = 32
    ):
    
    # Test Parameters:
    num_examples_per_generation_batch : int = 2048
    num_examples_per_batch : int = num_tests
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 1.0
    offsource_duration_seconds : float = 16.0
    crop_duration_seconds : float = 0.5
    scale_factor : float = 1.0E21
    
    max_populaton : int = 10
    max_num_inital_layers : int = 10
    
    num_train_examples : int = int(1.0E3)
    num_validate_examples : int = int(1.0E2)
    
    # Intilise gf.Scaling Method:
    scaling_method = \
        gf.ScalingMethod(
            gf.Distribution(min_=8.0,max_=15.0,type_=gf.DistributionType.UNIFORM),
            gf.ScalingTypes.SNR
        )
    
    # Define injection directory path:
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    injection_directory_path : Path = \
        Path(current_dir / "example_injection_parameters")

    # Load injection config:
    phenom_d_generator_high_mass : gf.cuPhenomDGenerator = \
        gf.WaveformGenerator.load(
            injection_directory_path / "phenom_d_parameters.json", 
            sample_rate_hertz, 
            onsource_duration_seconds,
            scaling_method=scaling_method
        )
    
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
            force_acquisition = True,
            cache_segments = False
        )
    
    # Initilise noise generator wrapper:
    noise_obtainer: gf.NoiseObtainer = \
        gf.NoiseObtainer(
            ifo_data_obtainer = ifo_data_obtainer,
            noise_type = gf.NoiseType.REAL,
            ifos = gf.IFO.L1
        )
    
    generator = gf.Dataset(
        # Random Seed:
        seed= 1000,
        # Temporal components:
        sample_rate_hertz=sample_rate_hertz,   
        onsource_duration_seconds=onsource_duration_seconds,
        offsource_duration_seconds=offsource_duration_seconds,
        crop_duration_seconds=crop_duration_seconds,
        # Noise: 
        noise_obtainer=noise_obtainer,
        # Injections:
        injection_generators=phenom_d_generator_high_mass, 
        # Output configuration:
        num_examples_per_batch=num_examples_per_batch,
        input_variables = [
            gf.ReturnVariables.WHITENED_ONSOURCE, 
            gf.ReturnVariables.INJECTION_MASKS, 
            gf.ReturnVariables.INJECTIONS,
            gf.ReturnVariables.WHITENED_INJECTIONS,
            gf.WaveformParameters.MASS_1_MSUN, 
            gf.WaveformParameters.MASS_2_MSUN
        ],
    )
        
    optimizer = gf.HyperParameter(
            {"type" : "list", "values" : ['adam']}
        )
    num_layers = gf.HyperParameter(
            {"type" : "int_range", "values" : [1, max_num_inital_layers]}
        )
    batch_size = gf.HyperParameter(
            {"type" : "list", "values" : [num_examples_per_batch]}
        )
    activations = gf.HyperParameter(
            {"type" : "list", "values" : ['relu', 'elu', 'sigmoid', 'tanh']}
        )
    d_units = gf.HyperParameter(
            {"type" : "power_2_range", "values" : [16, 256]}
        )
    filters = gf.HyperParameter(
            {"type" : "power_2_range", "values" : [16, 256]}
        )
    kernel_size = gf.HyperParameter(
            {"type" : "int_range", "values" : [1, 7]}
        )
    strides = gf.HyperParameter(
            {"type" : "int_range", "values" : [1, 7]}
        )

    param_limits = {
        "Dense" : gf.DenseLayer(d_units,  activations),
        "Convolutional":  gf.ConvLayer(
            filters, kernel_size, activations, strides
        )
    }

    genome_template = {
        'base' : {
            'optimizer'  : optimizer,
            'num_layers' : num_layers,
            'batch_size' : batch_size
        },
        'layers' : [
            (["Dense", "Convolutional"], param_limits) \
            for i in range(max_num_inital_layers)
        ]
    }
    
    population = gf.Population(
        10, 
        15, 
        genome_template,
        int(sample_rate_hertz*onsource_duration_seconds),
        2
    )
    population.train_population(
        100, 
        num_train_examples, 
        num_validate_examples, 
        num_examples_per_batch, 
        generator
    )
        
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
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    # Test Genetic Algorithm Optimiser:
    with strategy.scope():
        test_model()