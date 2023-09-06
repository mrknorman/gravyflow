from pathlib import Path
import logging

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import mixed_precision, layers
from tensorflow.keras import backend as K
from tensorflow.data.experimental import AutoShardPolicy

# Local imports:
from ..maths import Distribution, DistributionType
from ..setup import find_available_GPUs, setup_cuda, ensure_directory_exists
from ..injection import (cuPhenomDGenerator, InjectionGenerator, 
                         WaveformParameters, WaveformGenerator)
from ..plotting import generate_strain_plot
from ..acquisition import (IFODataObtainer, SegmentOrder, ObservingRun, 
                          DataQuality, DataLabel, IFO)
from ..noise import NoiseObtainer, NoiseType
from ..model import (ModelBuilder, HyperParameter, DenseLayer, ConvLayer, 
                    Population, randomizeLayer)
from ..dataset import get_ifo_data, ReturnVariables, get_ifo_data_generator

def test_model(
        num_tests : int = 32,
        output_diretory_path : Path = Path("./py_ml_data/tests/")
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

    # Load injection config:
    phenom_d_generator_high_mass : cuPhenomDGenerator = \
        WaveformGenerator.load(
            Path("./py_ml_tools/tests/injection_parameters.json"), 
            sample_rate_hertz, 
            onsource_duration_seconds,
            snr=Distribution(min_=8.0,max_=15.0,type_=DistributionType.UNIFORM)
        )
    
    # Setup ifo data acquisition object:
    ifo_data_obtainer : IFODataObtainer = \
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
    noise_obtainer: NoiseObtainer = \
        NoiseObtainer(
            ifo_data_obtainer = ifo_data_obtainer,
            noise_type = NoiseType.REAL
        )
    
    generator = get_ifo_data_generator(
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
            ReturnVariables.WHITENED_ONSOURCE, 
            ReturnVariables.INJECTION_MASKS, 
            ReturnVariables.INJECTIONS,
            ReturnVariables.WHITENED_INJECTIONS,
            WaveformParameters.MASS_1_MSUN, 
            WaveformParameters.MASS_2_MSUN
        ],
    )
        
    optimizer = HyperParameter(
            {"type" : "list", "values" : ['adam']}
        )
    num_layers = HyperParameter(
            {"type" : "int_range", "values" : [1, max_num_inital_layers]}
        )
    batch_size = HyperParameter(
            {"type" : "list", "values" : [num_examples_per_batch]}
        )
    activations = HyperParameter(
            {"type" : "list", "values" : ['relu', 'elu', 'sigmoid', 'tanh']}
        )
    d_units = HyperParameter(
            {"type" : "power_2_range", "values" : [16, 256]}
        )
    filters = HyperParameter(
            {"type" : "power_2_range", "values" : [16, 256]}
        )
    kernel_size = HyperParameter(
            {"type" : "int_range", "values" : [1, 7]}
        )
    strides = HyperParameter(
            {"type" : "int_range", "values" : [1, 7]}
        )

    param_limits = {
        "Dense" : DenseLayer(d_units,  activations),
        "Convolutional":  ConvLayer(filters, kernel_size, activations, strides)
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
    
    population = Population(10, 15, genome_template, int(sample_rate_hertz*onsource_duration_seconds), 2)
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
    gpus = find_available_GPUs(min_gpu_memory_mb, num_gpus_to_request)
    strategy = setup_cuda(
        gpus, 
        max_memory_limit = memory_to_allocate_tf, 
        logging_level=logging.WARNING
    )    
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    with strategy.scope():
        test_model()