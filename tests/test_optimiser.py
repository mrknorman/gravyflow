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
        num_tests : int = 5
    ):
        
    max_num_inital_layers : int = 4
        
    # Define injection directory path:
    injection_directory_path : Path = (
        gf.tests.PATH / "example_injection_parameters"
    )

    # Load injection config:
    phenom_d_generator : gf.cuPhenomDGenerator = gf.WaveformGenerator.load(
        injection_directory_path / "phenom_d_parameters.json"
    )
    
    # Setup ifo data acquisition object:
    ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
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
    noise_obtainer: gf.NoiseObtainer = gf.NoiseObtainer(
        ifo_data_obtainer = ifo_data_obtainer,
        noise_type = gf.NoiseType.REAL,
        ifos = gf.IFO.L1
    )
    
    generator = gf.Dataset(,
        # Noise: 
        noise_obtainer=noise_obtainer,
        # Injections:
        waveform_generators=phenom_d_generator, 
        # Output configuration:
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
            (["Dense", "Convolutional"], param_limits) for i in range(max_num_inital_layers)
        ]
    }

    num_generations : int = 5
    num_population_members : int = 5
    default_genome: gf.ModelGenome = None
    population_directory_path : Path = gf.PATH.parent() / "gravyflow_data/tests/optimiser_test_population"

    population = gf.Population(
        num_population_members=num_population_members,
        default_genome=default_genome,
        population_directory_path=population_directory_path
    )

    population.train(
        num_generations=num_generations
    )