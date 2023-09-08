# Built-In imports:
import logging
from pathlib import Path

# Library imports:
import numpy as np
import tensorflow as tf
from bokeh.io import output_file, save
from bokeh.layouts import gridplot

# Local imports:
from ..maths import Distribution, DistributionType
from ..setup import find_available_GPUs, setup_cuda, ensure_directory_exists
from ..injection import (cuPhenomDGenerator, InjectionGenerator, 
                         WaveformParameters, WaveformGenerator)
from ..acquisition import (IFODataObtainer, SegmentOrder, ObservingRun, 
                          DataQuality, DataLabel, IFO)
from ..noise import NoiseObtainer, NoiseType
from ..plotting import generate_strain_plot, generate_spectrogram
from ..dataset import get_ifo_dataset, ReturnVariables
from ..validate import Validator

def test_validate(
    output_diretory_path : Path = Path("./py_ml_data/tests/"),
    noise_directory_path : Path = Path("./py_ml_data/test_data/")
    ):
        
    # Test Parameters:
    num_examples_per_generation_batch : int = 2048
    num_examples_per_batch : int = 32
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 1.0
    offsource_duration_seconds : float = 16.0
    crop_duration_seconds : float = 0.5
    scale_factor : float = 1.0E21
    model_path : Path = Path("./py_ml_tools/tests/example_models")
    model_name : str = "example_cnn"
    
    efficiency_config : Dict[str, Union[float, int]] = \
        {
            "max_snr" : 15.0, 
            "num_snr_steps" : 61, 
            "num_examples_per_snr_step" : 16384
        }
    far_config : Dict[str, float] = \
        {
            "num_examples" : 1.0E6
        }
    roc_config : Dict[str, Union[float, List]] = \
        {
            "num_examples" : 1.0E6,
            "snr_ranges" :  [
                (8.0, 20.0),
                6.0,
                8.0,
                10.0,
                12.0
            ]
        }
    
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
            force_acquisition = False,
            cache_segments = True
        )
    
    # Initilise noise generator wrapper:
    noise_obtainer: NoiseObtainer = \
        NoiseObtainer(
            data_directory_path=noise_directory_path,
            ifo_data_obtainer=ifo_data_obtainer,
            noise_type=NoiseType.REAL,
        )
    
    dataset_args : Dict[str, Union[float, List, int]] = {
        # Random Seed:
        "seed" : 1000,
        # Temporal components:
        "sample_rate_hertz" : sample_rate_hertz,   
        "onsource_duration_seconds" : onsource_duration_seconds,
        "offsource_duration_seconds" : offsource_duration_seconds,
        "crop_duration_seconds" : crop_duration_seconds,
        # Noise: 
        "noise_obtainer" : noise_obtainer,
        # Injections:
        "injection_generators" : phenom_d_generator_high_mass, 
        # Output configuration:
        "num_examples_per_batch" : num_examples_per_batch,
        "input_variables" : [
            ReturnVariables.WHITENED_ONSOURCE
        ],
        "output_variables" : [
             ReturnVariables.INJECTION_MASKS
        ]
    }

    logging.info(f"Loading example model...")
    model = tf.keras.models.load_model(
        model_path / model_name
    )
    logging.info("Done.")

    # Validate model:
    validator = \
        Validator.validate(
            model, 
            model_name,
            dataset_args,
            num_examples_per_batch,
            efficiency_config,
            far_config,
            roc_config
        )

    # Save validation data:
    validator.save(
        output_data_directory / f"{model_name}_validation_data.h5", 
    )

    # Plot validation data:
    validator.plot(
        output_data_directory / f"{model_name}_validation_plots.html"
    )

    validators.append(validator)
            
    # Plot all model validation data comparison:            
    validators[0].plot(
        output_data_directory / "test_validation_plots.html",
        comparison_validators = validators[1:]
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
        test_validate()