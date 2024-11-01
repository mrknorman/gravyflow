# Built-In imports:
import logging
from pathlib import Path
import os

# Library imports:
import numpy as np
import tensorflow as tf
from bokeh.io import output_file, save
from bokeh.layouts import gridplot

# Local imports:
import gravyflow as gf

def test_validate(
    output_directory_path : Path = Path("./py_ml_data/tests/"),
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
    model_path : Path = Path("./gravitationalflow/tests/example_models")
    model_name : str = "example_cnn"
    
    efficiency_config : Dict[str, Union[float, int]] = \
        {
            "max_scaling" : 15.0, 
            "num_scaling_steps" : 31, 
            "num_examples_per_scaling_step" : 2048
        }
    far_config : Dict[str, float] = \
        {
            "num_examples" : 1.0E4
        }
    roc_config : Dict[str, Union[float, List]] = \
        {
            "num_examples" : 1.0E4,
            "scaling_ranges" :  [
                (8.0, 20.0),
                6.0
            ]
        }
    
    # Intilise Scaling Method:
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
            Path(injection_directory_path / "phenom_d_parameters.json"), 
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
                gfDataLabel.NOISE, 
                gfDataLabel.GLITCHES
            ],
            gf.SegmentOrder.RANDOM,
            force_acquisition = False,
            cache_segments = True
        )
    
    # Initilise noise generator wrapper:
    noise_obtainer: gf.NoiseObtainer = \
        gf.NoiseObtainer(
            data_directory_path=noise_directory_path,
            ifo_data_obtainer=ifo_data_obtainer,
            noise_type=gf.NoiseType.REAL,
            ifos=gf.IFO.L1
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
            gf.ReturnVariables.WHITENED_ONSOURCE
        ],
        "output_variables" : [
             gf.ReturnVariables.INJECTION_MASKS
        ]
    }

    logging.info(f"Loading example model...")
    model = tf.keras.models.load_model(
        model_path / model_name
    )
    logging.info("Done.")

    # Validate model:
    validator = \
        gf.Validator.validate(
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
        output_directory_path / f"{model_name}_validation_data.h5", 
    )

    # Plot validation data:
    validator.plot(
        output_directory_path / f"{model_name}_validation_plots.html"
    )
            
if __name__ == "__main__":
        
     # ---- User parameters ---- #
    
    # GPU setup:
    min_gpu_memory_mb : int = 10000
    num_gpus_to_request : int = 1
    memory_to_allocate_tf : int = 8000
    
    # Setup CUDA
    gpus = gf.find_available_GPUs(min_gpu_memory_mb, num_gpus_to_request)
    strategy = gf.setup_cuda(
        gpus, 
        max_memory_limit = memory_to_allocate_tf, 
        logging_level=logging.WARNING
    )    
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)

    with strategy.scope():
        test_validate()