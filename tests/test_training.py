# Built-In imports:
import logging
from pathlib import Path
from copy import deepcopy

# Library imports:
import numpy as np
import tensorflow as tf
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from tensorflow.keras import losses, metrics, optimizers, mixed_precision

# Local imports:
import gravyflow as gf

def test_training(
    num_train_examples : int = int(1.0E6),
    num_validation_examples : int = int(1.0E3),
    output_diretory_path : Path = Path("./py_ml_data/tests/")
    ):
    
    # Test Parameters:
    num_examples_per_generation_batch : int = 2048
    num_examples_per_batch : int = 32
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 1.0
    offsource_duration_seconds : float = 16.0
    crop_duration_seconds : float = 0.5
    
    training_config = \
    {
        "num_examples_per_epoc" : num_train_examples,
        "patience" : 3,
        "learning_rate" : 1e-4,
        "max_epochs" : 10,
        "model_path" : output_diretory_path / "example_cnn/"
    }
    
    # Define injection directory path:
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    injection_directory_path : Path = \
        Path(current_dir / "example_injection_parameters")
    
    # Intilise Scaling Method:
    scaling_method = \
        gf.ScalingMethod(
            gf.Distribution(min_=8.0,max_=15.0,type_=gf.DistributionType.UNIFORM),
            gf.ScalingTypes.SNR
        )
    
    # Load injection config:
    phenom_d_generator: gf.cuPhenomDGenerator = \
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
            force_acquisition = False,
            cache_segments = True
        )
    
    # Initilise noise generator wrapper:
    noise: gf.NoiseObtainer = \
        gf.NoiseObtainer(
            ifo_data_obtainer = ifo_data_obtainer,
            noise_type = gf.NoiseType.REAL,
            ifos = gf.IFO.L1
        )
    
    dataset_args = {
        # Temporal components:
        "sample_rate_hertz":sample_rate_hertz,   
        "onsource_duration_seconds":onsource_duration_seconds,
        "offsource_duration_seconds":offsource_duration_seconds,
        "crop_duration_seconds":crop_duration_seconds,
        #Noise:
        "noise_obtainer" : noise,
        # Injections:
        "injection_generators": phenom_d_generator, 
        # Output configuration:
        "num_examples_per_batch": num_examples_per_batch,
        "input_variables" : [
            gf.ReturnVariables.WHITENED_ONSOURCE, 
        ],
        "output_variables" : [
            gf.ReturnVariables.INJECTION_MASKS, 
        ]
    }
    
    train_args = deepcopy(dataset_args)
    train_args["seed"] = 1000
    train_args["group"] = "train"
    train_args["noise_obtainer"].ifo_data_obtainer.force_acquisition = True
    train_args["noise_obtainer"].ifo_data_obtainer.cache_segments = False
    
    def get_first_injection_features(features, labels):
        labels[gf.ReturnVariables.INJECTION_MASKS.name] = \
            tf.expand_dims(labels[gf.ReturnVariables.INJECTION_MASKS.name][0], 1)
        
        # Check for NaN in features
        for key, feature_tensor in features.items():
            tf.debugging.check_numerics(
                feature_tensor, 
                f"NaN detected in features under key '{key}'."
            )

        # Check for NaN in labels
        for key, label_tensor in labels.items():
            tf.debugging.check_numerics(
                label_tensor, 
                f"NaN detected in labels under key '{key}'."
            )
        
        return features, labels

    
    train_dataset : tf.data.Dataset = gf.Dataset(
        **train_args
    ).map(get_first_injection_features)
        
    validate_args = deepcopy(dataset_args)
    validate_args["seed"] = 1001
    validate_args["group"] = "validate"
    validate_args["noise_obtainer"].ifo_data_obtainer.force_acquisition = True
    validate_args["noise_obtainer"].ifo_data_obtainer.cache_segments = False
    
    validate_dataset : tf.data.Dataset = gf.Dataset(
        **validate_args
    ).map(get_first_injection_features).take(num_validation_examples//num_examples_per_batch)
    
    test_args = deepcopy(dataset_args)
    test_args["seed"] = 1002
    train_args["group"] = "test"
    
    test_dataset : tf.data.Dataset = gf.Dataset(
        **test_args
    ).map(get_first_injection_features)
    
    hidden_layers = [
        gf.ConvLayer(64, 8, "relu"),
        gf.PoolLayer(4),
        gf.ConvLayer(32, 8, "relu"),
        gf.ConvLayer(32, 16, "relu"),
        gf.PoolLayer(4),
        gf.ConvLayer(16, 16, "relu"),
        gf.ConvLayer(16, 32, "relu"),
        gf.ConvLayer(16, 32, "relu"),
        gf.DenseLayer(64)
    ]
    
    # Initilise model
    builder = gf.ModelBuilder(
        hidden_layers, 
        optimizer = \
            optimizers.Adam(learning_rate=training_config["learning_rate"]), 
        loss = losses.BinaryCrossentropy(), 
        batch_size = num_examples_per_batch
    )
    
    num_samples : int = int(sample_rate_hertz * onsource_duration_seconds)
    input_config = {
        "name" : gf.ReturnVariables.WHITENED_ONSOURCE.name,
        "shape" : (num_samples,)
    }
    
    output_config = {
        "name" : gf.ReturnVariables.INJECTION_MASKS.name,
        "type" : "binary"
    }
    
    builder.build_model(
        input_config,
        output_config
    )
        
    builder.summary()
    
    builder.train_model(
        train_dataset,
        validate_dataset,
        training_config
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
    
    # Test model_training:
    with strategy.scope():
        test_training()
    