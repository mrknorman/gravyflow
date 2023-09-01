from .dataset import get_ifo_data_generator, get_ifo_data, O3
from .model import ModelBuilder, HyperParameter, DenseLayer, ConvLayer, Population, randomizeLayer
from .setup import load_label_datasets, setup_cuda, get_element_shape

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import mixed_precision, layers
from tensorflow.keras import backend as K
from tensorflow.data.experimental import AutoShardPolicy
import tensorflow as tf
import numpy as np
import os

def load_cbc_datasets(data_directory_path, num_to_load):
    dataset_prefix = f"{data_directory_path}/datasets/cbc"
    dataset_paths = [f"{dataset_prefix}_{i}_v" for i in range(num_to_load)]
    
    # Check for existing dataset paths
    existing_paths = []
    for path in dataset_paths:
        if os.path.exists(path):
            existing_paths.append(path)
        else:
            print(f"Warning: {path} does not exist.")
    
    # Load the datasets with existing paths
    dataset = load_label_datasets(existing_paths, 1)
    
    return dataset

def gaussian_noise_generator(num_samples=8192):
    while True:
        noise = tf.random.normal([num_samples], dtype=tf.float16)
        constant = tf.constant(0.0, dtype=tf.float32)
        yield noise, constant

if __name__ == "__main__":
    
    gpus = find_available_GPUs(16000, 1)
    strategy = setup_cuda(gpus, verbose = True)
        
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    
    num_examples_per_batch = 32
    max_populaton = 10
    max_num_inital_layers = 10
    sample_rate_hertz = 8192.0
    onsource_duration_seconds = 1.0
    
    with strategy.scope():
        
        # Create TensorFlow dataset from the generator
        injection_configs = [
            {
                "type" : "cbc",
                "snr"  : \
                    {"min_value" : 0.0, "max_value": 50, "mean_value": 20, "std": 15,  "distribution_type": "normal"},
                "injection_chance" : 0.5,
                "padding_seconds" : {"front" : 0.2, "back" : 0.1},
                "args" : {
                    "approximant_enum" : \
                        {"value" : 1, "distribution_type": "constant", "dtype" : int}, 
                    "mass_1_msun" : \
                        {"min_value" : 5, "max_value": 95, "distribution_type": "uniform"},
                    "mass_2_msun" : \
                        {"min_value" : 5, "max_value": 95, "distribution_type": "uniform"},
                    "sample_rate_hertz" : \
                        {"value" : sample_rate_hertz, "distribution_type": "constant"},
                    "duration_seconds" : \
                        {"value" : onsource_duration_seconds, "distribution_type": "constant"},
                    "inclination_radians" : \
                        {"min_value" : 0, "max_value": np.pi, "distribution_type": "uniform"},
                    "distance_mpc" : \
                        {"min_value" : 10, "max_value": 1000, "distribution_type": "uniform"},
                    "reference_orbital_phase_in" : \
                        {"min_value" : 0, "max_value": 2*np.pi, "distribution_type": "uniform"},
                    "ascending_node_longitude" : \
                        {"min_value" : 0, "max_value": 2*np.pi, "distribution_type": "uniform"},
                    "eccentricity" : \
                        {"min_value" : 0, "max_value": 0.1, "distribution_type": "uniform"},
                    "mean_periastron_anomaly" : \
                        {"min_value" : 0, "max_value": 2*np.pi, "distribution_type": "uniform"},
                    "spin_1_in" : \
                        {"min_value" : -0.5, "max_value": 0.5, "distribution_type": "uniform", "num_values" : 3},
                    "spin_2_in" : \
                        {"min_value" : -0.5, "max_value": 0.5, "distribution_type": "uniform", "num_values" : 3}
                }
            }
        ]

        # Setting options for data distribution
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
        
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
        
        num_train_examples    = int(1.0E3)
        num_validate_examples = int(1.0E2)
        
        # Creating the noise dataset
        cbc_ds = get_ifo_data_generator(
            time_interval = O3,
            data_labels = ["noise", "glitches"],
            ifo = 'L1',
            injection_configs = injection_configs,
            sample_rate_hertz = sample_rate_hertz,
            onsource_duration_seconds = onsource_duration_seconds,
            max_segment_size = 3600,
            num_examples_per_batch = num_examples_per_batch,
            order = "random",
            seed = 123,
            apply_whitening = True,
            input_keys = ["onsource"], 
            output_keys = ["snr"]
        ).prefetch(tf.data.experimental.AUTOTUNE).with_options(options)
        
        population = Population(10, 15, genome_template, int(sample_rate_hertz*onsource_duration_seconds), 2)
        population.train_population(
            100, 
            num_train_examples, 
            num_validate_examples, 
            num_examples_per_batch, 
            cbc_ds
        )
        
