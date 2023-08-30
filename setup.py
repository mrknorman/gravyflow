import os
import logging
import json
import sys
import h5py
import subprocess
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf

from scipy.stats import truncnorm
from tensorflow.python.distribute.distribute_lib import Strategy
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.data import Dataset

def setup_cuda(
        device_num: str, 
        max_memory_limit: int, 
        logging_level: int = logging.WARNING
    ) -> Strategy:
    
    """
    Sets up CUDA for TensorFlow. Configures memory growth, logging verbosity, 
    and returns the strategy for distributed computing.

    Args:
        device_num (str): 
            The GPU device number to be made visible for TensorFlow.
        max_memory_limit (int): 
            The maximum GPU memory limit in MB.
        verbose (bool, optional):
            If True, prints the list of GPU devices. Defaults to False.

    Returns:
        tf.distribute.MirroredStrategy: The TensorFlow MirroredStrategy instance.
    """

    # Set up logging to file - this is beneficial in debugging scenarios and for 
    # traceability.
    logger = logging.getLogger("tensorflow_logger")
    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stream_handler)
    logger.setLevel(logging_level)
    
    try:
        # Set the device number for CUDA to recognize.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    except Exception as e:
        logging.error(
            f"Failed to set CUDA_VISIBLE_DEVICES environment variable: {e}"
        )
        raise

    # Confirm TensorFlow and CUDA version compatibility.
    tf_version = tf.__version__
    cuda_version = tf.sysconfig.get_build_info()['cuda_version']
    logging.info(
        f"TensorFlow version: {tf_version}, CUDA version: {cuda_version}"
    )

    # List all the physical GPUs.
    gpus = tf.config.list_physical_devices('GPU')

    # If any GPU is present.
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs.
            # Enable memory growth for each GPU and set memory limit.
            for gpu in gpus:
                # Limit the GPU memory.
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=max_memory_limit)])
                
                #tf.config.experimental.set_memory_growth(gpu, False)

        except RuntimeError as e:
            # This needs to be set before initializing GPUs.
            logging.error(
                f"Failed to set memory growth or set memory limit: GPUs must be"
                " initialized first. Error message: {e}")
            raise

    # MirroredStrategy performs synchronous distributed training on multiple GPUs on one machine.
    # It creates one replica of the model on each GPU available.
    strategy = tf.distribute.MirroredStrategy()

    # Set the logging level to ERROR to reduce logging noise.
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # If logging level = logging.INFO, print the list of GPUs.
    logging.info(tf.config.list_physical_devices("GPU"))

    # Return the MirroredStrategy instance.
    return strategy

def find_available_GPUs(
    min_memory_MB : int, 
    max_needed : int = -1
    ):
    """
    Finds the available GPUs that have memory available more than min_memory.

    Parameters
    ----------
    min_memory_MB : int
        The minimum free memory required.

    Returns
    -------
    available_gpus : str
        The list of indices of available GPUs ins string form for easy digestion
        by setup_cuda above.
    """
    
    # Run the NVIDIA-SMI command
    try:
        output = subprocess.check_output(
            [
                "/usr/bin/nvidia-smi", 
                 "--query-gpu=memory.free", 
                 "--format=csv,noheader,nounits"
            ], 
            stderr=subprocess.STDOUT, 
            universal_newlines=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Unable to run NVIDIA-SMI. Please check your environment. " \
              "Exiting! Error: {e.output}")
        return None

    # Split the output into lines
    memory_array = output.split("\n")
    # Remove the last empty line if it exists
    if memory_array[-1] == "":
        memory_array = memory_array[:-1]

    # Convert to integers
    memory_array = np.array(memory_array, dtype=int)
    
    # Find the indices of GPUs which have available memory more than min_memory_MB
    available_gpus = np.where(memory_array > min_memory_MB)[0].tolist()
    
    if (max_needed != -1) and (max_needed < len(available_gpus)):
        available_gpus = available_gpus[:-max_needed-1:-1]

    return ",".join(str(gpu) for gpu in available_gpus)

def load_datasets(paths):
    
    dataset = tf.data.experimental.load(paths[0])
    for path in paths[1:]:
        dataset = dataset.concatenate(dataset)
        
    return dataset

def add_labels(dataset, label):
    dataset_size = dataset.cardinality().numpy()

    labels = Dataset.from_tensor_slices(
        np.full(dataset_size, label, dtype=np.float32))
    
    return Dataset.zip((dataset, labels))
    
def load_label_datasets(paths, label):
    
    dataset = load_datasets(paths)
    return add_labels(dataset, label)

def load_noise_signal_datasets(noise_paths, signal_paths):
    
    noise  = load_label_datasets(noise_paths, 0)
    signal = load_label_datasets(signal_paths, 1)

    return signal.concatenate(noise)

def split_test_train(dataset, fraction):
    dataset_size = dataset.cardinality().numpy()
    test_size = fraction * dataset_size

    dataset = dataset.shuffle(dataset_size)
    test_dataset = dataset.take(test_size)
    train_dataset = dataset.skip(test_size)

    return test_dataset, train_dataset

def get_element_shape(dataset):
    for element in dataset.take(1):
        return element[0].shape
    
def randomise_dict(value):
    distribution_type = value.get('distribution_type', 'constant')
    dtype = value.get('dtype', 'float')
    num_values = value.get('num_values', 1)

    if distribution_type == 'constant':
        # Default value is 0 if not provided
        constant_value = float(value.get('value', 0.0)) 
        random_values = [constant_value] * num_values
    else:
        min_value = float(value.get('min_value', '-inf'))
        max_value = float(value.get('max_value', 'inf'))
        mean_value = float(value.get('mean_value', 0.0))
        std = float(value.get('std', 1.0))

        if distribution_type == 'uniform':
            random_values = np.random.uniform(min_value, max_value, num_values)
        elif distribution_type == 'normal':
            random_values = truncnorm.rvs(
                (min_value - mean_value) / std,
                (max_value - mean_value) / std,
                loc=mean_value,
                scale=std,
                size=num_values)
        else:
            raise ValueError('Unsupported distribution type')

    if dtype == 'int':
        random_values = [int(rv) for rv in random_values]
    
    random_values = random_values if num_values > 1 else random_values[0]
    return random_values
    
def randomise_arguments(input_dict, func):
    output_dict = {}
    for key, value in input_dict.items():
        output_dict[key] = randomise_dict(value)

    return func(**output_dict), output_dict
    
def read_injection_config_file(
    config_path: Path,
    sample_rate_hertz: float,
    onsource_duration_seconds: float
) -> dict:
    # Define replacement mapping
    replacements = {
        "sample_rate_hertz": sample_rate_hertz,
        "onsource_duration_seconds": onsource_duration_seconds,
        "pi": np.pi,
        "2*pi": 2.0 * np.pi
    }

    # Load injection config
    with open(config_path, "r") as file:
        config = json.load(file)
    
    # Replacing placeholders
    for key, value in config["args"].items():
        
        if "value" in value:
            value["value"] = \
                replacements.get(value["value"], value["value"])
        if "max_value" in value:
            value["max_value"] = \
                replacements.get(value.get("max_value"), value["max_value"])

    return config

def open_hdf5_file(
    file_path : Union[str, Path], 
    mode : str ='r+'
    ) -> h5py.File:
    
    file_path = Path(file_path)
    try:
        # Try to open the HDF5 file in the specified mode
        f = h5py.File(file_path, mode)
        f.close()
    except OSError:
        # The file does not exist, so create it in write mode
        f = h5py.File(file_path, 'w')
        f.close()
        logging.info(f'The file {file_path} was created in write mode.')
    else:
        logging.info(f'The file {file_path} was opened in {mode} mode.')
    return h5py.File(file_path, mode)

def ensure_directory_exists(
    directory: Union[str, Path]
    ):
    
    directory = Path(directory)  # Convert to Path if not already
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)