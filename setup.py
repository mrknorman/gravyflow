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

logging.basicConfig(level=logging.INFO)

def setup_cuda(
        device_num: str, 
        max_memory_limit: int, 
        logging_level: int = logging.WARNING
    ) -> tf.distribute.Strategy:
    """
    Sets up CUDA for TensorFlow. Configures memory growth, logging verbosity, 
    and returns the strategy for distributed computing.

    Args:
        device_num (str): 
            The GPU device number to be made visible for TensorFlow.
        max_memory_limit (int): 
            The maximum GPU memory limit in MB.
        logging_level (int, optional): 
            Sets the logging level. Defaults to logging.WARNING.

    Returns:
        tf.distribute.MirroredStrategy: 
            The TensorFlow MirroredStrategy instance.
    """

    # Set up logging to file - this is beneficial in debugging scenarios and for 
    # traceability.
    logging.basicConfig(filename='tensorflow_setup.log', level=logging_level)
    
    # Set the device number for CUDA to recognize.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)

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
        # Currently, memory growth needs to be the same across GPUs.
        # Enable memory growth for each GPU and set memory limit.
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=max_memory_limit
                    )
                ]
            )

    # Set the logging level to ERROR to reduce logging noise.
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # MirroredStrategy performs synchronous distributed training on multiple 
    # GPUs on one machine. It creates one replica of the model on each GPU 
    # available:
    strategy = tf.distribute.MirroredStrategy()

    # If verbose, print the list of GPUs.
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
        print(
            f"Unable to run NVIDIA-SMI. Please check your environment. Exiting!"
            " Error: {e.output}"
        )
        return None

    # Split the output into lines
    memory_array = output.split("\n")
    # Remove the last empty line if it exists
    if memory_array[-1] == "":
        memory_array = memory_array[:-1]

    # Convert to integers
    memory_array = np.array(memory_array, dtype=int)
    
    # Find the indices of GPUs which have available memory more than 
    # min_memory_MB
    available_gpus = np.where(memory_array > min_memory_MB)[0].tolist()
    
    if (max_needed != -1) and (max_needed < len(available_gpus)):
        available_gpus = available_gpus[:-max_needed-1:-1]

    return ",".join(str(gpu) for gpu in available_gpus)

def get_element_shape(dataset):
    for element in dataset.take(1):
        return element[0].shape
    
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
        
def get_tf_memory_usage() -> int:
    """Get TensorFlow's current GPU memory usage for a specific device.
    
    Returns
    -------
    int
        The current memory usage in megabytes.
    """
    
    # Extract device index
    device_index = int(
        tf.config.list_physical_devices("GPU")[0].name.split(":")[-1]
    )
    
    device_name = f"GPU:{device_index}"
    memory_info = tf.config.experimental.get_memory_info(device_name)
    return memory_info["current"] // (1024 * 1024)

def replace_placeholders(
        value: dict, 
        replacements: dict
    ) -> None:
        
    """Replace placeholders in the config dictionary with actual values."""
    for k in ["value", "max_", "min_", "type_"]:

        if isinstance(value, dict):
            if k in value:
                value[k] = replacements.get(value[k], value[k])
                
def env(
    min_gpu_memory_mb: int = 4000,
    num_gpus_to_request: int = 1,
    memory_to_allocate_tf: int = 2000
) -> tf.distribute.Strategy:
    
    # Check if there's already a strategy in scope:
    current_strategy = tf.distribute.get_strategy()
        
    if isinstance(current_strategy, tf.distribute.Strategy):
        logging.info("A TensorFlow distributed strategy is already in place.")
        return current_strategy.scope()

    # Setup CUDA
    gpus = find_available_GPUs(
        min_gpu_memory_mb, 
        num_gpus_to_request
    )
    strategy = setup_cuda(
        gpus, 
        max_memory_limit=memory_to_allocate_tf, 
        logging_level=logging.WARNING
    )    
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    return strategy.scope()
    
    