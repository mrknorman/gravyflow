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
from tensorflow.keras.callbacks import Callback

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
    # Set the TF_GPU_THREAD_MODE environment variable
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

    # Confirm TensorFlow and CUDA version compatibility.
    tf_version = tf.__version__
    cuda_version = tf.sysconfig.get_build_info()['cuda_version']
    logging.info(
        f"TensorFlow version: {tf_version}, CUDA version: {cuda_version}"
    )

    # Step 1: Set the mixed precision policy
    #tf.keras.mixed_precision.set_global_policy('mixed_float16')

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

def get_memory_array():
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
    return np.array(memory_array, dtype=int)

def find_available_GPUs(
    min_memory_MB : int = None, 
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
    
    memory_array = get_memory_array()
    
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
        logger = None,
        mode : str ='r+'
    ) -> h5py.File:
    
    file_path = Path(file_path)
    try:
        # Try to open the HDF5 file in the specified mode
        f = h5py.File(file_path, mode)
        f.close()
    except OSError:
        # The file does not exist, so create it in write mode
        f = h5py.File(file_path, 'w') #swmr=True)
        f.close()

        if logger is not None:
            logger.info(f'The file {file_path} was created in write mode.')
    else:
        if logger is not None:
            logger.info(f'The file {file_path} was opened in {mode} mode.')
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
    memory_to_allocate_tf: int = 2000,
    gpus: Union[int, str] = None
) -> tf.distribute.Strategy:
    
    # Check if there's already a strategy in scope:
    current_strategy = tf.distribute.get_strategy()
            
    def is_default_strategy(strategy):
        return "DefaultDistributionStrategy" in str(strategy)

    if not is_default_strategy(current_strategy):
        logging.info("A TensorFlow distributed strategy is already in place.")
        return current_strategy.scope()
    
    # Setup CUDA

    if gpus is None:
        gpus = find_available_GPUs(
            min_gpu_memory_mb, 
            num_gpus_to_request
        )
        print(gpus)

    strategy = setup_cuda(
        gpus, 
        max_memory_limit=memory_to_allocate_tf, 
        logging_level=logging.WARNING
    )    
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    return strategy.scope()
    
def is_redirected():
    return (
        not sys.stdin.isatty() or
        not sys.stdout.isatty() or
        not sys.stderr.isatty()
    )

def save_dict_to_hdf5(data_dict, filepath, force_overwrite=False):
    # If force_overwrite is False and the file exists, try to append the data
    if not force_overwrite and os.path.isfile(filepath):
        with h5py.File(filepath, 'a') as hdf:  # Open in append mode
            for key, data in data_dict.items():
                if key in hdf:
                    # Append the new data to the existing data
                    hdf[key].resize((hdf[key].shape[0] + len(data)), axis=0)
                    hdf[key][-len(data):] = data
                else:
                    # Create a new dataset if the key doesn't exist
                    hdf.create_dataset(key, data=data, maxshape=(None,))
            print(f"Data appended to {filepath}")
    else:
        # If the file doesn't exist or force_overwrite is True, create a new file
        with h5py.File(filepath, 'w') as hdf:  # Open in write mode
            for key, data in data_dict.items():
                # Create datasets, allowing them to grow in size (maxshape=(None,))
                hdf.create_dataset(key, data=data, maxshape=(None,))
            print(f"Data saved to new file {filepath}")

class CustomHistorySaver(Callback):
    def __init__(self, filepath, force_overwrite=False):
        super().__init__()
        self.filepath = filepath
        self.force_overwrite = force_overwrite

    def on_epoch_end(self, epoch, logs=None):
        # This method is called when the epoch ends
        if logs is not None:
            # Convert logs to a format that can be saved, which is a dict of lists
            history_dict = {k: [v] for k, v in logs.items()}
            # Call the save function

            ensure_directory_exists(self.filepath)
            save_dict_to_hdf5(history_dict, self.filepath / "history.hdf5", self.force_overwrite)