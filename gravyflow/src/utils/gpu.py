import os
import logging
import sys
import subprocess
from pathlib import Path
from typing import Union, List

import numpy as np
import tensorflow as tf
#from filelock import Timeout, FileLock

from scipy.stats import truncnorm
from tensorflow.python.distribute.distribute_lib import Strategy


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

    if "cuda_version" in tf.sysconfig.get_build_info():
        cuda_version = tf.sysconfig.get_build_info()['cuda_version']
        logging.info(
            f"TensorFlow version: {tf_version}, CUDA version: {cuda_version}"
        )
    else:
        logging.info("Running in CPU mode...")
    
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
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

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

    if not Path("/usr/bin/nvidia-smi").exists():
        return None
    
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
        raise CalledProcessError(
            f"Unable to run NVIDIA-SMI. Please check your environment. Exiting!"
            " Error: {e.output}"
        )
    
    # Split the output into lines
    memory_array = output.split("\n")
    # Remove the last empty line if it exists
    if memory_array[-1] == "":
        memory_array = memory_array[:-1]

    # Convert to integers
    return np.array(memory_array, dtype=int)

def get_gpu_utilization_array():
    # Run the NVIDIA-SMI command

    if not Path("/usr/bin/nvidia-smi").exists():
        return None

    try:
        output = subprocess.check_output(
            [
                "/usr/bin/nvidia-smi", 
                "--query-gpu=utilization.gpu",  # Querying GPU utilization
                "--format=csv,noheader,nounits"  # Formatting the output
            ], 
            stderr=subprocess.STDOUT, 
            universal_newlines=True
        )
    except subprocess.CalledProcessError as e:
        print((
            "Unable to run NVIDIA-SMI. Please check your environment. Exiting!",
            f" Error: {e.output}"
        ))
        return None

    # Split the output into lines
    utilization_array = output.split("\n")
    # Remove the last empty line if it exists
    if utilization_array[-1] == "":
        utilization_array = utilization_array[:-1]

    # Convert to integers
    return np.array(utilization_array, dtype=int)

def find_available_GPUs(
    min_memory_MB: int = None, 
    max_utilization_percentage: float = 80.0,
    max_needed: int = -1
):
    """
    Finds the available GPUs that have memory available more than min_memory.

    Parameters
    ----------
    min_memory_MB : int
        The minimum free memory required.
    max_utilization_percentage : float
        The maximum utilization percentage allowed.
    max_needed : int
        The maximum number of GPUs needed.

    Returns
    -------
    available_gpus : str
        The list of indices of available GPUs in string form for easy digestion
        by setup_cuda above, sorted by most free memory to least.
    """
    memory_array = get_memory_array()  # Assume this function exists and returns GPU memory array
    utilization_array = get_gpu_utilization_array()  # Assume this function exists and returns GPU utilization array

    if memory_array is None:
        raise MemoryError("No GPUs with requested memory available!")
    if utilization_array is None:
        raise MemoryError("No GPUs with requested utilization available!")

    # Filter GPUs based on memory and utilization criteria
    gpu_indices = np.arange(len(memory_array))
    filtered_indices = [
        i for i in gpu_indices
        if memory_array[i] > min_memory_MB and utilization_array[i] < max_utilization_percentage
    ]

    # Sort filtered GPUs by available memory in descending order
    sorted_gpus = sorted(filtered_indices, key=lambda i: memory_array[i], reverse=True)

    # Limit the number of GPUs if max_needed is specified
    if max_needed != -1:
        sorted_gpus = sorted_gpus[:max_needed]

    return ",".join(str(gpu) for gpu in sorted_gpus)

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
         
def env(
        min_gpu_memory_mb : int = 5000,
        max_gpu_utilization_percentage : float = 80,
        num_gpus_to_request : int = 1,
        memory_to_allocate_tf : int = 3000,
        gpus : Union[str, int, List[Union[int, str]], None]= None
    ) -> tf.distribute.Strategy:
    
    # Check if there's already a strategy in scope:
    current_strategy = tf.distribute.get_strategy()
            
    def is_default_strategy(strategy):
        return "DefaultDistributionStrategy" in str(strategy)

    if not is_default_strategy(current_strategy):
        logging.info("A TensorFlow distributed strategy is already in place.")
        return current_strategy.scope()

    if gpus is not None:
        # Verify type of gpus:
        if isinstance(gpus, int):
            gpus = str(int)
        elif isinstance(gpus, list) or isinstance(gpus, tuple):
            gpus = [str(gpu) for gpu in gpus].join(",")    
        elif not isinstance(gpus, str):
            raise ValueError("gpus should be int, str, or list of int or str")
    elif gpus is None:
        # Setup CUDA:
        gpus = find_available_GPUs(
            min_memory_MB=min_gpu_memory_mb, 
            max_utilization_percentage=max_gpu_utilization_percentage,
            max_needed=num_gpus_to_request
        )

    if not gpus:
        raise Exception("No GPUs avalible as they were too busy or not existant.")
    elif len(gpus) != num_gpus_to_request:
        logging.warning(f"Could not allocate requested number of GPUs ({num_gpus_to_request}) as they were too busy or not existant. Allocated {len(gpus) } instead.")
    
    strategy = setup_cuda(
        gpus, 
        max_memory_limit=memory_to_allocate_tf, 
        logging_level=logging.WARNING
    )
    
    return strategy.scope()