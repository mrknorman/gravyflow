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
        print(
            f"Unable to run NVIDIA-SMI. Please check your environment. Exiting! Error: {e.output}"
        )
        return None

    # Split the output into lines
    utilization_array = output.split("\n")
    # Remove the last empty line if it exists
    if utilization_array[-1] == "":
        utilization_array = utilization_array[:-1]

    # Convert to integers
    return np.array(utilization_array, dtype=int)

def find_available_GPUs(
    min_memory_MB : int = None, 
    max_utilization_percent : float = 50,
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
    utilization_array = get_gpu_utilization_array()

    if memory_array is None:
        return "-1"
    if utilization_array is None:
        return "-1"
    
    # Find the indices of GPUs which have available memory more than 
    # min_memory_MB
    available_gpus = list(set(np.where(memory_array > min_memory_MB)[0].tolist()).intersection(np.where(utilization_array < max_utilization_percent)[0].tolist()))
    
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
    gpus: Union[int, str] = None,
    max_needed=1
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
            num_gpus_to_request,
            max_needed=max_needed
        )

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

    ensure_directory_exists(filepath.parent)
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

def load_history(filepath):
    history_path = filepath / "history.hdf5"

    print("path", history_path)
    if os.path.exists(history_path):
        with h5py.File(history_path, 'r') as hfile:
            return {k: list(v) for k, v in hfile.items()}
    else:
        return {}

class CustomHistorySaver(Callback):
    def __init__(self, filepath, force_overwrite=False):
        super().__init__()
        self.filepath = filepath

        if not isinstance(filepath, Path):
            raise ValueError("Filepath must be Path!")

        self.force_overwrite = force_overwrite
        self.history = load_history(filepath) if not self.force_overwrite else {}

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            # Append logs to existing history
            for k, v in logs.items():
                if k in self.history:
                    self.history[k].append(v)
                else:
                    self.history[k] = [v]

            ensure_directory_exists(self.filepath)
            save_dict_to_hdf5(self.history, self.filepath / "history.hdf5", True)
            self.force_overwrite = False
            
class EarlyStoppingWithLoad(Callback):

    def __init__(
        self,
        model_path = None,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0,
    ):
        super().__init__()

        self.model_path = model_path
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.start_from_epoch = start_from_epoch

        if mode not in ["auto", "min", "max"]:
            logging.warning(
                "EarlyStopping mode %s is unknown, fallback to auto mode.",
                mode,
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if (
                self.monitor.endswith("acc")
                or self.monitor.endswith("accuracy")
                or self.monitor.endswith("auc")
            ):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        
        if self.model_path is not None:
            history_data = load_history(self.model_path) 
            # Assuming history_data is a dictionary containing your historical metrics
            last_epoch_metrics = {k: v for k, v in history_data.items()}

            print(last_epoch_metrics)

            if self.monitor in last_epoch_metrics:

                initial_epoch = len(last_epoch_metrics[self.monitor])
                
                if initial_epoch and last_epoch_metrics:
                    # Manually set their internal state
                    
                    # Assuming loss
                    best = min(last_epoch_metrics[self.monitor])
                    best_epoch = np.argmin(last_epoch_metrics[self.monitor]) + 1

                    self.wait = initial_epoch - best_epoch
                    self.stopped_epoch = 0
                    self.best = best
                    self.best_weights = tf.keras.models.load_model(self.model_path).get_weights()
                    self.best_epoch = best_epoch

                else:
                    print("Empty history!")

                    self.wait = 0
                    self.stopped_epoch = 0
                    self.best = np.Inf if self.monitor_op == np.less else -np.Inf
                    self.best_weights = None
                    self.best_epoch = 0
            else:
                raise ValueError("Key not in history dict!")
        else:
            self.wait = 0
            self.stopped_epoch = 0
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
            self.best_weights = None
            self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None or epoch < self.start_from_epoch:
            # If no monitor value exists or still in initial warm-up stage.
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous
            # best.
            if self.baseline is None or self._is_improvement(
                current, self.baseline
            ):
                self.wait = 0
            return

        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    io_utils.print_msg(
                        "Restoring model weights from "
                        "the end of the best epoch: "
                        f"{self.best_epoch + 1}."
                    )
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            io_utils.print_msg(
                f"Epoch {self.stopped_epoch + 1}: early stopping"
            )

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                "Early stopping conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)

class PrintWaitCallback(Callback):
    def __init__(self, early_stopping):
        super().__init__()
        self.early_stopping = early_stopping

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        wait = self.early_stopping.wait
        best = self.early_stopping.best
        best_epoch = self.early_stopping.best_epoch
        print(f"\nBest model so far had a value of: {best} at Epoch: {best_epoch} which was {wait} epochs ago.")