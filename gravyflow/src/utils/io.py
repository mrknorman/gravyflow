from typing import Union, Optional
from pathlib import Path
import inspect
import sys
import os
import logging

import numpy as np
import keras
from keras.callbacks import Callback

import h5py

def get_file_parent_path() -> Optional[Path]:
    """
    Returns the absolute path of the script/file that calls this function, with added error checking.

    Returns:
        Optional[Path]: The absolute path of the directory containing the caller file, or None if not found.
    """
    try:
        # Get the frame of the caller
        caller_frame = inspect.stack()[1]
        # Extract the file path from the frame
        caller_path = caller_frame.filename
        # Return the absolute path of the directory containing the file
        return Path(caller_path).parent.resolve()
    except IndexError:
        # This may occur if the call stack isn't accessible
        print("Error: Could not access the call stack.")
    except AttributeError:
        # This may occur if the caller frame does not have a 'filename' attribute
        print("Error: Caller frame does not have a 'filename' attribute.")
    except Exception as e:
        # Catch-all for any other unforeseen errors
        print(f"An unexpected error occurred: {e}")

    # Return None if we couldn't get the caller path
    return None

def is_redirected():
    return (
        not sys.stdin.isatty() or
        not sys.stdout.isatty() or
        not sys.stderr.isatty()
    )

def load_history(filepath):
    history_path = filepath / "history.hdf5"
    
    if os.path.exists(history_path):
        with h5py.File(history_path, 'r') as hfile:
            return {k: list(v) for k, v in hfile.items()}
    else:
        return {}

def replace_placeholders(
        input: dict, 
        replacements: dict
    ) -> None:
        
    """Replace placeholders in the config dictionary with actual values."""
    
    if isinstance(input, dict):
        for key, value in input.items():
            if isinstance(value, list) or isinstance(value, dict):
                input[key] = replace_placeholders(value, replacements)
            else:
                input[key] = replacements.get(value, value)
    elif isinstance(input, list):
        for index, item in enumerate(input):

            if isinstance(item, list) or isinstance(item, dict):
                input[index] = replace_placeholders(item, replacements)
            else:
                input[index] = replacements.get(item, item)
    else:
        raise ValueError('Item not list or dict')
    
    return input

def open_hdf5_file(
        file_path: Union[str, Path], 
        logger = None,
        mode: str ='r+'
    ) -> h5py.File:
    
    file_path = Path(file_path)

    try:
        # Try to open the HDF5 file in the specified mode
        f = h5py.File(file_path, mode, swmr=True)
        f.close()
    except OSError as e:
        # The file does not exist, so create it in write mode
        f = h5py.File(file_path, 'w', swmr=True)  # You can add swmr=True if needed
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

def snake_to_capitalized_spaces(snake_str: str) -> str:
    return ' '.join(word.capitalize() for word in snake_str.split('_'))

def transform_string(s):
    # Remove the 'perceptron_' prefix and split by underscore
    name = s.replace('model_', '')

    return snake_to_capitalized_spaces(name)

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
        restore_best_weights=True,
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
                    # Use keras.models.load_model instead of tf.keras.models.load_model
                    self.best_weights = keras.models.load_model(self.model_path).get_weights()
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
                    print(
                        "Restoring model weights from "
                        "the end of the best epoch: "
                        f"{self.best_epoch + 1}."
                    )
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(
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
