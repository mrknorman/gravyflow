from typing import Union
from pathlib import Path
import sys

import h5py

def is_redirected():
    return (
        not sys.stdin.isatty() or
        not sys.stdout.isatty() or
        not sys.stderr.isatty()
    )

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
        f = h5py.File(file_path, mode)
        f.close()
    except OSError as e:
        # The file does not exist, so create it in write mode
        f = h5py.File(file_path, 'w')  # You can add swmr=True if needed
        f.close()

        if logger is not None:
            logger.info(f'The file {file_path} was created in write mode.')
    else:
        if logger is not None:
            logger.info(f'The file {file_path} was opened in {mode} mode.')

    return h5py.File(file_path, mode)
    
"""
def open_hdf5_file_locking(
        file_path: Union[str, Path], 
        logger = None,
        mode: str ='r+',
        lock_timeout: int = 120,  # Time in seconds to wait for the file to become free
        poll_interval: float = 0.01  # Time in seconds between checks for file availability
    ) -> h5py.File:
    
    file_path = Path(file_path)
    lockfile_path = str(file_path) + ".lock"
    lock = FileLock(lockfile_path, timeout=lock_timeout)

    try:
        with lock.acquire(poll_interval=poll_interval):
            try:
                # Try to open the HDF5 file in the specified mode
                f = h5py.File(file_path, mode)
                f.close()
            except OSError as e:
                # The file does not exist, so create it in write mode
                f = h5py.File(file_path, 'w')  # You can add swmr=True if needed
                f.close()

                if logger is not None:
                    logger.info(f'The file {file_path} was created in write mode.')
            else:
                if logger is not None:
                    logger.info(f'The file {file_path} was opened in {mode} mode.')

            return h5py.File(file_path, mode)

    except Timeout:
        if logger is not None:
            logger.error(f'Timeout occurred. Could not acquire lock for {file_path}')
        raise Exception(f"Timeout: Unable to acquire lock for {file_path}")
"""

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

