from typing import Union
from pathlib import Path

import h5py
import logging

def replace_placeholders(
        value: dict, 
        replacements: dict
    ) -> None:
        
    """Replace placeholders in the config dictionary with actual values."""
    for k in ["value", "max_", "min_", "type_"]:

        if isinstance(value, dict):
            if k in value:
                value[k] = replacements.get(value[k], value[k])

def open_hdf5_file(
        file_path : Union[str, Path], 
        logger,
        mode : str ='r+',
        logging_level : int = logging.WARNING
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
        logger.info(f'The file {file_path} was created in write mode.')
    else:
        logger.info(f'The file {file_path} was opened in {mode} mode.')
    return h5py.File(file_path, mode)

def ensure_directory_exists(
    	directory: Union[str, Path]
    ):
	
    directory = Path(directory)  # Convert to Path if not already
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)