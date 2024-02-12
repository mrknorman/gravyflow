from typing import Dict
from pathlib import Path

import numpy as np
import h5py

def num_tests_from_config(pytestconfig : Dict):
    match pytestconfig.getoption("runsize"):
        case "small":
            num_tests = int(1E2)
        case "normal":
            num_tests = int(1E3)
        case "large":
            num_tests = int(1E4)
        case _:
            raise ValueError(f"Runsize {pytestconfig.runsize} not recognised!")

    return num_tests

def compare_and_save_parameters(
    current_parameters: Dict[str, np.ndarray], 
    parameters_file_path: Path
):
    """
    Compare current parameters with previously saved parameters and save them 
    if not present.

    Args:
        current_parameters (Dict[str, np.ndarray]): The current parameters.
        parameters_file_path (Path): Path to the parameters file.
    """
    tolerance : float = 1E-6  # Set an appropriate tolerance level
    if parameters_file_path.exists():
        with h5py.File(parameters_file_path, 'r') as hf:
            for key in current_parameters:
                previous_data = hf[key][:]
                np.testing.assert_allclose(
                    previous_data, 
                    current_parameters[key], 
                    atol=tolerance, 
                    err_msg=f"Parameter consistency check failed for {key}."
                )
    else:
        with h5py.File(parameters_file_path, 'w') as hf:
            for key, data in current_parameters.items():
                hf.create_dataset(key, data=data)