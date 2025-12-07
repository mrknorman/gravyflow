from typing import Dict
from pathlib import Path

import numpy as np
import h5py
import gravyflow as gf
from _pytest.config import Config

def num_tests_from_config(
        pytestconfig : Config
    ) -> int:

    match pytestconfig.getoption("runsize"):
        case "small":
            num_tests = int(2)
        case "normal":
            num_tests = int(1E2)
        case "large":
            num_tests = int(1E3)
        case _:
            raise ValueError(f"Runsize {pytestconfig.runsize} not recognised!")

    return num_tests

def compare_and_save_parameters(
        current_parameters: Dict[str, np.ndarray], 
        parameters_file_path: Path
    ) -> None:

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
from unittest.mock import patch, MagicMock

def test_gpu_utils():
    """Verify GPU discovery logic (mocked)."""
    
    with patch("gravyflow.src.utils.gpu.subprocess.check_output") as mock_subprocess:
        # Mock the output of nvidia-smi
        # Format: index, memory.total, memory.used, memory.free, utilization.gpu
        mock_subprocess.return_value = "0, 8000, 4000, 4000, 50\n1, 8000, 2000, 6000, 25"
        
        gpu_info = gf.get_gpu_memory_info()
        assert len(gpu_info) == 2
        assert gpu_info[0]['index'] == '0'
        assert gpu_info[0]['free'] == 4000
        assert gpu_info[1]['index'] == '1'
        assert gpu_info[1]['free'] == 6000

def test_io_utils():
    """Verify ensure_directory_exists and other IO helpers."""
    
    import tempfile
    import shutil
    
    # Test ensure_directory_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        target_dir = Path(tmpdir) / "subdir" / "subsubdir"
        assert not target_dir.exists()
        
        gf.ensure_directory_exists(target_dir)
        
        assert target_dir.exists()
        assert target_dir.is_dir()
