import os
import site
import glob
import sys

# Attempt to find nvidia libraries and set LD_LIBRARY_PATH
# This is required in conftest.py to ensure pytest finds the libraries
# before JAX is initialized by any plugins or test collection.
try:
    site_packages = site.getsitepackages()[0]
    nvidia_dir = os.path.join(site_packages, 'nvidia')
    
    if os.path.exists(nvidia_dir):
        libs = glob.glob(os.path.join(nvidia_dir, '*/lib'))
        path_to_add = ":".join(libs)
        
        current_ld = os.environ.get('LD_LIBRARY_PATH', '')
        if path_to_add not in current_ld:
            os.environ['LD_LIBRARY_PATH'] = f"{current_ld}:{path_to_add}"
except Exception as e:
    pass

# Set Keras backend to JAX
os.environ["KERAS_BACKEND"] = "jax"
import gravyflow as gf

import warnings
import pytest

# Autouse fixture to reset seeds before each test for isolation
@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for proper isolation."""
    import gravyflow as gf
    gf.set_random_seeds(gf.Defaults.seed)
    yield
    # Optionally reset after test too
    gf.set_random_seeds(gf.Defaults.seed)

def pytest_addoption(parser):
    parser.addoption(
        "--runsize", action="store", default="small",
        help="Run a specific size of tests: small, normal, or large"
    )
    parser.addoption(
        "--plot", action="store_true", default=False,
        help="Enable plotting of results"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (skipped on small runs)")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runsize") == "small":
        skip_slow = pytest.mark.skip(reason="Slow test skipped on small run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

@pytest.fixture
def data_obtainer():
    return gf.IFODataObtainer(
        observing_runs=gf.ObservingRun.O3, 
        data_quality=gf.DataQuality.BEST, 
        data_labels=[
            gf.DataLabel.NOISE, 
            gf.DataLabel.GLITCHES]
    )