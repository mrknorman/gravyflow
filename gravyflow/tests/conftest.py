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
    parser.addoption(
        "--network", action="store_true", default=False,
        help="Run tests marked 'network', which query GWOSC and Zenodo"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (skipped on small runs)")
    config.addinivalue_line(
        "markers", "network: needs internet access to GWOSC or Zenodo (opt-in: --network)"
    )

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runsize") == "small":
        skip_slow = pytest.mark.skip(reason="Slow test skipped on small run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if not config.getoption("--network"):
        skip_network = pytest.mark.skip(reason="Network test: run with --network")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)


#: Public sources that only tests run with --network may query.
_NETWORK_ONLY_URLS = (
    "gwosc.org/timeline/",
    "gwosc.org/api/v2/timelines",
    "zenodo.org/api/records/5649212",
)


@pytest.fixture(autouse=True, scope="session")
def public_sources_need_network_option(request):
    """
    Without --network, the GWOSC timeline and the Gravity Spy release are refused
    and their caches start empty, so no test depends on them, or on what this
    machine downloaded before, unless it is marked 'network'.
    """
    if request.config.getoption("--network"):
        yield
        return

    import tempfile
    import requests

    previous_cache_dir = os.environ.get("GRAVYFLOW_CACHE_DIR")
    cache_dir = tempfile.TemporaryDirectory(prefix="gravyflow-test-cache-")
    os.environ["GRAVYFLOW_CACHE_DIR"] = cache_dir.name

    original_request = requests.Session.request

    def guarded_request(session, method, url, *args, **kwargs):
        if any(fragment in str(url) for fragment in _NETWORK_ONLY_URLS):
            raise requests.ConnectionError(
                f"{url} is reachable only in test runs with --network."
            )
        return original_request(session, method, url, *args, **kwargs)

    requests.Session.request = guarded_request
    try:
        yield
    finally:
        requests.Session.request = original_request
        if previous_cache_dir is None:
            os.environ.pop("GRAVYFLOW_CACHE_DIR", None)
        else:
            os.environ["GRAVYFLOW_CACHE_DIR"] = previous_cache_dir
        cache_dir.cleanup()


@pytest.fixture
def data_obtainer():
    return gf.IFODataObtainer(
        observing_runs=gf.ObservingRun.O3, 
        data_quality=gf.DataQuality.BEST, 
        data_labels=[
            gf.DataLabel.NOISE, 
            gf.DataLabel.GLITCHES]
    )