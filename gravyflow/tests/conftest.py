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


def refuse_internet(monkeypatch) -> None:
    """
    Refuse every internet connection (loopback stays open), by name. Every network
    source the library uses
    (the GWOSC APIs, timeline, event list and strain files, Zenodo) connects through a
    socket, whatever client it goes through (requests, urllib, gwosc, gwpy), so one
    refusal covers them all, and any source added later.
    """
    import socket

    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def refuse(sock, address):
        loopback = str(address[0]) in ("localhost", "::1") or str(address[0]).startswith("127.")
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not loopback:
            raise ConnectionRefusedError(
                f"Offline test: refused a connection to {address}; the network is "
                "reachable only in test runs with --network."
            )

    def connect(sock, address, *args, **kwargs):
        refuse(sock, address)
        return original_connect(sock, address, *args, **kwargs)

    def connect_ex(sock, address, *args, **kwargs):
        refuse(sock, address)
        return original_connect_ex(sock, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


@pytest.fixture(autouse=True, scope="session")
def offline_unless_network_option(request):
    """
    Without --network, every internet connection is refused and the user cache starts
    empty, so no test depends on the network, or on what this machine downloaded
    before, unless it is marked 'network'.
    """
    if request.config.getoption("--network"):
        yield
        return

    import tempfile

    with tempfile.TemporaryDirectory(prefix="gravyflow-test-cache-") as cache_dir, \
            pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("GRAVYFLOW_CACHE_DIR", cache_dir)
        refuse_internet(monkeypatch)
        yield


@pytest.fixture
def data_obtainer():
    return gf.IFODataObtainer(
        observing_runs=gf.ObservingRun.O3, 
        data_quality=gf.DataQuality.BEST, 
        data_labels=[
            gf.DataLabel.NOISE, 
            gf.DataLabel.GLITCHES]
    )