import warnings
import pytest

import gravyflow as gf

def pytest_addoption(parser):
    parser.addoption(
        "--runsize", action="store", default="small",
        help="Run a specific size of tests: small, normal, or large"
    )
    parser.addoption(
        "--plot", action="store_true", default=False,
        help="Enable plotting of results"
    )

@pytest.fixture
def data_obtainer():
    return gf.IFODataObtainer(
        observing_runs=gf.ObservingRun.O3, 
        data_quality=gf.DataQuality.BEST, 
        data_labels=[
            gf.DataLabel.NOISE, 
            gf.DataLabel.GLITCHES]
    )