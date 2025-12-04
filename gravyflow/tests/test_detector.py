from typing import Optional, Dict, Tuple
from pathlib import Path
import logging

import pytest
import h5py
import numpy as np
import keras
from keras import ops
import jax.numpy as jnp
from pycbc.detector import add_detector_on_earth, _ground_detectors, Detector
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from timeit import Timer
from _pytest.config import Config

import gravyflow as gf

def zombie_antenna_pattern(
        right_ascension : float, 
        declination : float, 
        polarization : float
    ) -> Tuple[np.ndarray, np.ndarray]:

        d = Detector("L1")
        
        cosgha = np.cos(right_ascension)
        singha = np.sin(right_ascension)
        cosdec = np.cos(declination)
        sindec = np.sin(declination)
        cospsi = np.cos(polarization)
        sinpsi = np.sin(polarization)

        resp = d.response
        ttype = np.float64
        
        x0 = -cospsi * singha - sinpsi * cosgha * sindec
        x1 = -cospsi * cosgha + sinpsi * singha * sindec
        x2 =  sinpsi * cosdec

        x = np.array([x0, x1, x2], dtype=object)
        dx = resp.dot(x)

        y0 =  sinpsi * singha - cospsi * cosgha * sindec
        y1 =  sinpsi * cosgha + cospsi * singha * sindec
        y2 =  cospsi * cosdec

        y = np.array([y0, y1, y2], dtype=object)
        dy = resp.dot(y)

        if hasattr(dx, 'shape'):
            fplus = (x * dx - y * dy).sum(axis=0).astype(ttype)
            fcross = (x * dy + y * dx).sum(axis=0).astype(ttype)
        else:
            fplus = (x * dx - y * dy).sum()
            fcross = (x * dy + y * dx).sum()
        return fplus, fcross

# Network Fixture
@pytest.fixture
def network() -> gf.Network:
    return gf.Network([gf.IFO.L1, gf.IFO.H1, gf.IFO.V1])

def test_rotation_matrices():
    angle = ops.convert_to_tensor([0.0, np.pi/2.0])
    
    # Z-rotation
    # 0 -> Identity
    # pi/2 -> [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    rm_z = gf.src.dataset.conditioning.detector.rotation_matrix_z(angle)
    
    assert ops.shape(rm_z) == (2, 3, 3)
    
    # Check identity
    np.testing.assert_allclose(rm_z[0], np.eye(3), atol=1e-6)
    
    # Check pi/2
    expected_z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(rm_z[1], expected_z, atol=1e-6)

def test_network_init():
    # Initialize a simple network (L1)
    network = gf.Network([gf.IFO.L1])
    
    assert network.num_detectors == 1
    assert ops.shape(network.location) == (1, 3)
    
    # Check location (approximate check to ensure astropy ran)
    # L1 is in Louisiana, USA.
    loc = network.location[0]
    assert loc[0] != 0.0

# Test for Detector
def test_detector() -> None:

    with gf.env():
        latitude = 0.3
        longitude = 0.5
        network = gf.Network({
            "longitude_radians": latitude, 
            "latitude_radians": longitude,
            "y_angle_radians": 0.0,  
            "x_angle_radians": None,  
            "height_meters": 0.0,  
            "x_length_meters": 4000.0,  
            "y_length_meters": 4000.0
        })

        add_detector_on_earth("test", latitude, longitude)

        # Convert to numpy for comparison if it's a tensor
        x_vector = network.x_vector
        if hasattr(x_vector, "numpy"):
            x_vector = x_vector.numpy()
        else:
            x_vector = np.array(x_vector)

        np.testing.assert_allclose(
            x_vector[0], 
            _ground_detectors["test"]["xvec"],
            rtol=0, 
            atol=1e-07, 
            equal_nan=True, 
            err_msg="Gravyflow gf.Network construction does not equal pycbc method.", 
            verbose=True
        )

def test_response() -> None:

    with gf.env():

        test_detectors = [gf.IFO.L1, gf.IFO.H1, gf.IFO.V1]
        # Create from Enum:
        network = gf.Network(
            test_detectors
        )

        tolerance : float = 0.01

        l1 = Detector("L1")
        # Convert response to numpy
        response = network.response
        if hasattr(response, "numpy"):
            response = response.numpy()
        else:
            response = np.array(response)

        np.testing.assert_allclose(
            l1.response, 
            response[0], 
            atol=tolerance, 
            err_msg="L1 response check failed."
        )

        h1 = Detector("H1")
        np.testing.assert_allclose(
            h1.response, 
            response[1], 
            atol=tolerance, 
            err_msg="H1 response check failed."
        )

        v1 = Detector("V1")
        np.testing.assert_allclose(
            v1.response, 
            response[2], 
            atol=tolerance, 
            err_msg="V1 response check failed."
        )

# Test for Antenna Pattern
@pytest.mark.parametrize("num_tests", [int(100)])
def test_antenna_pattern(
        network : gf.Network, 
        num_tests : int
    ) -> None:

    with gf.env():
        right_ascension = ops.convert_to_tensor(
            np.random.uniform(0, 2 * np.pi, size=(num_tests,)), 
            dtype="float32"
        )
        declination = ops.convert_to_tensor(
            np.random.uniform(-np.pi / 2, np.pi / 2, size=(num_tests,)), 
            dtype="float32"
        )
        polarization = ops.convert_to_tensor(
            np.random.uniform(0, 2 * np.pi, size=(num_tests,)), 
            dtype="float32"
        )

        antenna_pattern = network.get_antenna_pattern(
            right_ascension, declination, polarization
        )
        f_plus = antenna_pattern[..., 0]
        f_cross = antenna_pattern[..., 1]

        np.testing.assert_equal(
            f_plus.shape,
            (num_tests, network.num_detectors), 
            err_msg=f"Unexpected shape for f_plus: {f_plus.shape}"
        )
        np.testing.assert_equal(
            f_cross.shape, 
            (num_tests, network.num_detectors), 
            err_msg=f"Unexpected shape for f_cross: {f_cross.shape}"
        )

# Test for Time Delay
def test_time_delay(
        network : gf.Network, 
        num_tests : Optional[int] = int(100)
    ) -> None:

    with gf.env():
        right_ascension = ops.convert_to_tensor(
            np.random.uniform(0, 2 * np.pi, size=(num_tests,)), dtype="float32"
        )
        declination = ops.convert_to_tensor(
            np.random.uniform(-np.pi / 2, np.pi / 2, size=(num_tests,)), dtype="float32"
        )

        delay = network.get_time_delay(right_ascension, declination)
        if hasattr(delay, "numpy"):
            delay = delay.numpy()
        else:
            delay = np.array(delay)
        np.testing.assert_allclose(delay[:, 0], delay[:, 1], atol=0.011)
        np.testing.assert_allclose(delay[:, 0], delay[:, 2], atol=0.032)

def save_and_compare_projected_injections(
        projected_injections: np.ndarray, 
        injections_file_path: Path,
        tolerance: float = 1e-6
    ) -> None:

    """
    Save and compare projected injections with previously saved injections.

    Args:
        projected_injections (np.ndarray): The projected injections data.
        injections_file_path (Path): Path to the file where injections are saved.
        tolerance (float): Tolerance level for comparison.
    """

    gf.ensure_directory_exists(injections_file_path.parent)

    if injections_file_path.exists():
        with h5py.File(injections_file_path, 'r') as hf:
            previous_injections = hf['projected_injections'][:]
            np.testing.assert_allclose(
                previous_injections, 
                projected_injections, 
                atol=tolerance, 
                err_msg="Projected injections consistency check failed."
            )
    else:
        with h5py.File(injections_file_path, 'w') as hf:
            hf.create_dataset('projected_injections', data=projected_injections)

def _test_projection(
        name : str,
        network : gf.Network, 
        output_directory_path : Path, 
        plot_file_name : str, 
        single_ifo : bool = False, 
        should_plot : bool = False
    ) -> None:

    injection_directory_path = Path(
        gf.tests.PATH / "example_injection_parameters"
    )

    phenom_d_generator = gf.WaveformGenerator.load(
        injection_directory_path / "phenom_d_parameters.json"
    )
    phenom_d_generator.injection_chance = 1.0

    injection_generator = gf.InjectionGenerator([phenom_d_generator])
    injections, _, _ = next(injection_generator())

    if single_ifo:
        network = gf.Network([gf.IFO.L1])  # Adjust for single IFO

    projected_injections = network.project_wave(injections[0])

    injections_file_path : Path = (
        gf.PATH / f"res/tests/projected_injections_{name}.hdf5"
    )

    save_and_compare_projected_injections(
        np.array(projected_injections),
        injections_file_path
    )

    injection_one = np.array(projected_injections)[0]

    layout = [[gf.generate_strain_plot(
        {"Injection Test": injection}, title="WNB injection example"
    ) for injection in injection_one]]

    if not should_plot:
        gf.ensure_directory_exists(output_directory_path)
        output_file(output_directory_path / plot_file_name)
        grid = gridplot(layout)
        save(grid)

@pytest.mark.slow
def test_project_wave(
        network : gf.Network, 
        pytestconfig : Config
    ) -> None:

    output_directory_path : Path = gf.PATH.parent.parent / "gravyflow_data/tests"

    with gf.env():
        _test_projection(
            "multi",
            network, 
            output_directory_path, 
            "projection_plots.html", 
            should_plot=pytestconfig.getoption("plot")
        )

@pytest.mark.slow
def test_project_wave_single(
        network : gf.Network, 
        pytestconfig : Config
    ) -> None:
    
    output_directory_path : Path = gf.PATH.parent.parent / "gravyflow_data/tests"
    
    with gf.env():
        _test_projection(
            "single",
            network, 
            output_directory_path, 
            "projection_plots_single.html", 
            single_ifo=True, 
            should_plot=pytestconfig.getoption("plot")
        )

def _test_antenna_pattern(
        num_tests : int = int(100)
    ) -> None:
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    with gf.env():
    
        test_detectors = [gf.IFO.L1]
        # Create from Enum:
        network = gf.Network(
            test_detectors
        )

        # Generating random values for our function
        right_ascension = ops.convert_to_tensor(
            np.random.uniform(0, 2 * np.pi, size=(num_tests,)), 
            dtype="float32"
        )
        declination = ops.convert_to_tensor(
            np.random.uniform(-np.pi / 2, np.pi / 2, size=(num_tests,)), 
            dtype="float32"
        )
        polarization = ops.convert_to_tensor(
            np.random.uniform(0, 2 * np.pi, size=(num_tests,)), 
            dtype="float32"
        )

        gps_reference = 0
        d = Detector("L1")

        # Compute the output
        antenna_pattern = network.get_antenna_pattern(
            d.gmst_estimate(gps_reference) - np.array(right_ascension), 
            declination, 
            polarization
        )

        f_plus = ops.squeeze(antenna_pattern[...,0])
        f_cross = ops.squeeze(antenna_pattern[...,1])

        f_plus_zomb, f_cross_zomb = zombie_antenna_pattern(
            d.gmst_estimate(gps_reference) - np.array(right_ascension),
            np.array(declination), 
            np.array(polarization)
        )

        f_plus_pycbc, f_cross_pycbc = d.antenna_pattern(
            np.array(right_ascension),
            np.array(declination), 
            np.array(polarization),
            gps_reference
        )
        
        np.testing.assert_allclose(
            f_plus, 
            f_plus_zomb,
            atol=0.001, 
            equal_nan=True, 
            err_msg="Gravyflow antenna pattern does not equal zombie method.", 
            verbose=True
        )

        np.testing.assert_allclose(
            f_cross, 
            f_cross_zomb,
            atol=0.001, 
            equal_nan=True, 
            err_msg="Gravyflow antenna pattern does not equal zombie method.", 
            verbose=True
        )

        np.testing.assert_allclose(
            f_cross, 
            f_cross_pycbc,
            atol=0.001, 
            equal_nan=True, 
            err_msg="Gravyflow antenna pattern does not equal pycbc method.", 
            verbose=True
        )

        np.testing.assert_allclose(
            f_plus,
            f_plus_pycbc,
            atol=0.001, 
            equal_nan=True, 
            err_msg="Gravyflow antenna pattern does not equal pycbc method.", 
            verbose=True
        )

def test_antenna_pattern(
        pytestconfig : Config
    ) -> None:

    _test_antenna_pattern(
        num_tests=gf.tests.num_tests_from_config(pytestconfig)
    )

def profile_atennnna_pattern() -> None:

    with gf.env():
    
        test_detectors = [gf.IFO.L1]
        # Create from Enum:
        network = gf.Network(
            test_detectors
        )

        # Generating random values for our function
        right_ascension = ops.convert_to_tensor(
            np.random.uniform(0, 2 * np.pi, size=(num_tests,)), 
            dtype="float32"
        )
        declination = ops.convert_to_tensor(
            np.random.uniform(-np.pi / 2, np.pi / 2, size=(num_tests,)), 
            dtype="float32"
        )
        polarization = ops.convert_to_tensor(
            np.random.uniform(0, 2 * np.pi, size=(num_tests,)), 
            dtype="float32"
        )

        gps_reference = 0
        d = Detector("L1")
                
        # Measure the time for network.get_antenna_pattern
        timer = Timer(
            lambda: network.get_antenna_pattern(
                right_ascension,
                declination, 
                polarization
            )
        )
        print(
            (f"Time for network.get_antenna_pattern: {timer.timeit(number=1)}"
             " seconds")
        )

        # Measure the time for zombie_antenna_pattern
        d = Detector("L1")
        timer = Timer(
            lambda: zombie_antenna_pattern(
                np.array(right_ascension), 
                np.array(declination), 
                np.array(polarization)
            )
        )
        print(f"Time for zombie_antenna_pattern: {timer.timeit(number=1)} seconds")

        # Measure the time for d.antenna_pattern using PyCBC
        timer = Timer(
            lambda: d.antenna_pattern(
                np.array(right_ascension), 
                np.array(declination), 
                np.array(polarization),
                630763213)
            )
        print(
            (f"Time for PyCBC's d.antenna_pattern: {timer.timeit(number=1)} "
             " seconds")
        )