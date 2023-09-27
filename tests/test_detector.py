# Built-In imports:
import logging
from pathlib import Path
from timeit import Timer

# Library imports:
import tensorflow as tf
import numpy as np
from pycbc.detector import (add_detector_on_earth, _ground_detectors, 
                            get_available_detectors, Detector)
from bokeh.io import output_file, save
from bokeh.layouts import gridplot

# Local imports:
from ..setup import find_available_GPUs, setup_cuda, ensure_directory_exists
from ..detector import Network, IFO
from ..injection import (cuPhenomDGenerator, WNBGenerator, InjectionGenerator, 
                         WaveformParameters, WaveformGenerator)
from ..plotting import generate_strain_plot

def test_detector():
    
    # Create from dictionary:
    latitude = 0.3
    longitude = 0.5
    
    network = Network({
        "longitude_radians" : latitude, 
        "latitude_radians" : longitude,
        "y_angle_radians" : 0.0,  # Batched tensor
        "x_angle_radians" : None,  # Batched tensor or None
        "height_meters" : 0.0,  # Batched tensor
        "x_length_meters" : 4000.0,  # Batched tensor
        "y_length_meters" : 4000.0   # Batched tensor
    })
    
    add_detector_on_earth(
        "test", 
        latitude, 
        longitude,              
        )
    
    np.testing.assert_allclose(
        network.x_vector.numpy()[0], 
        _ground_detectors["test"]["xvec"],
        rtol=0, 
        atol=1e-07, 
        equal_nan=True, 
        err_msg="Tensorflow Network construction does not equal pycbc method.", 
        verbose=True
    )
    
    # From loading:
    example_network_directory : Path = Path(
        "./py_ml_tools/tests/example_network_parameters/example_network.json"
    )
    
    network = Network.load(example_network_directory)
        
    # Create from Enum:
    network = Network(
        [IFO.L1, IFO.H1]
    )
    
    test_antenna_pattern()
    test_time_delay()
    
def test_antenna_pattern():
    # Generating random values for our function
    
    test_detectors = [IFO.L1, IFO.H1]
    # Create from Enum:
    network = Network(
        test_detectors
    )
    
    num_tests = 1000000
    
    right_ascension = tf.constant(
        np.random.uniform(0, 2 * np.pi, size=(num_tests,)), dtype=tf.float32
    )
    declination = tf.constant(
        np.random.uniform(-np.pi / 2, np.pi / 2, size=(num_tests,)), dtype=tf.float32
    )
    polarization = tf.constant(
        np.random.uniform(0, 2 * np.pi, size=(num_tests,)), dtype=tf.float32
    )

    # Compute the output
    f_plus, f_cross = network.get_antenna_pattern(
        right_ascension, 
        declination, 
        polarization
    )
    
    d = Detector("L1") 
    f_plus_zomb, f_cross_zomb = zomb_antenna_pattern(
        right_ascension.numpy(),
        declination.numpy(), 
        polarization.numpy(),
        630763213
    )
    
    f_plus_pycbc, f_cross_pycbc = d.antenna_pattern(
        right_ascension.numpy(),
        declination.numpy(), 
        polarization.numpy(),
        630763213
    )
        
    # Check the output shape
    assert f_plus.shape == (
        num_tests,
        len(test_detectors)), f"Unexpected shape for f_plus: {f_plus.shape}"
    assert f_cross.shape == (
        num_tests,
        len(test_detectors)), f"Unexpected shape for f_cross: {f_cross.shape}"
    print("Test passed!")
    
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

    # Measure the time for zomb_antenna_pattern
    d = Detector("L1")
    timer = Timer(
        lambda: zomb_antenna_pattern(
            right_ascension.numpy(), 
            declination.numpy(), 
            polarization.numpy(), 
            630763213
        )
    )
    print(f"Time for zomb_antenna_pattern: {timer.timeit(number=1)} seconds")

    # Measure the time for d.antenna_pattern using PyCBC
    timer = Timer(
        lambda: d.antenna_pattern(
            right_ascension.numpy(), 
            declination.numpy(), 
            polarization.numpy(),
            630763213)
        )
    print(
        (f"Time for PyCBC's d.antenna_pattern: {timer.timeit(number=1)} "
         " seconds")
    )
    
    
def test_time_delay():
    
    num_tests = 100000
    
    d = Detector("L1") 
        
    test_detectors = [IFO.L1, IFO.H1, IFO.V1]
    # Create from Enum:
    network = Network(
        test_detectors
    )
    
    right_ascension = tf.constant(
        np.random.uniform(0, 2 * np.pi, size=(num_tests,)), dtype=tf.float32
    )
    declination = tf.constant(
        np.random.uniform(-np.pi / 2, np.pi / 2, size=(num_tests,)), 
        dtype=tf.float32
    )
    
    delay = network.get_time_delay(right_ascension, declination).numpy()
        
    assert np.allclose(delay[:, 0], delay[:, 1], atol=0.011)
    
    assert np.allclose(delay[:, 0], delay[:, 2], atol=0.032)

def zomb_antenna_pattern(
        right_ascension, 
        declination, 
        polarization, 
        t_gps
    ):        
        d = Detector("L1")
        #gha = d.gmst_estimate(t_gps) - right_ascension
        
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
    
def test_project_wave(
    num_tests : int = 2049,
    output_diretory_path : Path = Path("./py_ml_data/tests/")
    ):
    
    # Test Parameters:
    num_examples_per_generation_batch : int = 2048
    num_examples_per_batch : int = num_tests
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 1.0
    crop_duration_seconds : float = 0.5
    scale_factor : float = 1.0E21
        
    # Define injection directory path:
    injection_directory_path : Path = \
        Path("./py_ml_tools/tests/example_injection_parameters")
    
    phenom_d_generator : cuPhenomDGenerator = \
        WaveformGenerator.load(
            injection_directory_path / "phenom_d_parameters.json", 
            sample_rate_hertz, 
            onsource_duration_seconds
        )
    
    phenom_d_generator.injection_chance = 1.0
    
    injection_generator : InjectionGenerator = \
        InjectionGenerator(
            [phenom_d_generator],
            sample_rate_hertz,
            onsource_duration_seconds,
            crop_duration_seconds,
            num_examples_per_generation_batch,
            num_examples_per_batch,
            variables_to_return = \
                [WaveformParameters.MASS_1_MSUN, WaveformParameters.MASS_2_MSUN]
        )
    
    total_onsource_duration_seconds : float = \
        onsource_duration_seconds + (crop_duration_seconds * 2.0)
        
    generator : Iterator = injection_generator.generate
    
    injections, mask, parameters = next(generator())
    
    network = Network([IFO.L1, IFO.H1, IFO.V1])
    
    projected_injections = \
        network.project_wave(injections[0], injections[0], sample_rate_hertz)
    
    injection_one = projected_injections.numpy()[0]
    
    total_onsource_duration_seconds : float = \
        onsource_duration_seconds + crop_duration_seconds
    
    layout = [
        [generate_strain_plot(
            {"Injection Test": injection},
            sample_rate_hertz,
            total_onsource_duration_seconds,
            title=f"WNB injection example",
            scale_factor=scale_factor
        )]
        for injection in injection_one
    ]
    
    # Ensure output directory exists
    ensure_directory_exists(output_diretory_path)
    
    # Define an output path for the dashboard
    output_file(output_diretory_path / "projection_plots.html")

    # Arrange the plots in a grid. 
    grid = gridplot(layout)
        
    save(grid)
    
if __name__ == "__main__":
        
    # ---- User parameters ---- #
    
    # GPU setup:
    min_gpu_memory_mb : int = 4000
    num_gpus_to_request : int = 1
    memory_to_allocate_tf : int = 2000
    
    # Setup CUDA
    gpus = find_available_GPUs(
        min_gpu_memory_mb, 
        num_gpus_to_request
    )
    strategy = setup_cuda(
        gpus, 
        max_memory_limit = memory_to_allocate_tf, 
        logging_level=logging.WARNING
    )    
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    # Test psd calculation:
    with strategy.scope():
        test_detector()
        test_project_wave()