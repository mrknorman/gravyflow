# Built-In imports:
import logging
from pathlib import Path

# Library imports:
import tensorflow as tf
import numpy as np
from pycbc.detector import (add_detector_on_earth, _ground_detectors, 
                            get_available_detectors, Detector)

# Local imports:
from ..setup import find_available_GPUs, setup_cuda
from ..detector import Network, IFO

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
        network.xvec.numpy()[0], 
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
    
    #print(network.response)
    
    # Create from Enum:
    network = Network(
        [IFO.L1, IFO.H1]
    )
    
    #print(network.response)
    #print(tf.shape(network.response))
    
    test_antenna_pattern()
    
def test_antenna_pattern():
    # Generating random values for our function
    
    test_detectors = [IFO.L1, IFO.H1]
    # Create from Enum:
    network = Network(
        test_detectors
    )
    
    num_tests = 10
    
    right_ascension = tf.constant(
        np.random.uniform(0, 2 * np.pi, size=(num_tests,)), dtype=tf.float32
    )
    declination = tf.constant(
        np.random.uniform(-np.pi / 2, np.pi / 2, size=(num_tests,)), dtype=tf.float32
    )
    polarization = tf.constant(
        np.random.uniform(0, 2 * np.pi, size=(num_tests,)), dtype=tf.float32
    )
    frequency = tf.constant(
        np.random.uniform(0, 2000, size=(num_tests,)), dtype=tf.float32
    )  # Example range, adjust as needed
    polarization_type = "tensor"  # Example type, can be changed to "vector" or "scalar"

    # Compute the output
    f_plus, f_cross = network.antenna_pattern(
        right_ascension, 
        declination, 
        polarization, 
        None, 
        polarization_type
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
    
    print(f_plus_zomb, f_plus)
    
    # Check the output shape
    assert f_plus.shape == (num_tests,len(test_detectors)), f"Unexpected shape for f_plus: {f_plus.shape}"
    assert f_cross.shape == (num_tests,len(test_detectors)), f"Unexpected shape for f_cross: {f_cross.shape}"
    print("Test passed!")
    
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