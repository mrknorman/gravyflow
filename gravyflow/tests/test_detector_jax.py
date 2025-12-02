import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import numpy as np
import keras
from keras import ops
from gravyflow.src.dataset.conditioning import detector as gf_detector

def test_rotation_matrices():
    angle = ops.convert_to_tensor([0.0, np.pi/2.0])
    
    # Z-rotation
    # 0 -> Identity
    # pi/2 -> [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    rm_z = gf_detector.rotation_matrix_z(angle)
    
    assert ops.shape(rm_z) == (2, 3, 3)
    
    # Check identity
    np.testing.assert_allclose(rm_z[0], np.eye(3), atol=1e-6)
    
    # Check pi/2
    expected_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(rm_z[1], expected_z, atol=1e-6)

def test_network_init():
    # Initialize a simple network (L1)
    network = gf_detector.Network([gf_detector.IFO.L1])
    
    assert network.num_detectors == 1
    assert ops.shape(network.location) == (1, 3)
    
    # Check location (approximate check to ensure astropy ran)
    # L1 is in Louisiana, USA.
    loc = network.location[0]
    assert loc[0] != 0.0

def test_project_wave():
    network = gf_detector.Network([gf_detector.IFO.L1, gf_detector.IFO.H1])
    
    sample_rate = 100.0
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Signal: Sine wave
    strain = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
    strain = ops.convert_to_tensor(strain.reshape(1, -1)) # (1, T)
    
    # Project
    projected = network.project_wave(
        strain,
        sample_rate_hertz=sample_rate
    )
    
    # Output should be (Batch, Detectors, Time)
    # Batch=1, Detectors=2
    assert ops.shape(projected) == (1, 2, 100)
    
    # Check that signals are not identical (due to time delay and antenna pattern)
    # Unless geometry aligns perfectly, they should differ.
    diff = ops.sum(ops.abs(projected[0, 0] - projected[0, 1]))
    assert diff > 0.0

def test_get_time_delay():
    network = gf_detector.Network([gf_detector.IFO.L1, gf_detector.IFO.H1])
    
    ra = ops.convert_to_tensor([0.0])
    dec = ops.convert_to_tensor([0.0])
    
    delays = network.get_time_delay(ra, dec)
    
    # Shape: (N_signals, N_detectors) -> Wait, get_time_delay_ returns (N, X) or (X, N)?
    # Implementation:
    # time_delay = ops.tensordot(location, ehat, axes=[[1], [0]]) -> (X, 1, N)
    # squeeze -> (X, N)
    # transpose -> (N, X)
    
    assert ops.shape(delays) == (1, 2)
