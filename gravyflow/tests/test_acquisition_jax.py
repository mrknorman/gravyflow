import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import numpy as np
import keras
from keras import ops
import jax
import jax.numpy as jnp
from gravyflow.src.dataset.noise import acquisition as gf_acq
import gravyflow as gf

def test_random_subsection():
    # Test random_subsection logic
    # Input data: (Batch, Time) or (Time,)
    # We simulate a long time series
    
    total_samples = 10000
    data = jnp.arange(total_samples, dtype=jnp.float32)
    # Make it list of tensors as expected by random_subsection wrapper
    data_list = [data]
    
    start_gps = [1000.0]
    time_interval = 1.0 # 1 Hz
    
    num_onsource = 100
    num_offsource = 200
    num_examples = 5
    seed = 42
    
    # random_subsection(data, start_gps_time, ...)
    # It returns (subarrays, backgrounds, start_times)
    
    # We need to test the internal function or the wrapper.
    # The wrapper expects list of data.
    
    subarrays, backgrounds, starts = gf_acq.random_subsection(
        data_list,
        start_gps,
        time_interval,
        num_onsource,
        num_offsource,
        num_examples,
        seed
    )
    
    # Check shapes
    # subarrays: (Batch, 1, Onsource)
    # backgrounds: (Batch, 1, Offsource)
    # starts: (Batch, 1)
    
    assert ops.shape(subarrays) == (num_examples, 1, num_onsource)
    assert ops.shape(backgrounds) == (num_examples, 1, num_offsource)
    assert ops.shape(starts) == (num_examples, 1)
    
    # Check content validity
    # Subarrays should be sequential chunks from data
    # We can check if differences are 1.0
    diffs = subarrays[:, 0, 1:] - subarrays[:, 0, :-1]
    np.testing.assert_allclose(diffs, 1.0)

def test_ifo_data_random_subsection():
    # Test IFOData.random_subsection
    total_samples = 5000
    data = np.arange(total_samples, dtype=np.float32)
    ifo_data = gf_acq.IFOData(
        data=[data],
        sample_rate_hertz=1.0,
        start_gps_time=[0.0]
    )
    
    subarrays, backgrounds, starts = ifo_data.random_subsection(
        num_onsource_samples=100,
        num_offsource_samples=100,
        num_examples_per_batch=2,
        seed=123
    )
    
    assert ops.shape(subarrays) == (2, 1, 100)
    assert ops.shape(backgrounds) == (2, 1, 100)

def test_concatenate_batches():
    # Test concatenation
    b1 = ops.ones((2, 1, 10))
    b2 = ops.ones((2, 1, 10))
    
    res, _, _ = gf_acq.concatenate_batches([b1, b2], [b1, b2], [b1, b2])
    # Should concat on axis 1?
    # Original code: tf.concat(subarrays, axis=1)
    # If input is list of (Batch, 1, Time), output is (Batch, 2, Time)
    
    assert ops.shape(res) == (2, 2, 10)
