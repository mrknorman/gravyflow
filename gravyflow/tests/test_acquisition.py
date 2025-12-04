import pytest
import numpy as np
import keras
from keras import ops
import jax.numpy as jnp
import gravyflow as gf
from gravyflow.src.dataset.noise import acquisition as gf_acq

@pytest.mark.slow
@pytest.mark.parametrize("seed, ifo", [(1000, gf.IFO.L1)])
def test_valid_data_segments_acquisition(
        data_obtainer : gf.IFODataObtainer, 
        seed : int, 
        ifo : gf.IFO
    ) -> None:
    
    """
    Test to ensure the acquisition of valid data segments meets expected criteria.
    """
    segments = data_obtainer.get_valid_segments(seed=seed, ifos=[ifo])
    
    np.testing.assert_array_less(
        10000, 
        len(segments), 
        err_msg=f"Num segments found, {len(segments)}, is too low!"
    )

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

def test_segment_ordering():
    # Mock IFODataObtainer to test order_segments
    
    # Create a dummy instance
    obtainer = gf.IFODataObtainer(
        gf.ObservingRun.O3, 
        gf.DataQuality.BEST, 
        gf.DataLabel.NOISE,
        force_acquisition=False,
        cache_segments=False
    )
    
    # Create dummy segments: (NumSegments, NumIFOs, 2)
    # 3 segments, 1 IFO
    # Seg 1: 10s duration
    # Seg 2: 30s duration
    # Seg 3: 20s duration
    segments = np.array([
        [[0, 10]], 
        [[20, 50]], 
        [[60, 80]]
    ])
    # Shape (3, 1, 2)
    
    # Test SHORTEST_FIRST
    sorted_segments = obtainer.order_segments(
        segments.copy(), 
        gf.SegmentOrder.SHORTEST_FIRST, 
        seed=42
    )
    # Expected order: 10s, 20s, 30s
    # Indices: 0, 2, 1
    # Check durations of first IFO
    durations = sorted_segments[:, 0, 1] - sorted_segments[:, 0, 0]
    assert np.all(durations == [10, 20, 30])
    
    # Test RANDOM
    # With seed 42
    random_segments = obtainer.order_segments(
        segments.copy(), 
        gf.SegmentOrder.RANDOM, 
        seed=42
    )
    # Just check it's a permutation
    assert ops.shape(random_segments) == ops.shape(segments)
    # Check sums to ensure same data
    assert np.sum(random_segments) == np.sum(segments)

def test_remove_short_segments():
    obtainer = gf.IFODataObtainer(
        gf.ObservingRun.O3, 
        gf.DataQuality.BEST, 
        gf.DataLabel.NOISE,
        force_acquisition=False,
        cache_segments=False
    )
    
    # Shape (NumSegments, NumIFOs, 2) -> (3, 1, 2)
    # Segments: 10s, 5s, 20s
    segments = np.array([
        [[0, 10]], 
        [[20, 25]], 
        [[30, 50]]
    ])
    
    # Remove segments shorter than 8s
    filtered = obtainer.remove_short_segments(segments, 8.0)
    
    # Should keep 1st and 3rd
    assert ops.shape(filtered) == (2, 1, 2)
    durations = filtered[:, 0, 1] - filtered[:, 0, 0]
    assert np.all(durations >= 8.0)
    assert durations[0] == 10.0
    assert durations[1] == 20.0

def test_compress_segments():
    obtainer = gf.IFODataObtainer(
        gf.ObservingRun.O3, 
        gf.DataQuality.BEST, 
        gf.DataLabel.NOISE,
        force_acquisition=False,
        cache_segments=False
    )
    
    # Segments with overlap
    # [0, 10], [5, 15], [20, 30]
    # Should merge to [0, 15], [20, 30]
    segments = np.array([
        [0, 10],
        [5, 15],
        [20, 30]
    ])
    
    compressed = obtainer.compress_segments(segments)
    
    assert ops.shape(compressed) == (2, 2)
    assert np.all(compressed[0] == [0, 15])
    assert np.all(compressed[1] == [20, 30])