import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import gravyflow as gf
import numpy as np
import keras
from keras import ops
from gravyflow.src.dataset.conditioning import pearson as gf_pearson

def test_pearson_basic():
    # Perfectly correlated
    x = ops.convert_to_tensor([[1.0, 2.0, 3.0]])
    y = ops.convert_to_tensor([[2.0, 4.0, 6.0]])
    corr = gf_pearson.pearson(x, y)
    np.testing.assert_allclose(corr, [1.0], atol=1e-5)
    
    # Anti-correlated
    y_neg = ops.convert_to_tensor([[-1.0, -2.0, -3.0]])
    corr_neg = gf_pearson.pearson(x, y_neg)
    np.testing.assert_allclose(corr_neg, [-1.0], atol=1e-5)

def test_rolling_pearson():
    sample_rate = 1.0
    max_diff = 2.0 # 2 samples. N_offsets = 2 * int(2*1) + 1 = 5
    
    # Create two correlated signals with a known shift
    # Channel 0: sine wave
    # Channel 1: same sine wave shifted
    
    t = np.linspace(0, 2*np.pi, 20, dtype=np.float32)
    c0 = np.sin(t)
    c1 = np.sin(t)  # Identical - max correlation at zero shift
    
    tensor = np.stack([c0, c1], axis=0) # (2, 20)
    tensor = tensor[np.newaxis, ...] # (1, 2, 20)
    tensor = ops.convert_to_tensor(tensor)
    
    corrs = gf_pearson.rolling_pearson(
        tensor,
        max_arrival_time_difference_seconds=max_diff,
        sample_rate_hertz=sample_rate
    )
    
    # Shape: (Batch, NumPairs, N_offsets)
    # NumPairs = 1 (0 vs 1)
    assert ops.shape(corrs)[0] == 1
    assert ops.shape(corrs)[1] == 1
    
    # For identical signals, max correlation should be 1.0 at some offset
    max_corr = float(ops.max(corrs))
    assert max_corr > 0.95, f"Max correlation should be ~1.0 for identical signals, got {max_corr}"

