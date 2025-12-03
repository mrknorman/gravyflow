import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
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
    max_diff = 2.0 # 2 samples. N_offsets = 4.
    
    # Batch=1, Channels=2, Time=10
    # Channel 0: [0, 1, 2, ...]
    # Channel 1: [0, 1, 2, ...] shifted by 2
    
    t = np.arange(10, dtype=np.float32)
    c0 = t
    c1 = np.roll(t, 2) # Right shift by 2
    
    tensor = np.stack([c0, c1], axis=0) # (2, 10)
    tensor = tensor[np.newaxis, ...] # (1, 2, 10)
    tensor = ops.convert_to_tensor(tensor)
    
    corrs = gf_pearson.rolling_pearson(
        tensor,
        max_arival_time_difference_seconds=max_diff,
        sample_rate_hertz=sample_rate
    )
    
    # Shape: (Batch, NumPairs, N_offsets)
    # NumPairs = 1 (0 vs 1)
    # N_offsets = 4
    assert ops.shape(corrs) == (1, 1, 4)
    
    # Offsets logic:
    # shift = N_offsets - offset
    # N_offsets = 4.
    # offset=0 -> shift=4
    # offset=1 -> shift=3
    # offset=2 -> shift=2 (Match!)
    # offset=3 -> shift=1
    
    # So index 2 should be max correlation (1.0)
    # But wait, c1 is rolled by 2.
    # We compute corr(c0, roll(c1, shift)).
    # We want roll(c1, shift) == c0.
    # c1 is roll(c0, 2).
    # roll(roll(c0, 2), shift) = roll(c0, 2+shift).
    # We want 2+shift = 0 (mod 10) -> shift = -2 = 8.
    # Or shift = -2.
    
    # Wait, my manual logic for `rolling_pearson` shifts:
    # shift = 2M - offset. Positive shifts.
    # We are shifting `y` (c1).
    # c1 is already shifted by +2.
    # If we shift it further right, it moves further away.
    
    # Maybe I should construct test such that `y` needs to be shifted to match `x`.
    # x = t.
    # y = t shifted by -2 (left).
    # Then shifting y by +2 (right) makes it match x.
    
    c1_left = np.roll(t, -2)
    tensor_left = np.stack([c0, c1_left], axis=0)[np.newaxis, ...]
    tensor_left = ops.convert_to_tensor(tensor_left)
    
    corrs = gf_pearson.rolling_pearson(
        tensor_left,
        max_arival_time_difference_seconds=max_diff,
        sample_rate_hertz=sample_rate
    )
    
    # We expect match at shift=+2.
    # shift = 4 - offset.
    # 2 = 4 - offset -> offset = 2.
    # So index 2 should be high.
    
    assert corrs[0, 0, 2] > 0.9
