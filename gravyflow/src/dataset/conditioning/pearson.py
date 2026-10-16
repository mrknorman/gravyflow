import math

import keras
from keras import ops
import jax.numpy as jnp
import jax

@jax.jit
def pearson(x, y):
    # x, y: (Batch, Time)
    
    mean_x = ops.mean(x, axis=-1, keepdims=True)
    mean_y = ops.mean(y, axis=-1, keepdims=True)
    
    numerator = ops.sum((x - mean_x) * (y - mean_y), axis=-1)
    
    var_x = ops.sum(ops.square(x - mean_x), axis=-1)
    var_y = ops.sum(ops.square(y - mean_y), axis=-1)
    
    denominator = ops.sqrt(var_x * var_y)
    
    return numerator / (denominator + 1e-5)

def max_lag_samples(max_arrival_time_difference_seconds: float, sample_rate_hertz: float) -> int:
    """
    The largest lag `rolling_pearson` computes, in samples: the largest arrival-time
    difference rounded up, so that every physically allowed delay is sampled.
    """
    return math.ceil(float(max_arrival_time_difference_seconds) * float(sample_rate_hertz))


def num_lags(max_arrival_time_difference_seconds: float, sample_rate_hertz: float) -> int:
    """
    The number of lags `rolling_pearson` computes for each detector pair: every lag
    from -M to +M samples, ``2 M + 1``, where M is ``max_lag_samples``.
    """
    return 2 * max_lag_samples(max_arrival_time_difference_seconds, sample_rate_hertz) + 1


@jax.jit(static_argnames=["max_arrival_time_difference_seconds", "sample_rate_hertz"])
def rolling_pearson(
        tensor, 
        max_arrival_time_difference_seconds: float,
        sample_rate_hertz: float
    ):
    
    """
    The Pearson correlation of every detector pair at every lag from -M to +M
    samples (``num_lags``), shape (Batch, Pairs, 2 M + 1). Lag index i compares the
    first detector with the second rolled by M - i samples.
    """
    tensor = ops.convert_to_tensor(tensor)
    # tensor: (Batch, Channels, Time)
    max_samples = max_lag_samples(max_arrival_time_difference_seconds, sample_rate_hertz)

    num_batches, num_arrays, array_size = ops.shape(tensor)
    
    # Create pairs
    # Use triu_indices for upper triangle (k=1 excludes diagonal)
    i_idxs, j_idxs = jnp.triu_indices(num_arrays, k=1)
    
    # Extract pairs
    # x_all: (NumPairs, Batch, Time) -> Transpose to (Batch, NumPairs, Time)
    # tensor is (Batch, Channels, Time)
    # We want to gather channels.
    # tensor_transposed: (Channels, Batch, Time)
    tensor_t = ops.transpose(tensor, (1, 0, 2))
    
    x_all = ops.take(tensor_t, i_idxs, axis=0) # (NumPairs, Batch, Time)
    y_all = ops.take(tensor_t, j_idxs, axis=0) # (NumPairs, Batch, Time)
    
    x_all = ops.transpose(x_all, (1, 0, 2)) # (Batch, NumPairs, Time)
    y_all = ops.transpose(y_all, (1, 0, 2)) # (Batch, NumPairs, Time)
    
    # Lags +M down to -M, inclusive.
    offsets = jnp.arange(2 * max_samples + 1)
    
    def compute_for_offset(offset, x, y):
        # offset is scalar; x, y are (Batch, Time).
        shift = max_samples - offset
        
        # tf.roll(y, shift, axis=-1)
        # jnp.roll
        y_shifted = jnp.roll(y, shift, axis=-1)
        
        return pearson(x, y_shifted)
        
    # vmap over offsets
    # We want to apply this to each pair.
    
    def compute_pair_corrs(x, y):
        # x, y: (Batch, Time)
        # vmap over offsets
        # return shape: (N_offsets, Batch) -> Transpose to (Batch, N_offsets)
        
        corrs = jax.vmap(lambda o: compute_for_offset(o, x, y))(offsets)
        return ops.transpose(corrs, (1, 0))
        
    # vmap over pairs
    # x_all: (Batch, NumPairs, Time)
    # We want to map over axis 1 (NumPairs).
    # But x, y arguments to compute_pair_corrs are (Batch, Time).
    # So we map over axis 1 of inputs, and stack results on axis 1.
    
    # jax.vmap(fun, in_axes=1, out_axes=1)
    
    all_correlations = jax.vmap(compute_pair_corrs, in_axes=1, out_axes=1)(x_all, y_all)
    # Result: (Batch, NumPairs, N_offsets)
    
    return all_correlations