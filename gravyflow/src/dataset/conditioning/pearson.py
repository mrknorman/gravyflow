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

@jax.jit(static_argnames=["max_arival_time_difference_seconds", "sample_rate_hertz"])
def rolling_pearson(
        tensor, 
        max_arival_time_difference_seconds: float,
        sample_rate_hertz: float
    ):
    
    tensor = ops.convert_to_tensor(tensor)
    # tensor: (Batch, Channels, Time)
    
    # Calculate max arrival time difference in samples
    max_samples = int(max_arival_time_difference_seconds * sample_rate_hertz)
    
    # Range of offsets: -max to +max (exclusive of +max? Original code: range(2*max))
    # Original:
    # max_arival_time_difference_samples *= 2
    # for offset in range(max_arival_time_difference_samples):
    #   offset_mag = offset - max_arival_time_difference_samples (Wait, original code used the *doubled* value?)
    
    # Original code:
    # max_arival_time_difference_samples = int(...)
    # max_arival_time_difference_samples *= 2  (Let's call this N_offsets)
    # for offset in range(N_offsets):
    #   offset_mag = offset - N_offsets (Wait, original code: offset - max_arival_time_difference_samples)
    #   If max was doubled, then offset - max is:
    #   0 - 2*M = -2M
    #   2*M-1 - 2*M = -1
    # This shifts are all negative?
    
    # Let's re-read original code carefully.
    # max_arival_time_difference_samples = int(...)
    # max_arival_time_difference_samples *= 2
    # ...
    # for offset in tf.range(max_arival_time_difference_samples):
    #    offset_mag = offset - max_arival_time_difference_samples
    
    # If M was original max. Variable becomes 2M.
    # Loop 0 to 2M-1.
    # offset_mag = offset - 2M.
    # Range: -2M to -1.
    
    # This seems odd. Usually one wants -M to +M.
    # If the variable name `max_arival_time_difference_samples` holds `2*M`.
    # Then `offset - 2*M` is always negative.
    
    # Maybe I should replicate exact behavior even if odd?
    # Or maybe I misread `offset - max_arival_time_difference_samples`.
    # Yes, `max_arival_time_difference_samples` is the DOUBLED value.
    
    # So shifts are indeed `offset - 2M`.
    # `tf.roll(..., shift=-offset_mag)`.
    # `shift = -(offset - 2M) = 2M - offset`.
    # Range: `2M` down to `1`.
    # So it shifts by positive amounts (right shift).
    
    # I will replicate this logic.
    
    num_batches, num_arrays, array_size = ops.shape(tensor)
    
    # Create pairs
    # jax.numpy.meshgrid or manual
    indices = jnp.arange(num_arrays)
    i, j = jnp.meshgrid(indices, indices, indexing='ij')
    
    mask = i < j
    i_idxs = i[mask]
    j_idxs = j[mask]
    
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
    
    # Define offset loop/map
    N_offsets = max_samples * 2
    offsets = jnp.arange(N_offsets)
    
    def compute_for_offset(offset, x, y):
        # offset is scalar
        # x, y are (Batch, Time)
        
        # Replicate original logic:
        # offset_mag = offset - N_offsets
        # shift = -offset_mag = N_offsets - offset
        
        shift = N_offsets - offset
        
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