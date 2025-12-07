from typing import Optional, Tuple
from functools import partial

import keras
from keras import ops
import jax.numpy as jnp
import jax

@partial(jax.jit, static_argnames=["n"])
def fftfreq(
        n : int, 
        d : float = 1.0
    ):
    val = 1.0 / (n * d)
    # ops.arange returns tensor
    results = ops.arange(0, n // 2 + 1, dtype="float32") 
    return results * val

@partial(jax.jit, static_argnames=["axis", "type", "bp", "overwrite_data"])
def detrend(data, axis=-1, type='linear', bp=0, overwrite_data=False):
    """
    Remove linear trend along axis from data using Keras Ops / JAX.
    """
    data = ops.convert_to_tensor(data)
    dtype = data.dtype
    
    if type not in ['linear', 'l', 'constant', 'c']:
        raise ValueError("Trend type must be 'linear' or 'constant'.")

    if type in ['constant', 'c']:
        # ops.mean with keepdims
        mean = ops.mean(data, axis=axis, keepdims=True)
        return data - mean
    else:
        # Linear detrending using JAX lstsq
        # This is a bit complex to port 1:1 with the exact TF logic if it uses dynamic shapes heavily.
        # However, typically we just want to fit a line y = mx + c to the data along axis.
        
        # Simplified linear detrend implementation for common case (no breakpoints)
        if bp != 0:
             raise NotImplementedError("Breakpoints not yet supported in JAX port of detrend.")
        
        N = ops.shape(data)[axis]
        
        # Create design matrix A = [1, t]
        # t is 0 to N-1
        t = ops.arange(N, dtype=dtype)
        # A needs to be (N, 2)
        ones = ops.ones((N,), dtype=dtype)
        A = ops.stack([ones, t], axis=-1) # (N, 2)
        
        # We need to solve A * x = y for each signal.
        # y is the data along axis.
        # If data is (Batch, N), we want to solve for each batch.
        
        # Move axis to end
        if axis != -1 and axis != len(ops.shape(data)) - 1:
            data_transposed = ops.moveaxis(data, axis, -1)
        else:
            data_transposed = data
            
        # Reshape to (Batch, N)
        original_shape = ops.shape(data_transposed)
        data_reshaped = ops.reshape(data_transposed, (-1, N))
        
        # JAX lstsq: jnp.linalg.lstsq(a, b) solves ax = b
        # a is (N, 2), b is (N, Batch) or (Batch, N)?
        # numpy.linalg.lstsq expects b to be (N, K)
        
        # Transpose data to (N, Batch)
        b = ops.transpose(data_reshaped, (1, 0))
        
        # Solve
        # A is (N, 2), b is (N, Batch)
        # x will be (2, Batch)
        coeffs, _, _, _ = jnp.linalg.lstsq(A, b, rcond=None)
        
        # Reconstruct trend
        # trend = A @ coeffs -> (N, 2) @ (2, Batch) -> (N, Batch)
        trend = ops.matmul(A, coeffs)
        
        # Transpose back to (Batch, N)
        trend = ops.transpose(trend, (1, 0))
        
        # Subtract trend
        detrended = data_reshaped - trend
        
        # Reshape back to original
        detrended = ops.reshape(detrended, original_shape)
        
        # Move axis back if needed
        if axis != -1 and axis != len(ops.shape(data)) - 1:
            detrended = ops.moveaxis(detrended, -1, axis)
            
        return detrended

@jax.jit(static_argnames=["nperseg", "sample_rate_hertz", "noverlap", "mode"])
def psd(
        signal, 
        nperseg : int, 
        sample_rate_hertz : float, 
        noverlap : Optional[int] = None, 
        mode : str ="mean"
    ):
    
    signal = ops.convert_to_tensor(signal)
    
    if noverlap is None:
        noverlap = nperseg // 2
        
    signal = detrend(signal, axis=-1, type='constant')
    
    # Step 1: Split the signal into overlapping segments
    # JAX/Keras doesn't have tf.signal.frame.
    # We can use jax.numpy stride tricks or manual extraction.
    # Or simply reshape if noverlap is 0, but it usually isn't.
    
    # Implementing frame using slicing
    step = nperseg - noverlap
    signal_len = ops.shape(signal)[-1]
    num_frames = (signal_len - nperseg) // step + 1
    
    # Indices: (num_frames, nperseg)
    # 0, 1, ..., nperseg-1
    # step, step+1, ...
    
    indices = ops.arange(nperseg)[None, :] + ops.arange(num_frames)[:, None] * step
    
    # If signal is (Batch, Time), we want (Batch, Frames, nperseg)
    # ops.take might work but indices need to be right.
    
    # Using jax.lax.gather or simply a loop if num_frames is small? No, loop is bad.
    # jnp.lib.stride_tricks.sliding_window_view is available in newer numpy/jax?
    # jnp doesn't have sliding_window_view yet in all versions.
    
    # Let's use indices gathering.
    # signal: (..., T)
    # indices: (F, W)
    # We want output: (..., F, W)
    
    # Flatten signal to 2D (Batch, T)
    original_shape = ops.shape(signal)
    if len(original_shape) > 1:
        flat_signal = ops.reshape(signal, (-1, signal_len))
    else:
        flat_signal = ops.reshape(signal, (1, signal_len))

        
    # Gather
    frames = ops.take(flat_signal, indices, axis=-1)
    
    # Step 2: Apply a window function to each segment
    window = jnp.hanning(nperseg) 
    window = ops.convert_to_tensor(window, dtype="float32")
    
    windowed_frames = frames * window
    
    # Step 3: Compute the periodogram
    # Use jnp.fft.rfft directly
    fft_vals = jnp.fft.rfft(windowed_frames)
    
    periodograms = ops.abs(fft_vals)**2 / ops.sum(window**2)
    
    # Step 4: Compute median or mean
    if mode == "median":
        # jnp.percentile
        pxx = jnp.percentile(periodograms, 50.0, axis=-2)
    elif mode == "mean":
        pxx = ops.mean(periodograms, axis=-2)
    else:
        raise ValueError("Mode not supported")
    
    # Step 5: Frequencies
    freqs = fftfreq(nperseg, d=1.0/sample_rate_hertz)
    
    # Step 6: Mask
    # pxx shape: (Batch, Freqs)
    X = ops.shape(pxx)[-1]
    
    # Construct mask
    # [1, 2, 2, ..., 2, 1]
    mask = ops.concatenate(
        [
            ops.convert_to_tensor([1.], dtype="float32"), 
            ops.ones((X-2,), dtype="float32") * 2.0, 
            ops.convert_to_tensor([1.], dtype="float32")
        ], 
        axis=0
    )
    
    pxx = mask * pxx / sample_rate_hertz
    
    # Reshape back to original dimensions
    if len(original_shape) > 1:
        new_shape = tuple(original_shape[:-1]) + (X,)
        pxx = ops.reshape(pxx, new_shape)
        
    return freqs, pxx