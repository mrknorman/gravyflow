import tensorflow as tf
import tensorflow_probability as tfp

@tf.function
def fftfreq(n, d=1.0):
    val = 1.0 / (n * d)
    results = tf.range(0, n // 2 + 1, dtype=tf.float32)  # Note the +1 here
    return results * val

def detrend(data, axis=-1, type='linear', bp=0, overwrite_data=False):
    """
    Remove linear trend along axis from data using TensorFlow.

    Parameters
    ----------
    data : Tensor
        The input data.
    axis : int, optional
        The axis along which to detrend the data. By default this is the
        last axis (-1).
    type : {'linear', 'constant'}, optional
        The type of detrending. If ``type == 'linear'`` (default),
        the result of a linear least-squares fit to `data` is subtracted
        from `data`.
        If ``type == 'constant'``, only the mean of `data` is subtracted.
    bp : int, optional
        A break point. If given, an individual linear fit is
        performed for each part of `data` between two break points.
        This parameter only has an effect when ``type == 'linear'``.
    overwrite_data : bool, optional
        If True, perform in place detrending and avoid a copy. Default is False

    Returns
    -------
    ret : Tensor
        The detrended input data.

    Examples
    --------
    >>> import numpy as np
    >>> import tensorflow as tf
    >>> npoints = 1000
    >>> noise = np.random.standard_normal(npoints)
    >>> x = 3 + 2*np.linspace(0, 1, npoints) + noise
    >>> x_tf = tf.convert_to_tensor(x)
    >>> (detrend(x_tf) - noise).numpy().max()
    0.06  # random

    """
    if type not in ['linear', 'l', 'constant', 'c']:
        raise ValueError("Trend type must be 'linear' or 'constant'.")
    dtype = data.dtype
    if type in ['constant', 'c']:
        ret = data - tf.math.reduce_mean(data, axis, keepdims=True)
        return ret
    else:
        N = data.shape[axis]
        bp = tf.sort(tf.unique(tf.concat([0, bp, N], axis=0)).y)
        if tf.math.reduce_any(bp > N):
            raise ValueError("Breakpoints must be less than length "
                             "of data along given axis.")
        Nreg = tf.shape(bp)[0] - 1
        newdims = tf.concat(
            [
                axis,
                tf.range(0, axis), tf.range(axis + 1, tf.rank(data))
            ], 
            axis=0
        )
        newdata = tf.reshape(tf.transpose(data, perm=newdims),
                             [N, tf.math.reduce_prod(tf.shape(data)) // N])
        if not overwrite_data:
            newdata = tf.identity(newdata)
        newdata = tf.cast(newdata, dtype)
        for m in range(Nreg):
            Npts = bp[m + 1] - bp[m]
            A = tf.ones((Npts, 2), dtype)
            A[:, 0] = tf.cast(tf.range(1, Npts + 1) * 1.0 / Npts, dtype)
            sl = slice(bp[m], bp[m + 1])
            coef = tf.linalg.lstsq(A, newdata[sl])
            newdata = newdata[sl] - tf.linalg.matvec(A, coef)
        tdshape = tf.gather(tf.shape(data), newdims)
        ret = tf.reshape(newdata, tdshape)
        olddims = list(
            range(1, tf.rank(data)))[:axis] \
            + [0] + list(range(1, tf.rank(data)))[axis:]
        ret = tf.transpose(ret, perm=olddims)
        return ret

@tf.function 
def psd(
        signal : tf.Tensor, 
        nperseg : int, 
        noverlap : int = None, 
        sample_rate_hertz : float = 1.0, 
        mode : str ="mean"
    ) -> (tf.Tensor, tf.Tensor):
    
    if noverlap is None:
        noverlap = nperseg // 2
        
    signal = detrend(signal, axis=-1, type='constant')
    
    # Step 1: Split the signal into overlapping segments
    signal_shape = tf.shape(signal)
    step = nperseg - noverlap
    frames = tf.signal.frame(signal, frame_length=nperseg, frame_step=step)
        
    # Step 2: Apply a window function to each segment
    # Hanning window is used here, but other windows can be applied as well
    window = tf.signal.hann_window(nperseg, dtype = tf.float32)
    windowed_frames = frames * window
    
    # Step 3: Compute the periodogram (scaled, absolute value of FFT) for each 
    # segment
    periodograms = \
        tf.abs(tf.signal.rfft(windowed_frames))**2 / tf.reduce_sum(window**2)
    
    # Step 4: Compute the median or mean of the periodograms based on the 
    #median_mode
    if mode == "median":
        pxx = tfp.stats.percentile(periodograms, 50.0, axis=-2)
    elif mode == "mean":
        pxx = tf.reduce_mean(periodograms, axis=-2)
    else:
        raise "Mode not supported"
    
    # Step 5: Compute the frequencies corresponding to the power spectrum values
    freqs = fftfreq(nperseg, d=1.0/sample_rate_hertz)
    
    #Create mask to multiply all but the 0 and nyquist frequency by 2
    X = pxx.shape[-1]
    mask = \
        tf.concat(
            [
                tf.constant([1.]), 
                tf.ones([X-2], dtype=tf.float32) * 2.0, 
                tf.constant([1.])
            ], 
            axis=0
        )
        
    return freqs, (mask*pxx / sample_rate_hertz)