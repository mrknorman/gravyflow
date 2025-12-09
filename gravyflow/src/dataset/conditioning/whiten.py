import keras
from keras import ops, Layer
import jax.numpy as jnp
import numpy as np
from gravyflow.src.dataset.tools.psd import psd
from gravyflow.src.dataset.config import Defaults
from gravyflow.src.utils.tensor import crop_samples

import jax

@jax.jit(static_argnames=["N", "nleft", "nright"])
def planck(N: int, nleft: int, nright: int):
    """
    Create a Planck-taper window.
    """
    # Creating left and right ranges
    left = ops.arange(nleft, dtype="float32")
    right = ops.arange(nright, dtype="float32") - nright + 1
    
    # Apply the Planck-taper function to left and right ranges
    taper_left = 1.0 / (ops.exp(-left/(nleft-1)) + 1)
    taper_right = 1.0 / (ops.exp(-right/(nright-1)) + 1)
    
    # Combine the left taper, a flat middle segment, and the right taper
    window = ops.concatenate([
        taper_left, 
        ops.ones((N-nleft-nright,), dtype="float32"), 
        ops.flip(taper_right, axis=0) # tf.reverse -> ops.flip
    ], axis=0)
    
    return window

@jax.jit(static_argnames=["ncorner"])
def truncate_transfer(
    transfer,
    ncorner: int = 0
    ):
    """
    Smoothly zero the edges of a frequency domain transfer function.
    """
    nsamp = ops.shape(transfer)[-1]
    ncorner = ncorner if ncorner else 0
    
    # Validate that ncorner is within the range of the array size
    if ncorner >= nsamp:
        raise ValueError(
            "ncorner must be less than the size of the transfer array"
        )
        
    plank = planck(nsamp-ncorner, nleft=5, nright=5)
    
    transfer_zeros = ops.zeros_like(transfer[...,:ncorner])
    transfer_mod = transfer[...,ncorner:nsamp] * plank
    new_transfer = ops.concatenate([transfer_zeros, transfer_mod], axis=-1)
    
    return new_transfer

@jax.jit(static_argnames=["ntaps", "window"])
def truncate_impulse(
    impulse, 
    ntaps: int, 
    window: str = 'hann'
    ):
    """
    Smoothly truncate a time domain impulse response.
    """
    
    # Ensure ntaps does not exceed the size of the impulse response
    if ntaps % 2 != 0:
        raise ValueError("ntaps must be an even number")
    
    trunc_start = int(ntaps / 2)
    trunc_stop = ops.shape(impulse)[-1] - trunc_start
        
    if window == 'hann':
        # jnp.hanning
        win = jnp.hanning(ntaps)
        win = ops.convert_to_tensor(win, dtype="float32")
    else:
        raise ValueError(f"Window function {window} not supported")
    
    impulse_start = impulse[...,:trunc_start] * win[trunc_start:ntaps]
    impulse_stop = impulse[...,trunc_stop:] * win[:trunc_start]
    impulse_middle = ops.zeros_like(impulse[...,trunc_start:trunc_stop])
    
    new_impulse = ops.concatenate([impulse_start, impulse_middle, impulse_stop], axis=-1)

    return new_impulse

@jax.jit(static_argnames=["ntaps", "window", "ncorner"])
def fir_from_transfer(
    transfer, 
    ntaps: int, 
    window: str = 'hann', 
    ncorner: int = 0
    ):
    """
    Design a Type II FIR filter given an arbitrary transfer function
    """

    if ntaps % 2 != 0:
        raise ValueError("ntaps must be an even number")
    
    transfer = truncate_transfer(transfer, ncorner=ncorner)
    
    # irfft
    # tf.signal.irfft expects complex input.
    # If transfer is real, casting to complex makes imaginary part 0.
    
    transfer_complex = ops.cast(transfer, "complex64")
    impulse = jnp.fft.irfft(transfer_complex)
    
    impulse = truncate_impulse(impulse, ntaps=ntaps, window=window)
    
    # roll
    shift = int(ntaps/2 - 1)
    impulse = ops.roll(impulse, shift=shift, axis=-1)[...,: ntaps]
    return impulse

def _centered(arr, newsize):
    # Ensure correct dimensionality
    # if len(arr.shape) == 1: ...
    # Keras ops.shape returns tensor or tuple?
    
    arr_shape = ops.shape(arr)
    # If 1D, expand dims?
    # ops.expand_dims
    
    start_ind = (arr_shape[-1] - newsize) // 2
    end_ind = start_ind + newsize
    return arr[..., start_ind:end_ind]

@jax.jit(static_argnames=["mode"])
def fftconvolve(in1, in2, mode="full"):
    # Extract shapes
    s1 = ops.shape(in1)[-1]
    s2 = ops.shape(in2)[-1]
    shape = s1 + s2 - 1

    # Compute convolution in Fourier space
    # jnp.fft.rfft(a, n=shape)
    sp1 = jnp.fft.rfft(in1, n=shape)
    sp2 = jnp.fft.rfft(in2, n=shape)
    ret = jnp.fft.irfft(sp1 * sp2, n=shape)

    # Crop according to mode
    if mode == "full":
        cropped = ret
    elif mode == "same":
        cropped = _centered(ret, s1)
    elif mode == "valid":
        cropped = _centered(ret, s1 - s2 + 1)
    else:
        raise ValueError("Acceptable mode flags are 'valid', 'same', or 'full'.")
    
    return cropped

@jax.jit(static_argnames=["window"])
def convolve(
    timeseries, 
    fir, 
    window: str = 'hann'
    ):
    """
    Perform convolution between the timeseries and the finite impulse response 
    filter.
    """
    pad = int(np.ceil(ops.shape(fir)[-1]/2))
    
    # Optimizing FFT size to power of 2 for efficiency
    # nfft = min(8*fir.shape[-1], timeseries.shape[-1])
    # Not used in 'same' mode logic below?
    
    if window == 'hann':
        win = jnp.hanning(ops.shape(fir)[-1])
        win = ops.convert_to_tensor(win, dtype="float32")
    else:
        raise ValueError(f"Window function {window} not supported")

    timeseries_new_front = timeseries[..., :pad] * win[:pad]
    timeseries_new_back = timeseries[..., -pad:] * win[-pad:]
    timeseries_new_middle = timeseries[..., pad:-pad]

    timeseries_new = ops.concatenate([
        timeseries_new_front, 
        timeseries_new_middle, 
        timeseries_new_back
    ], axis=-1)

    conv = ops.zeros_like(timeseries_new)
    conv = fftconvolve(timeseries_new, fir, mode='same')
    
    return conv

@jax.jit(static_argnames=["sample_rate_hertz", "fft_duration_seconds", "overlap_duration_seconds", "highpass_hertz", "detrend", "filter_duration_seconds", "window", "num_samples"])
def whiten(
    timeseries, 
    background,
    sample_rate_hertz: float, 
    fft_duration_seconds: float = 2.0, 
    overlap_duration_seconds: float = 1.0,
    highpass_hertz: float = None,
    detrend: str ='constant',
    filter_duration_seconds: float = 2.0,
    window: str = "hann",
    num_samples: int = None
    ):
    """
    Whiten a timeseries using the given parameters.
    """
    
    # Validate highpass frequency, if applicable
    if highpass_hertz:
        if highpass_hertz < 0 or highpass_hertz >= (sample_rate_hertz / 2):
            raise ValueError("Invalid highpass frequency.")

    # Validate filter duration
    if filter_duration_seconds <= 0:
        raise ValueError("Filter duration should be greater than zero.")

    # Validate window parameter
    if window not in ["hann"]:  # Extend list as needed
        raise ValueError("Invalid window type.")
    
    epsilon = 1e-8  # Small constant to avoid division by zero

    timeseries = ops.convert_to_tensor(timeseries)
    background = ops.convert_to_tensor(background)

    # Check if input is 1D or 2D
    is_1d = len(ops.shape(timeseries)) == 1
    if is_1d:
        # If 1D, add an extra dimension
        timeseries = ops.expand_dims(timeseries, axis=0)
        background = ops.expand_dims(background, axis=0)

    dt = 1.0 / sample_rate_hertz
    
    # Get num_samples - if not provided, extract from timeseries shape
    # When num_samples is passed as static arg, all derived values are concrete
    if num_samples is None:
        num_samples = int(timeseries.shape[-1])
    
    freqs, psd_val = psd(
        background, 
        nperseg=int(sample_rate_hertz*fft_duration_seconds), 
        noverlap=int(sample_rate_hertz*overlap_duration_seconds), 
        sample_rate_hertz=sample_rate_hertz
    )
    
    # Ensure psd doesn't contain negative values
    psd_val = ops.maximum(psd_val, 0.0)
    asd = ops.sqrt(psd_val)
    
    # Use static num_samples for df calculation to avoid JIT concretization errors
    df = sample_rate_hertz / float(num_samples)
    
    num_freqs = num_samples // 2 + 1
    fsamples = ops.arange(0, num_freqs, dtype="float32") * df
    freqs = ops.cast(freqs, "float32")
    
    # Interpolation
    # asd is (Batch, Freqs).
    # fsamples is (TargetFreqs,).
    
    def interp_fn(p):
        return jnp.interp(fsamples, freqs, p)
    
    # asd always has rank > 1 since input is always expanded to 2D before PSD
    asd_flat = ops.reshape(asd, (-1, ops.shape(asd)[-1]))
    asd_interp = jnp.vectorize(interp_fn, signature='(n)->(m)')(asd_flat)
    target_len = ops.shape(fsamples)[0]
    asd = ops.reshape(asd_interp, (*ops.shape(asd)[:-1], target_len))

    ncorner = int(highpass_hertz / df) if highpass_hertz else 0
    ntaps = int(filter_duration_seconds * sample_rate_hertz)
    
    # Ensure asd doesn't contain zeros to avoid division by zero
    asd = ops.maximum(asd, epsilon)
    transfer = 1.0 / asd

    tdw = fir_from_transfer(transfer, ntaps, window=window, ncorner=ncorner)
    out = convolve(timeseries, tdw)

    # If input was 1D, return 1D
    if is_1d:
        out = out[0]

    return out * ops.sqrt(2.0 * dt)

@keras.saving.register_keras_serializable(package="gravyflow")
class WhitenPass(Layer):
    def __init__(
            self, 
            sample_rate_hertz = None,
            onsource_duration_seconds = None,
            fft_duration_seconds=4, 
            overlap_duration_seconds=2,
            highpass_hertz=None, 
            detrend='constant', 
            filter_duration_seconds=2.0, 
            window="hann", 
            **kwargs
        ):
        super().__init__(**kwargs)

        if sample_rate_hertz is None:
            sample_rate_hertz = Defaults.sample_rate_hertz
        if onsource_duration_seconds is None:
            onsource_duration_seconds = Defaults.onsource_duration_seconds

        self.onsource_duration_seconds = onsource_duration_seconds
        self.sample_rate_hertz = sample_rate_hertz
        self.fft_duration_seconds = fft_duration_seconds
        self.overlap_duration_seconds = overlap_duration_seconds
        self.highpass_hertz = highpass_hertz
        self.detrend = detrend
        self.filter_duration_seconds = filter_duration_seconds
        self.window = window
        self.num_output_samples = int(self.onsource_duration_seconds*self.sample_rate_hertz)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        timeseries = inputs

        cropped = crop_samples(timeseries, self.onsource_duration_seconds, self.sample_rate_hertz)

        dynamic_shape = ops.shape(timeseries)
        return ops.reshape(cropped, (dynamic_shape[0], dynamic_shape[1], self.num_output_samples))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sample_rate_hertz': self.sample_rate_hertz,
            'fft_duration_seconds': self.fft_duration_seconds,
            'overlap_duration_seconds': self.overlap_duration_seconds,
            'highpass_hertz': self.highpass_hertz,
            'detrend': self.detrend,
            'filter_duration_seconds': self.filter_duration_seconds,
            'window': self.window,
        })
        return config

    def compute_output_shape(self, input_shape):
        timeseries_shape = input_shape
        return (timeseries_shape[0], timeseries_shape[1], self.onsource_duration_seconds*self.sample_rate_hertz) 

@keras.saving.register_keras_serializable(package="gravyflow")
class Whiten(Layer):
    def __init__(
            self, 
            sample_rate_hertz = None,
            onsource_duration_seconds = None,
            fft_duration_seconds=4, 
            overlap_duration_seconds=2,
            highpass_hertz=None, 
            detrend='constant', 
            filter_duration_seconds=2.0, 
            window="hann", 
            **kwargs
        ):
        super().__init__(**kwargs)

        if sample_rate_hertz is None:
            sample_rate_hertz = Defaults.sample_rate_hertz
        if onsource_duration_seconds is None:
            onsource_duration_seconds = Defaults.onsource_duration_seconds

        self.onsource_duration_seconds = onsource_duration_seconds
        self.sample_rate_hertz = sample_rate_hertz
        self.fft_duration_seconds = fft_duration_seconds
        self.overlap_duration_seconds = overlap_duration_seconds
        self.highpass_hertz = highpass_hertz
        self.detrend = detrend
        self.filter_duration_seconds = filter_duration_seconds
        self.window = window
        self.num_output_samples = int(self.onsource_duration_seconds*self.sample_rate_hertz)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        timeseries, background = inputs

        whitened = whiten(timeseries, background, self.sample_rate_hertz)
        cropped = crop_samples(whitened, self.onsource_duration_seconds, self.sample_rate_hertz)

        dynamic_shape = ops.shape(timeseries)
        return ops.reshape(cropped, (dynamic_shape[0], dynamic_shape[1], self.num_output_samples))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sample_rate_hertz': self.sample_rate_hertz,
            'fft_duration_seconds': self.fft_duration_seconds,
            'overlap_duration_seconds': self.overlap_duration_seconds,
            'highpass_hertz': self.highpass_hertz,
            'detrend': self.detrend,
            'filter_duration_seconds': self.filter_duration_seconds,
            'window': self.window,
        })
        return config

    def compute_output_shape(self, input_shape):
        timeseries_shape, _ = input_shape
        return (timeseries_shape[0], timeseries_shape[1], int(self.onsource_duration_seconds*self.sample_rate_hertz))