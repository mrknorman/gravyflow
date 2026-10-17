from dataclasses import dataclass
from enum import Enum, auto

import keras
from keras import ops
import jax
import jax.numpy as jnp
import numpy as np

@dataclass 
class ConditioningMethod:
    index : str
    shape: tuple = (1,)
    
class ConditioningMethods(Enum):
    WHITEN = ConditioningMethod(0)
    ROLLING_PEARSON = ConditioningMethod(1)
    SPECTROGRM = ConditioningMethod(2)
    
class DataStreams(Enum):
    ONSOURCE = auto()
    OFFSOURCE = auto()

@dataclass
class DataConditioner():
    type_ : ConditioningMethods
    data_stream : DataStreams = DataStreams.ONSOURCE
    
class DataWhitener(DataConditioner):
    None
    
class PearsonCorrelator(DataConditioner):
    None
    
class CreateSpectrogram(DataConditioner):
    num_frame_samples : int = 256
    num_step_samples : int = 128
    num_fft_samples : int = 256
    
    def apply(self, timeseries):
        
        return spectrogram(
            timeseries,
            self.num_frame_samples,
            self.num_step_samples,
            self.num_fft_samples
        )
    
# The smallest normal float32. XLA on CPU flushes anything below it to zero.
FLOAT32_TINY = float(np.finfo(np.float32).tiny)


def spectrogram(
        timeseries,
        num_frame_samples : int = 256,
        num_step_samples : int = 128,
        num_fft_samples : int = 256
    ):
    """
    The power spectrogram |STFT|^2 of each row of `timeseries` (..., T), in the
    input's units squared: symmetric Hann frames of num_frame_samples every
    num_step_samples, each a num_fft_samples-point rfft. Returns
    (..., frames, num_fft_samples // 2 + 1); a 1D input gives (frames, F).

    The power of physical strain (~1e-40) is below float32's range. The STFT is
    taken of each row over its peak, and the peak's square restored after; an
    input whose power would still underflow, such as raw strain, raises
    ValueError rather than return zeros. Take the spectrogram of whitened data
    (``WHITENED_ONSOURCE``, ``SPECTROGRAM_ONSOURCE`` or ``gf.whiten``). The check
    needs concrete values, so it does not run under ``jax.jit``.
    """
    timeseries = ops.convert_to_tensor(timeseries)
    signal_len = ops.shape(timeseries)[-1]
    num_frames = (signal_len - num_frame_samples) // num_step_samples + 1
    indices = ops.arange(num_frame_samples)[None, :] + ops.arange(num_frames)[:, None] * num_step_samples

    original_shape = ops.shape(timeseries)
    flat_signal = ops.reshape(timeseries, (-1, signal_len))
    peak = ops.max(ops.abs(flat_signal), axis=-1, keepdims=True)
    peak = ops.where(peak > 0, peak, ops.ones_like(peak))

    frames = ops.take(flat_signal / peak, indices, axis=-1)  # (Batch, NumFrames, FrameLen)
    window = ops.convert_to_tensor(jnp.hanning(num_frame_samples), dtype="float32")
    stfts = jnp.fft.rfft(frames * window, n=num_fft_samples)
    normalised = ops.real(stfts) ** 2 + ops.imag(stfts) ** 2

    peak = peak[..., None]
    if not isinstance(normalised, jax.core.Tracer):
        lost = (np.asarray(normalised) >= FLOAT32_TINY) & (
            np.asarray(normalised, np.float64) * np.asarray(peak, np.float64) ** 2 < FLOAT32_TINY)
        if lost.any():
            raise ValueError(
                f"gf.spectrogram: the power of this input (peak {float(np.max(peak)):.2g}) is below "
                "float32's range; physical strain's is ~1e-40. Take the spectrogram of whitened "
                "data (WHITENED_ONSOURCE, SPECTROGRAM_ONSOURCE or gf.whiten)."
            )
    # (power * peak) * peak: no intermediate leaves float32's range when the result is in it.
    spectrograms = (normalised * peak) * peak

    if len(original_shape) == 1:
        spectrograms = ops.squeeze(spectrograms, axis=0)
    else:
        spectrogram_shape = tuple(original_shape[:-1]) + (num_frames, num_fft_samples // 2 + 1)
        spectrograms = ops.reshape(spectrograms, spectrogram_shape)

    return spectrograms

def spectrogram_shape(
        input_shape, 
        num_frame_samples = 256, 
        num_step_samples = 128, 
        num_fft_samples = 256
    ):
    
    # Extract the last dimension of the input shape which is num_samples
    num_samples = input_shape[-1]

    if (num_frame_samples > num_samples):
        num_frame_samples = num_samples
    
    # Calculate the number of time frames (T)
    num_time_frames = 1 + (num_samples - num_frame_samples) // num_step_samples
    
    # Calculate the number of frequency bins (F)
    num_frequency_bins = num_fft_samples // 2 + 1
    
    # The resulting shape includes the original dimensions and the new spectrogram dimensions
    output_shape = input_shape[:-1] + (num_time_frames, num_frequency_bins)
    
    return output_shape