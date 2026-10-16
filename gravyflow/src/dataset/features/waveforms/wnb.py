import keras
from keras import ops
import jax.numpy as jnp
import jax
import numpy as np


def time_axis(num_samples: int, sample_rate_hertz: float):
    """Sample times in seconds, with t = 0 at sample num_samples / 2 (float32)."""
    return (jnp.arange(num_samples, dtype="float32") - num_samples / 2.0) / sample_rate_hertz


def hann_envelope(t, duration_seconds):
    """
    A Hann window of full width ``duration_seconds`` centred on t = 0.

    ``t`` has shape (time,) and ``duration_seconds`` (batch,); the result is
    (batch, time): cos^2(pi t / duration) inside |t| < duration / 2, else 0.
    """
    t = t[None, :]
    duration = jnp.asarray(duration_seconds, dtype="float32")[:, None]
    inside = jnp.abs(t) < duration / 2.0
    return jnp.where(inside, jnp.cos(jnp.pi * t / duration) ** 2, 0.0).astype("float32")


def band_mask(num_samples: int, sample_rate_hertz: float, min_frequency_hertz, max_frequency_hertz):
    """
    (batch, num_samples // 2 + 1) booleans: the rfft bins of a num_samples
    series whose frequency, k * sample_rate_hertz / num_samples, lies in
    [min_frequency_hertz, max_frequency_hertz].
    """
    frequencies = jnp.fft.rfftfreq(num_samples, 1.0 / sample_rate_hertz)
    low = jnp.asarray(min_frequency_hertz, dtype="float32")[:, None]
    high = jnp.asarray(max_frequency_hertz, dtype="float32")[:, None]
    return (frequencies[None, :] >= low) & (frequencies[None, :] <= high)


def adjust_envelopes_shape(filtered_noise, envelopes):
    # Determine the condition: whether envelopes have one extra sample compared to filtered_noise
    # filtered_noise: (Batch, 2, Time)
    # envelopes: (Batch, Time) -> expand to (Batch, 1, Time)

    env_shape = ops.shape(envelopes)
    noise_shape = ops.shape(filtered_noise)

    # Check last dim
    cond = env_shape[-1] == noise_shape[-1] + 1

    if cond:  # pragma: no cover
        return envelopes[..., :-1]
    return envelopes

def wnb(
    num_waveforms: int,
    sample_rate_hertz: float,
    max_duration_seconds: float,
    duration_seconds,
    min_frequency_hertz,
    max_frequency_hertz,
    seed : int,
    polarization_eccentricity = None
):
    """
    White noise bursts: band-limited Gaussian noise under a Hann envelope.

    Each polarisation is independent unit Gaussian noise, kept inside
    |t| < duration / 2 of the buffer's centre (sample N / 2), filtered to the
    rfft bins whose frequencies lie in [min_frequency_hertz,
    max_frequency_hertz], and multiplied by a Hann envelope of full width
    ``duration_seconds`` centred on the same sample. Returns (num_waveforms, 2,
    floor(max_duration_seconds * sample_rate_hertz)), float32.

    ``polarization_eccentricity`` (LAL's polarisation-ellipse eccentricity,
    per waveform) scales the plus and cross polarisations by sqrt(2) a and
    sqrt(2) b, with a = 1/sqrt(2 - e^2) and b = a sqrt(1 - e^2): e = 0 (the
    default) leaves two equal-power polarisations, e = 1 is plus only.
    """
    min_frequency_hertz = jnp.asarray(min_frequency_hertz, dtype="float32")
    max_frequency_hertz = jnp.asarray(max_frequency_hertz, dtype="float32")
    duration_seconds = jnp.asarray(duration_seconds, dtype="float32")

    max_num_samples = int(np.floor(max_duration_seconds * sample_rate_hertz))
    t = time_axis(max_num_samples, sample_rate_hertz)

    # Unit Gaussian noise for both polarisations, kept inside the burst.
    key = jax.random.PRNGKey(seed)
    gaussian_noise = jax.random.normal(
        key,
        shape=[num_waveforms, 2, max_num_samples],
        dtype="float32"
    )
    inside = jnp.abs(t)[None, :] < duration_seconds[:, None] / 2.0
    white_noise_burst = gaussian_noise * inside[:, None, :].astype("float32")

    band = band_mask(
        max_num_samples, sample_rate_hertz, min_frequency_hertz, max_frequency_hertz
    )
    filtered_noise = jnp.fft.irfft(
        jnp.fft.rfft(white_noise_burst) * band[:, None, :].astype("complex64"),
        n=max_num_samples
    )

    waveforms = filtered_noise * hann_envelope(t, duration_seconds)[:, None, :]

    if polarization_eccentricity is not None:
        eccentricity = jnp.asarray(polarization_eccentricity, dtype="float32")
        a = 1.0 / jnp.sqrt(2.0 - eccentricity ** 2)
        b = a * jnp.sqrt(1.0 - eccentricity ** 2)
        scale = jnp.sqrt(2.0) * jnp.stack([a, b], axis=-1)
        waveforms = waveforms * scale[:, :, None]

    return waveforms.astype("float32")

def generate_envelopes(
    num_samples_array,
    max_num_samples
    ):
    """
    Hann windows of ``num_samples_array`` samples each, centred in
    ``max_num_samples``; shape (batch, max_num_samples).
    """

    def create_envelope(num_samples):
        n = ops.arange(max_num_samples, dtype="float32")

        # Centered alignment
        start_idx = (max_num_samples - num_samples) // 2

        # Mask: start <= n < start + num
        mask = (n >= start_idx) & (n < (start_idx + num_samples))

        # Effective n for Hann window calculation (0 to num-1)
        effective_n = n - start_idx

        N_minus_1 = ops.maximum(num_samples - 1.0, 1.0)

        val = 0.5 * (1.0 - ops.cos(2.0 * np.pi * effective_n / N_minus_1))

        return val * ops.cast(mask, "float32")

    envelopes = jax.vmap(create_envelope)(num_samples_array)
    return envelopes
