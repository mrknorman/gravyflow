"""
Simple Waveform Generators for Testing and Validation

These generators produce simple, analytically-defined waveforms for testing
matched filtering, SNR calculations, and data pipeline validation.

Generators:
- PeriodicWaveGenerator: Sine, square, sawtooth, triangle waves
- SineGaussianGenerator: Sine-Gaussian wavelets, as lalsimulation's
  SimBurstSineGaussian
- ChirpletGenerator: Frequency-modulated chirps
- RingdownGenerator: Damped sinusoids (post-merger approximation)

Conventions, shared by every generator here:
- Each waveform's reference time, the time its ``get_signal_extent_seconds``
  is measured from, is sample ``N / 2`` of the generated buffer (t = 0). It is
  the envelope's centre for the sine-Gaussian, chirplet and periodic wave, and
  the start of the ringdown.
- The burst generators (sine-Gaussian, chirplet, ringdown) return both
  polarisations, with LAL's polarisation-ellipse convention: for an
  eccentricity e, a = 1/sqrt(2 - e^2) and b = a sqrt(1 - e^2), and the plus and
  cross polarisations have root-sum-square amplitudes a * amplitude and
  b * amplitude, so the total hrss, sqrt(sum(h+^2 + hx^2) dt), is
  ``amplitude``. e = 1 is linear (plus only) and e = 0 circular.
- Phases follow LAL's sine-Gaussian: h+ goes as cos(Phi(t) - phase) and hx as
  sin(Phi(t) - phase).
- Outputs are float32.
"""

from dataclasses import dataclass
from typing import Union, Tuple
from enum import Enum, auto

import jax.numpy as jnp
import numpy as np

import gravyflow as gf
from gravyflow.src.dataset.features.injection import get_support_bounds
from gravyflow.src.dataset.features.waveforms.wnb import time_axis, hann_envelope

# Signal extents are where the amplitude envelope falls to this fraction of its
# peak.
_ENVELOPE_FLOOR = 1.0e-3
# The sine-Gaussian envelope exp(-t^2 / tau^2) reaches the floor at this many tau.
_GAUSSIAN_EXTENT_TAUS = float(np.sqrt(np.log(1.0 / _ENVELOPE_FLOOR)))
# The ringdown envelope exp(-t / tau) reaches the floor at this many tau.
_RINGDOWN_EXTENT_TAUS = float(np.log(1.0 / _ENVELOPE_FLOOR))


class WaveShape(Enum):
    """Shape of periodic waveform."""
    SINE = auto()
    SQUARE = auto()
    SAWTOOTH = auto()
    TRIANGLE = auto()


def polarisation_axes(eccentricity):
    """LAL's polarisation-ellipse axes (a, b) for an eccentricity in [0, 1]."""
    a = 1.0 / jnp.sqrt(2.0 - eccentricity ** 2)
    b = a * jnp.sqrt(1.0 - eccentricity ** 2)
    return a, b


def _rss(h, sample_rate_hertz):
    """Root-sum-square amplitude sqrt(sum(h^2) dt) along the last axis."""
    return jnp.sqrt(jnp.sum(h ** 2, axis=-1) / sample_rate_hertz)


def _elliptical(envelope, phase, amplitude, eccentricity, sample_rate_hertz):
    """
    Plus and cross polarisations, (batch, 2, time), from an envelope and a phase.

    The plus polarisation is envelope * cos(phase) and the cross envelope *
    sin(phase), each scaled to root-sum-square a * amplitude and b * amplitude.
    """
    a, b = polarisation_axes(eccentricity)
    cosine = envelope * jnp.cos(phase)
    sine = envelope * jnp.sin(phase)
    cosine_rss = _rss(cosine, sample_rate_hertz)
    sine_rss = _rss(sine, sample_rate_hertz)
    plus = cosine * jnp.where(cosine_rss > 0, a * amplitude / cosine_rss, 0.0)[:, None]
    cross = sine * jnp.where(sine_rss > 0, b * amplitude / sine_rss, 0.0)[:, None]
    return jnp.stack([plus, cross], axis=1).astype("float32")


def _finite_bounds(generator, name):
    value = getattr(generator, name)
    low, high = get_support_bounds(value, name)
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError(
            f"{type(generator).__name__}.{name} has unbounded support; give its "
            "NORMAL distribution explicit min_ and max_ bounds."
        )
    return low, high


def _check_positive(generator, name):
    low, high = _finite_bounds(generator, name)
    if low <= 0.0:
        raise ValueError(
            f"{type(generator).__name__}.{name} must be positive; its support "
            f"reaches {low}."
        )
    return low, high


def _check_eccentricity(generator):
    low, high = _finite_bounds(generator, "polarization_eccentricity")
    if low < 0.0 or high > 1.0:
        raise ValueError(
            f"{type(generator).__name__}.polarization_eccentricity must lie in "
            f"[0, 1]; its support is [{low}, {high}]."
        )


def _check_below_nyquist(generator, names, sample_rate_hertz):
    nyquist = sample_rate_hertz / 2.0
    for name in names:
        _, high = _finite_bounds(generator, name)
        if high >= nyquist:
            raise ValueError(
                f"{type(generator).__name__}.{name} reaches {high} Hz, at or above "
                f"the Nyquist frequency {nyquist} Hz of a {sample_rate_hertz} Hz "
                "sample rate."
            )


def _sample(distribution, num_waveforms):
    return jnp.asarray(np.asarray(distribution.sample(num_waveforms), dtype=np.float32))


@dataclass
class PeriodicWaveGenerator(gf.WaveformGenerator):
    """
    Generator for periodic waveforms (sine, square, sawtooth, triangle).

    Useful for testing matched filter template identification and
    basic signal processing validation. Linearly polarised (hx = 0).

    Parameters:
        wave_shape: Type of periodic wave (SINE, SQUARE, SAWTOOTH, TRIANGLE)
        frequency_hertz: Oscillation frequency
        amplitude: Peak amplitude
        phase_radians: Phase at the envelope's centre (t = 0); the sine wave
            is sin(2 pi f t + phase)
        duration_seconds: Full width of the Hann envelope
    """
    wave_shape: WaveShape = WaveShape.SINE
    frequency_hertz: Union[float, gf.Distribution] = 100.0
    amplitude: Union[float, gf.Distribution] = 1.0
    phase_radians: Union[float, gf.Distribution] = 0.0
    duration_seconds: Union[float, gf.Distribution] = 0.5

    distributed_attributes: Tuple[str] = (
        "frequency_hertz",
        "amplitude",
        "phase_radians",
        "duration_seconds",
    )

    def __post_init__(self):
        self.frequency_hertz = self.ensure_float("frequency_hertz", self.frequency_hertz)
        self.amplitude = self.ensure_float("amplitude", self.amplitude)
        self.phase_radians = self.ensure_float("phase_radians", self.phase_radians)
        self.duration_seconds = self.ensure_float("duration_seconds", self.duration_seconds)

        super().__post_init__()
        _check_positive(self, "frequency_hertz")
        _check_positive(self, "duration_seconds")

    def get_max_generated_duration(self):
        return float(_check_positive(self, "duration_seconds")[1])

    def get_signal_extent_seconds(self, parameters):
        half = np.asarray(parameters[gf.WaveformParameters.DURATION_SECONDS]) / 2.0
        return half, half

    def generate(
        self,
        num_waveforms: int,
        sample_rate_hertz: float,
        duration_seconds: float,
        seed: int
    ):
        if num_waveforms <= 0:
            return None, {}

        if seed is not None:
            self.reseed(seed)
        _check_below_nyquist(self, ("frequency_hertz",), sample_rate_hertz)

        # Sample parameters
        freq = _sample(self.frequency_hertz, num_waveforms)
        amp = _sample(self.amplitude, num_waveforms)
        phase = _sample(self.phase_radians, num_waveforms)
        dur = _sample(self.duration_seconds, num_waveforms)

        num_samples = int(sample_rate_hertz * duration_seconds)
        t = time_axis(num_samples, sample_rate_hertz)

        # Phase referenced to the envelope's centre: (batch, time)
        phi = 2 * jnp.pi * freq[:, None] * t[None, :] + phase[:, None]

        # Generate waveform based on shape
        if self.wave_shape == WaveShape.SINE:
            waveform = jnp.sin(phi)
        elif self.wave_shape == WaveShape.SQUARE:
            waveform = jnp.sign(jnp.sin(phi))
        elif self.wave_shape == WaveShape.SAWTOOTH:
            # Sawtooth: 2 * (t*f mod 1) - 1
            waveform = 2 * (phi / (2 * jnp.pi) % 1) - 1
        elif self.wave_shape == WaveShape.TRIANGLE:
            # Triangle: 2 * |2 * (t*f mod 1) - 1| - 1
            sawtooth = 2 * (phi / (2 * jnp.pi) % 1) - 1
            waveform = 2 * jnp.abs(sawtooth) - 1
        else:
            waveform = jnp.sin(phi)

        waveform = amp[:, None] * waveform * hann_envelope(t, dur)

        # (batch, 2, time) for h+/hx; a periodic wave is linearly polarised.
        waveform = jnp.stack([waveform, jnp.zeros_like(waveform)], axis=1).astype("float32")

        # Apply injection chance
        waveform = self.apply_injection_chance(waveform, seed)

        params = {
            gf.WaveformParameters.FREQUENCY_HERTZ: freq,
            gf.WaveformParameters.AMPLITUDE: amp,
            gf.WaveformParameters.PHASE_RADIANS: phase,
            gf.WaveformParameters.DURATION_SECONDS: dur,
        }

        return waveform, params


@dataclass
class SineGaussianGenerator(gf.WaveformGenerator):
    """
    Generator for sine-Gaussian wavelets, as lalsimulation's SimBurstSineGaussian.

    Commonly used in gravitational wave burst searches. With phi = 2 pi f0 t,
    tau = Q / (sqrt(2) pi f0) and the envelope exp(-t^2 / tau^2):

        h+ = amplitude * a / cgrss * exp(-t^2 / tau^2) * cos(phi - phase)
        hx = amplitude * b / sgrss * exp(-t^2 / tau^2) * sin(phi - phase)

    where cgrss and sgrss are the root-sum-squares of the unit cosine- and
    sine-Gaussians, so ``amplitude`` is the hrss.

    Parameters:
        frequency_hertz: Central frequency f0
        quality_factor: Q = sqrt(2) pi f0 tau (dimensionless)
        amplitude: hrss, sqrt(sum(h+^2 + hx^2) dt)
        phase_radians: LAL's phase
        polarization_eccentricity: LAL's eccentricity: 1 is linear, 0 circular
    """
    frequency_hertz: Union[float, gf.Distribution] = 100.0
    quality_factor: Union[float, gf.Distribution] = 10.0
    amplitude: Union[float, gf.Distribution] = 1.0
    phase_radians: Union[float, gf.Distribution] = 0.0
    polarization_eccentricity: Union[float, gf.Distribution] = 1.0

    distributed_attributes: Tuple[str] = (
        "frequency_hertz",
        "quality_factor",
        "amplitude",
        "phase_radians",
        "polarization_eccentricity",
    )

    def __post_init__(self):
        self.frequency_hertz = self.ensure_float("frequency_hertz", self.frequency_hertz)
        self.quality_factor = self.ensure_float("quality_factor", self.quality_factor)
        self.amplitude = self.ensure_float("amplitude", self.amplitude)
        self.phase_radians = self.ensure_float("phase_radians", self.phase_radians)
        self.polarization_eccentricity = self.ensure_float(
            "polarization_eccentricity", self.polarization_eccentricity
        )

        super().__post_init__()
        _check_eccentricity(self)
        self.get_max_generated_duration()

    def get_max_generated_duration(self):
        # The envelope exp(-t^2/tau^2) falls to 1e-3 at 2.63 tau either side.
        _, q_max = _check_positive(self, "quality_factor")
        f_min, _ = _check_positive(self, "frequency_hertz")
        tau = q_max / (f_min * np.sqrt(2) * np.pi)
        return float(2.0 * _GAUSSIAN_EXTENT_TAUS * tau)

    def get_signal_extent_seconds(self, parameters):
        tau = np.asarray(parameters[gf.WaveformParameters.TAU_SECONDS])
        return _GAUSSIAN_EXTENT_TAUS * tau, _GAUSSIAN_EXTENT_TAUS * tau

    def generate(
        self,
        num_waveforms: int,
        sample_rate_hertz: float,
        duration_seconds: float,
        seed: int
    ):
        if num_waveforms <= 0:
            return None, {}

        if seed is not None:
            self.reseed(seed)
        _check_below_nyquist(self, ("frequency_hertz",), sample_rate_hertz)

        # Sample parameters
        freq = _sample(self.frequency_hertz, num_waveforms)
        q = _sample(self.quality_factor, num_waveforms)
        amp = _sample(self.amplitude, num_waveforms)
        phase = _sample(self.phase_radians, num_waveforms)
        eccentricity = _sample(self.polarization_eccentricity, num_waveforms)

        tau = q / (freq * jnp.sqrt(2.0) * jnp.pi)

        num_samples = int(sample_rate_hertz * duration_seconds)
        t = time_axis(num_samples, sample_rate_hertz)

        # LAL: phi = 2 pi f0 t; envelope exp(-phi^2 / (2 Q^2)) = exp(-t^2/tau^2).
        phi = 2 * jnp.pi * freq[:, None] * t[None, :]
        envelope = jnp.exp(-0.5 * phi ** 2 / q[:, None] ** 2)

        # Root-sum-squares of the unit cosine- and sine-Gaussians (LAL's cgrss
        # and sgrss, from K. Riles, LIGO-T040055).
        base = q / (4.0 * freq * jnp.sqrt(jnp.pi))
        cgrss = jnp.sqrt(base * (1.0 + jnp.exp(-q ** 2)))
        sgrss = jnp.sqrt(base * (1.0 - jnp.exp(-q ** 2)))
        a, b = polarisation_axes(eccentricity)
        h0_plus = amp * a / cgrss
        h0_cross = amp * b / sgrss

        argument = phi - phase[:, None]
        plus = h0_plus[:, None] * envelope * jnp.cos(argument)
        cross = h0_cross[:, None] * envelope * jnp.sin(argument)
        waveform = jnp.stack([plus, cross], axis=1).astype("float32")

        # Apply injection chance
        waveform = self.apply_injection_chance(waveform, seed)

        params = {
            gf.WaveformParameters.FREQUENCY_HERTZ: freq,
            gf.WaveformParameters.QUALITY_FACTOR: q,
            gf.WaveformParameters.AMPLITUDE: amp,
            gf.WaveformParameters.PHASE_RADIANS: phase,
            gf.WaveformParameters.TAU_SECONDS: tau,
            gf.WaveformParameters.POLARIZATION_ECCENTRICITY: eccentricity,
        }

        return waveform, params


@dataclass
class ChirpletGenerator(gf.WaveformGenerator):
    """
    Generator for chirplet signals: a linear chirp under a Hann envelope.

    With tau = t measured from the envelope's centre, the instantaneous
    frequency is f(tau) = (f0 + f1)/2 + k tau with k = (f1 - f0) / duration, so it
    is f0 at the envelope's start and f1 at its end. The phase is
    Phi = 2 pi ((f0 + f1)/2 tau + k tau^2 / 2) - phase, and h+ goes as cos(Phi),
    hx as sin(Phi), with LAL's polarisation-ellipse eccentricity.

    Parameters:
        start_frequency_hertz: Frequency at the envelope's start, f0
        end_frequency_hertz: Frequency at the envelope's end, f1
        amplitude: hrss, sqrt(sum(h+^2 + hx^2) dt)
        duration_seconds: Full width of the Hann envelope
        phase_radians: Phase at the envelope's centre
        polarization_eccentricity: LAL's eccentricity: 1 is linear, 0 circular
    """
    start_frequency_hertz: Union[float, gf.Distribution] = 50.0
    end_frequency_hertz: Union[float, gf.Distribution] = 200.0
    amplitude: Union[float, gf.Distribution] = 1.0
    duration_seconds: Union[float, gf.Distribution] = 0.5
    phase_radians: Union[float, gf.Distribution] = 0.0
    polarization_eccentricity: Union[float, gf.Distribution] = 1.0

    distributed_attributes: Tuple[str] = (
        "start_frequency_hertz",
        "end_frequency_hertz",
        "amplitude",
        "duration_seconds",
        "phase_radians",
        "polarization_eccentricity",
    )

    def __post_init__(self):
        self.start_frequency_hertz = self.ensure_float("start_frequency_hertz", self.start_frequency_hertz)
        self.end_frequency_hertz = self.ensure_float("end_frequency_hertz", self.end_frequency_hertz)
        self.amplitude = self.ensure_float("amplitude", self.amplitude)
        self.duration_seconds = self.ensure_float("duration_seconds", self.duration_seconds)
        self.phase_radians = self.ensure_float("phase_radians", self.phase_radians)
        self.polarization_eccentricity = self.ensure_float(
            "polarization_eccentricity", self.polarization_eccentricity
        )

        super().__post_init__()
        _check_positive(self, "start_frequency_hertz")
        _check_positive(self, "end_frequency_hertz")
        _check_positive(self, "duration_seconds")
        _check_eccentricity(self)

    def get_max_generated_duration(self):
        return float(_check_positive(self, "duration_seconds")[1])

    def get_signal_extent_seconds(self, parameters):
        half = np.asarray(parameters[gf.WaveformParameters.DURATION_SECONDS]) / 2.0
        return half, half

    def generate(
        self,
        num_waveforms: int,
        sample_rate_hertz: float,
        duration_seconds: float,
        seed: int
    ):
        if num_waveforms <= 0:
            return None, {}

        if seed is not None:
            self.reseed(seed)
        _check_below_nyquist(
            self, ("start_frequency_hertz", "end_frequency_hertz"), sample_rate_hertz
        )

        # Sample parameters
        f0 = _sample(self.start_frequency_hertz, num_waveforms)
        f1 = _sample(self.end_frequency_hertz, num_waveforms)
        amp = _sample(self.amplitude, num_waveforms)
        dur = _sample(self.duration_seconds, num_waveforms)
        phase = _sample(self.phase_radians, num_waveforms)
        eccentricity = _sample(self.polarization_eccentricity, num_waveforms)

        # Chirp rate k = (f1 - f0) / duration
        k = (f1 - f0) / dur
        centre_frequency = 0.5 * (f0 + f1)

        num_samples = int(sample_rate_hertz * duration_seconds)
        t = time_axis(num_samples, sample_rate_hertz)[None, :]

        argument = 2 * jnp.pi * (
            centre_frequency[:, None] * t + 0.5 * k[:, None] * t ** 2
        ) - phase[:, None]
        waveform = _elliptical(
            hann_envelope(t[0], dur), argument, amp, eccentricity, sample_rate_hertz
        )

        # Apply injection chance
        waveform = self.apply_injection_chance(waveform, seed)

        params = {
            gf.WaveformParameters.START_FREQUENCY_HERTZ: f0,
            gf.WaveformParameters.END_FREQUENCY_HERTZ: f1,
            gf.WaveformParameters.CHIRP_RATE: k,
            gf.WaveformParameters.AMPLITUDE: amp,
            gf.WaveformParameters.DURATION_SECONDS: dur,
            gf.WaveformParameters.PHASE_RADIANS: phase,
            gf.WaveformParameters.POLARIZATION_ECCENTRICITY: eccentricity,
        }

        return waveform, params


@dataclass
class RingdownGenerator(gf.WaveformGenerator):
    """
    Generator for ringdown (damped sinusoid) signals.

    Approximates the post-merger ringdown of a binary black hole merger with
    one damped mode starting at t = 0, the reference time:

        h+ ~ exp(-t / tau) cos(2 pi f t - phase),  hx ~ exp(-t / tau) sin(2 pi f t - phase)

    for t >= 0, with LAL's polarisation-ellipse eccentricity (a face-on source
    is circular, e = 0; an edge-on one linear, e = 1).

    Parameters:
        frequency_hertz: Ringdown frequency
        damping_time_seconds: Exponential decay time tau
        amplitude: hrss, sqrt(sum(h+^2 + hx^2) dt)
        phase_radians: Phase at the start
        polarization_eccentricity: LAL's eccentricity: 1 is linear, 0 circular
    """
    frequency_hertz: Union[float, gf.Distribution] = 200.0
    damping_time_seconds: Union[float, gf.Distribution] = 0.01
    amplitude: Union[float, gf.Distribution] = 1.0
    phase_radians: Union[float, gf.Distribution] = 0.0
    polarization_eccentricity: Union[float, gf.Distribution] = 1.0

    distributed_attributes: Tuple[str] = (
        "frequency_hertz",
        "damping_time_seconds",
        "amplitude",
        "phase_radians",
        "polarization_eccentricity",
    )

    def __post_init__(self):
        self.frequency_hertz = self.ensure_float("frequency_hertz", self.frequency_hertz)
        self.damping_time_seconds = self.ensure_float("damping_time_seconds", self.damping_time_seconds)
        self.amplitude = self.ensure_float("amplitude", self.amplitude)
        self.phase_radians = self.ensure_float("phase_radians", self.phase_radians)
        self.polarization_eccentricity = self.ensure_float(
            "polarization_eccentricity", self.polarization_eccentricity
        )

        super().__post_init__()
        _check_positive(self, "frequency_hertz")
        _check_positive(self, "damping_time_seconds")
        _check_eccentricity(self)

    def get_max_generated_duration(self):
        # The ringdown starts at the buffer's centre and falls to 1e-3 of its
        # start after ln(1000) tau, so the buffer needs twice that.
        _, tau_max = _check_positive(self, "damping_time_seconds")
        return float(2.0 * _RINGDOWN_EXTENT_TAUS * tau_max)

    def get_signal_extent_seconds(self, parameters):
        tau = np.asarray(parameters[gf.WaveformParameters.DAMPING_TIME_SECONDS])
        return np.zeros_like(tau), _RINGDOWN_EXTENT_TAUS * tau

    def generate(
        self,
        num_waveforms: int,
        sample_rate_hertz: float,
        duration_seconds: float,
        seed: int
    ):
        if num_waveforms <= 0:
            return None, {}

        if seed is not None:
            self.reseed(seed)
        _check_below_nyquist(self, ("frequency_hertz",), sample_rate_hertz)

        # Sample parameters
        freq = _sample(self.frequency_hertz, num_waveforms)
        tau = _sample(self.damping_time_seconds, num_waveforms)
        amp = _sample(self.amplitude, num_waveforms)
        phase = _sample(self.phase_radians, num_waveforms)
        eccentricity = _sample(self.polarization_eccentricity, num_waveforms)

        num_samples = int(sample_rate_hertz * duration_seconds)
        t = time_axis(num_samples, sample_rate_hertz)[None, :]

        envelope = jnp.where(t >= 0.0, jnp.exp(-jnp.maximum(t, 0.0) / tau[:, None]), 0.0)
        argument = 2 * jnp.pi * freq[:, None] * t - phase[:, None]
        waveform = _elliptical(envelope, argument, amp, eccentricity, sample_rate_hertz)

        # Apply injection chance
        waveform = self.apply_injection_chance(waveform, seed)

        params = {
            gf.WaveformParameters.FREQUENCY_HERTZ: freq,
            gf.WaveformParameters.DAMPING_TIME_SECONDS: tau,
            gf.WaveformParameters.AMPLITUDE: amp,
            gf.WaveformParameters.PHASE_RADIANS: phase,
            gf.WaveformParameters.POLARIZATION_ECCENTRICITY: eccentricity,
        }

        return waveform, params
