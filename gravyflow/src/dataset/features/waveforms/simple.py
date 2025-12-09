"""
Simple Waveform Generators for Testing and Validation

These generators produce simple, analytically-defined waveforms for testing
matched filtering, SNR calculations, and data pipeline validation.

Generators:
- PeriodicWaveGenerator: Sine, square, sawtooth, triangle waves
- SineGaussianGenerator: Sine-Gaussian wavelets (common in GW burst searches)
- ChirpletGenerator: Frequency-modulated chirps
- RingdownGenerator: Damped sinusoids (post-merger approximation)
"""

from dataclasses import dataclass
from typing import Union, Tuple
from enum import Enum, auto

import jax
import jax.numpy as jnp
from keras import ops
import numpy as np

import gravyflow as gf


class WaveShape(Enum):
    """Shape of periodic waveform."""
    SINE = auto()
    SQUARE = auto()
    SAWTOOTH = auto()
    TRIANGLE = auto()


@dataclass
class PeriodicWaveGenerator(gf.WaveformGenerator):
    """
    Generator for periodic waveforms (sine, square, sawtooth, triangle).
    
    Useful for testing matched filter template identification and 
    basic signal processing validation.
    
    Parameters:
        wave_shape: Type of periodic wave (SINE, SQUARE, SAWTOOTH, TRIANGLE)
        frequency_hertz: Oscillation frequency
        amplitude: Peak amplitude
        phase_radians: Initial phase offset
        duration_seconds: Duration of the burst portion
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
        
        if self.scale_factor is None:
            self.scale_factor = 1.0
        
        super().__post_init__()
    
    def get_max_generated_duration(self):
        if isinstance(self.duration_seconds, gf.Distribution):
            if self.duration_seconds.type_ == gf.DistributionType.CONSTANT:
                return self.duration_seconds.value
            return self.duration_seconds.max_
        return self.duration_seconds
    
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
        
        # Sample parameters
        freq = jnp.array(self.frequency_hertz.sample(num_waveforms))
        amp = jnp.array(self.amplitude.sample(num_waveforms))
        phase = jnp.array(self.phase_radians.sample(num_waveforms))
        dur = jnp.array(self.duration_seconds.sample(num_waveforms))
        
        # Generate time array
        num_samples = int(sample_rate_hertz * duration_seconds)
        t = jnp.linspace(0, duration_seconds, num_samples)
        
        # Compute phase for each waveform: (batch, time)
        # t: (time,), freq: (batch,), phase: (batch,)
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
        
        # Apply amplitude
        waveform = amp[:, None] * waveform
        
        # Apply duration envelope (Hann window centered)
        dur_samples = ops.cast(dur * sample_rate_hertz, "int32")
        envelope = self._generate_envelope(num_samples, dur_samples)
        waveform = waveform * envelope
        
        # Add channel dimension: (batch, 2, time) for h+/hx
        # For simple waves, h+ = waveform, hx = 0
        waveform = jnp.stack([waveform, jnp.zeros_like(waveform)], axis=1)
        
        # Apply injection chance
        waveform = self.apply_injection_chance(waveform, seed)
        
        params = {
            gf.WaveformParameters.FREQUENCY_HERTZ: freq,
            gf.WaveformParameters.AMPLITUDE: amp,
            gf.WaveformParameters.PHASE_RADIANS: phase,
            gf.WaveformParameters.DURATION_SECONDS: dur,
        }
        
        return waveform * self.scale_factor, params
    
    def _generate_envelope(self, max_samples, dur_samples):
        """Generate Hann window envelopes centered in the output."""
        def create_envelope(num_samples):
            n = jnp.arange(max_samples, dtype="float32")
            start_idx = (max_samples - num_samples) // 2
            mask = (n >= start_idx) & (n < (start_idx + num_samples))
            effective_n = n - start_idx
            N_minus_1 = jnp.maximum(num_samples - 1.0, 1.0)
            val = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * effective_n / N_minus_1))
            return val * mask.astype("float32")
        
        return jax.vmap(create_envelope)(dur_samples)


@dataclass
class SineGaussianGenerator(gf.WaveformGenerator):
    """
    Generator for sine-Gaussian wavelets.
    
    Commonly used in gravitational wave burst searches.
    h(t) = A * sin(2*pi*f0*(t-t0) + phi) * exp(-(t-t0)^2 / (2*tau^2))
    
    Parameters:
        frequency_hertz: Central frequency f0
        quality_factor: Q = f0 * tau * sqrt(2) * pi (dimensionless)
        amplitude: Peak amplitude A
        phase_radians: Initial phase phi
    """
    frequency_hertz: Union[float, gf.Distribution] = 100.0
    quality_factor: Union[float, gf.Distribution] = 10.0
    amplitude: Union[float, gf.Distribution] = 1.0
    phase_radians: Union[float, gf.Distribution] = 0.0
    
    distributed_attributes: Tuple[str] = (
        "frequency_hertz",
        "quality_factor",
        "amplitude",
        "phase_radians",
    )
    
    def __post_init__(self):
        self.frequency_hertz = self.ensure_float("frequency_hertz", self.frequency_hertz)
        self.quality_factor = self.ensure_float("quality_factor", self.quality_factor)
        self.amplitude = self.ensure_float("amplitude", self.amplitude)
        self.phase_radians = self.ensure_float("phase_radians", self.phase_radians)
        
        if self.scale_factor is None:
            self.scale_factor = 1.0
        
        super().__post_init__()
    
    def get_max_generated_duration(self):
        # Duration determined by Q and frequency
        # tau = Q / (f0 * sqrt(2) * pi), signal ~6*tau wide
        if isinstance(self.quality_factor, gf.Distribution):
            if self.quality_factor.type_ == gf.DistributionType.CONSTANT:
                q_max = self.quality_factor.value
            else:
                q_max = self.quality_factor.max_
        else:
            q_max = self.quality_factor
            
        if isinstance(self.frequency_hertz, gf.Distribution):
            if self.frequency_hertz.type_ == gf.DistributionType.CONSTANT:
                f_min = self.frequency_hertz.value
            else:
                f_min = self.frequency_hertz.min_
        else:
            f_min = self.frequency_hertz
        
        tau = q_max / (f_min * np.sqrt(2) * np.pi)
        return 6 * tau
    
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
        
        # Sample parameters
        freq = jnp.array(self.frequency_hertz.sample(num_waveforms))
        q = jnp.array(self.quality_factor.sample(num_waveforms))
        amp = jnp.array(self.amplitude.sample(num_waveforms))
        phase = jnp.array(self.phase_radians.sample(num_waveforms))
        
        # Compute tau from Q: Q = f0 * tau * sqrt(2) * pi
        tau = q / (freq * jnp.sqrt(2) * jnp.pi)
        
        # Generate time array centered at t=0
        num_samples = int(sample_rate_hertz * duration_seconds)
        t = jnp.linspace(-duration_seconds/2, duration_seconds/2, num_samples)
        
        # Compute sine-Gaussian: (batch, time)
        # h(t) = A * sin(2*pi*f0*t + phi) * exp(-t^2 / (2*tau^2))
        phi = 2 * jnp.pi * freq[:, None] * t[None, :] + phase[:, None]
        gaussian = jnp.exp(-t[None, :]**2 / (2 * tau[:, None]**2))
        waveform = amp[:, None] * jnp.sin(phi) * gaussian
        
        # Add channel dimension: (batch, 2, time)
        waveform = jnp.stack([waveform, jnp.zeros_like(waveform)], axis=1)
        
        # Apply injection chance
        waveform = self.apply_injection_chance(waveform, seed)
        
        params = {
            gf.WaveformParameters.FREQUENCY_HERTZ: freq,
            gf.WaveformParameters.QUALITY_FACTOR: q,
            gf.WaveformParameters.AMPLITUDE: amp,
            gf.WaveformParameters.PHASE_RADIANS: phase,
            gf.WaveformParameters.TAU_SECONDS: tau,
        }
        
        return waveform * self.scale_factor, params


@dataclass  
class ChirpletGenerator(gf.WaveformGenerator):
    """
    Generator for chirplet signals (frequency-modulated sinusoids).
    
    h(t) = A * sin(2*pi*(f0*t + 0.5*k*t^2) + phi) * envelope(t)
    
    Parameters:
        start_frequency_hertz: Initial frequency f0
        end_frequency_hertz: Final frequency (determines chirp rate k)
        amplitude: Peak amplitude
        duration_seconds: Duration of the chirp
        phase_radians: Initial phase
    """
    start_frequency_hertz: Union[float, gf.Distribution] = 50.0
    end_frequency_hertz: Union[float, gf.Distribution] = 200.0
    amplitude: Union[float, gf.Distribution] = 1.0
    duration_seconds: Union[float, gf.Distribution] = 0.5
    phase_radians: Union[float, gf.Distribution] = 0.0
    
    distributed_attributes: Tuple[str] = (
        "start_frequency_hertz",
        "end_frequency_hertz",
        "amplitude",
        "duration_seconds",
        "phase_radians",
    )
    
    def __post_init__(self):
        self.start_frequency_hertz = self.ensure_float("start_frequency_hertz", self.start_frequency_hertz)
        self.end_frequency_hertz = self.ensure_float("end_frequency_hertz", self.end_frequency_hertz)
        self.amplitude = self.ensure_float("amplitude", self.amplitude)
        self.duration_seconds = self.ensure_float("duration_seconds", self.duration_seconds)
        self.phase_radians = self.ensure_float("phase_radians", self.phase_radians)
        
        if self.scale_factor is None:
            self.scale_factor = 1.0
        
        super().__post_init__()
    
    def get_max_generated_duration(self):
        if isinstance(self.duration_seconds, gf.Distribution):
            if self.duration_seconds.type_ == gf.DistributionType.CONSTANT:
                return self.duration_seconds.value
            return self.duration_seconds.max_
        return self.duration_seconds
    
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
        
        # Sample parameters
        f0 = jnp.array(self.start_frequency_hertz.sample(num_waveforms))
        f1 = jnp.array(self.end_frequency_hertz.sample(num_waveforms))
        amp = jnp.array(self.amplitude.sample(num_waveforms))
        dur = jnp.array(self.duration_seconds.sample(num_waveforms))
        phase = jnp.array(self.phase_radians.sample(num_waveforms))
        
        # Chirp rate k = (f1 - f0) / duration
        k = (f1 - f0) / dur
        
        num_samples = int(sample_rate_hertz * duration_seconds)
        t = jnp.linspace(0, duration_seconds, num_samples)
        
        # Compute instantaneous phase: 2*pi*(f0*t + 0.5*k*t^2)
        phi = 2 * jnp.pi * (f0[:, None] * t[None, :] + 0.5 * k[:, None] * t[None, :]**2) + phase[:, None]
        waveform = amp[:, None] * jnp.sin(phi)
        
        # Apply duration envelope (Hann window)
        dur_samples = ops.cast(dur * sample_rate_hertz, "int32")
        envelope = self._generate_envelope(num_samples, dur_samples)
        waveform = waveform * envelope
        
        # Add channel dimension
        waveform = jnp.stack([waveform, jnp.zeros_like(waveform)], axis=1)
        
        # Apply injection chance
        waveform = self.apply_injection_chance(waveform, seed)
        
        params = {
            gf.WaveformParameters.START_FREQUENCY_HERTZ: f0,
            gf.WaveformParameters.END_FREQUENCY_HERTZ: f1,
            gf.WaveformParameters.CHIRP_RATE: k,
            gf.WaveformParameters.AMPLITUDE: amp,
            gf.WaveformParameters.DURATION_SECONDS: dur,
        }
        
        return waveform * self.scale_factor, params
    
    def _generate_envelope(self, max_samples, dur_samples):
        """Generate Hann window envelopes."""
        def create_envelope(num_samples):
            n = jnp.arange(max_samples, dtype="float32")
            start_idx = (max_samples - num_samples) // 2
            mask = (n >= start_idx) & (n < (start_idx + num_samples))
            effective_n = n - start_idx
            N_minus_1 = jnp.maximum(num_samples - 1.0, 1.0)
            val = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * effective_n / N_minus_1))
            return val * mask.astype("float32")
        
        return jax.vmap(create_envelope)(dur_samples)


@dataclass
class RingdownGenerator(gf.WaveformGenerator):
    """
    Generator for ringdown (damped sinusoid) signals.
    
    Approximates the post-merger ringdown phase of binary black hole mergers.
    h(t) = A * sin(2*pi*f*t + phi) * exp(-t/tau) for t >= 0
    
    Parameters:
        frequency_hertz: Ringdown frequency
        damping_time_seconds: Exponential decay time tau
        amplitude: Initial amplitude
        phase_radians: Initial phase
    """
    frequency_hertz: Union[float, gf.Distribution] = 200.0
    damping_time_seconds: Union[float, gf.Distribution] = 0.01
    amplitude: Union[float, gf.Distribution] = 1.0
    phase_radians: Union[float, gf.Distribution] = 0.0
    
    distributed_attributes: Tuple[str] = (
        "frequency_hertz",
        "damping_time_seconds",
        "amplitude",
        "phase_radians",
    )
    
    def __post_init__(self):
        self.frequency_hertz = self.ensure_float("frequency_hertz", self.frequency_hertz)
        self.damping_time_seconds = self.ensure_float("damping_time_seconds", self.damping_time_seconds)
        self.amplitude = self.ensure_float("amplitude", self.amplitude)
        self.phase_radians = self.ensure_float("phase_radians", self.phase_radians)
        
        if self.scale_factor is None:
            self.scale_factor = 1.0
        
        super().__post_init__()
    
    def get_max_generated_duration(self):
        # Ringdown effectively ends after ~5*tau
        if isinstance(self.damping_time_seconds, gf.Distribution):
            if self.damping_time_seconds.type_ == gf.DistributionType.CONSTANT:
                tau_max = self.damping_time_seconds.value
            else:
                tau_max = self.damping_time_seconds.max_
        else:
            tau_max = self.damping_time_seconds
        return 5 * tau_max
    
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
        
        # Sample parameters
        freq = jnp.array(self.frequency_hertz.sample(num_waveforms))
        tau = jnp.array(self.damping_time_seconds.sample(num_waveforms))
        amp = jnp.array(self.amplitude.sample(num_waveforms))
        phase = jnp.array(self.phase_radians.sample(num_waveforms))
        
        # Generate time array starting from 0
        num_samples = int(sample_rate_hertz * duration_seconds)
        t = jnp.linspace(0, duration_seconds, num_samples)
        
        # Compute ringdown: A * sin(2*pi*f*t + phi) * exp(-t/tau)
        phi = 2 * jnp.pi * freq[:, None] * t[None, :] + phase[:, None]
        decay = jnp.exp(-t[None, :] / tau[:, None])
        waveform = amp[:, None] * jnp.sin(phi) * decay
        
        # Add channel dimension
        waveform = jnp.stack([waveform, jnp.zeros_like(waveform)], axis=1)
        
        # Apply injection chance
        waveform = self.apply_injection_chance(waveform, seed)
        
        params = {
            gf.WaveformParameters.FREQUENCY_HERTZ: freq,
            gf.WaveformParameters.DAMPING_TIME_SECONDS: tau,
            gf.WaveformParameters.AMPLITUDE: amp,
            gf.WaveformParameters.PHASE_RADIANS: phase,
        }
        
        return waveform * self.scale_factor, params
