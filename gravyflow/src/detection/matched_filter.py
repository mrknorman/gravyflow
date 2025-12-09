"""
GPU Matched Filter

Main matched filtering class for gravitational wave detection using
on-the-fly template generation via ripple.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np

from gravyflow.src.detection.template_grid import TemplateGrid
from gravyflow.src.detection.snr import matched_filter_fft, template_sigma, find_triggers


@dataclass
class MatchedFilter:
    """
    GPU-accelerated matched filtering with on-the-fly ripple templates.
    
    This class provides a traditional matched filtering baseline for
    comparison with deep learning detectors. Templates are generated
    on-the-fly using ripple's JAX-based waveform generation, eliminating
    the need for stored template banks.
    
    Args:
        mass_1_range: (min, max) for primary mass in solar masses
        mass_2_range: (min, max) for secondary mass in solar masses
        num_templates_per_dim: Grid points per mass dimension
        sample_rate_hertz: Sample rate in Hz
        duration_seconds: Template duration in seconds
        f_low: Low frequency cutoff in Hz
    
    Usage:
        >>> mf = MatchedFilter(
        ...     mass_1_range=(5.0, 75.0),
        ...     mass_2_range=(5.0, 75.0),
        ...     num_templates_per_dim=16  # 136 templates
        ... )
        >>> snr = mf.filter(data, psd)
        >>> triggers = mf.detect(data, psd, threshold=8.0)
    """
    
    mass_1_range: Tuple[float, float] = (5.0, 75.0)
    mass_2_range: Tuple[float, float] = (5.0, 75.0)
    num_templates_per_dim: int = 16
    sample_rate_hertz: float = 8192.0
    duration_seconds: float = 2.0
    f_low: float = 20.0
    
    # Internal state
    _grid: TemplateGrid = field(init=False, repr=False)
    _templates: Optional[jnp.ndarray] = field(default=None, init=False, repr=False)
    _template_scale: float = field(default=1.0, init=False, repr=False)  # Amplitude scale factor
    _template_params: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = field(
        default=None, init=False, repr=False
    )
    
    def __post_init__(self):
        self._grid = TemplateGrid(
            mass_1_range=self.mass_1_range,
            mass_2_range=self.mass_2_range,
            num_mass_1_points=self.num_templates_per_dim,
        )
        self._template_params = self._grid.get_parameters()
    
    @property
    def num_templates(self) -> int:
        """Number of templates in the bank."""
        return self._grid.num_templates
    
    @property
    def template_masses(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return (mass_1, mass_2) arrays for all templates."""
        return self._template_params
    
    def generate_templates(self, force_regenerate: bool = False) -> jnp.ndarray:
        """
        Generate templates on-the-fly using ripple.
        
        Templates are cached after first generation unless force_regenerate=True.
        
        Args:
            force_regenerate: If True, regenerate even if cached.
        
        Returns:
            Template array of shape (num_templates, num_samples)
        """
        if self._templates is not None and not force_regenerate:
            return self._templates
        
        # Import ripple here to avoid circular imports
        try:
            from ripple.waveforms import IMRPhenomD
            from ripple import ms_to_Mc_eta
        except ImportError:
            raise ImportError(
                "ripple is required for template generation. "
                "Install with: pip install ripple-gw"
            )
        
        mass_1, mass_2 = self._template_params
        num_samples = int(self.duration_seconds * self.sample_rate_hertz)
        num_freqs = num_samples // 2 + 1
        
        # Frequency array for waveform generation
        f_u = self.sample_rate_hertz / 2.0
        freqs = jnp.linspace(0, f_u, num_freqs, dtype=jnp.float64)
        f_ref = self.f_low
        
        # Fixed parameters for templates (normalized)
        distance_mpc = 1.0  # Normalized - SNR scaling handles distance
        tc = self.duration_seconds * 0.9  # Merger at 90% of duration
        phi = 0.0  # Reference phase
        inc = 0.0  # Face-on (optimal orientation)
        
        # Generate templates using vmap
        @jax.vmap
        def generate_one(m1, m2):
            # Convert component masses to chirp mass and symmetric mass ratio
            Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
            
            # ripple IMRPhenomD expects theta = [Mc, eta, chi1, chi2, dist, tc, phi, inc]
            theta = jnp.array([Mc, eta, 0.0, 0.0, distance_mpc, tc, phi, inc])
            
            # Get frequency-domain waveform
            hp, hc = IMRPhenomD.gen_IMRPhenomD_hphc(freqs, theta, f_ref)
            
            # Apply low-frequency mask to avoid NaN (below where waveform exists)
            # Calculate minimum frequency for this mass pair
            from gravyflow.src.dataset.features.waveforms.cbc import calc_minimum_frequency
            f_min = calc_minimum_frequency(m1, m2, tc)
            
            # Apply taper
            taper_width = 8.0  # Hz
            f_taper_end = f_min + taper_width
            
            mask_zero = freqs < f_min
            mask_one = freqs >= f_taper_end
            mask_taper = (freqs >= f_min) & (freqs < f_taper_end)
            
            taper_factor = jnp.sin((jnp.pi / 2.0) * (freqs - f_min) / (f_taper_end - f_min)) ** 2
            final_mask = jnp.where(mask_zero, 0.0, jnp.where(mask_one, 1.0, taper_factor))
            
            hp = jnp.where(mask_zero, 0.0, hp * final_mask)
            
            # Convert to time domain
            h_time = jnp.fft.irfft(hp, n=num_samples)
            h_time = h_time * self.sample_rate_hertz  # Scale factor
            
            return h_time
        
        print(f"Generating {len(mass_1)} templates on GPU...")
        raw_templates = generate_one(mass_1, mass_2)
        
        # Scale templates by gf.Defaults.scale_factor for consistency with dataset
        # This is the same scaling applied to noise and injections
        import gravyflow as gf
        self._template_scale = 1.0 / gf.Defaults.scale_factor
        self._templates = raw_templates * gf.Defaults.scale_factor
        
        print(f"Templates generated: shape {self._templates.shape}, scaled by {gf.Defaults.scale_factor:.2e}")
        
        return self._templates
    
    def filter(
        self,
        data: jnp.ndarray,
        psd: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Compute SNR timeseries for all templates.
        
        Args:
            data: Whitened strain data, shape (samples,) or (batch, samples).
                  Expected to be O(1) amplitude (e.g., from GravyflowDataset).
            psd: One-sided PSD for template weighting. If None, uniform.
        
        Returns:
            SNR timeseries of shape:
            - (num_templates, samples) if data is 1D
            - (batch, num_templates, samples) if data is 2D
        """
        templates = self.generate_templates()
        
        # Templates are normalized to O(1) amplitude for numerical stability.
        # Input data should also be O(1) (e.g., whitened data).
        # matched_filter_fft already computes SNR = inner_product / sigma
        snr = matched_filter_fft(
            data, 
            templates, 
            psd=psd,
            sample_rate_hertz=self.sample_rate_hertz,
        )
            
        return snr
    
    def detect(
        self,
        data: jnp.ndarray,
        psd: Optional[jnp.ndarray] = None,
        threshold: float = 8.0,
        cluster_window_seconds: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Find triggers above SNR threshold.
        
        Args:
            data: Strain data, shape (samples,) or (batch, samples)
            psd: One-sided PSD
            threshold: Minimum SNR for trigger
            cluster_window_seconds: Time window (s) to cluster nearby peaks
        
        Returns:
            List of trigger dictionaries with keys:
            - 'snr': Peak SNR value
            - 'time_index': Sample index of peak
            - 'time_seconds': Time of peak in seconds
            - 'template_index': Which template triggered
            - 'mass_1': Primary mass of best template
            - 'mass_2': Secondary mass of best template
        """
        snr = self.filter(data, psd)
        
        cluster_window = int(cluster_window_seconds * self.sample_rate_hertz)
        triggers = find_triggers(snr, threshold, cluster_window)
        
        # Add mass parameters and time
        mass_1, mass_2 = self._template_params
        for trig in triggers:
            idx = trig['template_index']
            trig['mass_1'] = float(mass_1[idx])
            trig['mass_2'] = float(mass_2[idx])
            trig['time_seconds'] = trig['time_index'] / self.sample_rate_hertz
        
        return triggers
    
    def max_snr(
        self,
        data: jnp.ndarray,
        psd: Optional[jnp.ndarray] = None,
    ) -> Tuple[float, int, int]:
        """
        Get maximum SNR across all templates and times.
        
        Args:
            data: Strain data
            psd: One-sided PSD
        
        Returns:
            Tuple of (max_snr, best_template_index, best_time_index)
        """
        snr = self.filter(data, psd)
        
        if snr.ndim == 2:
            # (templates, samples)
            flat_idx = jnp.argmax(snr)
            template_idx = int(flat_idx // snr.shape[1])
            time_idx = int(flat_idx % snr.shape[1])
            max_val = float(snr[template_idx, time_idx])
        else:
            # (batch, templates, samples) - take max over batch
            max_val = float(jnp.max(snr))
            flat_idx = jnp.argmax(snr.reshape(-1))
            template_idx = int((flat_idx % (snr.shape[1] * snr.shape[2])) // snr.shape[2])
            time_idx = int(flat_idx % snr.shape[2])
        
        return max_val, template_idx, time_idx


# =============================================================================
# Keras Layer Wrapper for Pipeline Compatibility
# =============================================================================

import keras


@keras.saving.register_keras_serializable(package="gravyflow")
class MatchedFilterLayer(keras.Layer):
    """
    Keras-compatible layer wrapping MatchedFilter for pipeline integration.
    
    Enables matched filtering to be used with standard Keras `.predict()` interface,
    allowing direct comparison with neural network models on identical datasets.
    
    The layer converts matched filter SNR to binary detection probabilities:
    - Output shape: (batch, 2) for [P(noise), P(signal)]
    - P(signal) = sigmoid((max_snr - threshold) / temperature)
    
    Args:
        mass_1_range: (min, max) for primary mass in solar masses
        mass_2_range: (min, max) for secondary mass in solar masses
        num_templates_per_dim: Grid points per mass dimension
        sample_rate_hertz: Sample rate in Hz
        duration_seconds: Template duration in seconds
        f_low: Low frequency cutoff in Hz
        snr_threshold: SNR threshold for detection
        temperature: Sigmoid temperature (lower = sharper)
    
    Usage:
        >>> layer = MatchedFilterLayer(num_templates_per_dim=16)
        >>> model = keras.Sequential([layer])
        >>> probs = model.predict(data)  # (batch, 2)
    """
    
    def __init__(
        self,
        mass_1_range: Tuple[float, float] = (5.0, 75.0),
        mass_2_range: Tuple[float, float] = (5.0, 75.0),
        num_templates_per_dim: int = 16,
        sample_rate_hertz: float = 8192.0,
        duration_seconds: float = 2.0,
        f_low: float = 20.0,
        snr_threshold: float = 8.0,
        temperature: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.mass_1_range = tuple(mass_1_range)
        self.mass_2_range = tuple(mass_2_range)
        self.num_templates_per_dim = num_templates_per_dim
        self.sample_rate_hertz = sample_rate_hertz
        self.duration_seconds = duration_seconds
        self.f_low = f_low
        self.snr_threshold = snr_threshold
        self.temperature = temperature
        
        # Create underlying matched filter
        self._matched_filter = MatchedFilter(
            mass_1_range=self.mass_1_range,
            mass_2_range=self.mass_2_range,
            num_templates_per_dim=self.num_templates_per_dim,
            sample_rate_hertz=self.sample_rate_hertz,
            duration_seconds=self.duration_seconds,
            f_low=self.f_low,
        )
    
    def build(self, input_shape):
        """Pre-generate templates on first call."""
        self._matched_filter.generate_templates()
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Run matched filter and convert SNR to probabilities.
        
        Args:
            inputs: Strain data of shape (batch, samples) or (batch, channels, samples)
            training: Unused (no training for matched filter)
        
        Returns:
            Binary probabilities of shape (batch, 2) for [P(noise), P(signal)]
        """
        # Handle multi-channel input by taking mean across channels
        if len(inputs.shape) == 3:
            # (batch, channels, samples) -> (batch, samples)
            data = jnp.mean(inputs, axis=1)
        else:
            data = inputs
        
        # Get SNR for each batch element
        snr = self._matched_filter.filter(data)  # (batch, templates, samples) or (templates, samples)
        
        # Get maximum SNR over templates and time for each batch element
        if snr.ndim == 3:
            # (batch, templates, samples)
            max_snr = jnp.max(snr, axis=(1, 2))  # (batch,)
        elif snr.ndim == 2:
            # (templates, samples) - single sample
            max_snr = jnp.array([jnp.max(snr)])  # (1,)
        else:
            max_snr = jnp.max(snr)
            max_snr = jnp.atleast_1d(max_snr)
        
        # Convert SNR to detection probability via sigmoid
        prob_signal = jax.nn.sigmoid(
            (max_snr - self.snr_threshold) / self.temperature
        )
        prob_noise = 1.0 - prob_signal
        
        # Return in same format as classification models: (batch, 2)
        return jnp.stack([prob_noise, prob_signal], axis=-1)
    
    def get_config(self):
        """Return layer config for serialization."""
        config = super().get_config()
        config.update({
            "mass_1_range": self.mass_1_range,
            "mass_2_range": self.mass_2_range,
            "num_templates_per_dim": self.num_templates_per_dim,
            "sample_rate_hertz": self.sample_rate_hertz,
            "duration_seconds": self.duration_seconds,
            "f_low": self.f_low,
            "snr_threshold": self.snr_threshold,
            "temperature": self.temperature,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Reconstruct layer from config."""
        return cls(**config)
    
    def compute_output_shape(self, input_shape):
        """Output shape is (batch, 2) for binary classification."""
        if isinstance(input_shape, tuple):
            batch_size = input_shape[0]
        else:
            batch_size = None
        return (batch_size, 2)
    
    @property
    def matched_filter(self) -> MatchedFilter:
        """Access underlying MatchedFilter for advanced operations."""
        return self._matched_filter
    
    @property
    def num_templates(self) -> int:
        """Number of templates in the bank."""
        return self._matched_filter.num_templates
