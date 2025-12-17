from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto
from typing import Union, Iterator, List
from copy import deepcopy

import numpy as np
import keras
from keras import ops
import jax.numpy as jnp
import jax
from numpy.random import default_rng  

import gravyflow as gf
from gravyflow.src.dataset.features.event import get_events_with_params, EventType
from gravyflow.src.dataset.noise.acquisition import ifo_canonical_key

class NoiseType(Enum):
    WHITE = auto()
    COLORED = auto()
    PSEUDO_REAL = auto()
    REAL = auto()

def ensure_even(number):
    if number % 2 != 0:
        number -= 1
    return number
    
def _generate_white_noise(
    num_examples_per_batch: int,
    num_ifos : int,
    num_samples: int,
    seed : int
):
    """
    Optimized function to generate white Gaussian noise.
    """
    # Use jax.random directly for efficiency and correctness with JAX backend
    key = jax.random.PRNGKey(seed)
    
    return jax.random.normal(
        key,
        shape=(num_examples_per_batch, num_ifos, num_samples),
        dtype=jnp.float32
    )

def white_noise_generator(
    num_examples_per_batch: int,
    ifos : List[gf.IFO],
    onsource_duration_seconds: float,
    crop_duration_seconds : float,
    offsource_duration_seconds: float,
    sample_rate_hertz: float,
    seed : int
) -> Iterator:
    """
    Generator function that yields white Gaussian noise.
    """

    total_onsource_duration_seconds : float = onsource_duration_seconds + (crop_duration_seconds * 2.0)
    
    num_onsource_samples : int = ensure_even(int(total_onsource_duration_seconds * sample_rate_hertz))
    num_offsource_samples : int = ensure_even(int(offsource_duration_seconds * sample_rate_hertz))

    rng = default_rng(seed)

    while True:
        # Generate new seeds for each batch
        s1 = rng.integers(1000000000)
        s2 = rng.integers(1000000000)
        
        yield _generate_white_noise(
            num_examples_per_batch, 
            len(ifos),
            num_onsource_samples,
            seed=s1
        ), _generate_white_noise(
            num_examples_per_batch, 
            len(ifos),
            num_offsource_samples,
            seed=s2
        ), ops.full((num_examples_per_batch,), -1.0), None
        
def _generate_colored_noise(
    num_examples_per_batch: int,
    num_ifos : int,
    num_samples: int,
    interpolated_asd: jnp.ndarray, # JAX array
    seed : int
):
    """
    Function to generate colored Gaussian noise.
    """
    key = jax.random.PRNGKey(seed)
    
    white_noise = jax.random.normal(
        key,
        shape=(num_examples_per_batch, num_ifos, num_samples),
        dtype=jnp.float32
    )

    # FFT
    white_noise_fft = jnp.fft.rfft(white_noise)
    
    # Apply ASD
    # interpolated_asd shape: (1, IFOs, Freqs) or similar?
    # white_noise_fft shape: (Batch, IFOs, Freqs)
    
    colored_noise_fft = interpolated_asd * white_noise_fft
    
    # IFFT
    colored_noise = jnp.fft.irfft(colored_noise_fft, n=num_samples)
    
    return colored_noise

def interpolate_onsource_offsource_psd(
    num_samples_list: List[int],
    sample_rate_hertz: float,
    psd_freq: jnp.ndarray,
    psd_val: jnp.ndarray
) -> List[jnp.ndarray]:
    """
    Interpolates onsource and offsource PSDs to a new frequency axis.
    """

    interpolated_psds = []
    
    for num in num_samples_list:
        # New frequency grid
        new_freqs = jnp.linspace(
            0.0, 
            min(psd_freq[-1], sample_rate_hertz / 2),  # Nyquist frequency
            num // 2 + 1,
        )
        
        # Interpolate
        # jnp.interp(x, xp, fp)
        interp_val = jnp.interp(new_freqs, psd_freq, psd_val)
        
        # Cast to complex64 for FFT multiplication? 
        # Usually ASD is real, but we multiply with complex FFT.
        # Original code cast to complex64.
        interp_val = ops.cast(interp_val, "complex64")
        
        interpolated_psds.append(interp_val)
    
    return interpolated_psds[0], interpolated_psds[1]

def colored_noise_generator(
    num_examples_per_batch: int,
    onsource_duration_seconds: float,
    crop_duration_seconds : float,
    offsource_duration_seconds: float,
    ifos : List[gf.IFO],
    sample_rate_hertz: float,
    scale_factor : float,
    seed : int
) -> Iterator:
    """
    Generator function that yields colored Gaussian noise.
    """
    total_onsource_duration_seconds : float = onsource_duration_seconds + (crop_duration_seconds * 2.0)
    
    durations_seconds = [
        total_onsource_duration_seconds, 
        offsource_duration_seconds
    ]

    def load_psd(ifo, scale_factor):
        frequencies, asd = np.loadtxt(
            ifo.value.optimal_psd_path, delimiter=","
        ).T
        frequencies = jnp.array(frequencies, dtype=jnp.float32)
        asd = jnp.array(asd, dtype=jnp.float32)
        return frequencies, asd * scale_factor

    interpolated_onsource_asds = []
    interpolated_offsource_asds = []
    for ifo in ifos:
        frequencies, asd = load_psd(ifo, scale_factor)
        
        num_samples_list = [
            ensure_even(int(duration * sample_rate_hertz)) for duration in durations_seconds
        ]

        interpolated_asd_onsource, interpolated_asd_offsource = interpolate_onsource_offsource_psd(
                num_samples_list,
                sample_rate_hertz,
                frequencies,
                asd
            )

        interpolated_onsource_asds.append(
            interpolated_asd_onsource
        )
        interpolated_offsource_asds.append(
            interpolated_asd_offsource
        )

    interpolated_onsource_asds = ops.stack(interpolated_onsource_asds, axis=0) # (IFOs, Freqs)
    interpolated_offsource_asds = ops.stack(interpolated_offsource_asds, axis=0)
    
    # Expand dims for batch broadcasting
    interpolated_onsource_asds = ops.expand_dims(interpolated_onsource_asds, 0) # (1, IFOs, Freqs)
    interpolated_offsource_asds = ops.expand_dims(interpolated_offsource_asds, 0)

    rng = default_rng(seed)
    
    while True:
        s1 = rng.integers(1000000000)
        s2 = rng.integers(1000000000)
        
        yield _generate_colored_noise(
                num_examples_per_batch, 
                len(ifos),
                num_samples_list[0], 
                interpolated_onsource_asds,
                seed=s1
            ), _generate_colored_noise(
                num_examples_per_batch, 
                len(ifos),
                num_samples_list[1], 
                interpolated_offsource_asds,
                seed=s2
            ), ops.full((num_examples_per_batch,), -1.0), None
    

@dataclass
class Obtainer:
    """Base class for data obtainers (noise and transient)."""
    data_directory_path: Path = Path("./generator_data")
    ifo_data_obtainer: Union[None, gf.IFODataObtainer] = None
    ifos: List[gf.IFO] = None
    groups: Union[dict, None] = None
    
    def __post_init__(self):
        if self.ifos is None:
            self.ifos = [gf.IFO.L1]
        elif not isinstance(self.ifos, list) and not isinstance(self.ifos, tuple):
            self.ifos = [self.ifos]
        
        if not self.groups:
            self.groups = self._default_groups()
        
        self.rng = None
    
    def _default_groups(self) -> dict:
        """Override in subclasses for different default group splits."""
        return {"train": 0.89, "validate": 0.1, "test": 0.01}


@dataclass
class NoiseObtainer(Obtainer):
    """Obtainer for noise data (white, colored, pseudo-real, or real)."""
    noise_type: NoiseType = NoiseType.REAL
    
    def __post_init__(self):
        super().__post_init__()
    
    def __call__(
            self,
            sample_rate_hertz : float  = None,
            onsource_duration_seconds : float = None,
            crop_duration_seconds : float = None,
            offsource_duration_seconds : float = None,
            num_examples_per_batch : float = None,
            scale_factor : float = 1.0,
            group : str = "train",
            seed : int = None,
            sampling_mode : "gf.SamplingMode" = None
        ) -> Iterator:

        if sample_rate_hertz is None:
            sample_rate_hertz = gf.Defaults.sample_rate_hertz
        if onsource_duration_seconds is None:
            onsource_duration_seconds = gf.Defaults.onsource_duration_seconds
        if offsource_duration_seconds is None:
            offsource_duration_seconds = gf.Defaults.offsource_duration_seconds
        if crop_duration_seconds is None:
            crop_duration_seconds = gf.Defaults.crop_duration_seconds
        if scale_factor is None:
            scale_factor = gf.Defaults.scale_factor
        if num_examples_per_batch is None:
            num_examples_per_batch = gf.Defaults.num_examples_per_batch
        if seed is None:
            seed = gf.Defaults.seed
        if self.rng is None:
            self.rng = default_rng(seed)
        
        # Default to RANDOM mode if not specified
        if sampling_mode is None:
            sampling_mode = gf.SamplingMode.RANDOM

        self.generator = None

        seed_ = self.rng.integers(1000000000)
                
        match self.noise_type:
            case NoiseType.WHITE:
                self.generator = white_noise_generator(
                    num_examples_per_batch,
                    self.ifos,
                    onsource_duration_seconds,
                    crop_duration_seconds,
                    offsource_duration_seconds,
                    sample_rate_hertz,
                    seed=seed_
                )
                
            case NoiseType.COLORED:
                self.generator = colored_noise_generator(
                    num_examples_per_batch,
                    onsource_duration_seconds,
                    crop_duration_seconds,
                    offsource_duration_seconds,
                    self.ifos,
                    sample_rate_hertz,
                    scale_factor,
                    seed=seed_
                )
            
            case NoiseType.PSEUDO_REAL:
                self.generator = self.pseudo_real_noise_generator(
                    num_examples_per_batch,
                    onsource_duration_seconds,
                    crop_duration_seconds,
                    offsource_duration_seconds,
                    sample_rate_hertz,
                    group,
                    scale_factor,
                    seed=seed_
                )
            
            case NoiseType.REAL:
                canonical_ifos = ifo_canonical_key(self.ifos)

                if not self.ifo_data_obtainer:
                    raise ValueError("""
                        No IFO obtainer object present. In order to acquire real 
                        noise please parse a IFOObtainer object to NoiseObtainer
                        either during initlisation or through setting
                        NoiseObtainer.ifo_data_obtainer
                    """)
                
                # Deepcopy to ensure each generator has independent state
                ifo_data_obtainer_copy = deepcopy(self.ifo_data_obtainer)
                # Reset rng to ensure fresh random state (deepcopy carries over rng state)
                ifo_data_obtainer_copy.rng = None
                
                if ifo_data_obtainer_copy.valid_segments is None or canonical_ifos != ifo_data_obtainer_copy.ifos:
                        ifo_data_obtainer_copy.get_valid_segments(
                            self.ifos,
                            seed,
                            self.groups,
                            group,
                        )
                    
                        ifo_data_obtainer_copy.generate_file_path(
                            sample_rate_hertz,
                            group,
                            self.data_directory_path
                        )
                
                self.generator = ifo_data_obtainer_copy.get_onsource_offsource_chunks(
                        sample_rate_hertz,
                        onsource_duration_seconds,
                        crop_duration_seconds,
                        offsource_duration_seconds,
                        num_examples_per_batch,
                        self.ifos,
                        scale_factor,
                        seed=seed_,
                        sampling_mode=sampling_mode
                    )
                
            case _:
                raise ValueError(
                    f"NoiseType {self.noise_type} not recognised, please choose"
                    "from NoiseType.WHITE, NoiseType.COLORED, "
                    "NoiseType.PSEUDO_REAL, or NoiseType.REAL. "
                )
                
        if self.generator is None:  # pragma: no cover
            raise ValueError(
                "Noise generator failed to initilise..."
            )
                
        return self.generator
    
    def pseudo_real_noise_generator(
        self,
        num_examples_per_batch: int,
        onsource_duration_seconds: float,
        crop_duration_seconds : float,
        offsource_duration_seconds: float,
        sample_rate_hertz: float,
        group : str,
        scale_factor : float,
        seed : int
    ):

        rng = default_rng(seed)
        
        total_onsource_duration_seconds : float = onsource_duration_seconds + (crop_duration_seconds * 2.0)  
        
        durations_seconds = [
            total_onsource_duration_seconds, 
            offsource_duration_seconds
        ]
        
        num_samples_list = [
            int(duration * sample_rate_hertz) for duration in durations_seconds
        ]

        canonical_ifos = ifo_canonical_key(self.ifos)

        if not self.ifo_data_obtainer:
            raise ValueError("""
                No IFO obtainer object present. In order to acquire real 
                noise please parse a IFOObtainer object to NoiseObtainer
                either during initlisation or through setting
                NoiseObtainer.ifo_data_obtainer
            """)
        elif self.ifo_data_obtainer.valid_segments is None or canonical_ifos != self.ifo_data_obtainer.ifos:
            
            valid_segments = self.ifo_data_obtainer.get_valid_segments(
                self.ifos,
                seed,
                self.groups,
                group_name=group
            ) 

            self.ifo_data_obtainer.generate_file_path(
                sample_rate_hertz,
                group,
                self.data_directory_path
            )

        for segment in self.ifo_data_obtainer.acquire(
            sample_rate_hertz, 
            valid_segments, 
            self.ifos,
            scale_factor
        ):

            interpolated_onsource_psds = []
            interpolated_offsource_psds = []
            for data in segment.data:

                frequencies, psd = gf.psd(
                    data, 
                    nperseg=1024, 
                    sample_rate_hertz=sample_rate_hertz
                )
                
                # Squeeze to 1D - psd returns (batch, freqs), we need (freqs,)
                frequencies = ops.squeeze(frequencies)
                psd = ops.squeeze(psd)
                
                interpolated_psd_onsource, interpolated_psd_offsource = \
                    interpolate_onsource_offsource_psd(
                        num_samples_list,
                        sample_rate_hertz,
                        frequencies,
                        psd
                    )
                
                interpolated_onsource_psds.append(
                    interpolated_psd_onsource
                )
                interpolated_offsource_psds.append(
                    interpolated_psd_offsource
                )

            interpolated_onsource_psds = ops.stack(interpolated_onsource_psds, axis=0)
            interpolated_offsource_psds = ops.stack(interpolated_offsource_psds, axis=0)
            
            interpolated_onsource_psds = ops.expand_dims(interpolated_onsource_psds, 0)
            interpolated_offsource_psds = ops.expand_dims(interpolated_offsource_psds, 0)
            
            num_batches_in_segment : int = \
                int(
                      self.ifo_data_obtainer.max_segment_duration_seconds
                    * self.ifo_data_obtainer.saturation
                    / (num_examples_per_batch * onsource_duration_seconds)
                )
                        
            for _ in range(num_batches_in_segment):
                s1 = rng.integers(1000000000)
                s2 = rng.integers(1000000000)
                
                yield _generate_colored_noise(
                        num_examples_per_batch, 
                        len(self.ifos),
                        num_samples_list[0], 
                        ops.sqrt(interpolated_onsource_psds),
                        seed=s1
                    ), _generate_colored_noise(
                        num_examples_per_batch, 
                        len(self.ifos),
                        num_samples_list[1], 
                        ops.sqrt(interpolated_offsource_psds),
                        seed=s2
                    ), ops.full((num_examples_per_batch,), -1.0), None
                

@dataclass
class TransientObtainer(Obtainer):
    """
    Obtainer for transient events (GW mergers, glitches).
    
    Unlike NoiseObtainer, this requires an IFODataObtainer and operates
    in TRANSIENT mode (acquiring data around specific event times).
    
    Args:
        ifo_data_obtainer: Required. IFODataObtainer configured for transient acquisition.
        event_names: Optional event name(s) to fetch (e.g. "GW150914" or ["GW150914", "GW170817"]).
                    If set, supersedes default event fetching.
        ifos: List of IFOs to acquire data from.
        groups: Group splits (defaults to {"all": 1.0} for transients).
        data_labels: Must NOT include DataLabel.NOISE.
    """
    event_names: Union[None, str, List[str]] = None
    data_labels: List = None  # Will validate in __post_init__
    
    def __post_init__(self):
        # Validate ifo_data_obtainer is provided
        if self.ifo_data_obtainer is None:
            raise ValueError(
                "TransientObtainer requires an ifo_data_obtainer. "
                "Please provide a gf.IFODataObtainer configured for transient events."
            )
        
        # Validate data_labels don't include NOISE
        if self.data_labels is not None:
            if gf.DataLabel.NOISE in self.data_labels:
                raise ValueError(
                    "TransientObtainer cannot use DataLabel.NOISE. "
                    "Use NoiseObtainer for noise acquisition."
                )
        else:
            # Default to EVENTS
            self.data_labels = [gf.DataLabel.EVENTS]
        
        # Normalize event_names to list
        if isinstance(self.event_names, str):
            self.event_names = [self.event_names]
        
        super().__post_init__()
    
    def _default_groups(self) -> dict:
        """Transients include 'all' (100%) plus standard splits for classification."""
        return {"all": 1.0, "train": 0.89, "validate": 0.1, "test": 0.01}
    
    def __call__(
            self,
            sample_rate_hertz: float = None,
            onsource_duration_seconds: float = None,
            crop_duration_seconds: float = None,
            offsource_duration_seconds: float = None,
            num_examples_per_batch: int = None,
            scale_factor: float = 1.0,
            group: str = "all",
            seed: int = None,
            crop: bool = False,
            whiten: bool = False,
            precache_cap: int = None,  # If 0, skip precache (lazy download). If >0, limit cache size.
        ) -> Iterator:
        """
        Create a generator that yields transient event data.
        
        Args:
            sample_rate_hertz: Sample rate for data.
            onsource_duration_seconds: Duration of onsource window.
            crop_duration_seconds: Cropping buffer.
            offsource_duration_seconds: Duration of offsource window.
            num_examples_per_batch: Batch size.
            scale_factor: Amplitude scaling.
            group: Group name (default "all" for transients).
            seed: Random seed.
            crop: If True, crop onsource to remove padding (default False).
            whiten: If True, whiten onsource using offsource PSD (default False).
            precache_cap: If 0, disable precaching (lazy download). If >0, limit items cached.
            
        Yields:
            Tuples of (onsource, offsource, gps_times) tensors.
        """
        from copy import deepcopy
        from numpy.random import default_rng
        
        # Apply defaults
        if sample_rate_hertz is None:
            sample_rate_hertz = gf.Defaults.sample_rate_hertz
        if onsource_duration_seconds is None:
            onsource_duration_seconds = gf.Defaults.onsource_duration_seconds
        if offsource_duration_seconds is None:
            offsource_duration_seconds = gf.Defaults.offsource_duration_seconds
        if crop_duration_seconds is None:
            crop_duration_seconds = gf.Defaults.crop_duration_seconds
        if num_examples_per_batch is None:
            num_examples_per_batch = gf.Defaults.num_examples_per_batch
        if seed is None:
            seed = gf.Defaults.seed
        if self.rng is None:
            self.rng = default_rng(seed)
        
        # Deep copy to ensure independent state
        ifo_obtainer = deepcopy(self.ifo_data_obtainer)
        ifo_obtainer.rng = None
        
        # If event_names specified, filter to those events
        if self.event_names:
            self._filter_to_named_events(ifo_obtainer)
        
        # Get valid segments
        canonical_ifos = ifo_canonical_key(self.ifos)
        if ifo_obtainer.valid_segments is None or canonical_ifos != ifo_obtainer.ifos:
            ifo_obtainer.get_valid_segments(
                self.ifos,
                seed,
                self.groups,
                group,
            )
            ifo_obtainer.generate_file_path(
                sample_rate_hertz,
                group,
                self.data_directory_path
            )
            
            # For TRANSIENT mode with glitches: use precaching to GlitchCache
            if ifo_obtainer.cache_segments and ifo_obtainer.acquisition_mode == gf.AcquisitionMode.TRANSIENT:
                 ifo_obtainer._cache_valid_segments(ifo_obtainer.valid_segments, group)
        
        seed_ = self.rng.integers(1000000000)
        
        # Check if we should use precached data (for glitches)
        # precache_cap=0 forces this to False (lazy downloading)
        use_precache = (
            ifo_obtainer.acquisition_mode == gf.AcquisitionMode.TRANSIENT and
            gf.DataLabel.GLITCHES in ifo_obtainer.data_labels and
            (precache_cap is None or precache_cap > 0)
        )
        
        if use_precache:
            # Enforce standardized cache dimensions for reusability
            # This ensures the cache supports various window sizes without rebuilding
            # "32s padding either way" -> we ensure at least 4.0s onsource and 32.0s offsource
            target_onsource = onsource_duration_seconds + (crop_duration_seconds * 2)
            cache_onsource = max(target_onsource, 4.0)
            cache_offsource = max(offsource_duration_seconds, 32.0)
            
            # Call precache_transients - this will use cache if available and large enough
            # If existing cache is smaller than these dimensions, it will be rebuilt
            cache_path = ifo_obtainer.precache_transients(
                ifos=self.ifos,
                sample_rate_hertz=sample_rate_hertz,
                onsource_duration_seconds=cache_onsource,
                offsource_duration_seconds=cache_offsource,
                group_name=group,
                data_directory=self.data_directory_path,
                seed=seed_,
                cap=precache_cap
            )
            
            # Create generator that yields from cache
            base_generator = self._cache_generator(
                cache_path,
                num_examples_per_batch,
                sample_rate_hertz,
                onsource_duration_seconds,
                offsource_duration_seconds,
                crop_duration_seconds*2.0,
                scale_factor,
                seed_,
                allowed_segments=ifo_obtainer.valid_segments if group != 'all' else None
            )
        else:
            # Original per-batch download path (for events or non-cached)
            base_generator = ifo_obtainer.get_onsource_offsource_chunks(
                sample_rate_hertz,
                onsource_duration_seconds,
                crop_duration_seconds,
                offsource_duration_seconds,
                num_examples_per_batch,
                self.ifos,
                scale_factor,
                seed=seed_
            )
        
        # Wrap generator with cropping/whitening if requested
        if crop or whiten:
            self.generator = self._postprocess_generator(
                base_generator,
                sample_rate_hertz,
                crop_duration_seconds,
                scale_factor=scale_factor,
                crop=crop,
                whiten=whiten
            )
        else:
            self.generator = base_generator
        
        return self.generator
    
    def _postprocess_generator(
        self,
        generator,
        sample_rate_hertz: float,
        crop_duration_seconds: float,
        scale_factor: float = 1.0,
        crop: bool = False,
        whiten: bool = False
    ):
        """
        Wrap generator to apply cropping and/or whitening.
        
        Whitening requires scaling up by 1E21 to avoid float precision issues.
        Output is scaled to match scale_factor expectation.
        """
        WHITEN_SCALE = 1E21
        crop_samples = int(crop_duration_seconds * sample_rate_hertz)
        
        for onsource, offsource, gps_times, labels in generator:
            # Whiten first (before cropping)
            if whiten:
                # Scale up to avoid float precision issues
                onsource_scaled = onsource * WHITEN_SCALE
                offsource_scaled = offsource * WHITEN_SCALE
                
                # Whiten using default settings
                onsource = gf.whiten(
                    onsource_scaled, 
                    offsource_scaled, 
                    sample_rate_hertz,
                )
                
                # Scale output to match expected scale_factor
                onsource = onsource * (scale_factor / WHITEN_SCALE)
                
                # Check for NaNs after whitening
                import numpy as np
                if np.isnan(onsource).any():
                     # Identify which indices are NaN
                     import logging
                     nan_indices = np.where(np.isnan(onsource).any(axis=(1,2)))[0]
                     logging.error(f"NAN DETECTED: After whitening in _postprocess_generator! Indices: {nan_indices}")
                     if len(gps_times) >= max(nan_indices):
                         # Log GPS/Type info if available (assuming gps_times matches batch)
                         for idx in nan_indices:
                             gps = gps_times[idx] if idx < len(gps_times) else "Unknown"
                             lbl = labels[idx] if idx < len(labels) else "Unknown"
                             logging.error(f"  - Failed Index {idx}: GPS={gps}, Label={lbl}")
            
            # Crop padding from onsource
            if crop and crop_samples > 0:
                onsource = onsource[:, :, crop_samples:-crop_samples]
                
            yield onsource, offsource, gps_times, labels
    
    def _cache_generator(
        self,
        cache_path,
        num_examples_per_batch: int,
        sample_rate_hertz: float,
        onsource_duration: float,
        offsource_duration: float,
        crop_duration_seconds : float,
        scale_factor: float,
        seed: int,
        allowed_segments: np.ndarray = None
    ):
        """
        Generator that yields batches from a GlitchCache file.
        
        Delegates to GlitchCache.stream_batches for the actual streaming.
        """
        from gravyflow.src.dataset.features.glitch_cache import GlitchCache
        
        cache = GlitchCache(cache_path, mode='r')
        
        # Calculate padded onsource duration for cropping
        total_onsource_duration = onsource_duration + crop_duration_seconds
        
        # Delegate to GlitchCache.stream_batches
        yield from cache.stream_batches(
            batch_size=num_examples_per_batch,
            sample_rate_hertz=sample_rate_hertz,
            onsource_duration=total_onsource_duration,
            offsource_duration=offsource_duration,
            scale_factor=scale_factor,
            seed=seed,
            allowed_segments=allowed_segments
        )
    
    def _filter_to_named_events(self, ifo_obtainer):
        all_events = get_events_with_params(event_types=[EventType.CONFIDENT, EventType.MARGINAL])
        
        ifo_obtainer.event_names = self.event_names
        
        target_gps = []
        for event in all_events:
            if event.get("name") in self.event_names:
                target_gps.append(event["gps"])
        
        if not target_gps:
            raise ValueError(
                f"No events found matching names: {self.event_names}. "
                f"Available: {[e['name'] for e in all_events[:10]]}..."
            )
        
        # Override the feature_segments to only include these events
        import numpy as np
        padding = 32.0 + 0.2  # Add 0.2s buffer to compensate for epsilon trim in get_segment
        ifo_obtainer.feature_segments = np.array([
            [gps - padding, gps + padding] for gps in target_gps
        ])