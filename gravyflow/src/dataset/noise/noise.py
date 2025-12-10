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
        ), ops.full((num_examples_per_batch,), -1.0)
        
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
    
    # Stack or Concat?
    # Original: tf.concat(..., axis=1) -> (1, IFOs*Freqs) ??
    # Wait, original code:
    # interpolated_onsource_asds.append(interpolated_asd_onsource)
    # interpolated_onsource_asds = tf.concat(interpolated_onsource_asds, axis=1)
    
    # interpolated_asd_onsource shape from interpolate is (Freqs,).
    # If we append to list, we have [ (Freqs,), (Freqs,) ].
    # tf.concat(..., axis=1) would require rank 2?
    # Original interpolate returned shape?
    # tfp.math.interp_regular_1d_grid returns shape of query points.
    # If query points is (Freqs,), output is (Freqs,).
    # tf.concat([ (Freqs,), (Freqs,) ], axis=1) fails.
    # Maybe original code had expand_dims?
    # "interpolated_psd_onsource ... for num in num_samples_list"
    # It returns a list of tensors.
    
    # Let's assume we want (IFOs, Freqs) or (1, IFOs, Freqs) for broadcasting.
    # _generate_colored_noise expects `interpolated_asd * white_noise_fft`.
    # white_noise_fft is (Batch, IFOs, Freqs).
    # So interpolated_asd should be (1, IFOs, Freqs) or (IFOs, Freqs).
    
    # Let's stack them.
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
            ), ops.full((num_examples_per_batch,), -1.0)
    

@dataclass
class NoiseObtainer:
    data_directory_path : Path = Path("./generator_data")
    ifo_data_obtainer : Union[None, gf.IFODataObtainer] = None
    ifos : List[gf.IFO] = gf.IFO.L1
    noise_type : NoiseType = NoiseType.REAL
    groups : Union[dict, None] = None
    
    def __post_init__(self):
        
        if not isinstance(self.ifos, list) and not isinstance(self.ifos, tuple):
            self.ifos = [self.ifos]
        
        if not self.groups:
            self.groups = {
                "train" : 0.89,
                "validate" : 0.1,
                "test" : 0.01
            }

        self.rng = None
    
    def __call__(
            self,
            sample_rate_hertz : float  = None,
            onsource_duration_seconds : float = None,
            crop_duration_seconds : float = None,
            offsource_duration_seconds : float = None,
            num_examples_per_batch : float = None,
            scale_factor : float = 1.0,
            group : str = "train",
            seed : int = None
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
                canonical_ifos = tuple(sorted(ifo.name for ifo in self.ifos))

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
                        seed=seed_
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

        canonical_ifos = tuple(sorted(ifo.name for ifo in self.ifos))

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
                    / (
                        self.ifo_data_obtainer.saturation * 
                        num_examples_per_batch * onsource_duration_seconds
                    )
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
                    ), ops.full((num_examples_per_batch,), -1.0)
                

class FeatureObtainer(NoiseObtainer):
    data_directory_path : Path = Path("./generator_data")
    ifo_data_obtainer : Union[None, gf.IFODataObtainer] = None
    ifos : List[gf.IFO] = gf.IFO.L1
    noise_type : NoiseType = NoiseType.REAL
    groups : Union[dict, None] = None