import logging
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class WindowSpec:
    """
    Immutable specification for data acquisition time windows.
    
    This bundles all duration-related parameters that are frequently passed
    together, providing a single source of truth for window configuration.
    
    Terminology:
        - onsource: The final output window duration (e.g., 1.0s for inference)
        - offsource: The background/PSD estimation window (e.g., 16.0s)
        - crop: Extra duration on EACH SIDE of onsource for edge-effect handling
                (whitening artifacts, FFT edge effects, etc.)
        - total_onsource: onsource + (2 * crop) - the raw acquisition window
    
    Example:
        >>> spec = WindowSpec.default()  # Uses Defaults values
        >>> spec = WindowSpec(
        ...     sample_rate_hertz=2048.0,
        ...     onsource_duration_seconds=1.0,
        ...     offsource_duration_seconds=16.0,
        ...     crop_duration_seconds=0.5
        ... )
        >>> spec.total_onsource_duration_seconds  # 2.0
        >>> spec.num_onsource_samples  # 4096
        
        # For JAX JIT functions - use individual properties:
        >>> num_on = spec.num_onsource_samples  # Static int for JIT
    
    Attributes:
        sample_rate_hertz: Data sample rate in Hz
        onsource_duration_seconds: Final onsource window duration in seconds
        offsource_duration_seconds: Background window duration in seconds
        crop_duration_seconds: Cropping duration on each side in seconds
    
    Note:
        This class is frozen (immutable) to ensure it can be used as a 
        dictionary key and to prevent accidental modification.
    """
    sample_rate_hertz: float
    onsource_duration_seconds: float
    offsource_duration_seconds: float
    crop_duration_seconds: float
    
    @classmethod
    def default(cls) -> "WindowSpec":
        """Create WindowSpec from Defaults values."""
        return cls(
            sample_rate_hertz=Defaults.sample_rate_hertz,
            onsource_duration_seconds=Defaults.onsource_duration_seconds,
            offsource_duration_seconds=Defaults.offsource_duration_seconds,
            crop_duration_seconds=Defaults.crop_duration_seconds,
        )
    
    @classmethod
    def from_params(
        cls,
        sample_rate_hertz: float = None,
        onsource_duration_seconds: float = None,
        offsource_duration_seconds: float = None,
        crop_duration_seconds: float = None,
    ) -> "WindowSpec":
        """
        Create WindowSpec with partial overrides of defaults.
        
        Any parameter not specified will use the value from Defaults.
        """
        return cls(
            sample_rate_hertz=sample_rate_hertz if sample_rate_hertz is not None else Defaults.sample_rate_hertz,
            onsource_duration_seconds=onsource_duration_seconds if onsource_duration_seconds is not None else Defaults.onsource_duration_seconds,
            offsource_duration_seconds=offsource_duration_seconds if offsource_duration_seconds is not None else Defaults.offsource_duration_seconds,
            crop_duration_seconds=crop_duration_seconds if crop_duration_seconds is not None else Defaults.crop_duration_seconds,
        )

    @property
    def total_onsource_duration_seconds(self) -> float:
        """Onsource duration including crop padding on both ends."""
        return self.onsource_duration_seconds + (self.crop_duration_seconds * 2.0)
    
    @property
    def num_onsource_samples(self) -> int:
        """Number of onsource samples including cropping padding (ensured even)."""
        num = int(self.total_onsource_duration_seconds * self.sample_rate_hertz)
        return num - (num % 2)  # Ensure even
    
    @property
    def num_offsource_samples(self) -> int:
        """Number of offsource samples (ensured even)."""
        num = int(self.offsource_duration_seconds * self.sample_rate_hertz)
        return num - (num % 2)  # Ensure even
    
    @property
    def sample_counts(self) -> Tuple[int, int]:
        """
        Return (num_onsource_samples, num_offsource_samples) as a tuple.
        
        Useful when passing to functions that need both values.
        """
        return (self.num_onsource_samples, self.num_offsource_samples)
    
    def with_overrides(self, **kwargs) -> "WindowSpec":
        """
        Create a new WindowSpec with some values overridden.
        
        Example:
            >>> spec2 = spec.with_overrides(onsource_duration_seconds=2.0)
        """
        return WindowSpec(
            sample_rate_hertz=kwargs.get('sample_rate_hertz', self.sample_rate_hertz),
            onsource_duration_seconds=kwargs.get('onsource_duration_seconds', self.onsource_duration_seconds),
            offsource_duration_seconds=kwargs.get('offsource_duration_seconds', self.offsource_duration_seconds),
            crop_duration_seconds=kwargs.get('crop_duration_seconds', self.crop_duration_seconds),
        )


class Defaults:
    seed : int = 1000
    num_examples_per_generation_batch: int = 2048
    num_examples_per_batch: int = 32
    sample_rate_hertz: float = 2048.0
    onsource_duration_seconds: float = 1.0
    offsource_duration_seconds: float = 16.0
    crop_duration_seconds: float = 0.5
    scale_factor: float = 1.0E21
    
    # Standard tiers for JIT graph optimization
    # Using these values reduces the number of unique JAX graph compilations
    STANDARD_DURATIONS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    STANDARD_BATCH_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    STANDARD_SAMPLE_RATES = [2048.0, 4096.0, 8192.0, 16384.0]

    @classmethod
    def set(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise AttributeError(
                    f"{cls.__name__} has no attribute named '{key}'"
                )
    
    @staticmethod
    def quantize_duration(duration: float) -> float:
        """Round up to nearest standard duration tier to reduce JAX recompilations."""
        for std_dur in Defaults.STANDARD_DURATIONS:
            if duration <= std_dur:
                return std_dur
        # If larger than all tiers, return as-is but log warning
        logging.debug(f"Duration {duration}s exceeds standard tiers, using as-is.")
        return duration
    
    @staticmethod
    def quantize_batch_size(batch_size: int) -> int:
        """Round up to nearest standard batch size to reduce JAX recompilations."""
        for std_batch in Defaults.STANDARD_BATCH_SIZES:
            if batch_size <= std_batch:
                return std_batch
        logging.debug(f"Batch size {batch_size} exceeds standard tiers, using as-is.")
        return batch_size
    
    @staticmethod
    def quantize_sample_rate(sample_rate: float) -> float:
        """Round to nearest standard sample rate."""
        if sample_rate in Defaults.STANDARD_SAMPLE_RATES:
            return sample_rate
        # Find closest
        closest = min(Defaults.STANDARD_SAMPLE_RATES, key=lambda x: abs(x - sample_rate))
        if abs(closest - sample_rate) > 1.0:
            logging.debug(f"Sample rate {sample_rate} rounded to standard {closest}")
        return closest
    
    # Standard segment sizes (in samples) for reducing JIT recompilations
    # These correspond to powers of 2 in sample counts
    # At 2048 Hz: 16s, 32s, 64s, 128s, 256s, 512s, 1024s, 2048s durations
    STANDARD_SEGMENT_SAMPLES = [2**14, 2**15, 2**16, 2**17, 2**18, 2**19, 2**20, 2**21, 2**22]  # 32K to 4M samples
    
    @staticmethod
    def truncate_to_standard_segment_size(num_samples: int, min_required: int = 0) -> int:
        """
        Find the largest standard segment size that fits within num_samples.
        
        Returns the standard size to truncate to, or None if segment is too small.
        This ensures only a fixed number of segment sizes exist, reducing JIT compilations.
        
        Parameters
        ----------
        num_samples : int
            Actual number of samples in segment.
        min_required : int
            Minimum samples required (e.g., for batch extraction).
            
        Returns
        -------
        int or None
            Standard size to truncate to, or None if too small.
        """
        for std_size in reversed(Defaults.STANDARD_SEGMENT_SAMPLES):
            if std_size <= num_samples and std_size >= min_required:
                return std_size
        return None


class TransientDefaults:
    """
    Configuration constants for transient (glitch/event) acquisition.
    
    These constants are used throughout the transient pipeline:
    - Cache storage parameters
    - Download clustering limits
    - Segment padding
    - Batch save thresholds
    - Logging intervals
    
    Centralizing them here allows easy tuning and ensures consistency.
    """
    
    # ==========================================================================
    # CACHE PARAMETERS
    # ==========================================================================
    # Maximum sample rate stored in cache (can downsample but not upsample)
    CACHE_SAMPLE_RATE_HERTZ: float = 4096.0
    
    # Maximum duration windows (allows cropping to shorter durations)
    CACHE_ONSOURCE_DURATION: float = 32.0
    CACHE_OFFSOURCE_DURATION: float = 32.0
    CACHE_PADDING_DURATION: float = 1.0
    
    # ==========================================================================
    # SEGMENT PADDING
    # ==========================================================================
    # Default padding around transient events (ensures full event capture)
    DEFAULT_START_PADDING_SECONDS: float = 32.0
    DEFAULT_END_PADDING_SECONDS: float = 32.0
    
    # Epsilon buffer for segment boundary trimming
    SEGMENT_BOUNDARY_BUFFER: float = 0.2
    
    # ==========================================================================
    # DOWNLOAD CLUSTERING
    # ==========================================================================
    # Maximum segment duration for clustered downloads (prevents OOM)
    MAX_SEGMENT_SECONDS: float = 512.0
    
    # Network download overhead estimate (seconds per request)
    REQUEST_OVERHEAD_SECONDS: float = 15.0
    
    # Data download rate for estimation (seconds per second of data)
    DATA_DOWNLOAD_RATE: float = 0.01
    
    # ==========================================================================
    # BATCH THRESHOLDS
    # ==========================================================================
    # Save to cache every N glitches (balances I/O vs memory)
    BATCH_SAVE_THRESHOLD: int = 1000
    
    # Log cache stats every N samples
    CACHE_LOG_INTERVAL: int = 1000
    
    # LRU cache size for lazy-loaded transient segments
    LAZY_CACHE_MAXSIZE: int = 1000
    
    # HDF5 chunk size for streaming operations
    HDF5_CHUNK_SIZE: int = 10000
    
    # ==========================================================================
    # GPS PRECISION
    # ==========================================================================
    # GPS time precision for cache key generation (10ms = 0.01s)
    GPS_KEY_PRECISION: float = 0.01
    
    # GPS tolerance for matching transients (half second handles typical rounding)
    GPS_TOLERANCE_SECONDS: float = 0.5
    
    # Segment boundary epsilon for numerical safety
    SEGMENT_EPSILON_SECONDS: float = 0.1

    # ==========================================================================
    # UNIVERSAL CACHE STRATEGY
    # ==========================================================================
    # Cache downloads all LIGO detectors (H1, L1) regardless of what the user
    # requests. This ensures maximum cache reuse - a cache built with H1+L1 can
    # serve any subset (H1 only, L1 only, etc.) without redownloading.
    # 
    # NOTE: V1 (Virgo) is NOT included because:
    # - Only joined during O2 with limited coverage
    # - No data available for O1 events
    # - Would cause acquisition failures for older events
    #
    # Requested IFOs are filtered at output time, not download time.
    UNIVERSAL_IFOS: Tuple[str, ...] = ("H1", "L1")