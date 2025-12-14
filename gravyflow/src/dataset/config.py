import logging

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