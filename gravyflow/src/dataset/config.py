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