from dataclasses import dataclass, field
from typing import List, Tuple, Union
import numpy as np

@dataclass
class ValidationConfig:
    """Configuration for unified validation pipeline."""
    snr_range: Tuple[float, float] = (0.0, 15.0)
    num_examples: int = 100_000
    batch_size: int = 512
    snr_bin_width: float = 5.0
    num_worst_per_bin: int = 5
    
    # Validation settings for generic model
    far_thresholds: List[float] = field(
        default_factory=lambda: np.logspace(-1, -4.5, 50).tolist()
    )
    
    default_roc_min_snr: float = 5.0
    
    # Extra ROC pools config (e.g. low SNR, high SNR)
    # List of tuples (min_snr, max_snr) or floats (min_snr)
    extra_roc_pools: List[Union[Tuple[float, float], float]] = field(default_factory=list)
