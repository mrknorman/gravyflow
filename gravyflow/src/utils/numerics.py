"""
Numeric and type utilities for gravyflow.
"""
from dataclasses import dataclass
from typing import Union, List, Any, Optional

import gravyflow as gf


def ensure_even(number: int) -> int:
    """Ensures that a number is even by subtracting 1 if it is odd."""
    if number % 2 != 0:
        number -= 1
    return number


def ensure_list(x: Any) -> list:
    """
    Ensure input is a list.
    
    Converts None to empty list, tuples to list, and wraps scalars in a list.
    
    Args:
        x: Input value (None, list, tuple, or scalar)
        
    Returns:
        List representation of input
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def calculate_sample_counts(
    onsource_duration_seconds: float,
    offsource_duration_seconds: float,
    crop_duration_seconds: float,
    sample_rate_hertz: float
) -> tuple:
    """
    Calculate onsource and offsource sample counts with padding.
    
    Args:
        onsource_duration_seconds: Base onsource duration
        offsource_duration_seconds: Offsource duration  
        crop_duration_seconds: Padding/crop duration (applied to both ends)
        sample_rate_hertz: Sample rate
        
    Returns:
        Tuple of (total_onsource_duration, num_onsource_samples, num_offsource_samples)
    """
    total_padding = crop_duration_seconds * 2.0
    total_onsource_duration = onsource_duration_seconds + total_padding
    
    num_onsource_samples = ensure_even(int(total_onsource_duration * sample_rate_hertz))
    num_offsource_samples = ensure_even(int(offsource_duration_seconds * sample_rate_hertz))
    
    return total_onsource_duration, num_onsource_samples, num_offsource_samples


@dataclass
class AcquisitionParams:
    """
    Bundled parameters for data acquisition.
    
    Reduces parameter passing overhead by grouping related acquisition parameters.
    All parameters have sensible defaults from gf.Defaults.
    """
    sample_rate_hertz: float = None
    onsource_duration_seconds: float = None
    offsource_duration_seconds: float = None
    crop_duration_seconds: float = None
    num_examples_per_batch: int = None
    scale_factor: float = None
    seed: int = None
    ifos: List = None
    
    def __post_init__(self):
        """Apply defaults from gf.Defaults for any None values."""
        if self.sample_rate_hertz is None:
            self.sample_rate_hertz = gf.Defaults.sample_rate_hertz
        if self.onsource_duration_seconds is None:
            self.onsource_duration_seconds = gf.Defaults.onsource_duration_seconds
        if self.offsource_duration_seconds is None:
            self.offsource_duration_seconds = gf.Defaults.offsource_duration_seconds
        if self.crop_duration_seconds is None:
            self.crop_duration_seconds = gf.Defaults.crop_duration_seconds
        if self.num_examples_per_batch is None:
            self.num_examples_per_batch = gf.Defaults.num_examples_per_batch
        if self.scale_factor is None:
            self.scale_factor = gf.Defaults.scale_factor
        if self.seed is None:
            self.seed = gf.Defaults.seed
        if self.ifos is None:
            self.ifos = [gf.IFO.L1]
        else:
            self.ifos = ensure_list(self.ifos)
    
    @property
    def total_onsource_duration_seconds(self) -> float:
        """Onsource duration including crop padding on both sides."""
        return self.onsource_duration_seconds + (self.crop_duration_seconds * 2.0)
    
    @property
    def num_onsource_samples(self) -> int:
        """Number of onsource samples including padding."""
        return ensure_even(int(self.total_onsource_duration_seconds * self.sample_rate_hertz))
    
    @property
    def num_offsource_samples(self) -> int:
        """Number of offsource samples."""
        return ensure_even(int(self.offsource_duration_seconds * self.sample_rate_hertz))
    
    @property
    def num_ifos(self) -> int:
        """Number of IFOs."""
        return len(self.ifos)
