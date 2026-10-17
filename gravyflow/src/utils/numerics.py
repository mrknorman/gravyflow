"""
Numeric and type utilities for gravyflow.
"""
from dataclasses import dataclass
from typing import Union, List, Any, Optional

import gravyflow as gf
from gravyflow.src.dataset.config import batch_size, ignore_scale_factor


class NonFiniteWarning(UserWarning):
    """
    NaN or infinite values were replaced with zeros.

    It lives in this module, which imports no third-party package, so a warnings
    filter can name it before Keras's backend is chosen.
    """


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


@dataclass
class BatchConfig:
    """
    Configuration for batch generation parameters.
    
    This bundles batch-related parameters that are frequently passed
    together but are NOT related to window durations (those are in WindowSpec).
    
    Attributes:
        num_examples_per_batch: Number of examples in each batch
        scale_factor: Deprecated and ignored (data are physical strain); kept
            for one release so that positional arguments keep their places.
        seed: Random seed for reproducibility
        ifos: List of interferometers to use
    """
    num_examples_per_batch: int = None
    scale_factor: None = None
    seed: int = None
    ifos: List = None
    
    def __post_init__(self):
        """Apply defaults from gf.Defaults for any None values; refuse an empty batch by name."""
        self.num_examples_per_batch = batch_size(self.num_examples_per_batch)
        ignore_scale_factor(self.scale_factor)
        self.scale_factor = None
        if self.seed is None:
            self.seed = gf.Defaults.seed
        if self.ifos is None:
            self.ifos = [gf.IFO.L1]
        else:
            self.ifos = ensure_list(self.ifos)
    
    @property
    def num_ifos(self) -> int:
        """Number of IFOs."""
        return len(self.ifos)
