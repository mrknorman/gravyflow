"""
Data augmentation types for acquisition pipeline.

Each augmentation is a dataclass with its own probability and parameters,
enabling per-augmentation configuration.

Example:
    obtainer = TransientDataObtainer(
        augmentations=[
            SignReversal(probability=0.8),
            TimeReversal(probability=0.3),
            RandomShift(probability=0.5, shift_fraction=0.1)
        ]
    )
"""

from dataclasses import dataclass
from typing import List, Union


@dataclass
class SignReversal:
    """
    Flip sign of data (x -> -x).
    
    Useful for gravitational wave data since waveforms are symmetric
    under sign reversal (both + and - polarizations are physically valid).
    
    Args:
        probability: Probability of applying this augmentation (0.0-1.0)
    """
    probability: float = 0.5


@dataclass
class TimeReversal:
    """
    Reverse time axis.
    
    For noise data, time-reversed noise is statistically equivalent.
    For transients, this can help prevent overfitting to specific waveform shapes.
    
    Args:
        probability: Probability of applying this augmentation (0.0-1.0)
    """
    probability: float = 0.5


@dataclass
class RandomShift:
    """
    Shift event off-center by a random amount.
    
    TRANSIENT mode only. Prevents the model from learning that events
    are always centered in the window.
    
    Args:
        probability: Probability of applying this augmentation (0.0-1.0)
        shift_fraction: Maximum shift as fraction of window length (e.g., 0.25 = 25%)
    """
    probability: float = 0.5
    shift_fraction: float = 0.25


@dataclass
class AddNoise:
    """
    Add small noise perturbation to data.
    
    TRANSIENT mode only. Helps prevent overfitting to specific noise realizations.
    
    Args:
        probability: Probability of applying this augmentation (0.0-1.0)
        amplitude: Noise amplitude as fraction of data std (e.g., 0.1 = 10%)
    """
    probability: float = 0.5
    amplitude: float = 0.1


# Type alias for augmentation instances
Augmentation = Union[SignReversal, TimeReversal, RandomShift, AddNoise]


def default_augmentations() -> List[Augmentation]:
    """
    Default augmentations matching previous boolean defaults.
    
    Returns:
        List with SignReversal and TimeReversal at 50% probability.
    """
    return [SignReversal(), TimeReversal()]
