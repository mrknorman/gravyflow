"""
Pool-based dataset composition.

This module provides abstractions for creating datasets composed of 
multiple feature "pools", each with its own data source, class label,
and sampling probability.

Example usage:
    pools = [
        gf.FeaturePool(
            name="pure_noise",
            label=0,
            probability=0.5,
            noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.REAL, ...)
        ),
        gf.FeaturePool(
            name="glitches",
            label=1,
            probability=0.3,
            transient_obtainer=gf.TransientDataObtainer(
                data_labels=[gf.DataLabel.GLITCHES], ...
            )
        ),
        gf.FeaturePool(
            name="cbc_signals",
            label=2,
            probability=0.2,
            noise_obtainer=gf.NoiseObtainer(...),
            injection_generators=[cbc_generator]
        ),
    ]
    
    dataset = gf.ComposedDataset(pools=pools, ...)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from numpy.random import Generator, default_rng

if TYPE_CHECKING:
    from gravyflow.src.dataset.acquisition.noise import NoiseObtainer
    from gravyflow.src.dataset.acquisition.transient import TransientDataObtainer
    from gravyflow.src.dataset.features.injection import WaveformGenerator


@dataclass
class FeaturePool:
    """
    A pool of training examples with a class label and sampling probability.
    
    Each pool represents a distinct data source or feature type (e.g., pure noise,
    glitch-contaminated data, signal injections). Multiple pools can share the
    same label for hierarchical classification.
    
    Attributes:
        name: Human-readable identifier for the pool.
        label: Integer class label for supervised training.
        probability: Relative probability of sampling from this pool (will be normalized).
        noise_obtainer: Data source for pure noise or noise-with-injection pools.
        transient_obtainer: Data source for glitch/event pools.
        injection_generators: Optional waveform generators to add injections to noise.
        
    Note:
        Exactly one of `noise_obtainer` or `transient_obtainer` must be provided.
        `injection_generators` can only be used with `noise_obtainer`.
    """
    
    name: str
    label: int
    probability: float
    
    # Data sources (exactly one should be set)
    noise_obtainer: Optional["NoiseObtainer"] = None
    transient_obtainer: Optional["TransientDataObtainer"] = None
    
    # Optional injection layer (only valid with noise_obtainer)
    injection_generators: Optional[List["WaveformGenerator"]] = None
    
    def __post_init__(self):
        """Validate pool configuration."""
        # Must have at least one data source
        if self.noise_obtainer is None and self.transient_obtainer is None:
            raise ValueError(
                f"FeaturePool '{self.name}' must have either noise_obtainer or transient_obtainer"
            )
        
        # Cannot have both data sources
        if self.noise_obtainer is not None and self.transient_obtainer is not None:
            raise ValueError(
                f"FeaturePool '{self.name}' cannot have both noise_obtainer and transient_obtainer"
            )
        
        # Injection generators only valid with noise_obtainer
        if self.injection_generators is not None and self.transient_obtainer is not None:
            raise ValueError(
                f"FeaturePool '{self.name}': injection_generators can only be used with noise_obtainer, "
                "not transient_obtainer"
            )
        
        # Probability must be positive
        if self.probability <= 0:
            raise ValueError(
                f"FeaturePool '{self.name}': probability must be positive, got {self.probability}"
            )
        
        # Label must be non-negative
        if self.label < 0:
            raise ValueError(
                f"FeaturePool '{self.name}': label must be non-negative, got {self.label}"
            )
    
    @property
    def is_noise_pool(self) -> bool:
        """True if this pool uses a noise obtainer."""
        return self.noise_obtainer is not None
    
    @property
    def is_transient_pool(self) -> bool:
        """True if this pool uses a transient obtainer."""
        return self.transient_obtainer is not None
    
    @property
    def has_injections(self) -> bool:
        """True if this pool includes injections."""
        return self.injection_generators is not None and len(self.injection_generators) > 0


class PoolSampler:
    """
    Samples from multiple FeaturePools according to their probabilities.
    
    Probabilities are normalized to sum to 1.0.
    
    Attributes:
        pools: List of FeaturePool objects.
        probabilities: Normalized sampling probabilities.
        unique_labels: Set of unique class labels across all pools.
    """
    
    def __init__(self, pools: List[FeaturePool], seed: int = None):
        """
        Initialize the sampler.
        
        Args:
            pools: List of FeaturePool objects to sample from.
            seed: Random seed for reproducibility.
        """
        if not pools:
            raise ValueError("PoolSampler requires at least one pool")
        
        self.pools = pools
        self._validate_pools()
        
        # Always normalize probabilities
        total = sum(p.probability for p in pools)
        self._probabilities = np.array([p.probability / total for p in pools])
        
        # Track unique labels
        self._unique_labels = set(p.label for p in pools)
        
        # Initialize RNG
        self._rng = default_rng(seed)
    
    def _validate_pools(self):
        """Validate pool configuration."""
        names = [p.name for p in self.pools]
        if len(names) != len(set(names)):
            raise ValueError("Pool names must be unique")
    
    @property
    def probabilities(self) -> np.ndarray:
        """Normalized sampling probabilities."""
        return self._probabilities.copy()
    
    @property
    def unique_labels(self) -> set:
        """Set of unique class labels across all pools."""
        return self._unique_labels.copy()
    
    @property
    def num_classes(self) -> int:
        """Number of unique class labels."""
        return len(self._unique_labels)
    
    def sample_pool(self) -> FeaturePool:
        """
        Select a single pool based on probabilities.
        
        Returns:
            The selected FeaturePool.
        """
        idx = self._rng.choice(len(self.pools), p=self._probabilities)
        return self.pools[idx]
    
    def sample_pool_indices(self, batch_size: int) -> np.ndarray:
        """
        Sample pool indices for an entire batch.
        
        Args:
            batch_size: Number of samples to draw.
            
        Returns:
            Array of pool indices of shape (batch_size,).
        """
        return self._rng.choice(len(self.pools), size=batch_size, p=self._probabilities)
    
    def sample_batch_assignments(self, batch_size: int) -> List[Tuple[int, FeaturePool]]:
        """
        Sample pool assignments for an entire batch.
        
        Returns a list of (pool_index, pool) tuples for each example in the batch.
        
        Args:
            batch_size: Number of samples to draw.
            
        Returns:
            List of (pool_index, FeaturePool) tuples.
        """
        indices = self.sample_pool_indices(batch_size)
        return [(int(i), self.pools[i]) for i in indices]
    
    def get_pool_batch_sizes(self, batch_size: int) -> dict:
        """
        Sample and group examples by pool.
        
        Returns a dictionary mapping pool index to the number of examples
        to generate from that pool, along with the original positions.
        
        Args:
            batch_size: Total batch size.
            
        Returns:
            Dict mapping pool_index -> (count, list of original positions).
        """
        indices = self.sample_pool_indices(batch_size)
        
        pool_assignments = {}
        for pos, pool_idx in enumerate(indices):
            pool_idx = int(pool_idx)
            if pool_idx not in pool_assignments:
                pool_assignments[pool_idx] = (0, [])
            count, positions = pool_assignments[pool_idx]
            pool_assignments[pool_idx] = (count + 1, positions + [pos])
        
        return pool_assignments
    
    def reseed(self, seed: int):
        """Reset the random number generator with a new seed."""
        self._rng = default_rng(seed)
