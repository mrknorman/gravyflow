"""
Tests for pool-based dataset composition.

Tests the FeaturePool, PoolSampler, and ComposedDataset classes.
"""

import numpy as np
import pytest
from numpy.random import default_rng

import gravyflow as gf


class TestFeaturePool:
    """Tests for FeaturePool dataclass."""

    def test_valid_noise_pool(self):
        """Test creating a valid noise-based pool."""
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        pool = gf.FeaturePool(
            name="test_noise",
            label=0,
            probability=0.5,
            noise_obtainer=noise_obtainer
        )
        assert pool.name == "test_noise"
        assert pool.label == 0
        assert pool.probability == 0.5
        assert pool.is_noise_pool
        assert not pool.is_transient_pool
        assert not pool.has_injections

    def test_valid_noise_pool_with_injections(self):
        """Test creating a noise pool with injections."""
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        generator = gf.WNBGenerator(
            scaling_method=gf.ScalingMethod(
                value=gf.Distribution(min_=8, max_=15),
                type_=gf.ScalingTypes.SNR
            )
        )
        pool = gf.FeaturePool(
            name="test_injections",
            label=1,
            probability=0.3,
            noise_obtainer=noise_obtainer,
            injection_generators=[generator]
        )
        assert pool.is_noise_pool
        assert pool.has_injections
        assert len(pool.injection_generators) == 1

    def test_missing_data_source_raises(self):
        """Test that missing data source raises ValueError."""
        with pytest.raises(ValueError, match="must have either"):
            gf.FeaturePool(
                name="invalid",
                label=0,
                probability=0.5
            )

    def test_both_data_sources_raises(self):
        """Test that having both data sources raises ValueError."""
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        transient_obtainer = gf.TransientDataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.GLITCHES]
        )
        with pytest.raises(ValueError, match="cannot have both"):
            gf.FeaturePool(
                name="invalid",
                label=0,
                probability=0.5,
                noise_obtainer=noise_obtainer,
                transient_obtainer=transient_obtainer
            )

    def test_negative_probability_raises(self):
        """Test that negative probability raises ValueError."""
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        with pytest.raises(ValueError, match="must be positive"):
            gf.FeaturePool(
                name="invalid",
                label=0,
                probability=-0.5,
                noise_obtainer=noise_obtainer
            )

    def test_negative_label_raises(self):
        """Test that negative label raises ValueError."""
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        with pytest.raises(ValueError, match="must be non-negative"):
            gf.FeaturePool(
                name="invalid",
                label=-1,
                probability=0.5,
                noise_obtainer=noise_obtainer
            )


class TestPoolSampler:
    """Tests for PoolSampler class."""

    def test_sampler_initialization(self):
        """Test basic sampler initialization."""
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        pools = [
            gf.FeaturePool(name="pool_a", label=0, probability=0.5, noise_obtainer=noise_obtainer),
            gf.FeaturePool(name="pool_b", label=1, probability=0.3, noise_obtainer=noise_obtainer),
            gf.FeaturePool(name="pool_c", label=2, probability=0.2, noise_obtainer=noise_obtainer),
        ]
        sampler = gf.PoolSampler(pools, seed=42)
        
        assert len(sampler.pools) == 3
        assert sampler.num_classes == 3
        assert sampler.unique_labels == {0, 1, 2}

    def test_probability_normalization(self):
        """Test that probabilities are normalized."""
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        pools = [
            gf.FeaturePool(name="pool_a", label=0, probability=1.0, noise_obtainer=noise_obtainer),
            gf.FeaturePool(name="pool_b", label=1, probability=1.0, noise_obtainer=noise_obtainer),
        ]
        sampler = gf.PoolSampler(pools, seed=42)
        
        # Should be normalized to [0.5, 0.5]
        np.testing.assert_array_almost_equal(sampler.probabilities, [0.5, 0.5])

    def test_probability_distribution(self):
        """Test that sampling follows the probability distribution."""
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        pools = [
            gf.FeaturePool(name="pool_a", label=0, probability=0.7, noise_obtainer=noise_obtainer),
            gf.FeaturePool(name="pool_b", label=1, probability=0.3, noise_obtainer=noise_obtainer),
        ]
        sampler = gf.PoolSampler(pools, seed=42)
        
        # Sample many times
        n_samples = 10000
        indices = sampler.sample_pool_indices(n_samples)
        
        # Count occurrences
        unique, counts = np.unique(indices, return_counts=True)
        frequencies = counts / n_samples
        
        # Should be close to [0.7, 0.3]
        np.testing.assert_array_almost_equal(frequencies, [0.7, 0.3], decimal=1)

    def test_shared_labels(self):
        """Test that multiple pools can share the same label."""
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        pools = [
            gf.FeaturePool(name="noise", label=0, probability=0.5, noise_obtainer=noise_obtainer),
            gf.FeaturePool(name="signal_cbc", label=1, probability=0.25, noise_obtainer=noise_obtainer),
            gf.FeaturePool(name="signal_wnb", label=1, probability=0.25, noise_obtainer=noise_obtainer),
        ]
        sampler = gf.PoolSampler(pools, seed=42)
        
        # Only 2 unique labels
        assert sampler.num_classes == 2
        assert sampler.unique_labels == {0, 1}

    def test_empty_pools_raises(self):
        """Test that empty pool list raises ValueError."""
        with pytest.raises(ValueError, match="at least one pool"):
            gf.PoolSampler([])

    def test_duplicate_names_raises(self):
        """Test that duplicate pool names raise ValueError."""
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        pools = [
            gf.FeaturePool(name="pool_a", label=0, probability=0.5, noise_obtainer=noise_obtainer),
            gf.FeaturePool(name="pool_a", label=1, probability=0.5, noise_obtainer=noise_obtainer),
        ]
        with pytest.raises(ValueError, match="unique"):
            gf.PoolSampler(pools)

    def test_get_pool_batch_sizes(self):
        """Test batch size assignment by pool."""
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        pools = [
            gf.FeaturePool(name="pool_a", label=0, probability=0.5, noise_obtainer=noise_obtainer),
            gf.FeaturePool(name="pool_b", label=1, probability=0.5, noise_obtainer=noise_obtainer),
        ]
        sampler = gf.PoolSampler(pools, seed=42)
        
        assignments = sampler.get_pool_batch_sizes(32)
        
        # Should have assignments for pools
        total_assigned = sum(count for count, positions in assignments.values())
        assert total_assigned == 32
        
        # All positions 0-31 should be covered
        all_positions = set()
        for count, positions in assignments.values():
            all_positions.update(positions)
        assert all_positions == set(range(32))


class TestComposedDatasetInit:
    """Tests for ComposedDataset initialization."""

    def test_basic_initialization(self):
        """Test basic ComposedDataset initialization with white noise."""
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        pools = [
            gf.FeaturePool(
                name="noise",
                label=0,
                probability=1.0,
                noise_obtainer=noise_obtainer
            )
        ]
        
        dataset = gf.ComposedDataset(
            pools=pools,
            sample_rate_hertz=1024.0,
            num_examples_per_batch=4,
            input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
            output_variables=[gf.ReturnVariables.POOL_LABEL],
            steps_per_epoch=10
        )
        
        assert len(dataset) == 10
        assert dataset.pool_sampler.num_classes == 1

    def test_multi_pool_initialization(self):
        """Test initialization with multiple pools."""
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        pools = [
            gf.FeaturePool(name="noise", label=0, probability=0.5, noise_obtainer=noise_obtainer),
            gf.FeaturePool(name="signal", label=1, probability=0.5, noise_obtainer=noise_obtainer),
        ]
        
        dataset = gf.ComposedDataset(
            pools=pools,
            sample_rate_hertz=1024.0,
            num_examples_per_batch=4,
            input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
            output_variables=[gf.ReturnVariables.POOL_LABEL],
            steps_per_epoch=10
        )
        
        assert dataset.pool_sampler.num_classes == 2


@pytest.mark.slow
class TestComposedDatasetIteration:
    """Tests for ComposedDataset data generation."""

    def test_basic_iteration(self):
        """Test iterating through a composed dataset."""
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        pools = [
            gf.FeaturePool(
                name="noise",
                label=0,
                probability=1.0,
                noise_obtainer=noise_obtainer
            )
        ]
        
        with gf.env():
            dataset = gf.ComposedDataset(
                pools=pools,
                sample_rate_hertz=1024.0,
                num_examples_per_batch=4,
                input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
                output_variables=[gf.ReturnVariables.POOL_LABEL],
                steps_per_epoch=3
            )
            
            # Iterate a few batches
            for i, (inputs, outputs) in enumerate(dataset):
                if i >= 2:
                    break
                    
                # Check shapes
                assert "WHITENED_ONSOURCE" in inputs
                assert "POOL_LABEL" in outputs
                assert len(outputs["POOL_LABEL"]) == 4

    def test_multi_pool_labels(self):
        """Test that pool labels are correctly assigned."""
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        pools = [
            gf.FeaturePool(name="noise", label=0, probability=0.5, noise_obtainer=noise_obtainer),
            gf.FeaturePool(name="signal", label=1, probability=0.5, noise_obtainer=noise_obtainer),
        ]
        
        with gf.env():
            dataset = gf.ComposedDataset(
                pools=pools,
                sample_rate_hertz=1024.0,
                num_examples_per_batch=32,
                input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
                output_variables=[gf.ReturnVariables.POOL_LABEL],
                steps_per_epoch=10,
                seed=42
            )
            
            inputs, outputs = next(iter(dataset))
            labels = np.array(outputs["POOL_LABEL"])
            
            # Should have both label 0 and label 1 in the batch
            assert 0 in labels or 1 in labels
            # All labels should be valid
            assert np.all((labels == 0) | (labels == 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
