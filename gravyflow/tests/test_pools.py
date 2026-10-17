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
        sampler = gf.PoolSampler(pools)
        
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
        sampler = gf.PoolSampler(pools)
        
        # Should be normalized to [0.5, 0.5]
        np.testing.assert_array_almost_equal(sampler.probabilities, [0.5, 0.5])

    def test_probability_distribution(self):
        """Test that sampling follows the probability distribution."""
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        pools = [
            gf.FeaturePool(name="pool_a", label=0, probability=0.7, noise_obtainer=noise_obtainer),
            gf.FeaturePool(name="pool_b", label=1, probability=0.3, noise_obtainer=noise_obtainer),
        ]
        sampler = gf.PoolSampler(pools)
        
        # Sample many times
        n_samples = 10000
        indices = sampler.sample_pool_indices(n_samples, rng=default_rng(42))
        
        # Count occurrences
        unique, counts = np.unique(indices, return_counts=True)
        frequencies = counts / n_samples
        
        # Each frequency is a binomial proportion: 4 sigma is 4 sqrt(p (1 - p) / n), 0.018 here.
        np.testing.assert_allclose(frequencies, [0.7, 0.3], rtol=0, atol=4 * np.sqrt(0.7 * 0.3 / n_samples))

    def test_shared_labels(self):
        """Test that multiple pools can share the same label."""
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        pools = [
            gf.FeaturePool(name="noise", label=0, probability=0.5, noise_obtainer=noise_obtainer),
            gf.FeaturePool(name="signal_cbc", label=1, probability=0.25, noise_obtainer=noise_obtainer),
            gf.FeaturePool(name="signal_wnb", label=1, probability=0.25, noise_obtainer=noise_obtainer),
        ]
        sampler = gf.PoolSampler(pools)
        
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
        sampler = gf.PoolSampler(pools)
        
        assignments = sampler.get_pool_batch_sizes(32, rng=default_rng(42))
        
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
            steps_per_epoch=10,
            group="train"
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
            steps_per_epoch=10,
            group="train"
        )
        
        assert dataset.pool_sampler.num_classes == 2


def composed(pools, batch, steps, seed=42):
    return gf.ComposedDataset(
        pools=pools, sample_rate_hertz=1024.0, num_examples_per_batch=batch,
        input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
        output_variables=[gf.ReturnVariables.POOL_LABEL], steps_per_epoch=steps, seed=seed,
        group="train",
    )


class TestComposedDatasetIteration:
    """Tests for ComposedDataset data generation."""

    def test_basic_iteration(self):
        """An epoch is steps_per_epoch batches of whitened noise, every label the one pool's."""
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        pools = [gf.FeaturePool(name="noise", label=0, probability=1.0, noise_obtainer=noise_obtainer)]
        batches = list(composed(pools, batch=4, steps=3))
        assert len(batches) == 3
        for inputs, outputs in batches:
            whitened = np.asarray(inputs["WHITENED_ONSOURCE"])
            assert whitened.shape[:2] == (4, 1) and np.all(np.isfinite(whitened)) and np.any(whitened != 0)
            np.testing.assert_array_equal(np.asarray(outputs["POOL_LABEL"]).ravel(), [0] * 4)

    def test_multi_pool_labels(self):
        """Both pools serve a 32-example batch; the label sequence is the seed's."""
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        pools = [
            gf.FeaturePool(name="noise", label=0, probability=0.5, noise_obtainer=noise_obtainer),
            gf.FeaturePool(name="signal", label=1, probability=0.5, noise_obtainer=noise_obtainer),
        ]

        def labels(seed):
            _, outputs = next(iter(composed(pools, batch=32, steps=10, seed=seed)))
            return np.asarray(outputs["POOL_LABEL"]).ravel()

        first = labels(42)
        # At even odds, a batch of one label has probability 2 * 2**-32.
        assert set(first.tolist()) == {0, 1}
        np.testing.assert_array_equal(labels(42), first)
        assert not np.array_equal(labels(7), first)
