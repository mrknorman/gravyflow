"""
Lightweight determinism test suite for gravyflow.

Tests that deterministic outputs are produced when seeds are properly set.
These tests verify the core tenet of gravyflow: reproducible results.

Key design principles:
- Each test runs the same operation TWICE with same seed and verifies identical output
- Tests are lightweight (no slow markers unless absolutely necessary)
- Tests cover all major RNG sources in the codebase
"""

import pytest
import numpy as np
from keras import ops
import random

import gravyflow as gf


class TestRandomSeedBehavior:
    """Test that set_random_seeds properly affects all RNG sources."""
    
    def test_set_random_seeds_affects_numpy(self):
        """Verify set_random_seeds affects numpy random state."""
        gf.set_random_seeds(42)
        val1 = np.random.rand()
        
        gf.set_random_seeds(42)
        val2 = np.random.rand()
        
        assert val1 == val2, "Numpy random should be deterministic with same seed"
    
    def test_set_random_seeds_affects_python_random(self):
        """Verify set_random_seeds affects Python's random module."""
        gf.set_random_seeds(42)
        val1 = random.random()
        
        gf.set_random_seeds(42)
        val2 = random.random()
        
        assert val1 == val2, "Python random should be deterministic with same seed"
    
    def test_set_random_seeds_different_seeds(self):
        """Verify different seeds produce different results."""
        gf.set_random_seeds(42)
        np_val1 = np.random.rand()
        py_val1 = random.random()
        
        gf.set_random_seeds(123)
        np_val2 = np.random.rand()
        py_val2 = random.random()
        
        assert np_val1 != np_val2, "Different seeds should produce different numpy values"
        assert py_val1 != py_val2, "Different seeds should produce different Python random values"


class TestComponentDefaultSeeds:
    """Test that components use gf.Defaults.seed when no seed is provided."""
    
    def test_network_uses_default_seed(self):
        """Verify Network uses gf.Defaults.seed when seed=None."""
        # Create two networks without explicit seed - should both use gf.Defaults.seed
        network1 = gf.Network([gf.IFO.L1])
        network2 = gf.Network([gf.IFO.L1])
        
        # Their RNGs should be seeded identically, so first random from each should match
        val1 = network1.rng.integers(1000000)
        val2 = network2.rng.integers(1000000)
        
        assert val1 == val2, "Networks without explicit seed should use gf.Defaults.seed"
    
    def test_distribution_uses_default_seed(self):
        """Verify Distribution uses gf.Defaults.seed when seed=None."""
        dist1 = gf.Distribution(type_=gf.DistributionType.UNIFORM, min_=0.0, max_=1.0)
        dist2 = gf.Distribution(type_=gf.DistributionType.UNIFORM, min_=0.0, max_=1.0)
        
        samples1 = dist1.sample(10)
        samples2 = dist2.sample(10)
        
        np.testing.assert_allclose(samples1, samples2,
            err_msg="Distributions without explicit seed should use gf.Defaults.seed")
    
    def test_injection_generator_uses_default_seed(self):
        """Verify InjectionGenerator uses gf.Defaults.seed when seed=None."""
        config_path = gf.tests.PATH / "example_injection_parameters/phenom_d_parameters.json"
        
        # First generator
        gen1 = gf.WaveformGenerator.load(config_path)
        gen1.injection_chance = 1.0
        inj_gen1 = gf.InjectionGenerator([gen1])
        
        # Second generator - should produce identical results
        gen2 = gf.WaveformGenerator.load(config_path)
        gen2.injection_chance = 1.0
        inj_gen2 = gf.InjectionGenerator([gen2])
        
        # First random value from each RNG should match
        val1 = inj_gen1.rng.integers(1000000)
        val2 = inj_gen2.rng.integers(1000000)
        
        assert val1 == val2, "InjectionGenerators without explicit seed should use gf.Defaults.seed"


class TestComponentDeterminism:
    """Test that components produce deterministic results with same seed."""
    
    def test_network_determinism_with_explicit_seed(self):
        """Verify Network produces deterministic results with same seed."""
        seed = 42
        
        network1 = gf.Network([gf.IFO.L1], seed=seed)
        loc1 = np.array(network1.location)
        
        network2 = gf.Network([gf.IFO.L1], seed=seed)
        loc2 = np.array(network2.location)
        
        np.testing.assert_allclose(loc1, loc2, 
            err_msg="Network locations should be deterministic with same seed")
    
    def test_distribution_determinism_with_explicit_seed(self):
        """Verify Distribution sampling is deterministic with same seed."""
        seed = 42
        
        dist1 = gf.Distribution(type_=gf.DistributionType.UNIFORM, min_=0.0, max_=1.0, seed=seed)
        samples1 = dist1.sample(100)
        
        dist2 = gf.Distribution(type_=gf.DistributionType.UNIFORM, min_=0.0, max_=1.0, seed=seed)
        samples2 = dist2.sample(100)
        
        np.testing.assert_allclose(samples1, samples2,
            err_msg="Distribution samples should be deterministic with same seed")
    
    def test_injection_generator_determinism_full(self):
        """Verify InjectionGenerator produces deterministic waveforms."""
        config_path = gf.tests.PATH / "example_injection_parameters/phenom_d_parameters.json"
        seed = gf.Defaults.seed
        
        # First run
        gf.set_random_seeds(seed)
        gen1 = gf.WaveformGenerator.load(config_path)
        gen1.injection_chance = 1.0
        inj_gen1 = gf.InjectionGenerator([gen1])
        injections1, _, _ = next(inj_gen1())
        
        # Second run
        gf.set_random_seeds(seed)
        gen2 = gf.WaveformGenerator.load(config_path)
        gen2.injection_chance = 1.0
        inj_gen2 = gf.InjectionGenerator([gen2])
        injections2, _, _ = next(inj_gen2())
        
        np.testing.assert_allclose(np.array(injections1), np.array(injections2),
            err_msg="InjectionGenerator should produce identical waveforms with same seed")


class TestFullPipelineDeterminism:
    """Test determinism through complete pipelines."""
    
    def test_projection_pipeline_determinism(self):
        """Verify the full injection -> projection pipeline is deterministic."""
        config_path = gf.tests.PATH / "example_injection_parameters/phenom_d_parameters.json"
        seed = gf.Defaults.seed
        
        # Run 1
        gf.set_random_seeds(seed)
        network1 = gf.Network([gf.IFO.L1], seed=seed)
        gen1 = gf.WaveformGenerator.load(config_path)
        gen1.injection_chance = 1.0
        inj_gen1 = gf.InjectionGenerator([gen1])
        injections1, _, _ = next(inj_gen1())
        proj1 = np.array(network1.project_wave(injections1[0]))
        
        # Run 2
        gf.set_random_seeds(seed)
        network2 = gf.Network([gf.IFO.L1], seed=seed)
        gen2 = gf.WaveformGenerator.load(config_path)
        gen2.injection_chance = 1.0
        inj_gen2 = gf.InjectionGenerator([gen2])
        injections2, _, _ = next(inj_gen2())
        proj2 = np.array(network2.project_wave(injections2[0]))
        
        np.testing.assert_allclose(proj1, proj2, atol=1e-6,
            err_msg="Full injection->projection pipeline should be deterministic")


class TestEdgeCases:
    """Test edge cases and potential sources of non-determinism."""
    
    def test_multiple_sequential_samples_deterministic(self):
        """Verify that multiple sequential samples are deterministic."""
        seed = 42
        
        # Run 1: Sample 5 times
        gf.set_random_seeds(seed)
        results1 = [np.random.rand() for _ in range(5)]
        
        # Run 2: Sample 5 times
        gf.set_random_seeds(seed)
        results2 = [np.random.rand() for _ in range(5)]
        
        assert results1 == results2, "Sequential samples should be deterministic"
    
    def test_interleaved_rng_usage_deterministic(self):
        """Verify determinism when multiple RNGs are used interleaved."""
        seed = gf.Defaults.seed
        
        # Run 1
        gf.set_random_seeds(seed)
        network1 = gf.Network([gf.IFO.L1], seed=seed)
        dist1 = gf.Distribution(type_=gf.DistributionType.UNIFORM, min_=0.0, max_=1.0, seed=seed)
        
        val_net1 = network1.rng.integers(1000000)
        val_dist1 = dist1.sample(1)[0]
        val_net1b = network1.rng.integers(1000000)
        
        # Run 2
        gf.set_random_seeds(seed)
        network2 = gf.Network([gf.IFO.L1], seed=seed)
        dist2 = gf.Distribution(type_=gf.DistributionType.UNIFORM, min_=0.0, max_=1.0, seed=seed)
        
        val_net2 = network2.rng.integers(1000000)
        val_dist2 = dist2.sample(1)[0]
        val_net2b = network2.rng.integers(1000000)
        
        assert val_net1 == val_net2, "First network RNG values should match"
        assert val_dist1 == val_dist2, "Distribution values should match"
        assert val_net1b == val_net2b, "Second network RNG values should match"
