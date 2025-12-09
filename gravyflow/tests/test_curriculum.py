"""
Tests for Curriculum Learning functionality.
"""

import pytest
import numpy as np

import gravyflow as gf


class TestCurriculumSchedule:
    """Tests for CurriculumSchedule enum."""
    
    def test_schedule_types_exist(self):
        """Verify all expected schedule types exist."""
        assert gf.CurriculumSchedule.LINEAR
        assert gf.CurriculumSchedule.EXPONENTIAL
        assert gf.CurriculumSchedule.STEP
        assert gf.CurriculumSchedule.COSINE
        assert gf.CurriculumSchedule.CUSTOM


class TestCurriculumInit:
    """Tests for Curriculum initialization."""
    
    def test_basic_init(self):
        """Test basic curriculum creation."""
        start = gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM)
        end = gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM)
        
        curriculum = gf.Curriculum(start=start, end=end, num_epochs=10)
        
        assert curriculum.start == start
        assert curriculum.end == end
        assert curriculum.num_epochs == 10
        assert curriculum.schedule == gf.CurriculumSchedule.LINEAR
        assert curriculum._configured == False
    
    def test_init_with_constant_distributions(self):
        """Test curriculum with CONSTANT distribution type."""
        start = gf.Distribution(value=100.0, type_=gf.DistributionType.CONSTANT)
        end = gf.Distribution(value=10.0, type_=gf.DistributionType.CONSTANT)
        
        curriculum = gf.Curriculum(start=start, end=end, num_epochs=5)
        
        assert curriculum.type_ == gf.DistributionType.CONSTANT
    
    def test_mismatched_distribution_types_raises(self):
        """Test that mismatched distribution types raise an error."""
        start = gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM)
        end = gf.Distribution(value=10.0, type_=gf.DistributionType.CONSTANT)
        
        with pytest.raises(ValueError, match="same type"):
            gf.Curriculum(start=start, end=end)
    
    def test_schedule_option(self):
        """Test different schedule options."""
        start = gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM)
        end = gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM)
        
        curriculum = gf.Curriculum(
            start=start, 
            end=end, 
            schedule=gf.CurriculumSchedule.EXPONENTIAL
        )
        
        assert curriculum.schedule == gf.CurriculumSchedule.EXPONENTIAL


class TestCurriculumConfiguration:
    """Tests for Curriculum configuration from dataset."""
    
    def test_configure_sets_steps_per_epoch(self):
        """Test that configure() sets steps_per_epoch."""
        start = gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM)
        end = gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM)
        
        curriculum = gf.Curriculum(start=start, end=end, num_epochs=10)
        
        assert curriculum._configured == False
        assert curriculum.steps_per_epoch is None
        
        curriculum.configure(steps_per_epoch=1000)
        
        assert curriculum._configured == True
        assert curriculum.steps_per_epoch == 1000
    
    def test_current_epoch_before_configuration(self):
        """Test current_epoch returns 0 when not configured."""
        start = gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM)
        end = gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM)
        
        curriculum = gf.Curriculum(start=start, end=end)
        
        assert curriculum.current_epoch == 0
    
    def test_current_epoch_calculation(self):
        """Test current_epoch calculation based on steps."""
        start = gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM)
        end = gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM)
        
        curriculum = gf.Curriculum(start=start, end=end, num_epochs=10)
        curriculum.configure(steps_per_epoch=100)
        
        # Initially at epoch 0
        assert curriculum.current_epoch == 0
        
        # Simulate 50 steps (still epoch 0)
        curriculum._current_step = 50
        assert curriculum.current_epoch == 0
        
        # Simulate 100 steps (epoch 1)
        curriculum._current_step = 100
        assert curriculum.current_epoch == 1
        
        # Simulate 250 steps (epoch 2)
        curriculum._current_step = 250
        assert curriculum.current_epoch == 2


class TestCurriculumProgress:
    """Tests for curriculum progress calculation."""
    
    def test_linear_schedule(self):
        """Test LINEAR schedule progress calculation."""
        start = gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM)
        end = gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM)
        
        curriculum = gf.Curriculum(
            start=start, end=end, 
            num_epochs=10,
            schedule=gf.CurriculumSchedule.LINEAR
        )
        curriculum.configure(steps_per_epoch=100)
        
        # Epoch 0 -> progress 0
        assert curriculum._get_progress() == 0.0
        
        # Epoch 4 (step 450 // 100 = 4) -> progress = 4/9 = 0.444
        curriculum._current_step = 450
        progress = curriculum._get_progress()
        assert 0.40 < progress < 0.50
        
        # Epoch 9 -> progress 1.0
        curriculum._current_step = 900
        assert curriculum._get_progress() == 1.0
    
    def test_step_schedule(self):
        """Test STEP schedule progress calculation."""
        start = gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM)
        end = gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM)
        
        curriculum = gf.Curriculum(
            start=start, end=end, 
            num_epochs=10,
            schedule=gf.CurriculumSchedule.STEP
        )
        curriculum.configure(steps_per_epoch=100)
        
        # Before midpoint -> 0
        curriculum._current_step = 400
        assert curriculum._get_progress() == 0.0
        
        # After midpoint -> 1
        curriculum._current_step = 500
        assert curriculum._get_progress() == 1.0
    
    def test_custom_schedule(self):
        """Test CUSTOM schedule with user function."""
        start = gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM)
        end = gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM)
        
        # Custom function: jump to 1.0 at epoch 5
        def custom_fn(epoch, num_epochs):
            return 1.0 if epoch >= 5 else 0.0
        
        curriculum = gf.Curriculum(
            start=start, end=end, 
            num_epochs=10,
            schedule=gf.CurriculumSchedule.CUSTOM,
            custom_schedule_fn=custom_fn
        )
        curriculum.configure(steps_per_epoch=100)
        
        # Epoch 3 -> 0
        curriculum._current_step = 300
        assert curriculum._get_progress() == 0.0
        
        # Epoch 5 -> 1
        curriculum._current_step = 500
        assert curriculum._get_progress() == 1.0
    
    def test_custom_schedule_without_function_raises(self):
        """Test CUSTOM schedule without function raises error."""
        start = gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM)
        end = gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM)
        
        curriculum = gf.Curriculum(
            start=start, end=end,
            schedule=gf.CurriculumSchedule.CUSTOM
        )
        curriculum.configure(steps_per_epoch=100)
        
        with pytest.raises(ValueError, match="custom_schedule_fn required"):
            curriculum._get_progress()


class TestCurriculumInterpolation:
    """Tests for value interpolation."""
    
    def test_min_max_interpolation(self):
        """Test min/max interpolation over epochs."""
        start = gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM)
        end = gf.Distribution(min_=0.0, max_=20.0, type_=gf.DistributionType.UNIFORM)
        
        curriculum = gf.Curriculum(
            start=start, end=end,
            num_epochs=10,
            schedule=gf.CurriculumSchedule.LINEAR
        )
        curriculum.configure(steps_per_epoch=100)
        
        # At epoch 0: should be start values
        assert curriculum.min_ == 80.0
        assert curriculum.max_ == 100.0
        
        # At epoch 4.5 (half-way): should be interpolated
        curriculum._current_step = 450
        assert 35.0 < curriculum.min_ < 45.0
        assert 55.0 < curriculum.max_ < 65.0
        
        # At epoch 9 (end): should be end values
        curriculum._current_step = 900
        assert curriculum.min_ == 0.0
        assert curriculum.max_ == 20.0
    
    def test_constant_value_interpolation(self):
        """Test constant value interpolation."""
        start = gf.Distribution(value=100.0, type_=gf.DistributionType.CONSTANT)
        end = gf.Distribution(value=10.0, type_=gf.DistributionType.CONSTANT)
        
        curriculum = gf.Curriculum(
            start=start, end=end,
            num_epochs=10,
            schedule=gf.CurriculumSchedule.LINEAR
        )
        curriculum.configure(steps_per_epoch=100)
        
        # At epoch 0
        assert curriculum.value == 100.0
        
        # At epoch 9
        curriculum._current_step = 900
        assert curriculum.value == 10.0


class TestCurriculumSampling:
    """Tests for Curriculum sampling."""
    
    def test_sample_returns_array_like(self):
        """Test that sample() returns an array-like object."""
        start = gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM)
        end = gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM)
        
        curriculum = gf.Curriculum(start=start, end=end)
        curriculum.configure(steps_per_epoch=100)
        
        samples = curriculum.sample(10)
        
        # Distribution.sample() returns numpy array or list
        assert hasattr(samples, '__len__')
        assert len(samples) == 10
    
    def test_sample_advances_step(self):
        """Test that sample() advances the step counter."""
        start = gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM)
        end = gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM)
        
        curriculum = gf.Curriculum(start=start, end=end)
        curriculum.configure(steps_per_epoch=100)
        
        assert curriculum._current_step == 0
        
        curriculum.sample(5)
        assert curriculum._current_step == 1
        
        curriculum.sample(5)
        assert curriculum._current_step == 2
    
    def test_samples_in_expected_range(self):
        """Test that samples are within interpolated min/max range."""
        start = gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM)
        end = gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM)
        
        curriculum = gf.Curriculum(start=start, end=end, num_epochs=10)
        curriculum.configure(steps_per_epoch=100)
        
        # At epoch 0, samples should be in [80, 100]
        samples = curriculum.sample(100)
        assert all(80.0 <= s <= 100.0 for s in samples)
        
        # Move to end (epoch 9)
        curriculum._current_step = 899  # Will be 900 after next sample
        
        # At epoch 9, samples should be in [5, 15]
        for _ in range(5):
            curriculum._current_step = 900  # Reset to epoch 9
            samples = curriculum.sample(100)
            assert all(5.0 <= s <= 15.0 for s in samples), f"Got samples outside range: {min(samples)}, {max(samples)}"
            curriculum._current_step = 900  # Keep at epoch 9
    
    def test_reset(self):
        """Test that reset() clears the step counter."""
        start = gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM)
        end = gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM)
        
        curriculum = gf.Curriculum(start=start, end=end)
        curriculum.configure(steps_per_epoch=100)
        
        curriculum._current_step = 500
        assert curriculum._current_step == 500
        
        curriculum.reset()
        assert curriculum._current_step == 0


class TestCurriculumDatasetIntegration:
    """Tests for Curriculum integration with GravyflowDataset."""
    
    def test_dataset_configures_curriculum(self):
        """Test that GravyflowDataset auto-configures Curriculum."""
        start = gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM)
        end = gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM)
        
        curriculum = gf.Curriculum(start=start, end=end, num_epochs=10)
        
        scaling_method = gf.ScalingMethod(
            value=curriculum,
            type_=gf.ScalingTypes.SNR
        )
        
        waveform_gen = gf.WNBGenerator(
            duration_seconds=gf.Distribution(value=0.1, type_=gf.DistributionType.CONSTANT),
            min_frequency_hertz=gf.Distribution(value=100.0, type_=gf.DistributionType.CONSTANT),
            max_frequency_hertz=gf.Distribution(value=200.0, type_=gf.DistributionType.CONSTANT),
            scaling_method=scaling_method
        )
        
        # Curriculum should not be configured yet
        assert curriculum._configured == False
        
        # Create noise obtainer with simulated noise
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        
        # Create dataset - should auto-configure curriculum
        dataset = gf.GravyflowDataset(
            noise_obtainer=noise_obtainer,
            waveform_generators=waveform_gen,
            steps_per_epoch=500,
            num_examples_per_batch=4,
            sample_rate_hertz=1024,
            onsource_duration_seconds=1.0,
            offsource_duration_seconds=2.0,
            input_variables=[gf.ReturnVariables.ONSOURCE],
            output_variables=[gf.ReturnVariables.INJECTION_MASKS]
        )
        
        # Now curriculum should be configured
        assert curriculum._configured == True
        assert curriculum.steps_per_epoch == 500
    
    def test_curriculum_progresses_during_iteration(self):
        """Test that curriculum progresses as dataset is iterated."""
        start = gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM)
        end = gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM)
        
        curriculum = gf.Curriculum(start=start, end=end, num_epochs=10)
        
        scaling_method = gf.ScalingMethod(
            value=curriculum,
            type_=gf.ScalingTypes.SNR
        )
        
        waveform_gen = gf.WNBGenerator(
            duration_seconds=gf.Distribution(value=0.1, type_=gf.DistributionType.CONSTANT),
            min_frequency_hertz=gf.Distribution(value=100.0, type_=gf.DistributionType.CONSTANT),
            max_frequency_hertz=gf.Distribution(value=200.0, type_=gf.DistributionType.CONSTANT),
            scaling_method=scaling_method
        )
        
        # Create noise obtainer with simulated noise
        noise_obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        
        dataset = gf.GravyflowDataset(
            noise_obtainer=noise_obtainer,
            waveform_generators=waveform_gen,
            steps_per_epoch=10,  # Small for quick test
            num_examples_per_batch=2,
            sample_rate_hertz=1024,
            onsource_duration_seconds=1.0,
            offsource_duration_seconds=2.0,
            input_variables=[gf.ReturnVariables.ONSOURCE],
            output_variables=[gf.ReturnVariables.INJECTION_MASKS]
        )
        
        initial_step = curriculum._current_step
        
        # Get a batch - this should call sample() and advance step
        _ = dataset[0]
        
        # Step should have advanced
        assert curriculum._current_step > initial_step
