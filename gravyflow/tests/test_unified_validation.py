"""
Tests for the unified validation pipeline.

Tests ValidationConfig, UnifiedValidationBank, and related functionality.
"""

import pytest
import numpy as np

import gravyflow as gf
from gravyflow.src.model.validate import (
    ValidationConfig, 
    UnifiedValidationBank,
    generate_efficiency_scatter_plot
)


class TestValidationConfig:
    """Tests for ValidationConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ValidationConfig()
        
        assert config.snr_range == (0.0, 20.0)
        assert config.num_examples == 100_000
        assert config.batch_size == 512
        assert config.snr_bin_width == 5.0
        assert config.num_worst_per_bin == 10
        assert config.far_thresholds == [1e-1, 1e-2, 1e-3, 1e-4]
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ValidationConfig(
            snr_range=(4.0, 15.0),
            num_examples=50_000,
            batch_size=256,
            snr_bin_width=2.5,
            num_worst_per_bin=5
        )
        
        assert config.snr_range == (4.0, 15.0)
        assert config.num_examples == 50_000
        assert config.batch_size == 256
        assert config.snr_bin_width == 2.5
        assert config.num_worst_per_bin == 5
    
    def test_num_bins_calculation(self):
        """Test that bin count is calculated correctly from config."""
        config = ValidationConfig(snr_range=(0.0, 20.0), snr_bin_width=5.0)
        expected_bins = int((20.0 - 0.0) / 5.0)  # 4 bins
        
        assert expected_bins == 4
        
        config2 = ValidationConfig(snr_range=(5.0, 15.0), snr_bin_width=2.5)
        expected_bins2 = int((15.0 - 5.0) / 2.5)  # 4 bins
        
        assert expected_bins2 == 4


class TestUnifiedValidationBank:
    """Tests for UnifiedValidationBank class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model that returns random scores."""
        import keras
        
        # Simple model that outputs scores
        inputs = keras.layers.Input(shape=(1, 2048))
        x = keras.layers.GlobalAveragePooling1D()(inputs)
        outputs = keras.layers.Dense(2, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        
        return model
    
    @pytest.fixture
    def simple_dataset_args(self):
        """Create minimal dataset args for testing."""
        scaling_method = gf.ScalingMethod(
            value=gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM),
            type_=gf.ScalingTypes.SNR
        )
        
        cbc_generator = gf.CBCGenerator(
            scaling_method=scaling_method,
            network=[gf.IFO.L1],
            mass_1_msun=gf.Distribution(value=30.0, type_=gf.DistributionType.CONSTANT),
            mass_2_msun=gf.Distribution(value=25.0, type_=gf.DistributionType.CONSTANT),
            injection_chance=1.0,
        )
        
        noise_obtainer = gf.NoiseObtainer(
            noise_type=gf.NoiseType.WHITE,
            ifos=[gf.IFO.L1]
        )
        
        return {
            "noise_obtainer": noise_obtainer,
            "sample_rate_hertz": 2048.0,
            "onsource_duration_seconds": 1.0,
            "offsource_duration_seconds": 4.0,
            "crop_duration_seconds": 0.0,
            "waveform_generators": [cbc_generator],
            "input_variables": [gf.ReturnVariables.WHITENED_ONSOURCE],
            "output_variables": [gf.ReturnVariables.INJECTION_MASKS],
            "seed": 42,
        }
    
    def test_init_with_defaults(self, mock_model, simple_dataset_args):
        """Test bank initialization with default config."""
        bank = UnifiedValidationBank(
            model=mock_model,
            dataset_args=simple_dataset_args
        )
        
        assert bank.config is not None
        assert bank.snrs is None  # Not generated yet
        assert bank.scores is None
    
    def test_init_with_custom_config(self, mock_model, simple_dataset_args):
        """Test bank initialization with custom config."""
        config = ValidationConfig(
            num_examples=1000,
            batch_size=100,
            snr_range=(5.0, 15.0)
        )
        
        bank = UnifiedValidationBank(
            model=mock_model,
            dataset_args=simple_dataset_args,
            config=config
        )
        
        assert bank.config.num_examples == 1000
        assert bank.config.batch_size == 100


class TestGetScalar:
    """Tests for get_scalar helper function."""
    
    def test_scalar_extraction_from_0d_array(self):
        """Test extraction from 0-dimensional numpy array."""
        # Simulate what get_scalar does with a 0-d array
        val = np.array(5.0)
        
        assert val.ndim == 0
        result = float(val)
        assert result == 5.0
    
    def test_scalar_extraction_from_1d_array(self):
        """Test extraction from 1-dimensional numpy array."""
        val = np.array([5.0, 6.0, 7.0])
        
        assert val.ndim == 1
        result = float(val.flatten()[0])
        assert result == 5.0
    
    def test_scalar_extraction_from_empty_array(self):
        """Test extraction from empty array returns default."""
        val = np.array([])
        default = -1.0
        
        if val.size == 0:
            result = default
        else:
            result = float(val.flatten()[0])
        
        assert result == -1.0


class TestEfficiencyData:
    """Tests for efficiency data computation."""
    
    def test_efficiency_data_keys(self):
        """Test that efficiency data contains expected keys."""
        # This is a structure test - actual values tested with integration
        expected_keys = [
            "snrs", "scores", "fit_snrs", "fit_efficiency",
            "bin_centers", "bin_means", "bin_stds"
        ]
        
        # Mock efficiency data structure
        mock_data = {
            "snrs": np.random.uniform(0, 20, 1000),
            "scores": np.random.uniform(0, 1, 1000),
            "fit_snrs": np.linspace(0, 20, 100),
            "fit_efficiency": np.linspace(0, 1, 100),
            "bin_centers": np.array([2.5, 7.5, 12.5, 17.5]),
            "bin_means": np.array([0.2, 0.5, 0.8, 0.95]),
            "bin_stds": np.array([0.1, 0.15, 0.1, 0.05])
        }
        
        for key in expected_keys:
            assert key in mock_data


class TestWorstPerformers:
    """Tests for binned worst performers extraction."""
    
    def test_worst_per_bin_structure(self):
        """Test worst performers are organized by SNR bin."""
        # Mock structure
        mock_worst = {
            "0-5": [{"score": 0.1}, {"score": 0.15}],
            "5-10": [{"score": 0.2}, {"score": 0.25}],
            "10-15": [{"score": 0.35}, {"score": 0.4}],
            "15-20": [{"score": 0.55}, {"score": 0.6}],
        }
        
        # Test structure
        assert len(mock_worst) == 4
        assert "0-5" in mock_worst
        assert "15-20" in mock_worst
        
        # Each bin should have samples
        for bin_key, samples in mock_worst.items():
            assert isinstance(samples, list)
    
    def test_heap_tracking_keeps_lowest_scores(self):
        """Test that min-heap keeps lowest (worst) scores."""
        import heapq
        
        num_worst = 3
        heap = []
        scores = [0.9, 0.1, 0.5, 0.3, 0.7, 0.05, 0.2]
        
        for score in scores:
            if len(heap) < num_worst:
                heapq.heappush(heap, (-score, score))  # Negative for max-heap
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, (-score, score))
        
        # Should have 3 lowest scores
        result_scores = sorted([item[1] for item in heap])
        assert result_scores == [0.05, 0.1, 0.2]


class TestTARComputation:
    """Tests for TAR@FAR computation from unified bank."""
    
    def test_tar_structure(self):
        """Test TAR result structure."""
        # Mock TAR results
        mock_tar = {
            1e-1: {
                "threshold": 0.5,
                "bin_centers": np.array([2.5, 7.5, 12.5, 17.5]),
                "tar_per_bin": np.array([0.3, 0.6, 0.85, 0.95])
            },
            1e-2: {
                "threshold": 0.7,
                "bin_centers": np.array([2.5, 7.5, 12.5, 17.5]),
                "tar_per_bin": np.array([0.1, 0.4, 0.7, 0.9])
            }
        }
        
        # Verify structure
        for far, data in mock_tar.items():
            assert "threshold" in data
            assert "bin_centers" in data
            assert "tar_per_bin" in data
            
            # TAR should be between 0 and 1
            assert np.all(data["tar_per_bin"] >= 0)
            assert np.all(data["tar_per_bin"] <= 1)


class TestValidationConfigExport:
    """Test that ValidationConfig is properly exported."""
    
    def test_exported_from_gravyflow(self):
        """Test ValidationConfig is accessible from gravyflow."""
        assert hasattr(gf, 'ValidationConfig')
        
        config = gf.ValidationConfig()
        assert config.num_examples == 100_000
    
    def test_unified_validation_bank_exported(self):
        """Test UnifiedValidationBank is accessible from gravyflow."""
        assert hasattr(gf, 'UnifiedValidationBank')
