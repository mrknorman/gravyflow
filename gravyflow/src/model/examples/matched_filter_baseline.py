"""
Matched Filter Baseline Model

Wraps the MatchedFilter as an ExampleModel subclass for fair comparison 
with neural network detectors on identical datasets.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import keras
from keras.models import Model
from keras.layers import Input

import gravyflow as gf
from gravyflow.src.model.examples.base import ExampleModel
from gravyflow.src.detection.matched_filter import MatchedFilter, MatchedFilterLayer


@dataclass
class MatchedFilterBaselineConfig:
    """Configuration for MatchedFilterBaseline.
    
    Template Grid:
        - Mass ranges and grid density control template bank size
        - num_templates_per_dim=16 gives 136 templates
        - Increase for better coverage but slower inference
    
    Detection:
        - snr_threshold: Threshold for detection probability
        - temperature: Sigmoid sharpness (lower = sharper decision boundary)
    
    Dataset (for comparison with ML models):
        - Uses same CBC injection parameters as Gabbard2017
    """
    
    # Template grid parameters
    mass_1_range: Tuple[float, float] = (5.0, 95.0)
    mass_2_range: Tuple[float, float] = (5.0, 95.0)
    num_templates_per_dim: int = 16
    
    # Signal parameters
    sample_rate_hertz: float = 8192.0
    duration_seconds: float = 2.0
    f_low: float = 20.0
    
    # Detection parameters
    snr_threshold: float = 8.0
    temperature: float = 2.0
    
    # Dataset parameters (to match Gabbard2017 for comparison)
    injection_chance: float = 0.5
    snr_min: float = 10.0
    snr_max: float = 80.0
    batch_size: int = 32


class MatchedFilterBaseline(ExampleModel):
    """
    Matched Filter Baseline for Comparison with ML Models.
    
    Provides the traditional matched filtering approach wrapped as an ExampleModel,
    enabling fair comparison with neural network detectors on identical datasets.
    
    Key Differences from ML Models:
        - No training required (uses physics-based template bank)
        - `compile_model()` is a no-op
        - `pretrained()` returns model directly (no weights to load)
    
    Usage:
        >>> # Create model
        >>> model = MatchedFilterBaseline.model()
        >>> 
        >>> # Use with dataset (same as you would with Gabbard2017)
        >>> dataset = MatchedFilterBaseline.dataset()
        >>> probs = model.predict(dataset)
        >>> 
        >>> # Compare with ML model
        >>> ml_model = Gabbard2017.pretrained()
        >>> ml_probs = ml_model.predict(dataset)
    """
    
    config = MatchedFilterBaselineConfig()
    
    @classmethod
    def model(
        cls,
        input_shape_onsource: Optional[Tuple[int, int]] = None,
        input_shape_offsource: Optional[Tuple[int, int]] = None,
        config: Optional[MatchedFilterBaselineConfig] = None,
        **kwargs
    ) -> Model:
        """
        Create Keras model wrapping MatchedFilterLayer.
        
        Args:
            input_shape_onsource: Shape of onsource input (samples, channels).
            input_shape_offsource: Unused (no whitening layer needed).
            config: Model configuration. Uses class default if None.
        
        Returns:
            Keras model with MatchedFilterLayer.
        """
        if config is None:
            config = cls.config
        
        # Compute default input shape
        if input_shape_onsource is None:
            num_samples = int(config.sample_rate_hertz * config.duration_seconds)
            input_shape_onsource = (num_samples,)
        
        # Create input layer
        inputs = Input(shape=input_shape_onsource, name="onsource")
        
        # Create matched filter layer
        mf_layer = MatchedFilterLayer(
            mass_1_range=config.mass_1_range,
            mass_2_range=config.mass_2_range,
            num_templates_per_dim=config.num_templates_per_dim,
            sample_rate_hertz=config.sample_rate_hertz,
            duration_seconds=config.duration_seconds,
            f_low=config.f_low,
            snr_threshold=config.snr_threshold,
            temperature=config.temperature,
            name="matched_filter"
        )
        
        outputs = mf_layer(inputs)
        
        return Model(inputs=inputs, outputs=outputs, name="MatchedFilterBaseline")
    
    @classmethod
    def dataset(
        cls,
        noise_type: gf.NoiseType = gf.NoiseType.COLORED,
        config: Optional[MatchedFilterBaselineConfig] = None,
        seed: Optional[int] = None,
        group: str = "train",
        steps_per_epoch: int = 1000,
        **kwargs
    ) -> gf.GravyflowDataset:
        """
        Create dataset compatible with matched filter model.
        
        Uses similar CBC injection parameters as Gabbard2017 for fair comparison.
        
        Args:
            noise_type: Type of noise (COLORED, WHITE, REAL).
            config: Dataset configuration.
            seed: Random seed for reproducibility.
            group: Dataset group ("train", "validate", "test").
            steps_per_epoch: Number of batches per epoch.
        
        Returns:
            GravyflowDataset that outputs whitened onsource data.
        """
        if config is None:
            config = cls.config
        
        # CBC generator with same parameter space as Gabbard2017
        cbc_generator = gf.CBCGenerator(
            mass_1_msun=gf.Distribution(
                min_=config.mass_1_range[0],
                max_=config.mass_1_range[1],
                type_=gf.DistributionType.LOG
            ),
            mass_2_msun=gf.Distribution(
                min_=config.mass_2_range[0],
                max_=config.mass_2_range[1],
                type_=gf.DistributionType.LOG
            ),
            injection_chance=config.injection_chance,
            scaling_method=gf.ScalingMethod(
                value=gf.Distribution(
                    min_=config.snr_min,
                    max_=config.snr_max,
                    type_=gf.DistributionType.UNIFORM
                ),
                type_=gf.ScalingTypes.SNR
            ),
            network=[gf.IFO.L1],  # Single detector
        )
        
        # Setup noise
        noise_obtainer = gf.NoiseObtainer(
            noise_type=noise_type,
            ifos=[gf.IFO.L1]
        )
        
        # Dataset outputs whitened onsource for matched filter
        return gf.GravyflowDataset(
            noise_obtainer=noise_obtainer,
            sample_rate_hertz=config.sample_rate_hertz,
            onsource_duration_seconds=config.duration_seconds,
            offsource_duration_seconds=16.0,
            num_examples_per_batch=config.batch_size,
            waveform_generators=[cbc_generator],
            input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
            output_variables=[gf.ReturnVariables.INJECTION_MASKS],
            seed=seed,
            group=group,
            steps_per_epoch=steps_per_epoch,
            **kwargs
        )
    
    @classmethod
    def compile_model(
        cls,
        model: Model,
        learning_rate: float = 0.001,
        **kwargs
    ) -> Model:
        """
        No-op for matched filter (no training required).
        
        Returns the model unchanged. Included for API compatibility.
        """
        return model
    
    @classmethod
    def pretrained(cls, weights_path=None) -> Model:
        """
        Return model directly (no weights to load).
        
        Matched filters use physics-based templates, not learned weights.
        """
        return cls.model()
