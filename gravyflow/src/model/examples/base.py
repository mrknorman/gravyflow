"""
Base class for example model implementations.

All example models (George & Huerta 2017, Gabbard 2017, etc.) inherit from
ExampleModel to ensure a consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
from pathlib import Path

import keras
import gravyflow as gf


class ExampleModel(ABC):
    """
    Abstract base class for example GW detection models.
    
    All example models should inherit from this class and implement
    the required abstract methods to ensure a consistent interface.
    
    Required class attributes:
        config: Configuration dataclass for the model
    
    Required methods:
        model(): Create the neural network architecture
        dataset(): Create a training/validation dataset
        compile_model(): Compile the model with optimizer and loss
    
    Usage:
        >>> class MyModel(ExampleModel):
        ...     config = MyConfig()
        ...     
        ...     @classmethod
        ...     def model(cls, ...): ...
        ...     
        ...     @classmethod  
        ...     def dataset(cls, ...): ...
        ...     
        ...     @classmethod
        ...     def compile_model(cls, model): ...
    """
    
    # Subclasses must define their config
    config = None
    
    @classmethod
    @abstractmethod
    def model(
        cls,
        input_shape_onsource: Optional[Tuple[int, int]] = None,
        input_shape_offsource: Optional[Tuple[int, int]] = None,
        config = None,
        **kwargs
    ) -> keras.Model:
        """
        Create the neural network architecture.
        
        Args:
            input_shape_onsource: Shape of onsource input (samples, channels).
            input_shape_offsource: Shape of offsource input for whitening.
            config: Model configuration. Uses class default if None.
            **kwargs: Additional model-specific arguments.
        
        Returns:
            Uncompiled Keras model.
        """
        pass
    
    @classmethod
    @abstractmethod
    def dataset(
        cls,
        noise_type: gf.NoiseType = gf.NoiseType.COLORED,
        config = None,
        seed: Optional[int] = None,
        group: str = "train",
        steps_per_epoch: int = 1000,
        **kwargs
    ) -> gf.GravyflowDataset:
        """
        Create a dataset for training or evaluation.
        
        Args:
            noise_type: Type of noise to use.
            config: Dataset configuration. Uses class default if None.
            seed: Random seed for reproducibility.
            group: Dataset group ("train", "validate", "test").
            steps_per_epoch: Number of batches per epoch.
            **kwargs: Additional dataset-specific arguments.
        
        Returns:
            GravyflowDataset configured for this model.
        """
        pass
    
    @classmethod
    @abstractmethod
    def compile_model(
        cls,
        model: keras.Model,
        learning_rate: float = 0.001,
        **kwargs
    ) -> keras.Model:
        """
        Compile model with optimizer and loss function.
        
        Args:
            model: Keras model to compile.
            learning_rate: Learning rate for optimizer.
            **kwargs: Additional compilation arguments.
        
        Returns:
            Compiled model.
        """
        pass
    
    @classmethod
    def pretrained(cls, weights_path: Optional[Path] = None) -> keras.Model:
        """
        Load a pre-trained model.
        
        Default implementation loads from res/model/{model_name.lower()}.keras.
        Subclasses can override for custom behavior.
        
        Args:
            weights_path: Path to weights file. If None, uses default.
        
        Returns:
            Model with pre-trained weights.
        
        Raises:
            FileNotFoundError: If weights file not found.
        """
        if weights_path is None:
            model_name = cls.__name__.lower()
            weights_path = Path(gf.__file__).parent / "res" / "model" / f"{model_name}.keras"
        
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Pre-trained weights not found at {weights_path}. "
                f"Train the model first or download weights."
            )
        
        return keras.models.load_model(str(weights_path))


from dataclasses import dataclass
import numpy as np


@dataclass
class ValidationConfig:
    """
    Universal configuration for model validation datasets.
    
    This config is used across all models to ensure consistent
    validation benchmarks.
    """
    # Sampling parameters
    sample_rate_hertz: float = 2048.0
    onsource_duration_seconds: float = 1.0
    offsource_duration_seconds: float = 16.0
    
    # Detector configuration
    ifos: list = None  # Defaults to [gf.IFO.H1, gf.IFO.L1] if None
    
    # Mass distributions (broad BBH range)
    mass_1_min_msun: float = 5.0
    mass_1_max_msun: float = 95.0
    mass_2_min_msun: float = 5.0
    mass_2_max_msun: float = 95.0
    
    # SNR range for validation
    snr_min: float = 8.0
    snr_max: float = 20.0
    
    # Batch size
    batch_size: int = 32
    
    def __post_init__(self):
        if self.ifos is None:
            self.ifos = gf.IFO.L1  # Default to single detector



def validation_dataset_args(
    noise_type: gf.NoiseType = gf.NoiseType.COLORED,
    config: Optional[ValidationConfig] = None,
    seed: Optional[int] = None,
    **kwargs
) -> dict:
    """
    Create universal dataset arguments for model validation.
    
    This provides a consistent dataset configuration for validating
    all models, ensuring fair comparisons across architectures.
    
    Args:
        noise_type: Type of noise to use.
        config: Validation configuration. Uses defaults if None.
        seed: Random seed for reproducibility.
        **kwargs: Additional arguments passed to dataset.
    
    Returns:
        Dictionary of dataset arguments compatible with GravyflowDataset
        and Validator.validate().
    """
    cfg = config or ValidationConfig()
    
    # Mass distributions
    mass_1_distribution = gf.Distribution(
        min_=cfg.mass_1_min_msun,
        max_=cfg.mass_1_max_msun,
        type_=gf.DistributionType.LOG
    )
    
    mass_2_distribution = gf.Distribution(
        min_=cfg.mass_2_min_msun,
        max_=cfg.mass_2_max_msun,
        type_=gf.DistributionType.LOG
    )
    
    # Inclination distribution
    inclination_distribution = gf.Distribution(
        min_=0.0,
        max_=np.pi,
        type_=gf.DistributionType.UNIFORM
    )
    
    # SNR scaling
    scaling_method = gf.ScalingMethod(
        value=gf.Distribution(
            min_=cfg.snr_min,
            max_=cfg.snr_max,
            type_=gf.DistributionType.UNIFORM
        ),
        type_=gf.ScalingTypes.SNR
    )
    
    # Waveform generator
    waveform_generator = gf.CBCGenerator(
        mass_1_msun=mass_1_distribution,
        mass_2_msun=mass_2_distribution,
        inclination_radians=inclination_distribution,
        scaling_method=scaling_method,
        injection_chance=0.5,
        spin_1_in=(0.0, 0.0, 0.0),
        spin_2_in=(0.0, 0.0, 0.0),
    )
    
    # Noise configuration
    ifo_data_obtainer = None
    if noise_type == gf.NoiseType.REAL:
        ifo_data_obtainer = gf.IFODataObtainer(
            observing_runs=[gf.ObservingRun.O3],
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.NOISE]
        )

    noise_obtainer = gf.NoiseObtainer(
        noise_type=noise_type,
        ifos=cfg.ifos,
        ifo_data_obtainer=ifo_data_obtainer
    )
    
    # Build dataset args dict
    dataset_args = {
        "sample_rate_hertz": cfg.sample_rate_hertz,
        "onsource_duration_seconds": cfg.onsource_duration_seconds,
        "offsource_duration_seconds": cfg.offsource_duration_seconds,
        "noise_obtainer": noise_obtainer,
        "waveform_generators": waveform_generator,
        "num_examples_per_batch": cfg.batch_size,
        "seed": seed,
        "input_variables": [
            gf.ReturnVariables.ONSOURCE,
            gf.ReturnVariables.OFFSOURCE
        ],
        "output_variables": [
            gf.ReturnVariables.INJECTION_MASKS
        ],
        **kwargs
    }
    
    return dataset_args
