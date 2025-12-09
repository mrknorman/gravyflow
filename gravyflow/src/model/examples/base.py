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
