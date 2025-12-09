"""
Curriculum Learning for Gravitational Wave Detection Training.

Provides dynamic parameter schedules that adjust during training,
enabling curriculum learning strategies like the gradual noise increase
scheme from George & Huerta 2017.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union
from enum import Enum, auto

import numpy as np
from numpy.random import default_rng

import gravyflow as gf


class CurriculumSchedule(Enum):
    """Built-in schedule types for curriculum learning."""
    LINEAR = auto()       # Linear interpolation between start and end
    EXPONENTIAL = auto()  # Exponential decay (faster change initially)
    STEP = auto()         # Discrete step at midpoint
    COSINE = auto()       # Smooth cosine annealing
    CUSTOM = auto()       # User-provided function


@dataclass
class Curriculum:
    """
    Curriculum learning schedule for dynamic parameter adjustment.
    
    Wraps a Distribution and modifies its parameters based on training progress.
    Can be used anywhere a Distribution is accepted (e.g., ScalingMethod.value).
    
    The Curriculum automatically tracks progress by counting `.sample()` calls.
    When passed to a GravyflowDataset, it auto-configures `steps_per_epoch`
    to correctly compute epoch boundaries.
    
    Example - George & Huerta 2017 style (SNR decreasing from 100 to 5-15):
    
        >>> curriculum = gf.Curriculum(
        ...     start=gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM),
        ...     end=gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM),
        ...     num_epochs=20
        ... )
        >>> 
        >>> scaling_method = gf.ScalingMethod(
        ...     value=curriculum,
        ...     type_=gf.ScalingTypes.SNR
        ... )
        >>> 
        >>> waveform_gen = gf.CBCGenerator(..., scaling_method=scaling_method)
        >>> 
        >>> # Dataset auto-configures the curriculum with steps_per_epoch
        >>> dataset = gf.GravyflowDataset(
        ...     waveform_generators=waveform_gen,
        ...     steps_per_epoch=7000,
        ...     ...
        ... )
    
    Attributes:
        start: Initial distribution (at epoch 0).
        end: Final distribution (at num_epochs).
        num_epochs: Number of epochs over which to transition.
        schedule: Type of interpolation schedule.
        custom_schedule_fn: Optional function (epoch, num_epochs) -> float (0-1).
        seed: Random seed for sampling.
    """
    
    start: 'gf.Distribution'
    end: 'gf.Distribution'
    num_epochs: int = 10
    schedule: CurriculumSchedule = CurriculumSchedule.LINEAR
    custom_schedule_fn: Optional[Callable[[int, int], float]] = field(default=None, repr=False)
    seed: Optional[int] = None
    verbose: bool = False  # Print progress at epoch boundaries
    name: str = "Curriculum"  # Name for verbose output
    
    # Auto-configured by dataset
    steps_per_epoch: Optional[int] = field(default=None, repr=False)
    _current_step: int = field(default=0, repr=False)
    _configured: bool = field(default=False, repr=False)
    _last_printed_epoch: int = field(default=-1, repr=False)
    
    def __post_init__(self):
        # Inherit seed from start distribution if not provided
        if self.seed is None and hasattr(self.start, 'seed') and self.start.seed is not None:
            self.seed = self.start.seed
        elif self.seed is None:
            self.seed = gf.Defaults.seed
            
        self.rng = default_rng(self.seed)
        
        # Validate that start and end have compatible types
        if self.start.type_ != self.end.type_:
            raise ValueError(
                f"Start and end distributions must have same type. "
                f"Got {self.start.type_} and {self.end.type_}"
            )
    
    def configure(self, steps_per_epoch: int):
        """
        Configure the curriculum with dataset parameters.
        
        Called automatically by GravyflowDataset during initialization.
        
        Args:
            steps_per_epoch: Number of batches per epoch.
        """
        self.steps_per_epoch = steps_per_epoch
        self._configured = True
    
    def reset(self):
        """Reset progress counter (e.g., for a new training run)."""
        self._current_step = 0
    
    @property
    def current_epoch(self) -> int:
        """Current epoch based on step count."""
        if not self._configured or self.steps_per_epoch is None or self.steps_per_epoch == 0:
            return 0
        return self._current_step // self.steps_per_epoch
    
    @property
    def type_(self):
        """Distribution type (inherited from start)."""
        return self.start.type_
    
    def _get_progress(self) -> float:
        """
        Get interpolation progress (0.0 to 1.0) based on current epoch and schedule.
        """
        if not self._configured or self.num_epochs <= 1:
            return 0.0
        
        # Raw progress 0 to 1 based on epochs
        t = min(self.current_epoch / (self.num_epochs - 1), 1.0)
        
        match self.schedule:
            case CurriculumSchedule.LINEAR:
                return t
            
            case CurriculumSchedule.EXPONENTIAL:
                # Exponential: faster change initially
                # 1 - exp(-5*t) gives values from 0 to ~0.993
                return 1.0 - np.exp(-5.0 * t)
            
            case CurriculumSchedule.COSINE:
                # Cosine annealing: smooth S-curve
                return 0.5 * (1.0 - np.cos(np.pi * t))
            
            case CurriculumSchedule.STEP:
                # Discrete: 0 until halfway, then 1
                return 0.0 if t < 0.5 else 1.0
            
            case CurriculumSchedule.CUSTOM:
                if self.custom_schedule_fn is None:
                    raise ValueError("custom_schedule_fn required for CUSTOM schedule")
                return self.custom_schedule_fn(self.current_epoch, self.num_epochs)
            
            case _:
                return t
    
    def _interpolate(self, start_val: Optional[float], end_val: Optional[float]) -> Optional[float]:
        """Interpolate between start and end values based on progress."""
        if start_val is None or end_val is None:
            return start_val
        progress = self._get_progress()
        return start_val + progress * (end_val - start_val)
    
    @property
    def min_(self) -> Optional[float]:
        """Current minimum value (interpolated)."""
        return self._interpolate(self.start.min_, self.end.min_)
    
    @property
    def max_(self) -> Optional[float]:
        """Current maximum value (interpolated)."""
        return self._interpolate(self.start.max_, self.end.max_)
    
    @property
    def value(self) -> Optional[float]:
        """Current constant value (interpolated, for CONSTANT distributions)."""
        return self._interpolate(self.start.value, self.end.value)
    
    @property
    def mean(self) -> Optional[float]:
        """Current mean value (interpolated, for NORMAL distributions)."""
        return self._interpolate(self.start.mean, self.end.mean)
    
    @property
    def std(self) -> Optional[float]:
        """Current std value (interpolated, for NORMAL distributions)."""
        return self._interpolate(self.start.std, self.end.std)
    
    def sample(self, num_samples: int = 1) -> List[Union[int, float]]:
        """
        Sample from the current interpolated distribution.
        
        Creates an ephemeral Distribution with interpolated parameters
        and samples from it. Automatically advances the step counter.
        
        Args:
            num_samples: Number of samples to draw.
            
        Returns:
            List of sampled values.
        """
        # Create interpolated distribution based on current progress
        current_dist = gf.Distribution(
            min_=self.min_,
            max_=self.max_,
            value=self.value,
            mean=self.mean,
            std=self.std,
            type_=self.start.type_,
            possible_values=self.start.possible_values,
            dtype=self.start.dtype,
            seed=self.rng.integers(2**31)  # Use fresh seed each sample
        )
        
        samples = current_dist.sample(num_samples)
        
        # Auto-advance step counter
        self._current_step += 1
        
        # Auto-print on epoch change if verbose
        if self.verbose and self._configured:
            current_epoch = self.current_epoch
            if current_epoch != self._last_printed_epoch:
                self._last_printed_epoch = current_epoch
                progress = self._get_progress()
                
                # Build range string
                if self.min_ is not None and self.max_ is not None:
                    range_str = f"{self.min_:.1f} → {self.max_:.1f}"
                elif self.value is not None:
                    range_str = f"{self.value:.1f}"
                else:
                    range_str = "N/A"
                
                # Build progress bar (20 chars wide)
                bar_width = 20
                filled = int(progress * bar_width)
                bar = "█" * filled + "░" * (bar_width - filled)
                
                # Print with aligned formatting
                print(f"  📊 {self.name} │ Epoch {current_epoch + 1:2d}/{self.num_epochs} │ [{bar}] {progress*100:5.1f}% │ {range_str}")
        
        return samples
    
    def __repr__(self) -> str:
        return (
            f"Curriculum(start={self.start}, end={self.end}, "
            f"num_epochs={self.num_epochs}, schedule={self.schedule.name}, "
            f"current_epoch={self.current_epoch}, configured={self._configured})"
        )


class CurriculumProgressCallback:
    """
    Keras callback to print curriculum progress at epoch boundaries.
    
    Usage:
        >>> curriculum = gf.Curriculum(...)
        >>> callback = gf.CurriculumProgressCallback(curriculum, name="SNR")
        >>> model.fit(dataset, epochs=20, callbacks=[callback])
    
    Output example:
        Epoch 1: SNR range [80.0, 100.0] (progress: 0.0%)
        Epoch 2: SNR range [76.1, 95.5] (progress: 5.3%)
        ...
    """
    
    def __init__(
        self, 
        curriculum: Curriculum, 
        name: str = "Curriculum",
        verbose: bool = True
    ):
        """
        Args:
            curriculum: The Curriculum object to monitor.
            name: Name to display in log messages (e.g., "SNR").
            verbose: Whether to print progress messages.
        """
        self.curriculum = curriculum
        self.name = name
        self.verbose = verbose
        self._last_epoch = -1
    
    def set_model(self, model):
        """Called by Keras when the callback is attached to a model."""
        pass
    
    def set_params(self, params):
        """Called by Keras with training parameters."""
        pass
    
    def on_epoch_begin(self, epoch: int, logs=None):
        """Print curriculum progress at the start of each epoch."""
        if not self.verbose:
            return
        
        # Only print if epoch changed (avoid duplicate prints)
        if epoch == self._last_epoch:
            return
        self._last_epoch = epoch
        
        progress = self.curriculum._get_progress() * 100
        
        # Format based on distribution type
        if self.curriculum.min_ is not None and self.curriculum.max_ is not None:
            range_str = f"[{self.curriculum.min_:.1f}, {self.curriculum.max_:.1f}]"
        elif self.curriculum.value is not None:
            range_str = f"{self.curriculum.value:.1f}"
        elif self.curriculum.mean is not None:
            range_str = f"μ={self.curriculum.mean:.1f}, σ={self.curriculum.std:.1f}"
        else:
            range_str = "N/A"
        
        print(f"Epoch {epoch + 1}: {self.name} range {range_str} (progress: {progress:.1f}%)")
    
    def on_epoch_end(self, epoch: int, logs=None):
        """Called at the end of each epoch."""
        pass
    
    def on_train_begin(self, logs=None):
        """Called at the start of training."""
        if self.verbose:
            print(f"\n{self.name} Curriculum: {self.curriculum.num_epochs} epochs, "
                  f"{self.curriculum.schedule.name} schedule")
            print(f"  Start: [{self.curriculum.start.min_}, {self.curriculum.start.max_}]")
            print(f"  End:   [{self.curriculum.end.min_}, {self.curriculum.end.max_}]")
            print()
    
    def on_train_end(self, logs=None):
        """Called at the end of training."""
        pass
    
    def on_batch_begin(self, batch: int, logs=None):
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, batch: int, logs=None):
        """Called at the end of each batch."""
        pass

