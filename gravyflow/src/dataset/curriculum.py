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

import gravyflow as gf
from gravyflow.src.rng.legacy import warn_deprecated


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
    Curriculum learning schedule: a distribution that moves with the epoch.

    ``at(epoch)`` is the Distribution for one training epoch, interpolated
    from ``start`` (epoch 0) to ``end`` (epoch ``num_epochs - 1`` and after)
    by ``schedule``; ``at(None)`` is ``end``. It holds no counter and no
    random state, so a validation call cannot move training's schedule.

    Use it as a ScalingMethod's value. A dataset evaluates it at its own
    epoch: the training split at the current epoch (Keras', under
    ``model.fit`` with ``ds.epoch_callback()``), the validation and test
    splits at the end, so their metrics stay comparable across epochs.

    Example - George & Huerta 2017 style (SNR decreasing from 100 to 5-15):

        >>> curriculum = gf.Curriculum(
        ...     start=gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM),
        ...     end=gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM),
        ...     num_epochs=20
        ... )
        >>> scaling_method = gf.ScalingMethod(value=curriculum, type_=gf.ScalingTypes.SNR)
        >>> waveform_gen = gf.CBCGenerator(..., scaling_method=scaling_method)
        >>> dataset = gf.GravyflowDataset(waveform_generators=waveform_gen, ...)

    Attributes:
        start: Initial distribution (at epoch 0).
        end: Final distribution (at num_epochs - 1 and after).
        num_epochs: Number of epochs over which to transition.
        schedule: Type of interpolation schedule.
        custom_schedule_fn: Optional function (epoch, num_epochs) -> float (0-1).
        seed: Deprecated; the dataset's seed tree draws the samples.
    """
    start: 'gf.Distribution'
    end: 'gf.Distribution'
    num_epochs: int = 10
    schedule: CurriculumSchedule = CurriculumSchedule.LINEAR
    custom_schedule_fn: Optional[Callable[[int, int], float]] = field(default=None, repr=False)
    seed: Optional[int] = None  # Deprecated.
    verbose: bool = False  # Deprecated: use CurriculumProgressCallback.
    name: str = "Curriculum"  # Name for CurriculumProgressCallback's output

    def __post_init__(self):
        if self.seed is not None:
            warn_deprecated("Curriculum(seed=)", "the dataset's seed tree draws the samples")
        if self.verbose:
            warn_deprecated("Curriculum(verbose=)", "print progress with gf.CurriculumProgressCallback")
        if self.start.type_ != self.end.type_:
            raise ValueError(
                f"Start and end distributions must have same type. "
                f"Got {self.start.type_} and {self.end.type_}"
            )

    def configure(self, steps_per_epoch: int):
        warn_deprecated("Curriculum.configure", "a dataset passes its epoch to Curriculum.at")

    def reset(self):
        warn_deprecated("Curriculum.reset", "a Curriculum holds no progress; use at(epoch)")

    @property
    def type_(self):
        """Distribution type (inherited from start)."""
        return self.start.type_

    def progress(self, epoch: Optional[int]) -> float:
        """The schedule's progress, 0 to 1, at ``epoch``; None is the end (1)."""
        if epoch is None:
            return 1.0
        if self.num_epochs <= 1:
            return 0.0
        t = min(max(epoch, 0) / (self.num_epochs - 1), 1.0)

        match self.schedule:
            case CurriculumSchedule.LINEAR:
                return t
            case CurriculumSchedule.EXPONENTIAL:
                # Faster change initially: 1 - exp(-5 t) runs from 0 to ~0.993.
                return 1.0 - np.exp(-5.0 * t)
            case CurriculumSchedule.COSINE:
                # Cosine annealing: a smooth S-curve.
                return 0.5 * (1.0 - np.cos(np.pi * t))
            case CurriculumSchedule.STEP:
                # Discrete: 0 until halfway, then 1.
                return 0.0 if t < 0.5 else 1.0
            case CurriculumSchedule.CUSTOM:
                if self.custom_schedule_fn is None:
                    raise ValueError("custom_schedule_fn required for CUSTOM schedule")
                return self.custom_schedule_fn(epoch, self.num_epochs)
            case _:
                return t

    def at(self, epoch: Optional[int]) -> 'gf.Distribution':
        """The Distribution at ``epoch`` (None: the end distribution)."""
        progress = self.progress(epoch)

        def interpolate(start_val, end_val):
            if start_val is None or end_val is None:
                return start_val
            return start_val + progress * (end_val - start_val)

        return gf.Distribution(
            min_=interpolate(self.start.min_, self.end.min_),
            max_=interpolate(self.start.max_, self.end.max_),
            value=interpolate(self.start.value, self.end.value),
            mean=interpolate(self.start.mean, self.end.mean),
            std=interpolate(self.start.std, self.end.std),
            type_=self.start.type_,
            possible_values=self.start.possible_values,
            dtype=self.start.dtype,
        )

    def sample(self, num_samples: int = 1, rng=None, epoch: Optional[int] = None):
        """``num_samples`` values from ``at(epoch)``, drawn from ``rng``."""
        return self.at(epoch).sample(num_samples, rng)

    @staticmethod
    def resolve(value, epoch: Optional[int]):
        """``value.at(epoch)`` for a Curriculum; any other value unchanged."""
        return value.at(epoch) if isinstance(value, Curriculum) else value


class CurriculumProgressCallback:
    """
    Keras callback to print curriculum progress at epoch boundaries.

    Usage:
        >>> curriculum = gf.Curriculum(...)
        >>> callback = gf.CurriculumProgressCallback(curriculum, name="SNR")
        >>> model.fit(dataset, epochs=20, callbacks=[dataset.epoch_callback(), callback])

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
        if not self.verbose or epoch == self._last_epoch:
            return
        self._last_epoch = epoch

        progress = self.curriculum.progress(epoch) * 100
        current = self.curriculum.at(epoch)
        if current.min_ is not None and current.max_ is not None:
            range_str = f"[{current.min_:.1f}, {current.max_:.1f}]"
        elif current.value is not None:
            range_str = f"{current.value:.1f}"
        elif current.mean is not None:
            range_str = f"μ={current.mean:.1f}, σ={current.std:.1f}"
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
