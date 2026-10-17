from enum import Enum, auto
from typing import Union, Type, List
from dataclasses import dataclass
import random
import warnings

import numpy as np
import keras
from keras import ops
import jax.numpy as jnp

from scipy.stats import truncnorm
import gravyflow as gf
import jax

from gravyflow.src.utils.numerics import NonFiniteWarning


class DistributionType(Enum):
    CONSTANT = auto()         
    UNIFORM = auto()
    NORMAL = auto()
    CHOICE = auto()
    LOG = auto()
    POW_TWO = auto()

@dataclass(frozen=True)
class Distribution:
    """
    A distribution of one parameter: a frozen spec.

    ``sample(n, rng)`` draws n values from the NumPy Generator ``rng`` and
    returns an ndarray. A Distribution holds no random state and cannot be
    changed, so one object can serve any number of datasets.
    """
    value : Union[int, float] = None
    dtype : Type = float
    min_ : Union[int, float] = None
    max_ : Union[int, float] = None
    mean : float = None
    std : float = None
    possible_values : List = None
    type_ : DistributionType = DistributionType.CONSTANT

    def __post_init__(self):
        if self.min_ is not None or self.max_ is not None:
            if self.min_ is None or self.max_ is None:
                raise ValueError("Both min and max must be provided if one is provided.")

            if self.min_ > self.max_:
                # Bounds given in either order mean the same interval.
                low, high = self.max_, self.min_
                object.__setattr__(self, "min_", low)
                object.__setattr__(self, "max_", high)

    def sample(
        self,
        num_samples : int,
        rng : np.random.Generator
        ) -> np.ndarray:
        """``num_samples`` values drawn from ``rng`` (a NumPy Generator), as an ndarray."""
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "Distribution.sample needs an rng: a NumPy Generator, such as a "
                f"gf.SeedTree stream; got {type(rng).__name__}."
            )

        low, high = self.min_, self.max_

        match self.type_:

            case DistributionType.CONSTANT:
                if self.value is None:
                    raise ValueError(
                        "No constant value given in constant distribution."
                    )
                samples = np.asarray([self.value] * num_samples)

            case DistributionType.UNIFORM:
                if low is None:
                    raise ValueError("No minumum value given in uniform distribution.")
                if high is None:
                    raise ValueError("No maximum value given in uniform distribution.")
                if self.dtype == int:
                    # Integers in [min_, max_], both ends included.
                    samples = rng.integers(
                        int(np.ceil(low)), int(np.floor(high)) + 1, size=num_samples
                    )
                elif low == high:
                    samples = np.full(num_samples, low)
                else:
                    samples = rng.uniform(low, high, num_samples)

            case DistributionType.NORMAL:
                if self.mean is None:
                    raise ValueError("No mean value given in normal distribution.")
                if self.std is None:
                    raise ValueError("No std value given in normal distribution.")
                low = float("-inf") if low is None else low
                high = float("inf") if high is None else high
                if low == high:
                    samples = np.full(num_samples, low)
                else:
                    samples = truncnorm.rvs(
                        (low - self.mean) / self.std,
                        (high - self.mean) / self.std,
                        loc=self.mean,
                        scale=self.std,
                        size=num_samples,
                        random_state=rng
                    )

            case DistributionType.CHOICE:
                if self.possible_values is None:
                    raise ValueError(
                        "No possible values given in choice distribution."
                    )
                samples = rng.choice(self.possible_values, size=num_samples)

            case DistributionType.LOG:
                if low is None:
                    raise ValueError("No minumum value given in log distribution.")
                if high is None:
                    raise ValueError("No maximum value given in log distribution.")
                if low <= 0:
                    raise ValueError("LOG distribution requires min_ > 0.")
                # Uniform in log10, so log-uniform between min_ and max_.
                samples = 10 ** rng.uniform(np.log10(low), np.log10(high), num_samples)

            case DistributionType.POW_TWO:
                power_low, power_high = map(int, np.log2((low, high)))
                power = rng.integers(power_low, high=power_high + 1, size=num_samples)
                samples = 2**power

            case _:
                raise ValueError(f'Unsupported distribution type {self.type_}')

        samples = np.asarray(samples)
        if self.dtype == int and samples.dtype.kind != "i":
            samples = np.rint(samples.astype(np.float64)).astype(np.int64)
        return samples


@jax.jit
def _zero_non_finite(tensor):
    """(tensor with NaN and inf set to 0, how many were)."""
    finite = jnp.isfinite(tensor)
    return jnp.where(finite, tensor, jnp.zeros_like(tensor)), jnp.sum(~finite)


def replace_nan_and_inf_with_zero(tensor, name: str = None):
    """
    ``tensor`` with every NaN and infinite value replaced by 0: the library's
    one sanitiser of non-finite values.

    It never replaces silently: when it replaces anything it issues a
    NonFiniteWarning that names the tensor (``name``; else the caller's line)
    and counts the values replaced. A non-finite value past acquisition is a
    failure upstream, so make the warning an error to find it:
    ``warnings.simplefilter("error", gf.NonFiniteWarning)``.

    It runs eagerly, since it reports how many values it replaced: an eager
    ``jax.grad`` works (its values are concrete, and replaced values get zero
    gradient), but under ``jax.jit``, ``jax.vmap`` or any transformation whose
    values are abstract it raises a TypeError naming itself. Traced code that must
    zero non-finite values silently uses
    ``jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)``.
    """
    tensor = ops.convert_to_tensor(tensor)
    cleaned, count = _zero_non_finite(tensor)
    try:
        count = int(count)
    except TypeError:  # JAX's concretization errors: the count is abstract
        raise TypeError(
            "replace_nan_and_inf_with_zero runs eagerly (it reports what it replaces), "
            f"but was called on a traced value ({name or 'unnamed'}) under a JAX "
            "transformation such as jax.jit or jax.vmap. Call it outside the "
            "transformation, or use jnp.nan_to_num in traced code."
        ) from None
    if count:
        warnings.warn(
            f"{name or 'A tensor'}: {count} of {int(np.prod(tensor.shape))} values were NaN or "
            "infinite and were replaced with 0.",
            NonFiniteWarning,
            stacklevel=2,
        )
    return cleaned

def set_random_seeds(
    seed : int = 1000
    ):
    
    """
    Set random seeds for Keras, Numpy, and Core Python to ensure 
    deterministic results with the same seed.
    
    Args
    ---
    
    seed : int
        Random seed which will be used to set both Numpy and Keras seeds
    
    """
    
    # Set keras random seed (Keras 3 API):
    keras.utils.set_random_seed(seed)
    
    # Set Numpy random seed:
    np.random.seed(seed)
    
    # Set core Python.random seed:
    random.seed(seed)
    
def window_samples(duration_seconds: float, sample_rate_hertz: float) -> int:
    """The samples crop_samples keeps of a window of ``duration_seconds``."""
    return int(duration_seconds * sample_rate_hertz)

def even_samples(duration_seconds: float, sample_rate_hertz: float) -> int:
    """
    The samples of an acquired window (a padded onsource or an offsource) of
    ``duration_seconds``: whole samples, made even (WindowSpec's counts).
    """
    num = int(duration_seconds * sample_rate_hertz)
    return num - (num % 2)

def centre_index(num_samples: int) -> int:
    """
    The index of a window's centre sample: where a transient window holds its
    transient, in the stored onsource and in every window read from it.
    """
    return int(num_samples) // 2

def crop_start(num_samples: int, onsource_duration_seconds: float, sample_rate_hertz: float) -> int:
    """
    The index, in a window of ``num_samples``, of the first sample crop_samples keeps:
    the window's first returned sample after cropping (see WindowSpec.onsource_start_gps).
    """
    return (int(num_samples) - window_samples(onsource_duration_seconds, sample_rate_hertz)) // 2

def crop_samples(
        batched_onsource, 
        onsource_duration_seconds: float, 
        sample_rate_hertz: float
    ):
    """
    Crop to remove edge effects and ensure same data is retrieved in all cases.
    
    This function calculates the desired number of samples based on the duration 
    of examples in seconds and the sample rate, then it finds the start and end 
    index for cropping. It then crops the batched_onsource using these indices.
    
    Parameters
    ----------
    batched_onsource : tensor
        The batch of examples to be cropped.
    onsource_duration_seconds : float
        The duration of an example in seconds.
    sample_rate_hertz : float
        The sample rate in hertz.
    
    Returns
    -------
    tensor
        The cropped batched_onsource.
    """
    batched_onsource = ops.convert_to_tensor(batched_onsource)
    
    dims = len(ops.shape(batched_onsource))
    if dims == 1:
        batched_onsource = ops.expand_dims(batched_onsource, 0) 
    
    desired_num_samples = window_samples(onsource_duration_seconds, sample_rate_hertz)
    start = crop_start(ops.shape(batched_onsource)[-1], onsource_duration_seconds, sample_rate_hertz)
    end = start + desired_num_samples
    
    # Crop the batched_onsource
    batched_onsource = batched_onsource[..., start:end]
    
    if dims == 1:
        batched_onsource = ops.squeeze(batched_onsource, axis=0) 
    
    return batched_onsource

def rfftfreq(
        num_samples: int, 
        frequency_interval_hertz: Union[float, int] = 1.0
    ):
    """
    Return the Discrete Fourier Transform sample frequencies
    (for usage with rfft, irfft) using Keras operations.

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.

    Returns
    -------
    f : tensor
        Tensor of shape ``(n//2 + 1,)`` containing the sample frequencies.

    Examples
    --------
    >>> n = 10
    >>> sample_rate = 100
    >>> freq = rfftfreq(n, d=1./sample_rate)
    """
    
    # If num_samples is a static integer (not a tracer), keep it static for arange
    if isinstance(num_samples, int):
        num_frequency_samples = num_samples // 2 + 1
        results = ops.arange(0, num_frequency_samples, dtype="int32")
        
        # Cast for float calculation
        num_samples_float = ops.cast(num_samples, "float32")
        val = 1.0 / (num_samples_float * frequency_interval_hertz)
        
        frequency_tensor = ops.cast(results, dtype="float32") * val
        return frequency_tensor
        
    num_samples = ops.cast(num_samples, "float32")
    
    val = 1.0 / (num_samples * frequency_interval_hertz)
    num_frequency_samples = ops.cast(num_samples // 2 + 1, "int32")

    # Create a range tensor and scale it
    results = ops.arange(0, num_frequency_samples, dtype="int32")
    frequency_tensor = ops.cast(results, dtype="float32") * val
    
    return frequency_tensor
