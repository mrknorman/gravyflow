"""
Shape contracts for Gravyflow.

- ShapeEnforcer checks what the noise and transient obtainers yield:
  BIS (Batch, IFO, Samples) strain and BI (Batch, IFO) GPS times and labels.
- BatchContract is the one typed contract of a dataset's batch: every output
  is batch-first, with the documented shape its key declares, in float32
  (START_GPS_TIME in float64). A dataset checks every batch against it.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Generator, Any, Mapping, Iterable

import numpy as np
import jax.numpy as jnp

from gravyflow.src.utils.gps import as_gps_times


@dataclass(frozen=True)
class ContractSpec:
    """Specification for a tensor shape contract."""
    ndim: int
    axis_names: Tuple[str, ...]
    description: str


class ShapeContract(Enum):
    """The shape contracts of what the obtainers yield."""
    BIS = ContractSpec(3, ("Batch", "IFO", "Samples"), "Onsource/Offsource strain data")
    BI = ContractSpec(2, ("Batch", "IFO"), "GPS times")


# =============================================================================
# Semantic Axis Constants
# =============================================================================

# =============================================================================
# The batch contract
# =============================================================================

# The axes a documented batch shape is written in. A key declares its shape
# as a tuple of these symbols and integers (a fixed size), e.g. (BATCH,
# GENERATOR, 3); ANY ends a shape whose trailing axes are derived elsewhere.
BATCH = "B"              # examples in the batch
GENERATOR = "G"          # waveform generators (0 for a noise-only dataset)
IFO = "I"                # detectors
SAMPLES = "T"            # samples of the onsource window
OFFSOURCE_SAMPLES = "T_off"  # samples of the offsource window
ANY = ...

BATCH_FIRST = (BATCH,)
PER_GENERATOR = (BATCH, GENERATOR)


class BatchContractError(ValueError):
    """A dataset's batch does not match its documented contract."""


@dataclass(frozen=True)
class BatchContract:
    """
    The documented shape and dtype of every output of one dataset's batches.

    Each key declares its shape (``key.value.shape``): a ReturnVariable its
    whole shape; a WaveformParameter or ScalingType, a value per example and
    generator, (B, G) + its own shape. Every output is float32 but the GPS
    times, which are float64 (``key.value.dtype``).

    The sizes come from the dataset's settings: B examples, G generators, I
    detectors, T onsource samples. The offsource length is the obtainer's
    (its window rule), taken from the batch and checked for consistency.
    """
    batch: int
    generators: int
    ifos: int
    onsource_samples: int

    @staticmethod
    def template(key) -> Tuple:
        """The documented shape of ``key``'s output, in axis symbols."""
        from gravyflow.src.dataset.features.injection import ReturnVariables
        if isinstance(key, ReturnVariables):
            return tuple(key.value.shape)
        return PER_GENERATOR + tuple(key.value.shape)

    @staticmethod
    def dtype(key) -> str:
        return getattr(key.value, "dtype", "float32")

    def shape(self, key, offsource_samples: Optional[int] = None) -> Tuple:
        """``key``'s documented shape with this contract's sizes; ANY stays ANY."""
        sizes = {
            BATCH: self.batch, GENERATOR: self.generators, IFO: self.ifos,
            SAMPLES: self.onsource_samples, OFFSOURCE_SAMPLES: offsource_samples,
        }
        return tuple(sizes.get(axis, axis) if isinstance(axis, str) else axis
                     for axis in self.template(key))

    def describe(self, key) -> str:
        """The documented shape, e.g. "(B=32, G=2, I=2) float32"."""
        sizes = {BATCH: self.batch, GENERATOR: self.generators, IFO: self.ifos,
                 SAMPLES: self.onsource_samples}
        axes = []
        for axis in self.template(key):
            if axis is ANY:
                axes.append("...")
            elif isinstance(axis, str):
                axes.append(f"{axis}={sizes[axis]}" if axis in sizes else axis)
            else:
                axes.append(str(axis))
        return f"({', '.join(axes)}) {self.dtype(key)}"

    def check(self, requested: Iterable, *batches: Mapping[str, Any]) -> None:
        """
        Raise BatchContractError unless each of ``batches`` (name -> array)
        holds every key of ``requested`` it was asked for, each in its
        documented shape and dtype. ``requested`` is a list per batch dict.
        """
        offsource_samples = None
        for keys, batch in zip(requested, batches):
            for key in keys:
                if key.name not in batch:
                    raise BatchContractError(
                        f"{key.name} was requested but the batch has no {key.name}: this "
                        "dataset's noise obtainer and waveform generators do not provide it."
                    )
                value = batch[key.name]
                shape, dtype = tuple(np.shape(value)), str(value.dtype)
                if OFFSOURCE_SAMPLES in self.template(key) and offsource_samples is None:
                    offsource_samples = shape[self.template(key).index(OFFSOURCE_SAMPLES)]
                if not _matches(shape, self.shape(key, offsource_samples)) or dtype != self.dtype(key):
                    raise BatchContractError(
                        f"{key.name} is {shape} {dtype}; its contract is {self.describe(key)}."
                    )


def _matches(shape: Tuple, expected: Tuple) -> bool:
    if expected and expected[-1] is ANY:
        fixed = expected[:-1]
        return len(shape) >= len(fixed) and shape[:len(fixed)] == fixed
    return shape == expected


class ShapeEnforcer:
    """
    Central enforcement point for shape contracts (strict mode).
    
    Validates that tensors match expected shape contracts. Raises ValueError
    on mismatch - no auto-broadcasting.
    
    Example:
        enforcer = ShapeEnforcer(num_ifos=2)
        onsource = enforcer.validate(data, ShapeContract.BIS)
        
    For generator wrapping:
        for batch in ShapeEnforcer.wrap_generator(gen, num_ifos=2):
            on, off, gps = batch
    """
    
    def __init__(self, num_ifos: Optional[int] = None):
        """
        Args:
            num_ifos: Expected number of interferometers (IFO axis).
        """
        self.num_ifos = num_ifos
    
    def validate(self, x: Any, contract: ShapeContract, name: str = "tensor") -> jnp.ndarray:
        """
        Validate tensor against shape contract (strict mode).
        
        Args:
            x: Input tensor to validate.
            contract: Expected ShapeContract.
            name: Tensor name for error messages.
            
        Returns:
            The validated tensor (as jnp.ndarray).
            
        Raises:
            ValueError: If tensor doesn't match contract.
        """
        return self._check(jnp.asarray(x), contract, name)
    
    def validate_gps_times(self, x: Any, contract: ShapeContract, name: str = "gps_times"):
        """
        Validate GPS times against a shape contract, keeping them NumPy float64.
        
        A JAX array would round GPS times to float32, about 100 s near 1.3e9 s.
        """
        return self._check(as_gps_times(x), contract, name)
    
    def _check(self, x, contract: ShapeContract, name: str):
        """Raise ValueError unless ``x`` matches ``contract``; return ``x``."""
        spec = contract.value
        
        # Check ndim
        if x.ndim != spec.ndim:
            raise ValueError(
                f"Shape contract violation for '{name}': "
                f"expected {spec.ndim}D {spec.axis_names}, got {x.ndim}D with shape {x.shape}. "
                f"Contract: {contract.name} ({spec.description})"
            )
        
        # Check IFO axis if applicable
        if self.num_ifos is not None:
            ifo_axis = self._get_ifo_axis(contract)
            if ifo_axis is not None and x.shape[ifo_axis] != self.num_ifos:
                raise ValueError(
                    f"IFO axis mismatch for '{name}': "
                    f"expected {self.num_ifos}, got {x.shape[ifo_axis]}. "
                    f"Shape: {x.shape}, Contract: {contract.name}"
                )
        
        return x
    
    def _get_ifo_axis(self, contract: ShapeContract) -> Optional[int]:
        """Get the IFO axis index for a contract, or None if not applicable."""
        axis_names = contract.value.axis_names
        if "IFO" in axis_names:
            return axis_names.index("IFO")
        return None
    
    @staticmethod
    def wrap_generator(
        gen: Generator, 
        num_ifos: int,
        debug: bool = False
    ) -> Generator:
        """
        Wrap a generator to enforce shape contracts on yielded batches.
        
        Supports both dict and legacy tuple formats.
        
        Dict format (new):
            {
                ReturnVariables.ONSOURCE: tensor,  # Required
                ReturnVariables.OFFSOURCE: tensor, # Required
                ReturnVariables.START_GPS_TIME: tensor,  # Optional
                ...
            }
        
        Tuple format (legacy, deprecated):
            (onsource, offsource, gps_times, labels)
        
        Yields:
            Validated dict with shapes enforced.
        """
        # Import here to avoid circular dependency
        from gravyflow.src.dataset.features.injection import ReturnVariables as RV
        
        enforcer = ShapeEnforcer(num_ifos=num_ifos)
        
        # Shape contracts for each ReturnVariable
        SHAPE_CONTRACTS = {
            RV.ONSOURCE: ShapeContract.BIS,
            RV.OFFSOURCE: ShapeContract.BIS,
            RV.START_GPS_TIME: ShapeContract.BI,
            RV.TRANSIENT_GPS_TIME: ShapeContract.BI,
            RV.DATA_LABEL: ShapeContract.BI,
            RV.SUB_TYPE: ShapeContract.BI,
            RV.GLITCH_TYPE: ShapeContract.BI,
            RV.SOURCE_TYPE: ShapeContract.BI,
            RV.EVENT_TYPE: ShapeContract.BI,
        }
        
        REQUIRED_KEYS = {RV.ONSOURCE, RV.OFFSOURCE}
        GPS_KEYS = {RV.START_GPS_TIME, RV.TRANSIENT_GPS_TIME}
        
        for item in gen:
            # Handle dict format (new)
            if isinstance(item, dict):
                # Validate required keys
                for key in REQUIRED_KEYS:
                    if key not in item:
                        raise ValueError(f"Generator dict missing required key: {key.name}")
                
                # Validate and transform each tensor
                validated = {}
                for key, tensor in item.items():
                    if key in GPS_KEYS:
                        validated[key] = enforcer.validate_gps_times(
                            tensor, SHAPE_CONTRACTS[key], key.name
                        )
                    elif key in SHAPE_CONTRACTS:
                        validated[key] = enforcer.validate(
                            tensor, SHAPE_CONTRACTS[key], key.name
                        )
                    else:
                        # Pass through unknown keys without validation
                        validated[key] = tensor
                
                # Debug consistency checks
                if debug:
                    on_shape = validated[RV.ONSOURCE].shape
                    off_shape = validated[RV.OFFSOURCE].shape
                    if on_shape[0] != off_shape[0]:
                        raise ValueError(f"Batch mismatch: on={on_shape[0]}, off={off_shape[0]}")
                    if on_shape[1] != off_shape[1]:
                        raise ValueError(f"IFO mismatch: on={on_shape[1]}, off={off_shape[1]}")
                
                yield validated
            
            # Handle legacy tuple format
            else:
                if len(item) == 3:
                    on, off, gps = item
                    labels = None
                elif len(item) >= 4:
                    on, off, gps, labels = item[:4]
                else:
                    raise ValueError(f"Generator yielded tuple of length {len(item)}, expected 3 or 4")
                
                on = enforcer.validate(on, ShapeContract.BIS, "onsource")
                off = enforcer.validate(off, ShapeContract.BIS, "offsource")
                gps = enforcer.validate_gps_times(gps, ShapeContract.BI, "gps_times")
                
                if labels is not None:
                    labels = enforcer.validate(labels, ShapeContract.BI, "labels")
                
                if debug:
                    if on.shape[0] != off.shape[0]:
                        raise ValueError(f"Batch mismatch: on={on.shape[0]}, off={off.shape[0]}")
                    if on.shape[1] != off.shape[1]:
                        raise ValueError(f"IFO mismatch: on={on.shape[1]}, off={off.shape[1]}")
                
                # Convert to dict format for consistency
                result = {
                    RV.ONSOURCE: on,
                    RV.OFFSOURCE: off,
                    RV.START_GPS_TIME: gps,
                }
                if labels is not None:
                    result[RV.DATA_LABEL] = labels
                
                yield result
