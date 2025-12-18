"""
Shape enforcement utilities for Gravyflow.

Provides a unified ShapeEnforcer class with strict validation
for standard tensor shape contracts in the data pipeline.

Shape Contracts:
- BIS: (Batch, IFO, Samples) - Onsource/Offsource strain data
- BI: (Batch, IFO) - GPS times, Labels
- GBIS: (Generator, Batch, IFO, Samples) - Multi-generator injections
- GB: (Generator, Batch) - Injection masks per generator
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Generator, Any

import jax.numpy as jnp


@dataclass(frozen=True)
class ContractSpec:
    """Specification for a tensor shape contract."""
    ndim: int
    axis_names: Tuple[str, ...]
    description: str


class ShapeContract(Enum):
    """
    Standard tensor shape contracts for Gravyflow.
    
    Each contract defines expected dimensionality and axis semantics.
    """
    # Core data shapes (post-projection, per-IFO)
    BIS = ContractSpec(3, ("Batch", "IFO", "Samples"), "Onsource/Offsource strain data")
    BI = ContractSpec(2, ("Batch", "IFO"), "GPS times")

    # Multi-generator data (post-projection)
    GBIS = ContractSpec(4, ("Generator", "Batch", "IFO", "Samples"), "Multi-generator injections")
    GB = ContractSpec(2, ("Generator", "Batch"), "Injection masks per generator")
    
    # Waveform shapes (pre-projection, polarizations h+/hx)
    BPS = ContractSpec(3, ("Batch", "Polarization", "Samples"), "Single waveform (h+/hx)")
    GBPS = ContractSpec(4, ("Generator", "Batch", "Polarization", "Samples"), "Multi-generator waveforms")


# =============================================================================
# Semantic Axis Constants
# =============================================================================
# Use these instead of magic numbers like shape[0], shape[1], shape[2]
# Each class corresponds to a ShapeContract

class Axis_BIS:
    """Axis indices for BIS (Batch, IFO, Samples) tensors."""
    BATCH = 0
    IFO = 1
    SAMPLES = 2


class Axis_BI:
    """Axis indices for BI (Batch, IFO) tensors."""
    BATCH = 0
    IFO = 1


class Axis_GBIS:
    """Axis indices for GBIS (Generator, Batch, IFO, Samples) tensors."""
    GENERATOR = 0
    BATCH = 1
    IFO = 2
    SAMPLES = 3


class Axis_GB:
    """Axis indices for GB (Generator, Batch) tensors."""
    GENERATOR = 0
    BATCH = 1


class Axis_BPS:
    """Axis indices for BPS (Batch, Polarization, Samples) waveforms."""
    BATCH = 0
    POLARIZATION = 1  # 0=h+, 1=hx
    SAMPLES = 2


class Axis_GBPS:
    """Axis indices for GBPS (Generator, Batch, Polarization, Samples) waveforms."""
    GENERATOR = 0
    BATCH = 1
    POLARIZATION = 2  # 0=h+, 1=hx
    SAMPLES = 3


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
    
    def __init__(
        self, 
        num_ifos: Optional[int] = None, 
        num_generators: Optional[int] = None
    ):
        """
        Initialize enforcer with expected axis sizes.
        
        Args:
            num_ifos: Expected number of interferometers (IFO axis).
            num_generators: Expected number of waveform generators (G axis).
        """
        self.num_ifos = num_ifos
        self.num_generators = num_generators
    
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
        x = jnp.asarray(x)
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
        
        # Check Generator axis if applicable
        if self.num_generators is not None:
            gen_axis = self._get_generator_axis(contract)
            if gen_axis is not None and x.shape[gen_axis] != self.num_generators:
                raise ValueError(
                    f"Generator axis mismatch for '{name}': "
                    f"expected {self.num_generators}, got {x.shape[gen_axis]}. "
                    f"Shape: {x.shape}, Contract: {contract.name}"
                )
        
        return x
    
    def _get_ifo_axis(self, contract: ShapeContract) -> Optional[int]:
        """Get the IFO axis index for a contract, or None if not applicable."""
        axis_names = contract.value.axis_names
        if "IFO" in axis_names:
            return axis_names.index("IFO")
        return None
    
    def _get_generator_axis(self, contract: ShapeContract) -> Optional[int]:
        """Get the Generator axis index for a contract, or None if not applicable."""
        axis_names = contract.value.axis_names
        if "Generator" in axis_names:
            return axis_names.index("Generator")
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
            RV.GPS_TIME: ShapeContract.BI,  # Legacy alias
            RV.TRANSIENT_GPS_TIME: ShapeContract.BI,
            RV.DATA_LABEL: ShapeContract.BI,
            RV.GLITCH_TYPE: ShapeContract.BI,
            RV.EVENT_TYPE: ShapeContract.BI,
        }
        
        REQUIRED_KEYS = {RV.ONSOURCE, RV.OFFSOURCE}
        
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
                    if key in SHAPE_CONTRACTS:
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
                gps = enforcer.validate(gps, ShapeContract.BI, "gps_times")
                
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
    
    # =========================================================================
    # Static helper methods for common shape checks
    # =========================================================================
    
    @staticmethod
    def is_1d(x) -> bool:
        """Check if tensor is 1-dimensional."""
        return len(jnp.asarray(x).shape) == 1
    
    @staticmethod
    def is_2d(x) -> bool:
        """Check if tensor is 2-dimensional."""
        return len(jnp.asarray(x).shape) == 2
    
    @staticmethod
    def is_3d(x) -> bool:
        """Check if tensor is 3-dimensional."""
        return len(jnp.asarray(x).shape) == 3
    
    @staticmethod
    def is_4d(x) -> bool:
        """Check if tensor is 4-dimensional."""
        return len(jnp.asarray(x).shape) == 4
    
    @staticmethod
    def get_batch_size(x, contract: ShapeContract = None) -> int:
        """
        Get batch size from tensor based on contract.
        
        Args:
            x: Input tensor
            contract: Shape contract (default assumes batch is axis 0)
        
        Returns:
            Batch size as integer
        """
        x = jnp.asarray(x)
        if contract is None:
            return int(x.shape[0])
        
        axis_names = contract.value.axis_names
        if "Batch" in axis_names:
            batch_axis = axis_names.index("Batch")
            return int(x.shape[batch_axis])
        return int(x.shape[0])