"""
TransientSegment - Data structure for transient event acquisition.

Combines event metadata (from APIs) with acquisition parameters (boundaries)
in a single structure. Inherits from Segment base class.
"""
from dataclasses import dataclass, field
from typing import Optional, Union, List, TYPE_CHECKING

from .segment import Segment

if TYPE_CHECKING:
    import gravyflow as gf
    from gravyflow.src.dataset.features.glitch import GlitchType
    from gravyflow.src.dataset.features.event import SourceType, EventConfidence
    from gravyflow.src.dataset.acquisition.base import DataLabel, ObservingRun


# IFO Bitmask Helpers
# Bitmask encoding: bit 0=H1, bit 1=L1, bit 2=V1
def ifos_to_bitmask(ifos: List["gf.IFO"]) -> int:
    """
    Convert IFO list to bitmask.
    
    Args:
        ifos: List of IFO enums
        
    Returns:
        Integer bitmask where bit i corresponds to IFO index
    """
    from gravyflow.src.dataset.conditioning.detector import IFO
    ifo_map = {IFO.H1: 0, IFO.L1: 1, IFO.V1: 2}
    mask = 0
    for ifo in ifos:
        if ifo in ifo_map:
            mask |= (1 << ifo_map[ifo])
    return mask


def bitmask_to_ifos(mask: int) -> List["gf.IFO"]:
    """
    Convert bitmask to IFO list.
    
    Args:
        mask: Integer bitmask
        
    Returns:
        List of IFO enums represented by the mask
    """
    from gravyflow.src.dataset.conditioning.detector import IFO
    ifo_map = {0: IFO.H1, 1: IFO.L1, 2: IFO.V1}
    return [ifo_map[i] for i in range(3) if mask & (1 << i)]


@dataclass
class TransientSegment(Segment):
    """
    Transient event representation with metadata and acquisition parameters.
    
    Inherits GPS boundaries from Segment base class and adds transient-specific
    metadata like event type, observing run, and detection confidence.
    
    The GPS key is the immutable identity - it travels with this segment everywhere
    and is used for all matching operations.
    
    Note:
        The gps_key field is computed from transient_gps_time (event center),
        NOT from start_gps_time. This shadows the base Segment.gps_key property
        which uses start_gps_time. This is intentional - cache lookups should
        use the event center time, not the download window start.
    
    Attributes:
        # From Segment base class:
        start_gps_time: float              # Segment start (for data download)
        end_gps_time: float                # Segment end (for data download)
        
        # Transient-specific:
        gps_key: int                       # Integer GPS key at 10ms precision (PRIMARY ID)
        transient_gps_time: float          # Center GPS time (for display/extraction)
        label: DataLabel                   # GLITCHES or EVENTS
        kind: Union[GlitchType, SourceType]  # Specific type (WHISTLE, BBH, etc.)
        observing_run: ObservingRun        # O1, O2, O3, O4
        seen_in: List[IFO]                 # Detectors where transient was observed
        confidence: Optional[EventConfidence]  # Event confidence (events only)
        name: Optional[str]                # Event name like "GW150914" (events only)
        weight: float                      # Sampling weight for balancing
        
        # Extended metadata (NaN = not available)
        snr: float                         # Signal-to-noise ratio
        peak_freq: float                   # Peak frequency Hz
        duration: float                    # Glitch duration seconds
        ml_confidence: float               # ML classification confidence
        
        # Event PE data (NaN for glitches)
        mass1: float                       # Primary mass solar masses
        mass2: float                       # Secondary mass solar masses
        distance: float                    # Luminosity distance Mpc
        p_astro: float                     # Astrophysical probability
    """
    # Transient-specific identity (required fields first)
    gps_key: int = None
    transient_gps_time: float = None
    
    # Event Metadata (required fields)
    label: "DataLabel" = None
    kind: Union["GlitchType", "SourceType"] = None
    observing_run: "ObservingRun" = None
    
    # Optional metadata
    seen_in: List["gf.IFO"] = field(default_factory=list)
    confidence: Optional["EventConfidence"] = None
    name: Optional[str] = None
    weight: float = 1.0
    
    # Extended metadata (NaN = not available)
    snr: float = float('nan')
    peak_freq: float = float('nan')
    duration: float = float('nan')
    ml_confidence: float = float('nan')
    
    # Event PE data (NaN for glitches)
    mass1: float = float('nan')
    mass2: float = float('nan')
    distance: float = float('nan')
    p_astro: float = float('nan')
    
    @property
    def is_event(self) -> bool:
        """Check if this is an event."""
        from gravyflow.src.dataset.acquisition.base import DataLabel
        return self.label == DataLabel.EVENTS
    
    @property
    def is_glitch(self) -> bool:
        """Check if this is a glitch."""
        from gravyflow.src.dataset.acquisition.base import DataLabel
        return self.label == DataLabel.GLITCHES
    
    def __repr__(self) -> str:
        kind_str = self.kind.name if hasattr(self.kind, 'name') else str(self.kind)
        return (
            f"TransientSegment(gps_key={self.gps_key}, "
            f"type={kind_str}, "
            f"center={self.transient_gps_time:.3f}s, "
            f"duration={self.duration:.1f}s)"
        )
    
    def __hash__(self):
        """Hash by GPS key and metadata for deduplication."""
        return hash((
            self.gps_key,
            self.label,
            self.kind,
            tuple(sorted(ifo.name for ifo in self.seen_in))
        ))
    
    def __eq__(self, other) -> bool:
        """Equality based on GPS key and metadata (matches __hash__)."""
        if not isinstance(other, TransientSegment):
            return NotImplemented
        return (
            self.gps_key == other.gps_key and
            self.label == other.label and
            self.kind == other.kind and
            tuple(sorted(ifo.name for ifo in self.seen_in)) == 
            tuple(sorted(ifo.name for ifo in other.seen_in))
        )

