"""
Glitch times, labels and segments from the public Gravity Spy catalogue.

Glitches come from the Gravity Spy machine-learning classifications released on Zenodo
(Glanzer et al. 2023, Class. Quantum Grav. 40 065004; doi:10.5281/zenodo.5649212,
v1.0.0, CC-BY-4.0). See ``gravity_spy`` for the download, verification and cache.

Defaults: every class except ``No_Glitch`` (which marks a trigger that is not a
glitch), and ``ml_confidence >= 0.9``. Entries that overlap a GW event GWOSC publishes
(its in-band signal, see ``event.get_event_veto_segments``) are removed: Gravity Spy lists
some real events, GW190521 among them, as Blips.

A glitch's extent is its Omicron trigger, [start_time, start_time + duration], which
contains the peak time (``event_time``). Noise vetoes cover that extent plus a padding
on each side: ``GLITCH_VETO_PADDING_SECONDS`` (1 s), or more for the classes in
``GLITCH_VETO_PADDING_BY_TYPE`` (3 s for Extremely_Loud). In public GWOSC strain,
measured on 2,535 glitches of every class over 30 hours of O1 to O3 data, the excess
power around a glitch reaches beyond its Omicron extent by at most 0.44 s for 99%
of glitches; 99.88% lie within 1 s and the 99.9th percentile is 1.02 s. The only
glitches beyond 1.25 s were Extremely_Loud, at up to 2.5 s.
"""
from enum import IntEnum
from typing import Dict, List, Optional, Union

import logging
logger = logging.getLogger(__name__)

import numpy as np

from gravyflow.src.dataset.acquisition.base import ObservingRun
from gravyflow.src.dataset.conditioning.detector import IFO
from gravyflow.src.dataset.features import gravity_spy
from gravyflow.src.dataset.features import event as event_module
from gravyflow.src.dataset.features.gravity_spy import DEFAULT_MIN_ML_CONFIDENCE
from gravyflow.src.dataset.acquisition.gwosc_data import coalesce_segments

#: Seconds added on each side of a glitch's Omicron extent when it is vetoed from noise
#: (see the module docstring for the measurement behind it).
GLITCH_VETO_PADDING_SECONDS = 1.0


class GlitchType(IntEnum):
    """
    Glitch classes from Gravity Spy.

    Using IntEnum allows direct .value access for integer labels,
    eliminating the need for list(GlitchType).index() lookups.

    O1 and O2 use 22 classes. O3 adds BLIP_LOW_FREQUENCY and FAST_SCATTERING.
    The values of the first twenty classes are fixed; later classes are appended.
    """
    AIR_COMPRESSOR = 0
    BLIP = 1
    CHIRP = 2
    EXTREMELY_LOUD = 3
    HELIX = 4
    KOI_FISH = 5
    LIGHT_MODULATION = 6
    LOW_FREQUENCY_BURST = 7
    LOW_FREQUENCY_LINES = 8
    NO_GLITCH = 9
    NONE_OF_THE_ABOVE = 10
    PAIRED_DOVES = 11
    POWER_LINE = 12
    REPEATING_BLIPS = 13
    SCATTERED_LIGHT = 14
    SCRATCHY = 15
    TOMTE = 16
    VIOLIN_MODE = 17
    WANDERING_LINE = 18
    WHISTLE = 19
    LINES_1080 = 20
    RIPPLES_1400 = 21
    BLIP_LOW_FREQUENCY = 22
    FAST_SCATTERING = 23

    @classmethod
    def from_gravity_spy_label(cls, label: str) -> "GlitchType":
        """The GlitchType for a Gravity Spy ``ml_label``, e.g. "1080Lines"."""
        try:
            return _GLITCH_TYPE_BY_LABEL[label]
        except KeyError:
            raise ValueError(f"Unknown Gravity Spy label {label!r}.") from None

# Mapping from GlitchType enum to Gravity Spy ml_label values
GLITCH_TYPE_DB_NAMES = {
    GlitchType.AIR_COMPRESSOR: "Air_Compressor",
    GlitchType.BLIP: "Blip",
    GlitchType.CHIRP: "Chirp",
    GlitchType.EXTREMELY_LOUD: "Extremely_Loud",
    GlitchType.HELIX: "Helix",
    GlitchType.KOI_FISH: "Koi_Fish",
    GlitchType.LIGHT_MODULATION: "Light_Modulation",
    GlitchType.LOW_FREQUENCY_BURST: "Low_Frequency_Burst",
    GlitchType.LOW_FREQUENCY_LINES: "Low_Frequency_Lines",
    GlitchType.NO_GLITCH: "No_Glitch",
    GlitchType.NONE_OF_THE_ABOVE: "None_of_the_Above",
    GlitchType.PAIRED_DOVES: "Paired_Doves",
    GlitchType.POWER_LINE: "Power_Line",
    GlitchType.REPEATING_BLIPS: "Repeating_Blips",
    GlitchType.SCATTERED_LIGHT: "Scattered_Light",
    GlitchType.SCRATCHY: "Scratchy",
    GlitchType.TOMTE: "Tomte",
    GlitchType.VIOLIN_MODE: "Violin_Mode",
    GlitchType.WANDERING_LINE: "Wandering_Line",
    GlitchType.WHISTLE: "Whistle",
    GlitchType.LINES_1080: "1080Lines",
    GlitchType.RIPPLES_1400: "1400Ripples",
    GlitchType.BLIP_LOW_FREQUENCY: "Blip_Low_Frequency",
    GlitchType.FAST_SCATTERING: "Fast_Scattering",
}

_GLITCH_TYPE_BY_LABEL = {label: glitch_type for glitch_type, label in GLITCH_TYPE_DB_NAMES.items()}

#: Classes used when none are requested: every class but NO_GLITCH.
DEFAULT_GLITCH_TYPES = [glitch_type for glitch_type in GlitchType if glitch_type != GlitchType.NO_GLITCH]

#: Entries vetoed from real noise: every Gravity Spy entry, whatever its class and
#: classification confidence. Each is an Omicron trigger above SNR 7.5; the class and
#: its confidence say what kind of excess power it is, not whether there is any.
NOISE_VETO_GLITCH_TYPES = list(GlitchType)
NOISE_VETO_MIN_ML_CONFIDENCE = 0.0

#: Classes whose excess power reaches further beyond the Omicron extent: their veto
#: padding in seconds, used instead of ``GLITCH_VETO_PADDING_SECONDS``.
GLITCH_VETO_PADDING_BY_TYPE = {GlitchType.EXTREMELY_LOUD: 3.0}


def glitch_veto_padding(catalogue: Dict[str, np.ndarray]) -> np.ndarray:
    """Per-entry veto padding (s) for a catalogue from ``get_glitch_catalogue``."""
    labels = np.asarray(catalogue["label"])
    padding = np.full(len(labels), GLITCH_VETO_PADDING_SECONDS, dtype=np.float64)
    for glitch_type, seconds in GLITCH_VETO_PADDING_BY_TYPE.items():
        padding[labels == int(glitch_type)] = seconds
    return padding


def _resolve_bounds(observing_run, start_gps_time, end_gps_time):
    if start_gps_time is None:
        start_gps_time = observing_run.value.start_gps_time
    if end_gps_time is None:
        end_gps_time = observing_run.value.end_gps_time
    return start_gps_time, end_gps_time


def _resolve_types(glitch_types) -> List[GlitchType]:
    if not glitch_types:
        return list(DEFAULT_GLITCH_TYPES)
    if not isinstance(glitch_types, (list, tuple)):
        glitch_types = [glitch_types]
    return [GlitchType(glitch_type) for glitch_type in glitch_types]


def get_glitch_catalogue(
    ifo: IFO,
    observing_run: ObservingRun = ObservingRun.O3,
    glitch_types: Union[List[GlitchType], GlitchType] = None,
    start_gps_time: float = None,
    end_gps_time: float = None,
    min_ml_confidence: float = DEFAULT_MIN_ML_CONFIDENCE,
    exclude_events: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Glitches of one detector with their Gravity Spy metadata.

    Args:
        ifo: Detector (H1 or L1).
        observing_run: Run whose bounds are used when start/end are not given.
        glitch_types: Classes to keep. None keeps every class but NO_GLITCH.
        start_gps_time, end_gps_time: Keep glitches whose peak time lies inside.
        min_ml_confidence: Keep classifications with at least this confidence.
        exclude_events: Drop entries whose extent overlaps a catalogued GW event's
            veto segment (confident or marginal), so no real event is ever a glitch.

    Returns:
        Arrays sorted by time: ``event_time`` (peak GPS), ``start_time``, ``duration``,
        ``peak_frequency``, ``snr``, ``ml_confidence`` and ``label`` (GlitchType values).
    """
    start_gps_time, end_gps_time = _resolve_bounds(observing_run, start_gps_time, end_gps_time)
    glitch_types = _resolve_types(glitch_types)
    rows = gravity_spy.query(
        ifo.name,
        start_gps_time,
        end_gps_time,
        labels=[GLITCH_TYPE_DB_NAMES[glitch_type] for glitch_type in glitch_types],
        min_ml_confidence=min_ml_confidence,
    )
    labels = np.array(
        [_GLITCH_TYPE_BY_LABEL[label].value for label in rows.pop("ml_label")],
        dtype=np.int32,
    )
    rows["label"] = labels
    if exclude_events and len(labels) > 0:
        keep = ~overlaps_segments(glitch_extents(rows), event_module.get_event_veto_segments())
        rows = {key: value[keep] for key, value in rows.items()}
    return rows


def glitch_extents(catalogue: Dict[str, np.ndarray], padding_seconds=0.0) -> np.ndarray:
    """
    (N, 2) extents of catalogue glitches: the Omicron trigger
    [start_time, start_time + duration], widened if needed to contain the peak time,
    plus ``padding_seconds`` (a number, or one per entry) on each side.
    """
    peak = np.asarray(catalogue["event_time"], dtype=np.float64)
    start = np.asarray(catalogue["start_time"], dtype=np.float64)
    end = start + np.asarray(catalogue["duration"], dtype=np.float64)
    start = np.minimum(start, peak) - padding_seconds
    end = np.maximum(end, peak) + padding_seconds
    return np.stack([start, end], axis=1).reshape(-1, 2)


def overlaps_segments(intervals: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Boolean mask: which (N, 2) ``intervals`` overlap any of ``segments``."""
    intervals = np.reshape(np.asarray(intervals, dtype=np.float64), (-1, 2))
    segments = coalesce_segments(segments)
    if len(segments) == 0 or len(intervals) == 0:
        return np.zeros(len(intervals), dtype=bool)
    # The last segment starting before each interval ends is the only candidate,
    # because coalesced segments are disjoint and sorted.
    index = np.searchsorted(segments[:, 0], intervals[:, 1], side="left") - 1
    valid = index >= 0
    overlaps = np.zeros(len(intervals), dtype=bool)
    overlaps[valid] = segments[index[valid], 1] > intervals[valid, 0]
    return overlaps


def get_glitch_times(
    ifo: IFO,
    observing_run: ObservingRun = ObservingRun.O3,
    glitch_types: Union[List[GlitchType], GlitchType] = None,
    start_gps_time : float = None,
    end_gps_time : float = None,
    min_ml_confidence: float = DEFAULT_MIN_ML_CONFIDENCE,
) -> np.ndarray:
    """Peak GPS times of the selected glitches (see ``get_glitch_catalogue``)."""
    return get_glitch_catalogue(
        ifo, observing_run, glitch_types, start_gps_time, end_gps_time, min_ml_confidence
    )["event_time"]


def get_glitch_times_with_labels(
    ifo: IFO,
    observing_run: ObservingRun = ObservingRun.O3,
    glitch_types: Union[List[GlitchType], GlitchType] = None,
    start_gps_time: float = None,
    end_gps_time: float = None,
    balanced: bool = False,
    min_ml_confidence: float = DEFAULT_MIN_ML_CONFIDENCE,
) -> tuple:
    """
    Get glitch times with type labels for classification.

    Args:
        ifo: Detector to query
        observing_run: Observing run for default time bounds
        glitch_types: Classes to include. None includes every class but NO_GLITCH.
        start_gps_time: Start GPS time (defaults to observing run start)
        end_gps_time: End GPS time (defaults to observing run end)
        balanced: If True, oversample every non-empty class to the largest
        min_ml_confidence: Keep classifications with at least this confidence

    Returns:
        Tuple of (gps_times, type_indices), where type_indices are GlitchType values
    """
    catalogue = get_glitch_catalogue(
        ifo, observing_run, glitch_types, start_gps_time, end_gps_time, min_ml_confidence
    )
    times, labels = catalogue["event_time"], catalogue["label"]

    if balanced and len(times) > 0:
        classes = [value for value in np.unique(labels)]
        max_count = max(int(np.sum(labels == value)) for value in classes)
        balanced_times, balanced_labels = [], []
        for value in classes:
            class_times = times[labels == value]
            indices = np.random.choice(len(class_times), max_count, replace=True)
            balanced_times.append(class_times[indices])
            balanced_labels.append(np.full(max_count, value, dtype=np.int32))
        times = np.concatenate(balanced_times)
        labels = np.concatenate(balanced_labels)

    return times, labels

def get_glitch_segments(
    ifo: IFO,
    observing_run: ObservingRun = ObservingRun.O3,
    glitch_types: Union[List[GlitchType], GlitchType] = None,
    start_gps_time : float = None,
    end_gps_time : float = None,
    min_ml_confidence: float = DEFAULT_MIN_ML_CONFIDENCE,
    padding_seconds: Optional[float] = None,
    exclude_events: bool = True,
    ) -> np.ndarray:
    """
    (N, 2) veto segments for the selected glitches: each glitch's whole extent (see
    ``glitch_extents``) plus ``padding_seconds`` on each side, sorted by peak time.
    None pads by class (``glitch_veto_padding``).
    """
    catalogue = get_glitch_catalogue(
        ifo, observing_run, glitch_types, start_gps_time, end_gps_time, min_ml_confidence,
        exclude_events=exclude_events,
    )
    if padding_seconds is None:
        padding_seconds = glitch_veto_padding(catalogue)
    return glitch_extents(catalogue, padding_seconds)

def get_glitch_type_from_index(index: int) -> Union[GlitchType, None]:
    """
    Convert integer index back to GlitchType enum.

    Args:
        index: GlitchType value, or -1 for unknown.

    Returns:
        GlitchType enum member or None if index is -1.

    Raises:
        ValueError: If index is out of range.
    """
    if index == -1:
        return None

    try:
        return GlitchType(int(index))
    except ValueError:
        raise ValueError(f"Invalid glitch type index: {index}. Must be between 0 and {len(GlitchType)-1}.")
