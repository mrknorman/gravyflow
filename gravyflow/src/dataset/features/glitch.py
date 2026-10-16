"""
Glitch times, labels and segments from the public Gravity Spy catalogue.

Glitches come from the Gravity Spy machine-learning classifications released on Zenodo
(Glanzer et al. 2023, Class. Quantum Grav. 40 065004; doi:10.5281/zenodo.5649212,
v1.0.0, CC-BY-4.0). See ``gravity_spy`` for the download, verification and cache.

Defaults: every class except ``No_Glitch`` (which marks a trigger that is not a
glitch), and ``ml_confidence >= 0.9``.
"""
from enum import IntEnum
from typing import Dict, List, Optional, Union

import logging
logger = logging.getLogger(__name__)

import numpy as np

from gravyflow.src.dataset.acquisition.base import ObservingRun
from gravyflow.src.dataset.conditioning.detector import IFO
from gravyflow.src.dataset.features import gravity_spy
from gravyflow.src.dataset.features.gravity_spy import DEFAULT_MIN_ML_CONFIDENCE


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
) -> Dict[str, np.ndarray]:
    """
    Glitches of one detector with their Gravity Spy metadata.

    Args:
        ifo: Detector (H1 or L1).
        observing_run: Run whose bounds are used when start/end are not given.
        glitch_types: Classes to keep. None keeps every class but NO_GLITCH.
        start_gps_time, end_gps_time: Keep glitches whose peak time lies inside.
        min_ml_confidence: Keep classifications with at least this confidence.

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
    return rows


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
    ) -> np.ndarray:
    """
    (N, 2) array of [peak time, peak time + duration] for the selected glitches.
    """
    catalogue = get_glitch_catalogue(
        ifo, observing_run, glitch_types, start_gps_time, end_gps_time, min_ml_confidence
    )
    return np.stack(
        [catalogue["event_time"], catalogue["event_time"] + catalogue["duration"]], axis=1
    )

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
