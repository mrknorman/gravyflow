"""
Gravitational-wave events from the GWOSC event API.

``get_event_catalogue`` is the one event list behind the noise vetoes, the glitch
catalogue's event exclusion and the event-time functions: every event in every
catalogue GWOSC's event API lists (https://gwosc.org/eventapi/json/). That is the GWTC
releases (confident, marginal and auxiliary, and the cumulative GWTC table) and also
the other lists GWOSC hosts, such as O3_IMBH_marginal, IAS-O3a, O3_Discovery_Papers,
O4_Discovery_Papers and O1_O2-Preliminary: the conservative reading of "real events",
at a negligible cost in livetime.

Also provides EventConfidence (CONFIDENT/MARGINAL) and the parameter-estimation queries
used to build event examples.
"""

from enum import Enum, IntEnum, auto
from typing import List, Optional, Union
import logging

import numpy as np
from gwpy.table import EventTable


class EventConfidence(IntEnum):
    """
    Confidence level of gravitational wave event detection.
    
    Mirrors GlitchType for symmetry. Use DataLabel.EVENTS to get all types.
    """
    CONFIDENT = 1  # GWTC *-confident catalogs (confirmed events)
    MARGINAL = 0    # GWTC *-marginal catalogs (sub-threshold triggers)

class SourceType(IntEnum):
    """
    Type of gravitational wave source.
    
    Using IntEnum allows direct .value access for integer labels.
    """
    UNKNOWN = -1  # The catalogue gives no masses
    BBH = 0   # Binary Black Hole
    BNS = 1   # Binary Neutron Star
    NSBH = 2  # Neutron Star - Black Hole


#: The source-frame mass, in solar masses, below which a component is a neutron star.
NEUTRON_STAR_MAX_MASS = 3.0


def source_type_from_masses(mass1: float, mass2: float) -> SourceType:
    """
    The source type of a binary from its components' median source-frame masses, in
    either order: BNS if both are below ``NEUTRON_STAR_MAX_MASS``, NSBH if one is,
    BBH otherwise, and UNKNOWN when either mass is missing (NaN).
    """
    if not (np.isfinite(mass1) and np.isfinite(mass2)):
        return SourceType.UNKNOWN
    neutron_stars = int(mass1 < NEUTRON_STAR_MAX_MASS) + int(mass2 < NEUTRON_STAR_MAX_MASS)
    return (SourceType.BBH, SourceType.NSBH, SourceType.BNS)[neutron_stars]

# Catalog mappings by event type
CONFIDENT_CATALOGS = [
    "GWTC",
    "GWTC-1-confident",
    "GWTC-2",
    "GWTC-2.1-confident",
    "GWTC-3-confident",
]

MARGINAL_CATALOGS = [
    "GWTC-1-marginal",
    "GWTC-2.1-auxiliary",
    "GWTC-2.1-marginal",
    "GWTC-3-marginal",
]


#: An event's veto starts where its signal enters the band at this frequency.
EVENT_VETO_LOW_FREQUENCY_HERTZ = 20.0
#: The leading-order chirp time is multiplied by this factor to cover higher-order and
#: spin terms. With it, every catalogued event's veto (with masses) is at least as long
#: as LALSuite's upper bound on the chirp time (``XLALSimInspiralChirpTimeBound``) at
#: its source-frame masses and maximal spins; 1.1, LALSuite's own factor, falls up to
#: 2 s short for extreme mass ratios such as GW191219_163120.
EVENT_VETO_CHIRP_TIME_FACTOR = 1.25
#: Seconds added before the in-band signal and after the merger: merger and ringdown,
#: the catalogue's 0.1 s GPS rounding and the light travel time between detectors.
EVENT_VETO_PADDING_SECONDS = 2.0
#: Chirp mass (solar masses) assumed when a catalogue gives no masses: a 1 + 1 solar
#: mass binary, lighter than any catalogued neutron-star binary, so its duration is
#: an upper bound.
UNKNOWN_EVENT_CHIRP_MASS_SOLAR = 0.87

#: Version of the cached event table's format; bumping it re-fetches every cache.
#: 3 added the parameter-estimation columns (``EVENT_PARAMETER_COLUMNS``).
_EVENT_CATALOGUE_FORMAT = 3

#: Parameter-estimation columns the event list carries, as GWOSC names them (NaN
#: where a catalogue gives none), and the keys ``get_events_with_params`` gives them.
EVENT_PARAMETER_COLUMNS = {
    "mass_1_source": "mass1",
    "mass_2_source": "mass2",
    "luminosity_distance": "distance",
    "p_astro": "p_astro",
}

_SOLAR_MASS_SECONDS = 4.925490947e-6  # G M_sun / c^3


def chirp_time_seconds(
    chirp_mass_solar,
    low_frequency_hertz: float = EVENT_VETO_LOW_FREQUENCY_HERTZ,
):
    """
    Leading-order time from ``low_frequency_hertz`` to merger for a chirp mass.

    tau = (5 / 256) (pi f)^(-8/3) (G Mc / c^3)^(-5/3). Source-frame masses give an
    upper bound on the observed duration, since the detector-frame mass (1 + z) Mc is
    larger. GW170817 (Mc = 1.186) gives about 165 s from 20 Hz.
    """
    chirp_mass_seconds = np.asarray(chirp_mass_solar, dtype=np.float64) * _SOLAR_MASS_SECONDS
    return (5.0 / 256.0) * (np.pi * low_frequency_hertz) ** (-8.0 / 3.0) \
        * chirp_mass_seconds ** (-5.0 / 3.0)


def _row_value(row, column):
    try:
        value = row[column]
    except (KeyError, IndexError, ValueError):
        return np.nan
    if np.ma.is_masked(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _row_chirp_mass(row) -> float:
    """Source-frame chirp mass of a catalogue row, or NaN if it has no masses."""
    chirp_mass = _row_value(row, "chirp_mass_source")
    if np.isfinite(chirp_mass):
        return chirp_mass
    m1, m2 = _row_value(row, "mass_1_source"), _row_value(row, "mass_2_source")
    if np.isfinite(m1) and np.isfinite(m2) and m1 > 0 and m2 > 0:
        return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    # Detector-frame chirp mass: larger than the source-frame one, so it is used only
    # when nothing else is published.
    return _row_value(row, "chirp_mass")


def fetch_event_catalogue_names() -> List[str]:
    """Every catalogue GWOSC's event API lists, sorted by name."""
    from gwosc.api import fetch_cataloglist_json

    return sorted(fetch_cataloglist_json())


def _observing_runs_key() -> dict:
    """The GWOSC runs this gravyflow knows, as stored in the event cache."""
    from gravyflow.src.dataset.acquisition.gwosc_data import GWOSC_RUNS

    return {name: [run["gps_start"], run["gps_end"]] for name, run in GWOSC_RUNS.items()}


def event_catalogue_cache_path():
    """Where ``get_event_catalogue`` caches the event table."""
    from gravyflow.src.dataset.acquisition.gwosc_data import user_cache_dir

    return user_cache_dir("gwosc_events") / f"event_catalogue_v{_EVENT_CATALOGUE_FORMAT}.json"


def get_event_catalogue(use_cache: bool = True) -> dict:
    """
    Every event GWOSC's event API publishes, one row per catalogue entry.

    The catalogues are discovered from the API (``fetch_event_catalogue_names``), so a
    list GWOSC adds is vetoed without a code change. Returns arrays ``gps``,
    ``chirp_mass_source`` (solar masses; NaN when the catalogue gives no masses),
    ``name``, ``catalog`` and the parameter estimates ``EVENT_PARAMETER_COLUMNS``
    (NaN where the catalogue gives none).

    The table is cached in the user cache (``~/.gravyflow/cache/gwosc_events``) with
    the catalogue names and the GWOSC runs it was fetched under (``GWOSC_RUNS``). It
    is fetched again when gravyflow's runs change, as they do when a run is added, so
    events published with a new run are vetoed. To pick up events GWOSC adds inside
    known runs, call with ``use_cache=False``: the table is fetched afresh and the
    cache rewritten, so every later call (the noise vetoes among them) reads it. A
    catalogue that cannot be fetched raises and leaves the cache as it was: a partial
    list would silently leave events in noise.
    """
    import json
    from gravyflow.src.dataset.acquisition.gwosc_data import _atomic_write_text

    cache_file = event_catalogue_cache_path()
    runs = _observing_runs_key()
    if use_cache and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if cached["runs"] == runs:
                return _event_rows_to_arrays(cached["events"])
            logging.info(f"GWOSC runs changed since {cache_file} was written; re-fetching.")
        except (ValueError, KeyError) as error:
            logging.warning(f"Ignoring unreadable event cache {cache_file}: {error}")

    catalogs = fetch_event_catalogue_names()
    events = []
    for catalog in catalogs:
        table = EventTable.fetch_open_data(catalog)
        for row in table:
            gps = _row_value(row, "GPS")
            if not np.isfinite(gps):
                continue
            chirp_mass = _row_chirp_mass(row)
            name = row["commonName"] if "commonName" in table.colnames else row["name"]
            parameters = {column: _row_value(row, column) for column in EVENT_PARAMETER_COLUMNS}
            events.append({
                "gps": gps,
                "chirp_mass_source": chirp_mass,
                "name": str(name),
                "catalog": catalog,
                **parameters,
            })
            # JSON has no NaN: a missing value is stored as null.
            events[-1] = {key: (None if isinstance(value, float) and not np.isfinite(value) else value)
                          for key, value in events[-1].items()}

    _atomic_write_text(
        cache_file,
        json.dumps({"runs": runs, "catalogs": catalogs, "events": events}),
    )
    return _event_rows_to_arrays(events)


def _event_rows_to_arrays(events: List[dict]) -> dict:
    def numbers(key):
        # A value a row lacks (null, or absent) reads as NaN.
        return np.array([np.nan if e.get(key) is None else e[key] for e in events], dtype=np.float64)

    return {
        "gps": np.array([e["gps"] for e in events], dtype=np.float64),
        "chirp_mass_source": numbers("chirp_mass_source"),
        "name": np.array([e["name"] for e in events], dtype=str),
        "catalog": np.array([e["catalog"] for e in events], dtype=str),
        **{column: numbers(column) for column in EVENT_PARAMETER_COLUMNS},
    }


def get_event_veto_segments(
    low_frequency_hertz: float = EVENT_VETO_LOW_FREQUENCY_HERTZ,
    padding_seconds: float = EVENT_VETO_PADDING_SECONDS,
    catalogue: Optional[dict] = None,
) -> np.ndarray:
    """
    One veto segment per catalogued event, covering its whole in-band signal.

    Each segment is [t - 1.25 tau - padding, t + padding], where t is the catalogue's
    merger time and tau the leading-order chirp time from ``low_frequency_hertz`` for
    the event's source-frame chirp mass (see ``chirp_time_seconds``). Events without
    masses use ``UNKNOWN_EVENT_CHIRP_MASS_SOLAR``. Confident, marginal and auxiliary
    events are all included.

    Returns:
        A sorted, coalesced (N, 2) array of GPS segments.
    """
    from gravyflow.src.dataset.acquisition.gwosc_data import coalesce_segments

    if catalogue is None:
        catalogue = get_event_catalogue()
    gps = np.asarray(catalogue["gps"], dtype=np.float64)
    if len(gps) == 0:
        return np.empty((0, 2))
    chirp_mass = np.asarray(catalogue["chirp_mass_source"], dtype=np.float64)
    chirp_mass = np.where(np.isfinite(chirp_mass) & (chirp_mass > 0), chirp_mass,
                          UNKNOWN_EVENT_CHIRP_MASS_SOLAR)
    in_band = EVENT_VETO_CHIRP_TIME_FACTOR * chirp_time_seconds(chirp_mass, low_frequency_hertz)
    segments = np.stack(
        [gps - in_band - padding_seconds, gps + padding_seconds], axis=1
    )
    return coalesce_segments(segments)


def _event_times(catalogs: Optional[List[str]], use_cache: bool) -> np.ndarray:
    """GPS times from ``get_event_catalogue``, of the given catalogues or of all."""
    catalogue = get_event_catalogue(use_cache)
    if catalogs is None:
        return catalogue["gps"]
    return catalogue["gps"][np.isin(catalogue["catalog"], catalogs)]


def get_confident_event_times(use_cache: bool = True) -> np.ndarray:
    """GPS times of the events in the GWTC confident catalogues (``CONFIDENT_CATALOGS``)."""
    return _event_times(CONFIDENT_CATALOGS, use_cache)


def get_marginal_event_times(use_cache: bool = True) -> np.ndarray:
    """GPS times of the events in the GWTC marginal catalogues (``MARGINAL_CATALOGS``)."""
    return _event_times(MARGINAL_CATALOGS, use_cache)


def get_all_event_times(use_cache: bool = True) -> np.ndarray:
    """GPS times of every event GWOSC publishes: the times the noise vetoes remove."""
    return _event_times(None, use_cache)


def get_event_times_by_type(
    event_types: List[EventConfidence],
    use_cache: bool = True
) -> np.ndarray:
    """GPS times of the events in the GWTC catalogues of the given confidences."""
    catalogs = []
    if EventConfidence.CONFIDENT in event_types:
        catalogs += CONFIDENT_CATALOGS
    if EventConfidence.MARGINAL in event_types:
        catalogs += MARGINAL_CATALOGS
    return _event_times(catalogs, use_cache)


def _observing_run_name(gps: float) -> str:
    """The name of the ``ObservingRun`` whose bounds hold ``gps``, or "Unknown"."""
    from gravyflow.src.dataset.acquisition.base import ObservingRun

    for run in ObservingRun:
        if run.value.start_gps_time <= gps <= run.value.end_gps_time:
            return run.name
    return "Unknown"


def get_events_with_params(
    observing_runs: List = None,
    event_types: List[EventConfidence] = None,
    event_names: List[str] = None
) -> List[dict]:
    """
    GW events of the GWTC catalogues with their parameter estimates, from the one
    event list (``get_event_catalogue``).

    Args:
        observing_runs: Optional list of ObservingRun enums to filter by.
                       If None, returns events of every run.
        event_types: Optional list of EventConfidence (CONFIDENT, MARGINAL), whose
                    catalogues (``CONFIDENT_CATALOGS``, ``MARGINAL_CATALOGS``) are read.
                    Defaults to [EventConfidence.CONFIDENT].
        event_names: Optional list of event names to keep.

    Returns:
        One dict per event, sorted by GPS time, with keys name, gps, catalog,
        observing_run and the parameter estimates mass1, mass2, distance and p_astro
        (NaN where the catalogue gives none). An event listed by several catalogues
        is given once, by its catalogue name (``commonName``), from the first
        catalogue in the order above: one physical event is one record, though its
        catalogues' GPS times differ.
    """
    if event_types is None:
        event_types = [EventConfidence.CONFIDENT]
    catalogs = []
    if EventConfidence.CONFIDENT in event_types:
        catalogs += CONFIDENT_CATALOGS
    if EventConfidence.MARGINAL in event_types:
        catalogs += MARGINAL_CATALOGS
    runs = {run.name for run in observing_runs} if observing_runs else None

    catalogue = get_event_catalogue()
    missing = np.full(len(catalogue["gps"]), np.nan)
    parameters = {key: catalogue.get(column, missing) for column, key in EVENT_PARAMETER_COLUMNS.items()}

    events, seen = [], set()
    for catalog in catalogs:
        for index in np.flatnonzero(catalogue["catalog"] == catalog):
            gps = float(catalogue["gps"][index])
            event = {
                "name": str(catalogue["name"][index]),
                "gps": gps,
                "catalog": catalog,
                "observing_run": _observing_run_name(gps),
                **{key: float(values[index]) for key, values in parameters.items()},
            }
            if runs is not None and event["observing_run"] not in runs:
                continue
            if event_names and event["name"] not in event_names:
                continue
            if event["name"] not in seen:
                seen.add(event["name"])
                events.append(event)

    events.sort(key=lambda event: event["gps"])
    logging.info(f"{len(events)} events with parameter estimates")
    return events


def get_confident_events_with_params(observing_runs: List = None) -> List[dict]:
    """Wrapper for backward compatibility."""
    return get_events_with_params(observing_runs, event_types=[EventConfidence.CONFIDENT])


def search_events(
    mass1_range: tuple = None,
    mass2_range: tuple = None,
    total_mass_range: tuple = None,
    distance_range: tuple = None,
    observing_runs: List = None,
    event_types: List[EventConfidence] = None,
    source_type: Union[SourceType, str] = None,
    name_contains: str = None,
) -> List[str]:
    """
    Search for event names matching specified conditions.
    
    Args:
        mass1_range: (min, max) for primary mass in solar masses
        mass2_range: (min, max) for secondary mass in solar masses  
        total_mass_range: (min, max) for total mass (mass1 + mass2)
        distance_range: (min, max) for luminosity distance in Mpc
        observing_runs: List of ObservingRun enums to filter by
        event_types: List of EventConfidence (CONFIDENT, MARGINAL). Default: [CONFIDENT]
        source_type: SourceType enum or string ("BBH", "BNS", "NSBH")
        name_contains: Substring to match in event name
        
    Returns:
        List of event names matching all conditions
        
    Example:
        # Find all BBH events
        names = gf.search_events(source_type=gf.SourceType.BBH)
        
        # Find marginal events in O3
        names = gf.search_events(
            observing_runs=[gf.ObservingRun.O3],
            event_types=[gf.EventConfidence.MARGINAL]
        )
    """
    events = get_events_with_params(observing_runs, event_types)
    
    def in_range(value, range_tuple):
        if range_tuple is None:
            return True
        if np.isnan(value):
            return False
        min_val, max_val = range_tuple
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True
    
    # Handle string input for source_type for backward compatibility/ease of use
    if isinstance(source_type, str):
        try:
            source_type = SourceType[source_type.upper()]
        except KeyError:
            raise ValueError(f"Unknown source type {source_type!r}: choose from "
                             f"{[member.name for member in SourceType]}.") from None

    matching_names = []
    for event in events:
        # Mass filters
        if not in_range(event.get("mass1", np.nan), mass1_range):
            continue
        if not in_range(event.get("mass2", np.nan), mass2_range):
            continue
        
        # Total mass filter
        if total_mass_range is not None:
            total_mass = event.get("mass1", 0) + event.get("mass2", 0)
            if not in_range(total_mass, total_mass_range):
                continue
        
        # Distance filter
        if not in_range(event.get("distance", np.nan), distance_range):
            continue
        
        # Source type filter
        if source_type is not None:
            if source_type_from_masses(event.get("mass1", np.nan), event.get("mass2", np.nan)) != source_type:
                continue
        
        # Name filter
        if name_contains is not None:
            if name_contains.lower() not in event.get("name", "").lower():
                continue
        
        matching_names.append(event["name"])
    
    return matching_names
