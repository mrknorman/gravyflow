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
from pathlib import Path
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
    BBH = 0   # Binary Black Hole
    BNS = 1   # Binary Neutron Star
    NSBH = 2  # Neutron Star - Black Hole

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
_EVENT_CATALOGUE_FORMAT = 2

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
    ``name`` and ``catalog``.

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
            events.append({
                "gps": gps,
                "chirp_mass_source": chirp_mass if np.isfinite(chirp_mass) else None,
                "name": str(name),
                "catalog": catalog,
            })

    _atomic_write_text(
        cache_file,
        json.dumps({"runs": runs, "catalogs": catalogs, "events": events}),
    )
    return _event_rows_to_arrays(events)


def _event_rows_to_arrays(events: List[dict]) -> dict:
    return {
        "gps": np.array([e["gps"] for e in events], dtype=np.float64),
        "chirp_mass_source": np.array(
            [np.nan if e["chirp_mass_source"] is None else e["chirp_mass_source"] for e in events],
            dtype=np.float64,
        ),
        "name": np.array([e["name"] for e in events], dtype=str),
        "catalog": np.array([e["catalog"] for e in events], dtype=str),
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


def get_events_with_params(
    observing_runs: List = None,
    event_types: List[EventConfidence] = None,
    event_names: List[str] = None
) -> List[dict]:
    """
    Fetch GW events with full parameter estimation data.
    
    Returns a list of dictionaries containing event metadata and PE parameters.
    
    Args:
        observing_runs: Optional list of ObservingRun enums to filter by.
                       If None, returns all O1/O2/O3/O4 events.
        event_types: Optional list of EventConfidence (CONFIDENT, MARGINAL).
                    Defaults to [EventConfidence.CONFIDENT].
    
    Returns:
        List of dicts with keys: name, gps, mass1, mass2, distance, catalog, observing_run...
    """
    if event_types is None:
        event_types = [EventConfidence.CONFIDENT]
    
    events = []
    
    # Map catalogs to observing runs
    # Define observing run GPS ranges (approximate)
    O1_START, O1_END = 1126051217, 1137254417
    O2_START, O2_END = 1164556817, 1187733618
    O3_START, O3_END = 1238166018, 1269363618
    # O4: May 24 2023 - Jan 16 2024 (O4a) + ...
    O4_START, O4_END = 1368000000, 1400000000

    def get_run_from_gps(gps):
        if O1_START <= gps <= O1_END:
            return "O1"
        elif O2_START <= gps <= O2_END:
            return "O2"
        elif O3_START <= gps <= O3_END:
            return "O3"
        elif O4_START <= gps <= O4_END:
            return "O4"
        return "Unknown"

    # Determine which catalogs to fetch
    catalogs_to_fetch = []
    if EventConfidence.CONFIDENT in event_types:
        catalogs_to_fetch.extend(CONFIDENT_CATALOGS)
    if EventConfidence.MARGINAL in event_types:
        catalogs_to_fetch.extend(MARGINAL_CATALOGS)
    
    # Remove duplicates by GPS time (some events appear in multiple catalogs)
    seen_gps = set()
    unique_events = []
    
    # Simple caching implementation to avoid repeated catalog fetching
    import json
    import os
    from hashlib import md5
    
    # Create cache dir
    cache_dir = Path(__file__).parent.parent.parent / "res" / "event_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_key = md5(str(sorted(event_types, key=lambda x: x.value)).encode('utf-8')).hexdigest()
    cache_file = cache_dir / f"events_{cache_key}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                events = json.load(f)
            if len(events) < 10:
                logging.warning(
                    f"Event cache has only {len(events)} events (expected many more). "
                    f"Discarding stale cache at {cache_file}"
                )
                events = []
        except Exception as e:
            logging.warning(f"Failed to load event cache: {e}. Re-fetching.")
            events = []
            
    if not events: # Fetch if no cache or empty
        for catalog in catalogs_to_fetch:
            try:
                table = EventTable.fetch_open_data(catalog)
                
                for row in table:
                    try:
                        run_name = get_run_from_gps(float(row["GPS"]))
                        
                        event = {
                            "name": str(row.get("commonName", row.get("name", "Unknown"))),
                            "gps": float(row["GPS"]),
                            "mass1": float(row.get("mass_1_source", row.get("m1", np.nan))),
                            "mass2": float(row.get("mass_2_source", row.get("m2", np.nan))),
                            "distance": float(row.get("luminosity_distance", row.get("distance", np.nan))),
                            # Source classification probabilities
                            "p_bbh": float(row.get("p_astro", row.get("pastro", np.nan))),
                            "p_bns": float(row.get("p_BNS", np.nan)),
                            "p_nsbh": float(row.get("p_NSBH", np.nan)),
                            "catalog": catalog,
                            "observing_run": run_name,
                        }
                        
                        # Only add if GPS is valid
                        if not np.isnan(event["gps"]):
                             # Convert np.nan to None for JSON serialization
                            for k, v in event.items():
                                if isinstance(v, float) and np.isnan(v):
                                    event[k] = None
                                    
                            events.append(event)
                            
                    except (KeyError, ValueError, TypeError) as e:
                        logging.debug(f"Skipping event row: {e}")
                        continue
                        
            except Exception as e:
                logging.warning(f"Failed to fetch catalog {catalog}: {e}")
                continue
        
        # Save cache
        try:
            with open(cache_file, 'w') as f:
                json.dump(events, f)
        except Exception as e:
             logging.warning(f"Failed to save event cache: {e}")
    
    # Filter by observing run if requested
    target_runs = set()
    if observing_runs:
        for run in observing_runs:
            if hasattr(run, 'value'):
                target_runs.add(run.value.name if hasattr(run.value, 'name') else str(run))
            else:
                target_runs.add(str(run))
                
    for event in events:
        gps_key = round(event["gps"], 1)  # Round to 0.1s for deduplication
        
        # Check if event belongs to requested run
        if target_runs and event["observing_run"] not in target_runs:
            continue
            
        # Check if event matches requested names
        if event_names and event["name"] not in event_names:
            continue
            
        if gps_key not in seen_gps:
            seen_gps.add(gps_key)
            unique_events.append(event)
    
    # Sort by GPS time
    unique_events.sort(key=lambda x: x["gps"])
    
    logging.info(f"Fetched {len(unique_events)} events with PE parameters")
    return unique_events


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
            source_type = SourceType(source_type.upper())
        except ValueError:
            logging.warning(f"Unknown source type: {source_type}. Ignoring filter.")
            source_type = None

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
            m1 = event.get("mass1", np.nan)
            m2 = event.get("mass2", np.nan)
            if np.isnan(m1) or np.isnan(m2):
                continue
            # Neutron star mass threshold ~3 solar masses
            ns_threshold = 3.0
            is_ns1 = m1 < ns_threshold
            is_ns2 = m2 < ns_threshold
            
            is_bbh = (not is_ns1) and (not is_ns2)
            is_bns = is_ns1 and is_ns2
            is_nsbh = is_ns1 != is_ns2
            
            if source_type == SourceType.BBH and not is_bbh:
                continue
            elif source_type == SourceType.BNS and not is_bns:
                continue
            elif source_type == SourceType.NSBH and not is_nsbh:
                continue
        
        # Name filter
        if name_contains is not None:
            if name_contains.lower() not in event.get("name", "").lower():
                continue
        
        matching_names.append(event["name"])
    
    return matching_names
