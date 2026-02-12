"""
Event acquisition for gravitational wave events from GWTC catalogs.

Provides EventConfidence enum (CONFIDENT/MARGINAL) for catalog confidence level,
plus functions to fetch event times from GWTC catalogs.
"""

from enum import Enum, IntEnum, auto
from typing import List, Optional, Union
from pathlib import Path
import logging

import numpy as np
from gwpy.table import EventTable

import gravyflow as gf


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


def _fetch_event_times_from_catalogs(
    catalogs: List[str],
    cache_file: Optional[Path] = None
) -> np.ndarray:
    """
    Fetch GPS times from specified GWTC catalogs.
    
    Args:
        catalogs: List of catalog names to query
        cache_file: Optional path to cache results
        
    Returns:
        Array of GPS times
    """
    if cache_file is not None and cache_file.exists():
        logging.info(f"Loading event times from cache: {cache_file}")
        return np.load(cache_file)
    
    logging.info(f"Fetching event times from catalogs: {catalogs}")
    gps_times = np.array([])
    
    for catalogue in catalogs:
        try:
            events = EventTable.fetch_open_data(catalogue)
            gps_times = np.append(gps_times, events["GPS"].data.compressed())
        except Exception as e:
            logging.warning(f"Failed to fetch {catalogue}: {e}")
            continue
    
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, gps_times)
        logging.info(f"Cached event times to: {cache_file}")
    
    return gps_times


def get_confident_event_times(use_cache: bool = True) -> np.ndarray:
    """
    Fetch GPS times of confirmed GW events from GWTC *-confident catalogs.
    
    Args:
        use_cache: Whether to use/update cache file
        
    Returns:
        Array of GPS times for confident events
    """
    cache_file = gf.PATH / "res/cached_confident_event_times.npy" if use_cache else None
    return _fetch_event_times_from_catalogs(CONFIDENT_CATALOGS, cache_file)


def get_marginal_event_times(use_cache: bool = True) -> np.ndarray:
    """
    Fetch GPS times of marginal/candidate events from GWTC *-marginal catalogs.
    
    Args:
        use_cache: Whether to use/update cache file
        
    Returns:
        Array of GPS times for marginal events
    """
    cache_file = gf.PATH / "res/cached_marginal_event_times.npy" if use_cache else None
    return _fetch_event_times_from_catalogs(MARGINAL_CATALOGS, cache_file)


def get_all_event_times(use_cache: bool = True) -> np.ndarray:
    """
    Fetch GPS times of all events (confident + marginal).
    
    Backward compatible with original get_all_event_times behavior.
    
    Args:
        use_cache: Whether to use/update cache file
        
    Returns:
        Array of GPS times for all events
    """
    confident = get_confident_event_times(use_cache)
    marginal = get_marginal_event_times(use_cache)
    return np.concatenate([confident, marginal])


def get_event_times_by_type(
    event_types: List[EventConfidence],
    use_cache: bool = True
) -> np.ndarray:
    """
    Fetch GPS times for specified event types.
    
    Args:
        event_types: List of EventConfidence values to fetch
        use_cache: Whether to use cache
        
    Returns:
        Array of GPS times for requested event types
    """
    times = []
    
    for event_type in event_types:
        if event_type == EventConfidence.CONFIDENT:
            times.append(get_confident_event_times(use_cache))
        elif event_type == EventConfidence.MARGINAL:
            times.append(get_marginal_event_times(use_cache))
    
    if not times:
        return np.array([])
    
    return np.concatenate(times)


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
