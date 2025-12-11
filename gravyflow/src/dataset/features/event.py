"""
Event acquisition for gravitational wave events from GWTC catalogs.

Provides EventType enum (CONFIDENT/MARGINAL) mirroring GlitchType,
plus functions to fetch event times from GWTC catalogs.
"""

from enum import Enum, auto
from typing import List, Optional
from pathlib import Path
import logging

import numpy as np
from gwpy.table import EventTable

import gravyflow as gf


class EventType(Enum):
    """
    Type of gravitational wave event.
    
    Mirrors GlitchType for symmetry. Use DataLabel.EVENTS to get all types.
    """
    CONFIDENT = 'confident'  # GWTC *-confident catalogs (confirmed events)
    MARGINAL = 'marginal'    # GWTC *-marginal catalogs (sub-threshold triggers)


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
    event_types: List[EventType],
    use_cache: bool = True
) -> np.ndarray:
    """
    Fetch GPS times for specified event types.
    
    Args:
        event_types: List of EventType values to fetch
        use_cache: Whether to use cache
        
    Returns:
        Array of GPS times for requested event types
    """
    times = []
    
    for event_type in event_types:
        if event_type == EventType.CONFIDENT:
            times.append(get_confident_event_times(use_cache))
        elif event_type == EventType.MARGINAL:
            times.append(get_marginal_event_times(use_cache))
    
    if not times:
        return np.array([])
    
    return np.concatenate(times)


def get_confident_events_with_params(
    observing_runs: List = None
) -> List[dict]:
    """
    Fetch confirmed GW events with full parameter estimation data.
    
    Returns a list of dictionaries containing event metadata and PE parameters
    suitable for validation against model predictions.
    
    Args:
        observing_runs: Optional list of ObservingRun enums to filter by.
                       If None, returns all O1/O2/O3 events.
    
    Returns:
        List of dicts with keys:
            - name: Event name (e.g., "GW150914")
            - gps: GPS time of event
            - mass1: Primary mass (Msun)
            - mass2: Secondary mass (Msun)
            - distance: Luminosity distance (Mpc)
            - catalog: Source catalog name
            - observing_run: O1/O2/O3
    """
    events = []
    
    # Map catalogs to observing runs
    catalog_run_map = {
        "GWTC": "O1",
        "GWTC-1-confident": "O1",
        "GWTC-2": "O2",
        "GWTC-2.1-confident": "O2",
        "GWTC-3-confident": "O3",
    }
    
    # Filter catalogs by observing run if specified
    catalogs_to_fetch = CONFIDENT_CATALOGS
    if observing_runs:
        run_names = set()
        for run in observing_runs:
            if hasattr(run, 'value'):
                run_names.add(run.value.name if hasattr(run.value, 'name') else str(run))
            else:
                run_names.add(str(run))
        
        catalogs_to_fetch = [
            cat for cat in CONFIDENT_CATALOGS 
            if catalog_run_map.get(cat) in run_names
        ]
    
    for catalog in catalogs_to_fetch:
        try:
            table = EventTable.fetch_open_data(catalog)
            
            for row in table:
                try:
                    event = {
                        "name": str(row.get("commonName", row.get("name", "Unknown"))),
                        "gps": float(row["GPS"]),
                        "mass1": float(row.get("mass_1_source", row.get("m1", np.nan))),
                        "mass2": float(row.get("mass_2_source", row.get("m2", np.nan))),
                        "distance": float(row.get("luminosity_distance", row.get("distance", np.nan))),
                        # Source classification probabilities
                        "p_bbh": float(row.get("p_astro", row.get("pastro", np.nan))) if "p_astro" in row.colnames or "pastro" in row.colnames else np.nan,
                        "p_bns": float(row.get("p_BNS", np.nan)) if "p_BNS" in row.colnames else np.nan,
                        "p_nsbh": float(row.get("p_NSBH", np.nan)) if "p_NSBH" in row.colnames else np.nan,
                        "catalog": catalog,
                        "observing_run": catalog_run_map.get(catalog, "Unknown"),
                    }
                    
                    # Only add if GPS is valid
                    if not np.isnan(event["gps"]):
                        events.append(event)
                        
                except (KeyError, ValueError, TypeError) as e:
                    logging.debug(f"Skipping event row: {e}")
                    continue
                    
        except Exception as e:
            logging.warning(f"Failed to fetch catalog {catalog}: {e}")
            continue
    
    # Remove duplicates by GPS time (some events appear in multiple catalogs)
    seen_gps = set()
    unique_events = []
    for event in events:
        gps_key = round(event["gps"], 1)  # Round to 0.1s for deduplication
        if gps_key not in seen_gps:
            seen_gps.add(gps_key)
            unique_events.append(event)
    
    # Sort by GPS time
    unique_events.sort(key=lambda x: x["gps"])
    
    logging.info(f"Fetched {len(unique_events)} confirmed events with PE parameters")
    return unique_events
