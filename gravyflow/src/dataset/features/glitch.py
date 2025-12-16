from enum import Enum, auto
from typing import Union, List
from pathlib import Path
import time
import hashlib
#import threading
import queue

import numpy as np
import pandas as pd
from gwpy.table import GravitySpyTable

import gravyflow as gf


def _get_cache_path(selection: str) -> Path:
    """Generate cache file path based on selection hash."""
    cache_dir = gf.PATH / "res" / "glitch_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Hash the selection string for a unique filename
    selection_hash = hashlib.md5(selection.encode()).hexdigest()[:16]
    return cache_dir / f"glitch_query_{selection_hash}.csv"


def fetch_event_times(selection, max_retries=10, use_cache=True):
    """
    Fetch glitch event times from GravitySpy with local caching.
    
    Cache files are stored in gravyflow/res/glitch_cache/ for fast reuse.
    """
    # Check cache first
    if use_cache:
        cache_path = _get_cache_path(selection)
        if cache_path.exists():
            try:
                data = pd.read_csv(cache_path)
                return data
            except Exception as e:
                print(f"Cache read failed: {e}, fetching from server...")
    
    attempts = 0
    while True:
        try:
            # Attempt to fetch the data
            data = GravitySpyTable.fetch(
                "gravityspy",
                "glitches",
                columns=["event_time", "duration"],
                selection=selection
            ).to_pandas()

            if attempts == 0 and "ml_label" in selection:
                 print(f"DEBUG: Found {len(data)} events for selection='{selection}'")

            # Cache the result
            if use_cache and len(data) > 0:
                try:
                    cache_path = _get_cache_path(selection)
                    data.to_csv(cache_path, index=False)
                except Exception as e:
                    print(f"Cache write failed: {e}")

            return data

        except Exception as e:
            msg = str(e)
            print(f"Failed to acquire gravity spy data because: {msg} retrying...")
            
            # Don't retry on syntax errors
            if "Cannot parse filter definition" in msg or "SyntaxError" in msg or "unterminated string literal" in msg:
                raise e
                
            # If an exception occurs, increment the attempts counter
            attempts += 1
            # Check if the maximum number of retries has been reached
            if attempts >= max_retries:
                raise Exception(f"Max retries reached: {max_retries}") from e
            
            # Wait for 30 seconds before retrying
            time.sleep(30)
    
    return -1  # If successful, return the data

class GlitchType(Enum):
    AIR_COMPRESSOR = 'Air_Compressor'
    BLIP = 'Blip'
    CHIRP = 'Chirp'
    EXTREMELY_LOUD = 'Extremely_Loud'
    HELIX = 'Helix'
    KOI_FISH = 'Koi_Fish'
    LIGHT_MODULATION = 'Light_Modulation'
    LOW_FREQUENCY_BURST = 'Low_Frequency_Burst'
    LOW_FREQUENCY_LINES = 'Low_Frequency_Lines'
    NO_GLITCH = 'No_Glitch'
    NONE_OF_THE_ABOVE = 'None_of_the_Above'
    PAIRED_DOVES = 'Paired_Doves'
    POWER_LINE = 'Power_Line'
    REPEATING_BLIPS = 'Repeating_Blips'
    SCATTERED_LIGHT = 'Scattered_Light'
    SCRATCHY = 'Scratchy'
    TOMTE = 'Tomte'
    VIOLIN_MODE = 'Violin_Mode'
    WANDERING_LINE = 'Wandering_Line'
    WHISTLE = 'Whistle'

def get_glitch_times(
    ifo: gf.IFO,
    observing_run: gf.ObservingRun = gf.ObservingRun.O3,
    glitch_types: Union[List[GlitchType], GlitchType] = None,
    start_gps_time : float = None,
    end_gps_time : float = None
) -> np.ndarray:
    
    if start_gps_time is None:
        start_gps_time = observing_run.value.start_gps_time
    if end_gps_time is None:
        end_gps_time = observing_run.value.end_gps_time
    
    ifo = ifo.name

    # Initialize an empty list to hold the data from each query.
    all_data = []
    
    # This condition checks if glitch_types is not None and not an empty list.
    if glitch_types:
        # Ensure glitch_types is a list for easier processing.
        if not isinstance(glitch_types, list):
            glitch_types = [glitch_types]

        # Loop over each glitch type and perform individual queries.
        for glitch_type in glitch_types:
            glitch_name = glitch_type.value
            selection = f"ifo = '{ifo}' AND event_time >= {start_gps_time} AND event_time <= {end_gps_time} AND ml_label = '{glitch_name}' AND No_Glitch < 0.1"
            
            data = fetch_event_times(selection, max_retries=10)['event_time'].to_numpy()
            
            # Append the results to the all_data list.
            all_data.append(data)

        # Concatenate all the numpy arrays from each query into one array.
        return np.concatenate(all_data)
    else:
        # If glitch_types is None or an empty list, it selects all glitch types.
        selection = f"ifo={ifo} && event_time>{start_gps_time} & event_time<{end_gps_time} && No_Glitch<0.1"

        data = fetch_event_times(selection, max_retries=10)['event_time'].to_numpy()

        return data


def get_glitch_times_with_labels(
    ifo: gf.IFO,
    observing_run: gf.ObservingRun = gf.ObservingRun.O3,
    glitch_types: Union[List[GlitchType], GlitchType] = None,
    start_gps_time: float = None,
    end_gps_time: float = None,
    balanced: bool = False
) -> tuple:
    """
    Get glitch times with type labels for classification.
    
    Args:
        ifo: Detector to query
        observing_run: Observing run for default time bounds
        glitch_types: List of glitch types to include. If None, includes all types.
        start_gps_time: Start GPS time (defaults to observing run start)
        end_gps_time: End GPS time (defaults to observing run end)
        balanced: If True, sample equally from each type
        
    Returns:
        Tuple of (gps_times, type_indices) where type_indices are integers
        corresponding to GlitchType enum order (0-19)
    """
    if start_gps_time is None:
        start_gps_time = observing_run.value.start_gps_time
    if end_gps_time is None:
        end_gps_time = observing_run.value.end_gps_time
    
    # Get all GlitchType values or use provided subset
    # Exclude NO_GLITCH by default as it's a placeholder (always returns 0 results with No_Glitch < 0.1 filter)
    if glitch_types is None:
        glitch_types = [gt for gt in GlitchType if gt != GlitchType.NO_GLITCH]
    elif not isinstance(glitch_types, list):
        glitch_types = [glitch_types]
    
    ifo_name = ifo.name
    all_times = []
    all_labels = []
    
    for glitch_type in glitch_types:
        glitch_name = glitch_type.value
        # Get enum index (0-19)
        type_index = list(GlitchType).index(glitch_type)
        
        selection = f"ifo = '{ifo_name}' AND event_time >= {start_gps_time} AND event_time <= {end_gps_time} AND ml_label = '{glitch_name}' AND No_Glitch < 0.1"
        
        try:
            data = fetch_event_times(selection, max_retries=10)['event_time'].to_numpy()
            all_times.append(data)
            all_labels.append(np.full(len(data), type_index, dtype=np.int32))
        except Exception as e:
            print(f"Warning: Failed to fetch {glitch_name}: {e}")
            all_times.append(np.array([]))
            all_labels.append(np.array([], dtype=np.int32))
            continue
    
    if not all_times:
        return np.array([]), np.array([], dtype=np.int32)
    
    times = np.concatenate(all_times)
    labels = np.concatenate(all_labels)
    
    # Balance classes if requested
    if balanced and len(all_times) > 1:
        # Filter out empty classes for balancing purposes
        non_empty_indices = [i for i, t in enumerate(all_times) if len(t) > 0]
        
        if non_empty_indices:
             # Find MAX count across NON-EMPTY types for oversampling
            max_count = max(len(all_times[i]) for i in non_empty_indices)
            
            print(f"DEBUG: Non-empty indices: {non_empty_indices}")
            print(f"DEBUG: Max count: {max_count}")
            
            balanced_times = []
            balanced_labels = []
            
            for i, (type_times, type_idx) in enumerate(zip(all_times, [list(GlitchType).index(gt) for gt in glitch_types])):
                if len(type_times) > 0:
                    # Oversample to max_count with replacement
                    indices = np.random.choice(len(type_times), max_count, replace=True)
                    balanced_times.append(type_times[indices])
                    balanced_labels.append(np.full(max_count, type_idx, dtype=np.int32))
            
            times = np.concatenate(balanced_times)
            labels = np.concatenate(balanced_labels)
            print(f"DEBUG: Balanced total times (Oversampled): {len(times)}")
        else:
            # All empty
            print("DEBUG: All classes empty")
            times = np.array([])
            labels = np.array([], dtype=np.int32)
    
    return times, labels
    
def get_glitch_segments(
    ifo: gf.IFO,
    observing_run: gf.ObservingRun = gf.ObservingRun.O3,
    glitch_types: Union[List[GlitchType], GlitchType] = None,
    start_gps_time : float = None,
    end_gps_time : float = None,
    ) -> np.ndarray:
    
    if start_gps_time is None:
        start_gps_time = observing_run.value.start_gps_time
    if end_gps_time is None:
        end_gps_time = observing_run.value.end_gps_time
    
    ifo = ifo.name

    # Initialize an empty list to hold the data from each query.
    all_data = []
    
    # This condition checks if glitch_types is not None and not an empty list.
    if glitch_types:
        # Ensure glitch_types is a list for easier processing.
        if not isinstance(glitch_types, list):
            glitch_types = [glitch_types]

        # Loop over each glitch type and perform individual queries.
        for glitch_type in glitch_types:
            glitch_name = glitch_type.value
            selection = f"ifo = '{ifo}' AND event_time >= {start_gps_time} AND event_time <= {end_gps_time} AND ml_label = '{glitch_name}' AND No_Glitch < 0.1"
            
            data = fetch_event_times(selection, max_retries=10)
            
            # Calculate 'end_time' by adding 'duration' to 'start_time'
            data['end_time'] = data['event_time'] + data['duration'] 
            data['start_time'] = data['event_time']

            # Select the 'start_time' and 'end_time' and convert to a NumPy array
            data = data[['start_time', 'end_time']].to_numpy()
            
            # Append the results to the all_data list.
            all_data.append(data)

        # Concatenate all the numpy arrays from each query into one array.
        return np.concatenate(all_data)
    else:
        # If glitch_types is None or an empty list, it selects all glitch types.
        selection = f"ifo={ifo} && event_time>{start_gps_time} & event_time<{end_gps_time} && No_Glitch<0.1"

        data = fetch_event_times(selection, max_retries=10)
        
        # Calculate 'end_time' by adding 'duration' to 'start_time'
        
        data['end_time'] = data['event_time'] + data['duration']
        data['start_time'] = data['event_time']

        # Select the 'start_time' and 'end_time' and convert to a NumPy array
        data = data[['start_time', 'end_time']].to_numpy()

        return data
def get_glitch_type_from_index(index: int) -> Union[GlitchType, None]:
    """
    Convert integer index back to GlitchType enum.
    
    Args:
        index: Integer index (0-19) or -1 for unknown.
        
    Returns:
        GlitchType enum member or None if index is -1.
        
    Raises:
        ValueError: If index is out of range.
    """
    if index == -1:
        return None
        
    try:
        return list(GlitchType)[index]
    except IndexError:
        raise ValueError(f"Invalid glitch type index: {index}. Must be between 0 and {len(GlitchType)-1}.")
