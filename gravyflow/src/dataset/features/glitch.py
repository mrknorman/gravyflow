from enum import Enum, auto
from typing import Union, List
import time
#import threading
import queue

import numpy as np
from gwpy.table import GravitySpyTable

import gravyflow as gf

def fetch_event_times(selection, max_retries=10):
        
    attempts = 0
    while True:
        try:
            # Attempt to fetch the data
            data = GravitySpyTable.fetch(
                "gravityspy",
                "glitches",
                columns=["event_time", "duration"],  # Assuming we're only interested in the event times.
                selection=selection
            ).to_pandas()

            return data

        except Exception as e:
            print(f"Failed to acquire gravity spy data because: {e} retrying...")
            # If an exception occurs, increment the attempts counter
            attempts += 1
            # Check if the maximum number of retries has been reached
            if attempts >= max_retries:
                raise Exception(f"Max retries reached: {max_retries}") from e
            
            # Wait for 10 seconds before retrying
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
            selection = f"ifo={ifo} && event_time>{start_gps_time} & event_time<{end_gps_time} && ml_label={glitch_name} && No_Glitch<0.1"
            
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
            selection = f"ifo={ifo} && event_time>{start_gps_time} & event_time<{end_gps_time} && ml_label={glitch_name} && No_Glitch<0.1"
            
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