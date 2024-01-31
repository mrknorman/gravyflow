from pathlib import Path
from typing import Dict

import gravyflow as gf

def test_acquisition(
        num_examples_per_batch : int = 32,
        max_segment_duration_seconds : float = 2048.0,
        onsource_duration_seconds : float = 1.0,
        offsource_duration_seconds : float = 16.0,
        window_duration_seconds : float = 1.0,
        sample_rate_hertz : float = 2048.0,
        data_directory_path : Path = Path("./test_directory"),
        groups : Dict = {
            "train" : 0.98,
            "validate" : 0.01,
            "test" : 0.01
        }
    ) -> bool:
     
    # Derived parameters:
    min_segment_duration_seconds : float = (
        (onsource_duration_seconds + window_duration_seconds) 
        * num_examples_per_batch + offsource_duration_seconds
    )
    
    # Setup data IFOConfig:
    data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
        observing_runs=gf.ObservingRun.O3, 
        data_quality=gf.DataQuality.BEST, 
        data_labels=[
            gf.DataLabel.NOISE, 
            gf.DataLabel.GLITCHES
        ]
    )

    # Get valid data segments:
    segments = data_obtainer.get_valid_segments(
        ifos=[gf.IFO.L1],
        seed=1000,
        groups=groups,
        group_name="train",
        segment_order=gf.SegmentOrder.RANDOM
    )
    
    # Assert expected number of data segments found:
    assert len(segments) > 10000, f"Num segments found, {len(segments)}, is too low!"
    
    return True

if __name__ == "__main__":
    test_acquisition()