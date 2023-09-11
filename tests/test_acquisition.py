from ..acquisition import (IFODataObtainer, ObservingRun, DataQuality,
                               DataLabel, SegmentOrder, IFO)
from pathlib import Path

if __name__ == "__main__":
    # User parameters:
    num_examples_per_batch = 32
    max_segment_duration_seconds = 2048.0
    onsource_duration_seconds = 1.0
    offsource_duration_seconds = 16.0
    window_duration_seconds = 1.0
    sample_rate_hertz = 2048.0
    data_directory_path = Path("./test_directory")
    groups = \
        {
            "train" : 0.98,
            "validate" : 0.01,
            "test" : 0.01
        }
    
    # Derived parameters:
    min_segment_duration_seconds = \
        (onsource_duration_seconds + window_duration_seconds) * \
        num_examples_per_batch + offsource_duration_seconds
    
    # Setup data IFOConfig:
    data_obtainer = \
        IFODataObtainer(
            ObservingRun.O3, 
            DataQuality.BEST, 
            [
                DataLabel.NOISE, 
                DataLabel.GLITCHES
            ]
        )

    # Get valid data segments:
    data_obtainer.get_valid_segments(
        IFO.L1,
        groups,
        "train",
        SegmentOrder.RANDOM
    )
    
        