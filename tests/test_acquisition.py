from pathlib import Path

import gravyflow as gf

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
        gf.IFODataObtainer(
            gf.ObservingRun.O3, 
            gf.DataQuality.BEST, 
            [
                gf.DataLabel.NOISE, 
                gf.DataLabel.GLITCHES
            ]
        )

    # Get valid data segments:
    segments = data_obtainer.get_valid_segments(
        [gf.IFO.L1],
        groups,
        "train",
        gf.SegmentOrder.RANDOM
    )
    
    assert len(segments) > 10000, \
        f"Num segments found {len(segments)} is too low!"
    

    
    
        