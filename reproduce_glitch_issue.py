import logging
import numpy as np
import gravyflow as gf
from gravyflow.src.dataset.noise.acquisition.transient import TransientDataObtainer
from gravyflow.src.dataset.noise.acquisition.base import DataQuality, DataLabel, ObservingRun
from gravyflow.src.dataset.features.glitch import GlitchType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_glitch_retrieval():
    print("Initializing TransientDataObtainer for Glitches...")
    obtainer = TransientDataObtainer(
        data_quality=DataQuality.BEST,
        data_labels=[DataLabel.GLITCHES],
        observing_runs=[ObservingRun.O1], # Use O1 for speed
        saturation=1.0 # 1 sample per glitch
    )
    
    print("\nCalling get_valid_segments...")
    # This should trigger build_feature_index
    segments = obtainer.get_valid_segments(
        ifos=[gf.IFO.L1],
        seed=42,
        group_name='train',
        groups={'train': 1.0}
    )
    print(f"Got {len(segments)} segments")
    
    if len(segments) == 0:
        print("No segments found! Check build_feature_index.")
        return

    print(f"\nFeatureIndex populated: {obtainer._feature_index is not None}")
    if obtainer._feature_index:
        print(f"Records in index: {len(obtainer._feature_index)}")
    
    print(f"GPS Key Cache size: {len(obtainer._gps_key_to_record)}")

    # Check lookup for first segment
    seg_start, seg_end = segments[0, 0]
    center_gps = (seg_start + seg_end) / 2.0
    print(f"\nChecking lookup for GPS {center_gps} (derived from segment)")
    
    # Try exact record lookup
    record = obtainer.get_record_for_gps(center_gps)
    print(f"Record found via get_record_for_gps: {record is not None}")
    if record:
        print(f"Record GPS: {record.transient_gps_time}")
        print(f"Record Kind: {record.kind}")
    
    # Try label lookup
    labels = obtainer._lookup_labels([center_gps])
    print(f"Label lookup result: {labels}")
    
    # Check if _get_sample_from_cache would work (logic check)
    from gravyflow.src.utils.gps import gps_to_key
    key = gps_to_key(center_gps)
    print(f"GPS Key: {key}")
    print(f"Key in cache: {key in obtainer._gps_key_to_record}")
    
    # Try getting a batch (simplified)
    print("\nAttempting to fetch data batch...")
    try:
        batch = next(obtainer.get_batch(
            batch_size=4,
            sample_rate_hertz=2048.0,
            onsource_duration_seconds=1.0,
            offsource_duration_seconds=1.0,
            ifos=[gf.IFO.L1], 
            scale_factor=1.0
        ))
        
        X, y = batch
        print(f"Batch X shape: {X.shape}")
        print(f"Batch y shape: {y.shape}")
        print(f"Batch y values: {y}")
        
        # Check for Nones or Zeros
        if np.all(X == 0):
            print("WARNING: Batch data is all zeros!")
        else:
            print("Batch data contains non-zero values.")
            
    except Exception as e:
        print(f"Batch retrieval failed: {e}")

if __name__ == "__main__":
    test_glitch_retrieval()
