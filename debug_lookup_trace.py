"""
Debug script to trace exactly where label lookup fails.
Adds logging to confirm the hypothesis.
"""
import logging
import numpy as np
import gravyflow as gf

# Enable DEBUG level
logging.basicConfig(level=logging.DEBUG, format='%(name)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

# Configure for glitch acquisition
ifo_data_obtainer = gf.IFODataObtainer(
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.GLITCHES],
    observing_runs=[gf.ObservingRun.O1],
    saturation=1.0,
    balanced_glitch_types=True,
)

print("="*80)
print("INITIAL STATE CHECK")
print("="*80)
print(f"valid_segments: {ifo_data_obtainer.valid_segments}")
print(f"Has _feature_index attr: {hasattr(ifo_data_obtainer, '_feature_index')}")
if hasattr(ifo_data_obtainer, '_feature_index'):
    print(f"_feature_index is None: {ifo_data_obtainer._feature_index is None}")
print(f"Has _gps_key_to_record attr: {hasattr(ifo_data_obtainer, '_gps_key_to_record')}")
if hasattr(ifo_data_obtainer, '_gps_key_to_record'):
    print(f"_gps_key_to_record length: {len(ifo_data_obtainer._gps_key_to_record)}")

# Create wrapper
glitch_obtainer = gf.TransientObtainer(
    ifo_data_obtainer=ifo_data_obtainer,
    ifos=[gf.IFO.L1],
)

# MANUALLY call get_valid_segments to see what happens
print("\n" + "="*80)
print("CALLING get_valid_segments DIRECTLY")
print("="*80)
segments = ifo_data_obtainer.get_valid_segments(
    ifos=[gf.IFO.L1],
    seed=42,
    groups={'train': 1.0},
    group_name='train'
)
print(f"Returned {len(segments)} segments")

print("\n" + "="*80)
print("POST get_valid_segments STATE")
print("="*80)
print(f"_feature_index is None: {ifo_data_obtainer._feature_index is None}")
if ifo_data_obtainer._feature_index:
    print(f"_feature_index length: {len(ifo_data_obtainer._feature_index)}")
print(f"_gps_key_to_record length: {len(ifo_data_obtainer._gps_key_to_record)}")

# Test lookup on first segment
if len(segments) > 0:
    test_gps = np.mean(segments[0, 0])
    print(f"\nTesting lookup for GPS: {test_gps}")
    
    from gravyflow.src.utils.gps import gps_to_key
    key = gps_to_key(test_gps)
    print(f"GPS key: {key}")
    print(f"Key in map: {key in ifo_data_obtainer._gps_key_to_record}")
    
    if key in ifo_data_obtainer._gps_key_to_record:
        record = ifo_data_obtainer._gps_key_to_record[key]
        print(f"Found record: {record.kind} @ {record.transient_gps_time}")
    
    # Try the lookup function
    labels = ifo_data_obtainer._lookup_labels([test_gps])
    print(f"_lookup_labels returned: {labels}")

print("\n" + "="*80)
print("NOW TESTING WITH WRAPPER (deepcopy path)")
print("="*80)

# Create generator
glitch_generator = glitch_obtainer(
    precache_cap=0,
    sample_rate_hertz=2048.0,
    onsource_duration_seconds=1.0,
    offsource_duration_seconds=1.0,
    num_examples_per_batch=4,
    group="train",
    scale_factor=1.0,
    seed=42,
    crop=False,
    whiten=False
)

# Get batch
batch = next(glitch_generator)
glitch_types = batch.get(gf.ReturnVariables.GLITCH_TYPE)

print(f"\nGlitch Types from batch: {glitch_types}")
print(f"Number of -1s: {np.sum(glitch_types == -1)}")
print(f"Number of valid: {np.sum(glitch_types != -1)}")
