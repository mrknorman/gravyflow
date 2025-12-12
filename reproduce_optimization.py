
import time
import gravyflow as gf
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# 4. Verify Catalog Cache Speed
print("\n--- Testing Catalog Caching ---")
from gravyflow.src.dataset.features.event import get_confident_events_with_params

print("Fetching catalog (Run 1 - should be slow if not cached)...")
start = time.time()
events_1 = get_confident_events_with_params()
end = time.time()
print(f"Run 1 Time: {end - start:.4f}s")

print("Fetching catalog (Run 2 - should be instant)...")
start = time.time()
events_2 = get_confident_events_with_params()
end = time.time()
print(f"Run 2 Time: {end - start:.4f}s")

if (end - start) < 1.0:
    print("SUCCESS: Catalog caching effective.")
else:
    print("FAILURE: Catalog caching faulty?")

# 1. Setup specific event request
target_event = "GW190412" # A confident event

# Setup IFOObtainer
ifo_obt = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.EVENTS],
    event_types=[gf.EventType.CONFIDENT],
    force_acquisition=True # Force new acquisition to test cache vs fresh
)

# Setup TransientObtainer
transient_obt = gf.TransientObtainer(
    ifo_data_obtainer=ifo_obt,
    event_names=[target_event]
)

# 2. Trigger acquisition (should only fetch GW190412)
print(f"Acquiring {target_event}...")
generator = transient_obt(
    sample_rate_hertz=2048.0,
    onsource_duration_seconds=1.0,
    offsource_duration_seconds=1.0,
    num_examples_per_batch=1
)

# Fetch one batch
batch = next(generator)
onsource, offsource, gps = batch

print(f"Batch GPS shape: {gps.shape}")
print(f"Batch GPS value: {gps[0].item()}")

# Verify we got an event
expected_gps = 1239082262.1 # GW190412 GPS (approx)
# Check if close
diff = abs(gps[0].item() - expected_gps)
print(f"GPS Diff from GW190412: {diff:.4f}s")

if diff < 1.0:
    print("SUCCESS: Retrieved targeted event.")
else:
    print("FAILURE: Retrieved wrong event?")

# 3. Verify Cache Structure (Dictionary/GPS based)
print("\n--- Testing Precache Keys manually ---")
# Manually simulate what TransientObtainer does
# We need to manually set feature segments on ifo_obt
ifo_obt.feature_segments = np.array([[1234567890.0, 1234567900.0]])
cache_path = ifo_obt.precache_features(
    ifos=[gf.IFO.L1],
    sample_rate_hertz=2048.0,
    onsource_duration_seconds=1.0
)

print(f"Cache created at: {cache_path}")

# Open and check keys
import h5py
with h5py.File(cache_path, 'r') as f:
    print("Keys in cache file:")
    found_gps_key = False
    for k in f.keys():
        print(f" - {k}")
        if "segments" in k:
            # Check subgroup
            print(f"   Subgroups in {k}:")
            for sk in f[k].keys():
                print(f"    - {sk}")
                if "segment_1234567890.0" in sk:
                    found_gps_key = True

if found_gps_key:
    print("SUCCESS: Found GPS-based key in cache.")
else:
    print("FAILURE: Did not find GPS-based key.")

