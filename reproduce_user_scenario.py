
import time
import gravyflow as gf
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)

print("--- Testing User Scenario (O3 Config + O1/O2 Events) ---")
print("Expected: Warning about Observing Run Mismatch.")

start_total = time.time()

# User provided snippet:
# transient_obtainer = next(gf.TransientObtainer(
#     ifo_data_obtainer=gf.IFODataObtainer(
#         observing_runs=gf.ObservingRun.O3,
#         data_quality=gf.DataQuality.BEST,
#         data_labels=[gf.DataLabel.EVENTS]
#     ),
#     ifos=[gf.IFO.H1],
#     event_names=["GW150914", "GW170817"] 
# )())

print("Initializing...")
ifo_obt = gf.IFODataObtainer(
    observing_runs=gf.ObservingRun.O3,
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.EVENTS],
    force_acquisition=True # Force to ensure we hit acquisition path
)

transient_obt = gf.TransientObtainer(
    ifo_data_obtainer=ifo_obt,
    ifos=[gf.IFO.H1],
    event_names=["GW150914", "GW170817"]
)

print("Generating...")
start_gen = time.time()
generator = transient_obt()
batch = next(generator)
end_gen = time.time()

print(f"Generation took: {end_gen - start_gen:.4f}s")
print(f"Total time: {end_gen - start_total:.4f}s")

# Inspect result
gps = batch[2]
print(f"Batch GPS: {gps}")
