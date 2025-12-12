
import matplotlib.pyplot as plt
import numpy as np
import gravyflow as gf

# Example setup from your request
# Correct usage: Initialize -> Call -> Next
obtainer = gf.TransientObtainer(
    ifo_data_obtainer=gf.IFODataObtainer(
        observing_runs=gf.ObservingRun.O1, # or O3
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.EVENTS]
    ),
    ifos=[gf.IFO.H1, gf.IFO.L1],
    event_names=["GW150914", "GW170817"]
)

# Create generator
generator = obtainer()

# Fetch one batch
print("Fetching batch...")
batch = next(generator)
# batch format: (strain_data, background_data, gps_times)
strain_data = batch[0] # Shape: (Batch, Ifo, Time)
gps_times = batch[2]

print(f"Strain Data Shape: {strain_data.shape}")

# Plotting with gf.generate_strain_plot (Existing Utility)
from bokeh.io import save, output_file
import numpy as np

print("Generating Bokeh plot...")

# generate_strain_plot expects a dictionary where keys are names and values are arrays
# We need to constructing the dictionary from the (Batch, Detector, Time) array.
# For simplicity, let's plot just the FIRST batch item if batch size > 1, 
# or loop to create multiple plots.

# Let's plot the FIRST event in the batch for this example:
event_idx = 0
strain_sample = strain_data[event_idx] # Shape: (2, Time) for H1, L1
ifo_names = ["H1", "L1"]

# Construct dictionary: {"H1": array, "L1": array}
strain_dict = {
    name: strain_sample[i] 
    for i, name in enumerate(ifo_names)
    if i < strain_sample.shape[0]
}

plot = gf.generate_strain_plot(
    strain=strain_dict,
    sample_rate_hertz=4096.0, # Default or from config
    title=f"Event GPS: {gps_times[event_idx, 0]:.2f}",
    # generate_strain_plot can handle multiple events if passed a list of dicts,
    # but let's keep it simple for this example.
)

output_file("transient_plot_example.html")
save(plot)
print("Plot saved to transient_plot_example.html")
