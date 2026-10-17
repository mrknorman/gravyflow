# Transient Obtainer Documentation

## Overview
The `TransientObtainer` is a specialized data acquisition class designed to fetch discrete, localized segments of time-series data. Unlike the `NoiseObtainer` (which generates continuous or random noise), the `TransientObtainer` focuses on specific "interesting" times, such as:
*   **Gravitational Wave Events**: Known mergers (e.g., GW150914).
*   **Glitches**: Short-duration noise artifacts identified by GravitySpy (e.g., "Blip", "Scattered_Light").

It acts as a high-level wrapper around `IFODataObtainer`, managing the logic of identifying these times, splitting them and planning each detector's read, reading the strain through the store, and serving it in batches.

## Key Concepts

### 1. Identify -> Split and plan -> Read
When initialized, the `TransientObtainer` performs three main steps:
1.  **Identify**: It queries a metadata source (GWOSC event catalogues for events, the public Gravity Spy release for glitches) to find a list of GPS times and labels.
2.  **Split and plan**: Each record takes its footprint's split (`acquisition.md`, section 2). Each requested detector is planned one read: the transient's own detectors at its time, every other detector at a partner time of its own, in a block of the record's split (a glitch's own; an event's nearest in its run), never at the transient's time (`acquisition.md`, Transient detectors and partners). A record with no partner time is not served, and is counted under `"no partner"`.
3.  **Read**: Each example reads each detector through the strain store, as noise does: the source's own files (tiles), fetched whole the first time a read needs them and kept under the user cache (`$GRAVYFLOW_CACHE_DIR`, else `~/.gravyflow/cache`), then viewed and resampled to the request's rate. A tile is fetched once and shared by every rate, split and configuration. A planned transient whose strain the source does not deliver is refused by name (`TransientUnavailable`, naming the `exclusions` entry that plans around it); nothing is skipped or zero-filled.

### 2. Balancing (`balanced_glitch_types`)
For validation and training, it is often critical to have a balanced dataset.
*   **`balanced_glitch_types=False` (Default)**: The obtainer provides access to *all* available segments. If "Scattered_Light" has 20,000 samples and "Helix" has 50, the dataset will be heavily imbalanced.
*   **`balanced_glitch_types=True`**: The obtainer calculates the count of the *rarest* class (e.g., 61 samples) and limits *all* other classes to this count. This ensures every batch has a uniform probability of containing any given class, but significantly reduces the total dataset size.

---

## Configuration

The behavior is primarily controlled by the `IFODataObtainer` passed to the constructor.

### Acquiring Events (GWs)

To fetch known gravitational waves:

```python
import gravyflow as gf

# 1. Configure the source
ifo_obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_labels=[gf.DataLabel.EVENTS],      # Target Events
    event_types=[gf.EventType.CONFIDENT]    # Only confirmed detections
)

# 2. Create the TransientObtainer
transient_obt = gf.TransientObtainer(
    ifo_data_obtainer=ifo_obtainer,
    ifos=[gf.IFO.L1]
)

# 3. Generate data
generator = transient_obt(
    sample_rate_hertz=2048.0,
    onsource_duration_seconds=1.0, # Center on event
    crop=True
)
```

### Acquiring Glitches (Noise Artifacts)

To fetch specific noise shapes (glitches):

```python
# 1. Configure for Glitches
ifo_obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_labels=[gf.DataLabel.GLITCHES],     # Target Glitches
    balanced_glitch_types=True               # Optional: Balance classes
)

# 2. Create Obtainer
transient_obt = gf.TransientObtainer(
    ifo_data_obtainer=ifo_obtainer,
    ifos=[gf.IFO.L1]
)
```

---

## Advanced Usage

### Reading strain

Nothing is downloaded ahead: a batch's reads fetch the tiles they need, together, and later reads of the same time, at any rate, read the stored tiles. Downloading the tiles of the whole O3 glitch catalogue (~250k records) takes hours and much disk space, so a quick test should serve a few named events or a small split.

### Handling Glitch Labels

The generator returns: `(onsource, offsource, gps_times, labels)`.
*   `labels`: An array of integers representing the `GlitchType` index.

To convert these back to readable names:

```python
label_index = labels[0]
glitch_enum = gf.get_glitch_type_from_index(label_index)
print(f"Glitch Type: {glitch_enum.name}") 
# Output: "Glitch Type: KOI_FISH"
```

## Troubleshooting

*   **"Min count: X" Log**: When using `balanced_glitch_types=True`, look for this log line. It tells you the bottleneck class limit. If this number is too small, consider disabling balancing or manually excluding rare glitch types.
*   **Gravity Spy download or md5 errors**: Glitches come from the public Gravity Spy release on Zenodo (doi:10.5281/zenodo.5649212; Glanzer et al. 2023, *Class. Quantum Grav.* 40 065004). A failed download or checksum raises an error and caches nothing; retry, or check access to zenodo.org.
