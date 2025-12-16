# Transient Obtainer Documentation

## Overview
The `TransientObtainer` is a specialized data acquisition class designed to fetch discrete, localized segments of time-series data. Unlike the `NoiseObtainer` (which generates continuous or random noise), the `TransientObtainer` focuses on specific "interesting" times, such as:
*   **Gravitational Wave Events**: Known mergers (e.g., GW150914).
*   **Glitches**: Short-duration noise artifacts identified by GravitySpy (e.g., "Blip", "Scattered_Light").

It acts as a high-level wrapper around `IFODataObtainer`, managing the logic of identifying these times, clustering them to avoid redundancy, caching the data locally, and serving it in batches.

## Key Concepts

### 1. Identify -> Cluster -> Cache
When initialized, the `TransientObtainer` performs three main steps:
1.  **Identify**: It queries a metadata source (like GraceDB for events or GravitySpy for glitches) to find a list of GPS times and labels.
2.  **Cluster**: It groups improved segments that are close together (within `max_segment_seconds`) into single continuous blocks. This is efficient for fetching multiple events that occur in quick succession.
3.  **Cache**: 
    *   **Full Precache**: If `precache_cap` is set (default behavior), it downloads *all* identified segments immediately and stores them in `res/glitch_cache` or `res/event_cache`.
    *   **Lazy Loading**: If `precache_cap=0`, it skips the immediate download. Segments are fetched on-the-fly during generation.

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

### Lazy Loading vs. Precaching

Downloading the entire O3 glitch catalog (~250k events) can take hours and consume significant disk space.

*   **Production Training**: Use the default (Full Precache). This ensures input/output is fast during training because all data is strictly local.
*   **Quick Tests / Debugging**: Use `precache_cap=0` (Lazy Loading).

```python
# Force lazy loading by passing precache_cap=0 to the generator call
generator = transient_obt(
    precache_cap=0, 
    ...
)
```

*Note: In some wrapper scripts (like `small_glitch.py`), you may need to subclass `TransientObtainer` to force this behavior if the standard interface doesn't expose it explicitly.*

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
*   **"Unterminated string literal"**: Warning from GravitySpy queries. Usually harmless/transient, but if persistent, it means a specific glitch class query is malformed.
