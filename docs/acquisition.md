# Gravyflow Acquisition Process

This document provides a detailed explanation of the data acquisition process in `gravyflow`, specifically focusing on how the `IFODataObtainer` selects, splits, and groups time segments for noise and feature acquisition.

## High-Level Overview

The acquisition process is designed to retrieve valid, high-quality data segments from gravitational wave detectors (IFOs) like LIGO. The process ensures that:
1.  Data is available for all requested detectors simultaneously (coincident time).
2.  Data meets specific quality criteria (state flags).
3.  Segments are split into manageable chunks for training, validation, and testing.
4.  Unwanted periods (glitches, known events) are vetoed or handled appropriately.
5.  For noise acquisition, long segments are prioritized to maximize the number of background samples.

## The Segment Selection Pipeline

The core logic resides in the `get_valid_segments` method of the `IFODataObtainer` class. Here is the step-by-step flow:

### 1. Initial Retrieval (`get_all_segment_times`)
The process begins by reading the public GWOSC timeline for each requested interferometer (e.g., H1, L1). No LVK credentials are needed.
-   **Input**: The GPS bounds of each observing run, from the GWOSC run API (O3 is GWOSC's O3a plus O3b), the detector (IFO), and the GWOSC flags to apply.
-   **Flags**: GWOSC defines each data-quality flag as the times that *pass* it, so applying a category means intersecting with its flag. The default is `DATA` with `BURST_CAT1` and `BURST_CAT2` applied (`gf.GWOSC_BURST_FLAGS`); `gwosc_flags=gf.GWOSC_DATA_FLAGS` or `gf.GWOSC_CBC_FLAGS` selects others.
-   **Operation**: Returns a list of `[start, end)` time pairs where the detector passes every flag.
-   **Cache**: Timeline segments are cached in `~/.gravyflow/cache/gwosc_timeline` (or `$GRAVYFLOW_CACHE_DIR`).
-   **Result**: A list of valid segments for *one* detector.

Livetime per option (days), measured from the GWOSC timeline:

| Run | IFO | `DATA` | `+BURST_CAT1` | `+BURST_CAT2` (default) | `DATA+CBC_CAT1+CBC_CAT2` |
|---|---|---|---|---|---|
| O1 | H1 | 76.70 | 76.70 | 76.22 | 75.89 |
| O1 | L1 | 65.81 | 65.81 | 65.78 | 65.80 |
| O2 | H1 | 157.81 | 157.81 | 157.24 | 157.38 |
| O2 | L1 | 153.96 | 153.96 | 153.56 | 153.72 |
| O3 | H1 | 245.21 | 245.21 | 243.53 | 244.70 |
| O3 | L1 | 251.93 | 251.93 | 250.47 | 251.47 |

Strain comes from local frames when a frame server is configured, and otherwise from GWOSC through `TimeSeries.fetch_open_data`, at 4096 Hz when that suffices and at 16384 Hz otherwise. Sample rates above 16384 Hz are refused: real data is never upsampled.

### Glitch catalogue

Glitch times, classes and metadata (SNR, peak frequency, duration, confidence) come from the public Gravity Spy machine-learning classifications: Glanzer et al. 2023, *Class. Quantum Grav.* 40 065004, released on Zenodo (doi:10.5281/zenodo.5649212, v1.0.0, CC-BY-4.0). Please cite both if you use glitches from gravyflow. Each detector and run's file (16 to 216 MB) is downloaded on first use, checked against its published md5, and cached in compact form in `~/.gravyflow/cache/gravityspy`. The defaults keep every class except `No_Glitch`, with `ml_confidence >= 0.9`.

### 2. Coincidence and Cutting (`cut_segments`)
To train a model on multi-detector data, we need times where *all* detectors were active.
-   **Splitting**: Before finding intersections, segments are often split into smaller, fixed-duration chunks (e.g., 8196 seconds) to facilitate consistent grouping later.
-   **Operation**: The `cut_segments` method takes long continuous segments and chops them into these fixed-size blocks.

### 3. Group Assignment (`get_segments_for_group`)
Data is deterministically assigned to `train`, `validate`, or `test` groups to ensure no leakage between them.
-   **Logic**: The timeline is divided into blocks (e.g., 8196s). Each block is hashed or assigned based on its time to a specific group.
-   **Example**: If the split is 98% train, 1% validate, 1% test, then for every 100 blocks, 98 go to training.
-   **Result**: A filtered list of segments that belong *only* to the requested group (e.g., "train").

### 4. Vetoing Unwanted Times (`remove_unwanted_segments`)
We often want to exclude times containing known glitches or gravitational wave events (unless we are specifically training on them).
-   **Inputs**: Lists of known event times (from catalogues like GWTC) and glitch times.
-   **Operation**:
    -   `pad_gps_times_with_veto_window`: Adds a safety buffer around each event/glitch time.
    -   `veto_time_segments`: Subtracts these veto windows from the valid segments. If a segment overlaps with a veto, the overlapping part is removed, potentially splitting the segment into two smaller ones.

### 5. Noise Acquisition Specifics (`merge_bins` & `largest_segments_per_bin`)
For noise background estimation, we prefer long, continuous segments to generate many random realizations.
-   **Problem**: After vetoing, we might have many tiny, fragmented segments.
-   **Binning**: The timeline is divided into "bins" of a maximum desired duration (e.g., 2048s).
-   **Merging (`merge_bins`)**: Segments from all detectors are aligned into these bins.
-   **Selection (`largest_segments_per_bin`)**: Within each bin, the algorithm finds the largest common interval where *all* detectors have valid data. This maximizes the usable coincident time for that specific window.
-   **Result**: A list of `[start, end]` pairs representing the best available coincident noise data for each time bin.

### 6. Final Ordering (`order_segments`)
Finally, the segments are ordered based on the user's preference:
-   **Random**: Shuffled for training variety.
-   **Chronological**: Useful for sequential analysis.
-   **Shortest First**: Sometimes used for debugging.

## Summary Diagram

```mermaid
graph TD
    A["Read GWOSC timeline flags"] --> B["Raw Segments per IFO"]
    B --> C["Cut into Fixed Blocks (e.g. 8196s)"]
    C --> D{"Assign Group"}
    D -- Train --> E["Train Segments"]
    D -- Test --> F["Test Segments"]
    E --> G["Veto Events/Glitches"]
    G --> H["Fragmented Segments"]
    H --> I["Bin & Merge Coincident Times"]
    I --> J["Select Largest Segment per Bin"]
    J --> K["Final Valid Segments"]
```

This rigorous process ensures that the data fed into the machine learning model is clean, consistent, and strictly separated by dataset split.
