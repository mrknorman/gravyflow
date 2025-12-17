
### User Request
The user is reporting that "Chirp" glitch streams are returning all NaNs. The debugging script was hanging. We need to investigate why "Chirp" data is invalid and fix it.

### Proposed Plan

#### 1. Isolate "Chirp" Data Acquisition
Create a minimal script `debug_chirp_raw.py` that:
- Uses `gf.get_glitch_times` to fetch a list of "Chirp" events.
- Prints the first few GPS times found.
- Attempts to download raw data for the first "Chirp" event using `gwpy.timeseries.TimeSeries.fetch_open_data` (bypassing `gravyflow` wrappers temporarily to check source).
- Identifying if the issue is with `GravitySpy` queries, `gwdatafind`, or the data files themselves.

#### 2. Investigate `gwdatafind` "Unterminated String" Warning
The logs show `(1, 12): unterminated string literal` from `gwdatafind` or `GravitySpyTable`.
- This might be breaking the `fetch` call for certain glitch types (like Chirp).
- We will verify if the query string construction in `gravyflow/src/dataset/features/glitch.py` is compatible with the `gwdatafind`/`GravitySpy` backend.

#### 3. Debug `TransientObtainer` Processing
If raw data exists, the issue might be in `TransientObtainer`:
- Whitening logic might be encountering zeros or gaps.
- Cropping logic might be misaligned.
- We will trace the data flow for a single known "Chirp" event through `TransientObtainer`.

#### 4. Fix and Verify
- Apply fixes to `glitch.py` or acquisition logic.
- Run `debug_glitch_nans.py` (modified to be faster) to verify no NaNs are returned.
