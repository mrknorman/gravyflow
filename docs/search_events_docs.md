## Searching for Events

GravyFlow provides a powerful `search_events` function to filter gravitational wave events from GWTC catalogs based on astrophysical properties, observing runs, and more.

### gf.search_events

```python
def search_events(
    mass1_range: tuple = None,
    mass2_range: tuple = None,
    total_mass_range: tuple = None,
    distance_range: tuple = None,
    observing_runs: List[gf.ObservingRun] = None,
    event_types: List[gf.EventType] = None,
    source_type: Union[gf.SourceType, str] = None,
    name_contains: str = None,
) -> List[str]
```

**Parameters:**

- `source_type` : `Union[gf.SourceType, str]` = `None`:
  > Filter by astrophysical source type. 
  > **Enums (Recommended):**
  > - `gf.SourceType.BBH`: Binary Black Hole (both masses ≥ 3 M☉)
  > - `gf.SourceType.BNS`: Binary Neutron Star (both masses < 3 M☉)
  > - `gf.SourceType.NSBH`: Neutron Star - Black Hole (one < 3 M☉, one ≥ 3 M☉)
  >
  > **Strings (Supported):** `"BBH"`, `"BNS"`, `"NSBH"` (case-insensitive).

- `observing_runs` : `List[gf.ObservingRun]` = `None`:
  > Filter by specific observing runs (e.g., `[gf.ObservingRun.O3]`).

- `event_types` : `List[gf.EventType]` = `[gf.EventType.CONFIDENT]`:
  > Filter by event confidence.
  > **Options:**
  > - `gf.EventType.CONFIDENT`: Confirmed detections (Default).
  > - `gf.EventType.MARGINAL`: Marginal triggers/candidates.

- `mass1_range` : `tuple` = `None`:
  > (min, max) range for primary mass in solar masses. Use `None` for unbounded limits. 
  > *Example:* `(30, None)` finds events with m1 > 30 M☉.

- `mass2_range` : `tuple` = `None`:
  > (min, max) range for secondary mass in solar masses.

- `total_mass_range` : `tuple` = `None`:
  > (min, max) range for total system mass (m1 + m2).

- `distance_range` : `tuple` = `None`:
  > (min, max) range for luminosity distance in Mpc.
  > *Example:* `(None, 500)` finds events closer than 500 Mpc.

- `name_contains` : `str` = `None`:
  > Substring to search for in the event name (case-insensitive).
  > *Example:* `"GW17"` matches all 2017 events.

**Returns:**
- `List[str]`: A list of event names matching all specified conditions.

---

### Examples

#### 1. Filter by Source Type
```python
import gravyflow as gf

# Find all Binary Neutron Star events
bns_events = gf.search_events(source_type=gf.SourceType.BNS)
print(bns_events)
# Output: ['GW170817', 'GW190425']
```

#### 2. Include Marginal Events
```python
# Find marginal/candidate events
marginal_events = gf.search_events(event_types=[gf.EventType.MARGINAL])
```

#### 3. Complex Physical Queries
```python
# Find heavy BBHs (Total Mass > 80 M☉) that are relatively close (< 1000 Mpc)
heavy_nearby = gf.search_events(
    source_type=gf.SourceType.BBH,
    total_mass_range=(80, None),
    distance_range=(None, 1000)
)
```
