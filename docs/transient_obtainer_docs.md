# Transient Obtainer Documentation

## Overview
The `TransientObtainer` is designed to acquire data for transient gravitational wave events (e.g., BBH, BNS mergers) or glitches. It simplifies the interface for configuring `IFODataObtainer` for these specific use cases.

## Class: `TransientObtainer`

```python
class TransientObtainer(Obtainer):
    def __init__(
        self,
        ifo_data_obtainer: gf.IFODataObtainer,
        event_names: Union[str, List[str]] = None,
        # ... other args
    )
```

### Key Features
- **Targeted Acquisition**: Can fetch specific events by name (e.g., `event_names=["GW150914"]`).
- **Broad Acquisition**: If `event_names` is None, it iterates over all available events defined by the `ifo_data_obtainer` configuration.
- **Event Types**: Controls whether to fetch `CONFIDENT` or `MARGINAL` events via the `IFODataObtainer`.

## Configuration via `IFODataObtainer`

The underlying `IFODataObtainer` determines the pool of events and data quality.

```python
ifo_obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_labels=[gf.DataLabel.EVENTS],
    event_types=[gf.EventType.CONFIDENT]  # Default: CONFIDENT only
)
```

### Parameters
- **`event_types`**: List of `gf.EventType`.
    - `[gf.EventType.CONFIDENT]`: (Default) Fetch only confirmed GW events.
    - `[gf.EventType.MARGINAL]`: Fetch marginal candidates.
    - `[gf.EventType.CONFIDENT, gf.EventType.MARGINAL]`: Fetch both.

## Examples

### 1. Fetching All Confident O3 Events
```python
ifo_obt = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_labels=[gf.DataLabel.EVENTS],
    event_types=[gf.EventType.CONFIDENT]
)
transient_obt = gf.TransientObtainer(ifo_data_obtainer=ifo_obt)

# Generator yields badges of O3 events
generator = transient_obt(sample_rate_hertz=2048, ...)
```

### 2. Fetching Marginal Events
```python
ifo_obt_marginal = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_labels=[gf.DataLabel.EVENTS],
    event_types=[gf.EventType.MARGINAL]
)
transient_obt = gf.TransientObtainer(ifo_data_obtainer=ifo_obt_marginal)
```

### 3. Fetching Specific Events (Overrides Random Order)
```python
transient_obt = gf.TransientObtainer(
    ifo_data_obtainer=ifo_obt,
    event_names=["GW190521", "GW190425"]
)
```
