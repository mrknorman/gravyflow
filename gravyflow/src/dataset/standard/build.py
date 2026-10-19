"""
How a standard dataset's pinned files are made: its segments and its events, computed
from GWOSC's public timelines and event lists and gravyflow's partition of time.

The segments are the time of the split (``test``) on which every detector passes the
data-quality flags, less every published event's veto: each detector's flagged time
(``gwosc_data.get_gwosc_segments``) less the events (``event.noise_event_vetoes``,
the vetoes real noise is drawn under), inside the split's own time
(``SplitSpec.admissible``: its blocks, less the guard at every edge with another
split), intersected over the detectors, cut inward to the whole GPS seconds a read of
it at the dataset's rate serves (``store.read.interiors``: the read's resampling margin
stays inside the segment, so no read of a segment touches time outside it, a data gap
included), and kept where at least ``minimum_seconds`` long. No glitch is vetoed: a search's background holds the
glitches the detectors had.

``build_files`` is the one entry point: it queries GWOSC (timelines and the event API,
about sixty requests; no strain) and writes the files a ``StandardDataset`` pins. Its
inputs are the definition's own fields, so rerunning it after a change to the partition,
the flags or the event lists shows exactly what the pins must become.
"""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from gravyflow.src.dataset.acquisition.gwosc_data import (
    GWOSC_RUNS, coalesce_segments, get_gwosc_segments, intersect_segments, subtract_segments, within_segments,
)
from gravyflow.src.dataset.features.event import CONFIDENT_CATALOGS, noise_event_vetoes
from gravyflow.src.dataset.features.transient_index import event_footprints
from gravyflow.src.dataset.store.read import interiors

#: The columns of a standard dataset's event list, in order.
EVENT_COLUMNS = ("name", "gps", "catalog", "confident", "chirp_mass_source", "p_astro", "detectors")


def split_segments(
    readable: Mapping[str, np.ndarray],
    vetoes: np.ndarray,
    split,
    split_name: str,
    minimum_seconds: float,
    source,
    sample_rate_hertz: float,
) -> np.ndarray:
    """
    The time of split ``split_name`` every detector holds: each detector's ``readable``
    time ((N, 2) GPS, by detector name) less ``vetoes``, inside the split's own time
    (``split.admissible``), intersected over the detectors, cut inward to the whole GPS
    seconds a read of it from ``source`` at ``sample_rate_hertz`` serves with its margin
    inside it (``store.read.interiors``), and kept where at least ``minimum_seconds``
    long. Returns sorted (S, 2) int64 GPS seconds.
    """
    if not readable:
        raise ValueError("A standard dataset needs at least one detector.")
    common = None
    for segments in readable.values():
        held = split.admissible(subtract_segments(segments, vetoes), split_name)
        common = held if common is None else intersect_segments(common, held)
    first, count = interiors(source, sample_rate_hertz, coalesce_segments(common))
    rate = float(sample_rate_hertz)
    whole = np.stack([np.ceil(first / rate), np.floor((first + count) / rate)], axis=1).astype(np.int64)
    return whole[whole[:, 1] - whole[:, 0] >= minimum_seconds].reshape(-1, 2)


def split_events(catalogue: Mapping[str, np.ndarray], readable: Mapping[str, np.ndarray], split,
                 split_name: str, span: Sequence[float]) -> list:
    """
    The events of split ``split_name`` in GPS ``span``: one row per event name (the
    entry of the first confident catalogue that lists it, else its first entry), whose
    footprint (``event_footprints``) the split holds whole (``split.assign``),
    with the detectors whose ``readable`` time holds the footprint. Rows of
    ``EVENT_COLUMNS``, sorted by GPS.
    """
    gps = np.asarray(catalogue["gps"], dtype=np.float64)
    footprints = event_footprints(gps, catalogue["chirp_mass_source"])
    rank = {name: i for i, name in enumerate(CONFIDENT_CATALOGS)}
    order = sorted(range(len(gps)), key=lambda i: (rank.get(str(catalogue["catalog"][i]), len(rank)), i))
    rows, seen = [], set()
    for i in order:
        name = str(catalogue["name"][i])
        start, end = footprints[i]
        if name in seen or not (span[0] <= start and end <= span[1]):
            continue
        if split.assign(start, end) != split_name:
            continue
        seen.add(name)
        held = [ifo for ifo, segments in readable.items()
                if within_segments(footprints[i:i + 1], segments)[0]]
        rows.append({
            "name": name, "gps": float(gps[i]), "catalog": str(catalogue["catalog"][i]),
            "confident": str(catalogue["catalog"][i]) in rank,
            "chirp_mass_source": float(catalogue["chirp_mass_source"][i]),
            "p_astro": float(catalogue["p_astro"][i]), "detectors": "+".join(held),
        })
    return sorted(rows, key=lambda row: row["gps"])


def segments_csv(segments: np.ndarray) -> bytes:
    """The bytes of a segment list's pinned file: ``start,end`` GPS seconds, one per line."""
    return ("start,end\n" + "".join(f"{int(a)},{int(b)}\n" for a, b in segments)).encode()


def events_csv(rows: list) -> bytes:
    """The bytes of an event list's pinned file (``EVENT_COLUMNS``; NaN for a value none gives)."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(EVENT_COLUMNS)
    for row in rows:
        writer.writerow([repr(row[c]) if isinstance(row[c], float) else row[c] for c in EVENT_COLUMNS])
    return out.getvalue().encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute(definition) -> Dict[str, bytes]:
    """
    A ``StandardDataset``'s segment and event files, from GWOSC: {file name: bytes}.
    Network: GWOSC's timelines (cached in gravyflow's user cache) and its event API.
    """
    from gravyflow.src.dataset.features.event import get_event_catalogue

    from gravyflow.src.dataset.acquisition.sources import GWOSC

    readable = {ifo: get_gwosc_segments(ifo, definition.gwosc_runs, definition.flags)
                for ifo in definition.detectors}
    catalogue = get_event_catalogue()
    split = definition.split
    vetoes = noise_event_vetoes(split.guard_seconds, catalogue)
    span = (min(GWOSC_RUNS[run]["gps_start"] for run in definition.gwosc_runs),
            max(GWOSC_RUNS[run]["gps_end"] for run in definition.gwosc_runs))
    segments = split_segments(readable, vetoes, split, definition.split_name, definition.minimum_segment_seconds,
                              GWOSC(), definition.sample_rate_hertz)
    events = split_events(catalogue, readable, split, definition.split_name, span)
    return {"segments.csv": segments_csv(segments), "events.csv": events_csv(events)}


def build_files(definition, directory: Path) -> Dict[str, str]:
    """
    Compute ``definition``'s segment and event files (``compute``) and write them into
    ``directory``; returns {file name: sha256}, the values its pins must hold.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pins = {}
    for name, data in compute(definition).items():
        (directory / name).write_bytes(data)
        pins[name] = sha256(data)
    return pins


#: The aligned spins the best case is searched over, per component: the extremes and none.
BEST_CASE_SPINS = (-0.99, 0.0, 0.99)


def best_case_snr_grid(mass_grid_msun: Sequence[float], noise_profiles: Sequence[str], detectors: Sequence[str],
                       lower_frequency_hertz: float, upper_frequency_hertz: float,
                       approximant: str = "IMRPhenomXAS", spins: Sequence[float] = BEST_CASE_SPINS,
                       frequency_step_hertz: float = 0.125) -> np.ndarray:
    """
    The best-case network optimal SNR at 1 Gpc of every pair of ``mass_grid_msun``
    (detector frame): the (G, G) symmetric table of ``population.Reach``.

    The best case is a face-on source directly above every detector (F+ = 1, Fx = 0, so
    each sees |h+|, which face-on equals |hx|), each detector's noise the lowest, per
    frequency, of the ``noise_profiles`` curves (``gf.NoiseProfile`` names) of all
    ``detectors``, and the spins the loudest of every aligned pair from ``spins``. The
    SNR is gravyflow's definition (``tools.snr``) in the frequency domain,
    rho^2 = 4 df sum |h+|^2 / S over [lower, upper), and the network SNR is
    sqrt(len(detectors)) times one detector's. The waveforms are ripple's
    ``approximant`` from ``lower_frequency_hertz``.
    """
    import jax
    import jax.numpy as jnp

    from gravyflow.src.dataset.conditioning.detector import IFO, NoiseProfile
    from gravyflow.src.dataset.features.waveforms.cbc import (
        _as_approximant, _frequency_domain_polarisations, _in_x64, _source_parameters,
    )
    from gravyflow.src.detection.template_grid import noise_curve_psd

    model = _as_approximant(approximant)
    frequencies = np.arange(0.0, upper_frequency_hertz, frequency_step_hertz)
    band = frequencies >= lower_frequency_hertz
    inverse_psd = 1.0 / np.min([noise_curve_psd(IFO[ifo], frequencies[band], NoiseProfile[profile])
                                for profile in noise_profiles for ifo in detectors], axis=0)

    def one(m1, m2, s1, s2):
        zero = jnp.zeros(())
        parameters = _source_parameters(m1, m2, jnp.stack([zero, zero, s1]), jnp.stack([zero, zero, s2]),
                                        0.0, 0.0, 1000.0, 0.0, 0.0)
        plus, _ = _frequency_domain_polarisations(model, jnp.asarray(frequencies), parameters,
                                                  lower_frequency_hertz, 0.0)
        return jnp.abs(plus) ** 2

    plus_power = _in_x64(jax.jit(jax.vmap(one)))
    grid = np.asarray(mass_grid_msun, dtype=np.float64)
    s1, s2 = np.array([(a, b) for a in spins for b in spins], dtype=np.float64).T
    table = np.zeros((len(grid), len(grid)))
    for i in range(len(grid)):
        for j in range(i + 1):
            power = np.asarray(plus_power(np.full(len(s1), grid[i]), np.full(len(s1), grid[j]), s1, s2))
            single = np.sqrt(4.0 * frequency_step_hertz * np.sum(power[:, band] * inverse_psd, axis=-1))
            table[i, j] = table[j, i] = np.sqrt(len(detectors)) * single.max()
    return table
