"""
Fixture layouts: where the fake detectors have data, what the catalogues hold, and
the golden block table that judges a decoded footprint's split.

The golden table is literal. Its values were read from the library one block at a
time (``get_segments_for_group`` with a one-second segment inside each block) and
blessed by hand; the harness never re-implements the split. A self-test checks that
the library still agrees with every literal, so a change of split is a deliberate
re-blessing, never a silent one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .fakes import EventRecord, GlitchRecord, HARDWARE_INJECTION_FLAGS
from .provenance import Encoding

#: The split block, in seconds.
BLOCK_SECONDS = 8192.0
#: The fractions the golden table was blessed with (NoiseObtainer's defaults).
GOLDEN_GROUPS = {"train": 0.89, "validate": 0.1, "test": 0.01}
_LETTERS = {"r": "train", "v": "validate", "t": "test"}

#: {origin run: {first block index: letters}}, one letter per block from that index;
#: block k is [origin + k * BLOCK_SECONDS, origin + (k + 1) * BLOCK_SECONDS), where
#: the origin is the run's start (the first configured run's start, on this trunk).
GOLDEN_BLOCKS: Dict[str, Dict[int, str]] = {
    "O3": {
        1: (
            "rrvrvrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrvrrrrvrrrrrrrrrrrrrrvvrrrrrrrrrr"
            "vrrrvrrtrrrrrrrrrrvrrrvrvrvrrrvrrrrrrrrrrvrrrrrrrrrvrrrrrrrrrrrrrrrrrrrrrvr"
            "rrrrrrrrrrrrrvrrrrrrrrvrrrrrrrvrrrvrrrrrrrrrrrrrrrrrrrvrrrrrvrrrrrrvrvrrrrr"
            "rrrrrrrrrrrrrrrvvrrrrrrrrrrrrtvrrrrrrrrvvvrrrrrrrrrrrrrrrrrrrrrrvrrrrrrrrvr"
        ),
    },
    "O1": {
        1: (
            "rrvrvrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrvrrrrvrrrrrrrrrrrrrrvvrrrrrrrrrr"
            "vrrrvrrtrrrrrrrrrrvrrrvrvrvrrrvrrrrrrrrrrvrrrrrrrrrvrrrrrrrrrrrrrrrrrrrrrvr"
            "rrrrrrrrrrrrrvrrrrrrrrvrrrrrrrvrrrvrrrrrrrrrrrrrrrrrrrvrrrrrvrrrrrrvrvrrrrr"
            "rrrrrrrrrrrrrrrvvrrrrrrrrrrrrtvrrrrrrrrvvvrrrrrrrrrrrrrrrrrrrrrrvrrrrrrrrvr"
        ),
        4701: (
            "rrrvrrrrrrrrrrrrrrrrrvrrrvrrrrrvrrrrrrvrrrrrrrrrrrvrrrrrrrrvrvrvrrrrrrrrrrr"
            "rrrvrrrrrrrrrrrrrrrrrrrtrrrrrrrvrrrrrrrvrrvrrrrrrrrrrrrvrrrrrrrrrrrrvrrrrvr"
            "rrrrrrrrrvrrrrrrrvrrrrrvrrrrrrrrrrrrrrrrvrrrvrrrrrrvrrrrrrrrrrrrvvrvrrrrrrv"
            "rrrvrrrrrrrrvrrrrrrrrrrrvrrrrrrrrrrrrrrrrrrrvrrrrrrrrrrrrrvrrrvrrrrrrrvrrrr"
        ),
        13686: (
            "rvrrrrrrrrrvrrrrrrrrrrrrrrrrrrrvrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrvrrrrr"
            "rrrrvrrrrrrrrrrrrvrrrrrrrrrrrrrrrrrrrrrrrrrrvvrrtrrvrrrrrrrrrrrrrrvrrrrrrrr"
            "rrrrrrvrrrrrvrrrrrrrrrrrrrrrrrrrrrrrrrrrrvrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrv"
            "rrrrrrrrrrrrrrvrrrtrrvrrrvrrrrrrrrrrrvrvrrrrrrrrrrrvrrrrrrrrrrrrrrvrrrrrrrr"
        ),
    },
}


def origin_gps(origin: str) -> float:
    import gravyflow as gf

    return float(gf.ObservingRun[origin].value.start_gps_time)


def golden_blocks(origin: str) -> Dict[int, str]:
    """{block index: split} for every literal of ``origin``."""
    table = {}
    for first, letters in GOLDEN_BLOCKS[origin].items():
        for offset, letter in enumerate(letters):
            table[first + offset] = _LETTERS[letter]
    return table


def block_of(origin: str, gps: float) -> int:
    return int(np.floor((float(gps) - origin_gps(origin)) / BLOCK_SECONDS))


def block_start(origin: str, block: int) -> float:
    return origin_gps(origin) + block * BLOCK_SECONDS


def golden_split(origin: str, start: float, end: float) -> str:
    """
    The split of the time [start, end): it must lie in one block of the table.
    Raises when it crosses a block edge or lies outside the table.
    """
    first, last = block_of(origin, start), block_of(origin, np.nextafter(float(end), -np.inf))
    if first != last:
        raise AssertionError(f"[{start}, {end}) crosses the edge of blocks {first} and {last}")
    table = golden_blocks(origin)
    if first not in table:
        raise AssertionError(f"[{start}, {end}) is in block {first}, outside the golden table")
    return table[first]


def boundary(origin: str, before: str, after: str, run: Optional[str] = None, skip: int = 0) -> float:
    """GPS of the first block edge (after ``skip`` such edges) with ``before`` then ``after``."""
    import gravyflow as gf

    table = golden_blocks(origin)
    bounds = None
    if run is not None:
        value = gf.ObservingRun[run].value
        bounds = (value.start_gps_time, value.end_gps_time)
    found = 0
    for block in sorted(table):
        if table.get(block) == before and table.get(block + 1) == after:
            edge = block_start(origin, block + 1)
            if bounds is not None and not (bounds[0] + BLOCK_SECONDS < edge < bounds[1] - BLOCK_SECONDS):
                continue
            if found == skip:
                return edge
            found += 1
    raise LookupError(f"No {before}|{after} block edge in the golden table of {origin} ({run}).")


# =============================================================================
# FIXTURES
# =============================================================================

@dataclass(frozen=True)
class Fixture:
    """A fake world: data segments, flag gaps, catalogues, and its strain code."""

    encoding: Encoding
    segments: Dict[str, np.ndarray]
    origin: str
    runs: Tuple[str, ...]
    flag_gaps: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    glitches: Tuple[GlitchRecord, ...] = ()
    events: Tuple[EventRecord, ...] = ()

    def observing_runs(self):
        import gravyflow as gf

        return [gf.ObservingRun[name] for name in self.runs]

    def with_catalogues(self, glitches=(), events=()) -> "Fixture":
        return Fixture(self.encoding, self.segments, self.origin, self.runs, self.flag_gaps,
                       tuple(glitches), tuple(events))


def _fixture(intervals, rate, origin, runs, ifos=("H1", "L1"), **kwargs) -> Fixture:
    intervals = [(float(a), float(b)) for a, b in intervals]
    segments = {ifo: np.array(intervals, dtype=np.float64) for ifo in ifos}
    encoding = Encoding.from_intervals(intervals, rate)
    return Fixture(encoding, segments, origin, tuple(runs), **kwargs)


def straddle(split_a: str, split_b: str, half_width: float = 1024.0, rate: float = 256.0,
             origin: str = "O3") -> Fixture:
    """
    One continuous segment per detector across an edge between a block of
    ``split_a`` and a block of ``split_b``: each split's piece touches the edge.
    """
    edge = boundary(origin, split_a, split_b)
    return _fixture([(edge - half_width, edge + half_width)], rate, origin, (origin,))


def odd_pieces(rate: float = 256.0, origin: str = "O3") -> Fixture:
    """
    Pieces of 512.2 s, 1030 s and 2048 s, and a short piece with an odd sample count,
    each in its own 2048 s bin of one train block.
    """
    edge = boundary(origin, "train", "train", skip=8)
    pieces = [
        (edge + 100.0, edge + 612.2),
        (edge + 2048.0 + 100.0, edge + 2048.0 + 1130.0),
        (edge + 4096.0, edge + 6144.0),
        (edge + 6144.0 + 100.0, edge + 6144.0 + 100.0 + 40.0 + 0.2 + 1.0 / rate),
    ]
    return _fixture(pieces, rate, origin, (origin,))


def two_splits_per_run(runs: Sequence[str], half_width: float = 320.0, rate: float = 256.0,
                       origin: str = "O1") -> Tuple[Fixture, List[float]]:
    """
    In each run, a segment across a train|test edge and one across a validate|train
    edge: every split has data in every run, touching an edge. Returns the fixture
    and the edges.
    """
    edges = []
    for run in runs:
        edges.append(boundary(origin, "train", "test", run=run))
        edges.append(boundary(origin, "validate", "train", run=run))
    intervals = [(edge - half_width, edge + half_width) for edge in edges]
    return _fixture(intervals, rate, origin, tuple(runs)), edges


def hw_injection_gaps(rate: float = 256.0) -> Fixture:
    """
    ``two_splits_per_run`` over O3, with a gap of each hardware-injection flag in
    every split's piece, including a coherent H1+L1 stochastic gap.
    """
    base, edges = two_splits_per_run(("O3",), half_width=512.0, rate=rate, origin="O3")
    gaps: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}

    def add(ifo, flag, start, duration):
        gaps.setdefault((ifo, flag), []).append((start, start + duration))

    for edge in edges:
        for side in (-1.0, 1.0):
            centre = edge + side * 256.0
            add("H1", "NO_CBC_HW_INJ", centre - 150.0, 20.0)
            add("L1", "NO_BURST_HW_INJ", centre - 100.0, 10.0)
            add("H1", "NO_STOCH_HW_INJ", centre - 60.0, 100.0)
            add("L1", "NO_STOCH_HW_INJ", centre - 60.0, 100.0)
            add("H1", "NO_DETCHAR_HW_INJ", centre + 120.0, 30.0)
    flag_gaps = {key: np.array(value) for key, value in gaps.items()}
    assert {flag for _, flag in flag_gaps} == set(HARDWARE_INJECTION_FLAGS)
    return Fixture(base.encoding, base.segments, base.origin, base.runs, flag_gaps)


def glitches(edges: Sequence[float]) -> Tuple[GlitchRecord, ...]:
    """
    Gravity Spy rows around each block edge in ``edges`` (within 300 s of it):
    confidences 0.1, 0.5 and 0.95; a No_Glitch row; a cluster 5 s apart; H1 and L1
    rows 7 ms apart; rows 10 s and 70 s from the edge on both sides; an Extremely_Loud
    row; a row whose extent begins well before its peak. Deliberately unsorted.
    """
    rows: List[GlitchRecord] = []
    for edge in edges:
        for side in (-1.0, 1.0):
            centre = edge + side * 160.0
            rows += [
                GlitchRecord("H1", centre + 0.0, centre - 0.5, 1.0, "Blip", 0.95),
                GlitchRecord("L1", centre + 0.007, centre - 0.493, 1.0, "Blip", 0.95),
                GlitchRecord("H1", centre + 40.0, centre + 39.0, 2.0, "Koi_Fish", 0.5),
                GlitchRecord("L1", centre + 80.0, centre + 79.0, 2.0, "Tomte", 0.1),
                GlitchRecord("H1", centre + 120.0, centre + 119.5, 1.0, "No_Glitch", 0.99),
                GlitchRecord("L1", centre - 60.0, centre - 68.0, 9.0, "Scattered_Light", 0.97),
                GlitchRecord("H1", centre - 120.0, centre - 121.0, 2.0, "Extremely_Loud", 0.99),
            ]
            rows += [
                GlitchRecord("H1", centre - 30.0 + 5.0 * k, centre - 30.25 + 5.0 * k, 0.5, "Blip", 0.92)
                for k in range(4)
            ]
        for offset in (-70.0, -10.0, 10.0, 70.0):
            rows.append(GlitchRecord("L1", edge + offset, edge + offset - 0.25, 0.5, "Whistle", 0.96))
    order = np.random.default_rng(20261016).permutation(len(rows))
    return tuple(rows[i] for i in order)


#: Padding added to every glitch's whole extent, in seconds, by class (measured:
#: excess power beyond the Omicron extent stays within 1 s except Extremely_Loud).
GLITCH_PADDING_SECONDS = {"Extremely_Loud": 3.0}
DEFAULT_GLITCH_PADDING_SECONDS = 1.0


def glitch_forbidden(rows: Iterable[GlitchRecord]) -> Dict[str, List[Tuple[float, float]]]:
    """Per detector: each row's whole extent plus its class padding."""
    out: Dict[str, List[Tuple[float, float]]] = {}
    for row in rows:
        pad = GLITCH_PADDING_SECONDS.get(row.label, DEFAULT_GLITCH_PADDING_SECONDS)
        start, end = row.extent
        out.setdefault(row.ifo, []).append((start - pad, end + pad))
    return out


def events(edges: Sequence[float], in_band: Sequence[float] = (2.0, 60.0, 165.0)) -> Tuple[
        Tuple[EventRecord, ...], Tuple[GlitchRecord, ...]]:
    """
    One event per in-band duration on each side of each edge (within 820 s of it,
    so the edge's segment needs a half-width of at least 820 s), with its companions in
    the glitch table: a Blip twin 0.05 s after the merger, a Chirp at the merger and
    Koi_Fish rows 40 s and 10 s before and 30 s after.
    """
    found: List[EventRecord] = []
    companions: List[GlitchRecord] = []
    for e_index, edge in enumerate(edges):
        for side in (-1.0, 1.0):
            for k, duration in enumerate(in_band):
                gps = edge + side * (150.0 + 250.0 * k) + 0.4
                name = f"GWFAKE{e_index}{int(side > 0)}{k}"
                found.append(EventRecord(name, gps, float(duration), confident=(k != 1)))
                companions += [
                    GlitchRecord("H1", gps + 0.05, gps - 0.2, 0.5, "Blip", 0.95),
                    GlitchRecord("L1", gps, gps - 0.5, 0.6, "Chirp", 0.95),
                    GlitchRecord("H1", gps - 40.0, gps - 40.5, 1.0, "Koi_Fish", 0.95),
                    GlitchRecord("L1", gps - 10.0, gps - 10.5, 1.0, "Koi_Fish", 0.95),
                    GlitchRecord("H1", gps + 30.0, gps + 29.5, 1.0, "Koi_Fish", 0.95),
                ]
    return tuple(found), tuple(companions)


# =============================================================================
# LIBRARY OBJECTS FOR A FIXTURE
# =============================================================================

#: The harness's default window, in seconds: onsource, crop each side, offsource.
ONSOURCE_SECONDS, CROP_SECONDS, OFFSOURCE_SECONDS = 1.0, 0.5, 4.0


def max_piece_seconds(rate: float) -> float:
    """The longest piece the decoder is exact for: 2^18 samples (see ``provenance``)."""
    return float(2 ** 18) / float(rate)


def ifo_data_obtainer(fixture: Fixture, **overrides):
    """
    A REAL-noise IFODataObtainer over the fixture's runs: no augmentation, no
    prefetch, no segment cache, and pieces of at most ``max_piece_seconds``.
    """
    import gravyflow as gf

    kwargs = dict(
        max_segment_duration_seconds=max_piece_seconds(fixture.encoding.sample_rate_hertz),
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.NOISE],
        observing_runs=fixture.observing_runs(),
        cache_segments=False,
        augmentations=[],
        prefetch_segments=0,
    )
    kwargs.update(overrides)
    return gf.IFODataObtainer(**kwargs)


def noise_obtainer(fixture: Fixture, directory, ifos=("H1", "L1"), **overrides):
    """A REAL NoiseObtainer over the fixture, with the golden table's fractions."""
    import gravyflow as gf

    return gf.NoiseObtainer(
        ifo_data_obtainer=ifo_data_obtainer(fixture, **overrides),
        ifos=[gf.IFO[name] for name in ifos],
        noise_type=gf.NoiseType.REAL,
        groups=dict(GOLDEN_GROUPS),
        data_directory_path=Path(directory),
    )


def noise_batches(fixture: Fixture, directory, split: str, rate: float = 256.0,
                  mode=None, batch: int = 8, seed: int = 7, ifos=("H1", "L1"), **overrides):
    """The obtainer's raw batches (uncropped onsource) for one split."""
    import gravyflow as gf

    obtainer = noise_obtainer(fixture, directory, ifos, **overrides)
    return obtainer(
        sample_rate_hertz=rate,
        onsource_duration_seconds=ONSOURCE_SECONDS,
        crop_duration_seconds=CROP_SECONDS,
        offsource_duration_seconds=OFFSOURCE_SECONDS,
        num_examples_per_batch=batch,
        scale_factor=1.0,
        group=split,
        seed=seed,
        sampling_mode=mode or gf.SamplingMode.RANDOM,
    )


def dataset(fixture: Fixture, directory, split: str, rate: float = 256.0, mode=None,
            batch: int = 8, seed: int = 7, ifos=("H1", "L1"), input_variables=None,
            output_variables=None, waveform_generators=None, **overrides):
    """A Dataset of the fixture's REAL noise (cropped onsource, unit scale)."""
    import gravyflow as gf

    return gf.Dataset(
        noise_obtainer=noise_obtainer(fixture, directory, ifos, **overrides),
        sample_rate_hertz=rate,
        onsource_duration_seconds=ONSOURCE_SECONDS,
        crop_duration_seconds=CROP_SECONDS,
        offsource_duration_seconds=OFFSOURCE_SECONDS,
        num_examples_per_batch=batch,
        scale_factor=1.0,
        group=split,
        seed=seed,
        ifos=[gf.IFO[name] for name in ifos],
        sampling_mode=mode or gf.SamplingMode.RANDOM,
        waveform_generators=waveform_generators,
        input_variables=input_variables or [
            gf.ReturnVariables.ONSOURCE, gf.ReturnVariables.OFFSOURCE,
        ],
        output_variables=output_variables or [gf.ReturnVariables.START_GPS_TIME],
    )


def dataset_batches(ds):
    """A Dataset's batches, in order, as (input, output) pairs."""
    k = 0
    while True:
        yield ds[k]
        k += 1


def window(rate: float = 256.0):
    from .provenance import Window

    return Window.from_seconds(rate, ONSOURCE_SECONDS, CROP_SECONDS, OFFSOURCE_SECONDS)


def served_segments(fixture: Fixture, ifos, split: str, seed: int = 7, **overrides):
    """
    The library's served-time set for one split: {ifo: (N, 2)} from a fresh
    obtainer's ``get_valid_segments`` (the time the generator samples from).
    """
    import gravyflow as gf

    obtainer = ifo_data_obtainer(fixture, **overrides)
    valid = obtainer.get_valid_segments([gf.IFO[name] for name in ifos], seed, dict(GOLDEN_GROUPS), split)
    valid = np.asarray(valid, dtype=np.float64)
    return {name: valid[:, d] for d, name in enumerate(ifos)}


__all__ = [
    "ONSOURCE_SECONDS", "CROP_SECONDS", "OFFSOURCE_SECONDS", "max_piece_seconds", "ifo_data_obtainer",
    "noise_obtainer", "noise_batches", "dataset", "dataset_batches", "window", "served_segments",
    "BLOCK_SECONDS", "GOLDEN_GROUPS", "GOLDEN_BLOCKS", "origin_gps", "golden_blocks", "block_of",
    "block_start", "golden_split", "boundary", "Fixture", "straddle", "odd_pieces",
    "two_splits_per_run", "hw_injection_gaps", "glitches", "glitch_forbidden",
    "GLITCH_PADDING_SECONDS", "DEFAULT_GLITCH_PADDING_SECONDS", "events",
]
