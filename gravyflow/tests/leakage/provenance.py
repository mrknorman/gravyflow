"""
The provenance code: every fake strain sample carries its own GPS time.

A fake sample is a positive integer *code*, at most 2^21, times ``2^-exponent``. Codes
are split between detectors (H1 takes [1, 2^20], L1 the next 2^20), so the detector of
a row is read from its values, never from its position. Inside a detector's range,
codes are laid out island by island: an island is a run of consecutive samples at the
fixture's served rate, and its codes increase by one per sample. Decoding therefore
needs no labels and no fetch log: a value names its detector, its island and its
absolute sample index.

Exactness. Integer codes survive float32 storage, casts, crops and strides exactly.
The library resamples every real-noise piece with a float32 FFT, even at an equal
rate; its round trip moves a code by at most 0.08 for pieces of up to 2^18 samples
with codes up to 2^21 (0.22 at 2^19 samples, 0.38 at 2^20). The decoder rounds to the
nearest code and refuses any value more than 0.25 from it, so fixtures keep pieces at
or below 2^18 samples (``fixtures.max_piece_seconds``), and a larger error refuses
loudly rather than shifting a row.

The decoder refuses, never skips: a row it cannot decode raises ``DecodeRefused``
with the row and the reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

#: The lowest code: no fake sample is zero, so a zero-filled row never decodes.
CODE_FLOOR = 1
#: The detectors the code distinguishes, in the order of their code ranges.
CODE_DETECTORS: Tuple[str, ...] = ("H1", "L1")
#: Codes available to each detector; the largest code is 2^21.
CODES_PER_DETECTOR = 2 ** 20
CODE_CEILING = CODE_FLOOR + len(CODE_DETECTORS) * CODES_PER_DETECTOR
#: The largest distance from an integer that a decoded value may have.
INTEGER_TOLERANCE = 0.25


class DecodeRefused(ValueError):
    """A row is not fake strain that the fixture can place in time."""


class OutsideFixture(ValueError):
    """A request for samples the fixture does not hold."""


@dataclass(frozen=True)
class Island:
    """``num_samples`` consecutive samples from absolute sample index ``first``."""

    first: int
    num_samples: int
    code_offset: int

    @property
    def end(self) -> int:
        return self.first + self.num_samples


@dataclass(frozen=True)
class Encoding:
    """The code of one fixture: its served rate, its islands and its amplitude."""

    sample_rate_hertz: float
    islands: Tuple[Island, ...]
    exponent: int = 0

    @classmethod
    def from_intervals(
        cls,
        intervals_seconds: Iterable[Tuple[float, float]],
        sample_rate_hertz: float,
        exponent: int = 0,
    ) -> "Encoding":
        """
        Islands covering each [start, end) interval, in seconds.

        Refuses a rate that is not a power of two (sample times must be exact in
        float64), overlapping intervals, and a layout with more samples than one
        detector has codes.
        """
        rate = float(sample_rate_hertz)
        if rate <= 0 or 2 ** round(np.log2(rate)) != rate:
            raise ValueError(f"The served rate must be a power of two, got {rate} Hz.")
        spans = sorted(
            (int(np.floor(start * rate)), int(np.ceil(end * rate)))
            for start, end in intervals_seconds
        )
        if not spans:
            raise ValueError("A fixture needs at least one island.")
        islands, offset, previous_end = [], 0, None
        for first, end in spans:
            if end <= first:
                raise ValueError(f"Empty island [{first}, {end}) samples.")
            if previous_end is not None and first < previous_end:
                raise ValueError("Fixture islands overlap; merge them first.")
            islands.append(Island(first, end - first, offset))
            offset += end - first
            previous_end = end
        if offset > CODES_PER_DETECTOR:
            raise ValueError(
                f"The fixture holds {offset} samples per detector at {rate:g} Hz; the code "
                f"has {CODES_PER_DETECTOR} ({CODES_PER_DETECTOR / rate:g} s). Use smaller "
                "islands or a lower rate."
            )
        return cls(rate, tuple(islands), int(exponent))

    # ------------------------------------------------------------------ encode

    @property
    def unit(self) -> float:
        return float(2.0 ** -self.exponent)

    def detector_base(self, ifo_name: str) -> int:
        if ifo_name not in CODE_DETECTORS:
            raise OutsideFixture(f"The code has no range for detector {ifo_name}.")
        return CODE_FLOOR + CODE_DETECTORS.index(ifo_name) * CODES_PER_DETECTOR

    def encode(self, ifo_name: str, first: int, num_samples: int) -> np.ndarray:
        """Values of samples [first, first + num_samples) of one detector, float64."""
        samples = np.arange(first, first + num_samples, dtype=np.int64)
        codes = np.full(num_samples, -1, dtype=np.int64)
        for island in self.islands:
            inside = (samples >= island.first) & (samples < island.end)
            codes[inside] = island.code_offset + samples[inside] - island.first
        if num_samples == 0 or np.any(codes < 0):
            missing = samples[codes < 0]
            raise OutsideFixture(
                f"{ifo_name} samples {first}..{first + num_samples} at "
                f"{self.sample_rate_hertz:g} Hz are not in the fixture (first missing GPS "
                f"{(missing[0] if len(missing) else first) / self.sample_rate_hertz:.6f})."
            )
        return (self.detector_base(ifo_name) + codes).astype(np.float64) * self.unit

    # ------------------------------------------------------------------ decode

    def sample_index(self, codes: np.ndarray) -> np.ndarray:
        """Absolute sample indices of island codes (detector range removed)."""
        offsets = np.array([island.code_offset for island in self.islands])
        index = np.searchsorted(offsets, codes, side="right") - 1
        island_first = np.array([island.first for island in self.islands])[index]
        island_size = np.array([island.num_samples for island in self.islands])[index]
        within = codes - offsets[index]
        if np.any(within >= island_size):
            raise DecodeRefused("value outside the fixture's islands")
        return island_first + within


@dataclass(frozen=True)
class Span:
    """A decoded row: its detector, first absolute sample index, length and stride."""

    ifo: str
    first: int
    num_samples: int
    stride: int = 1

    @property
    def end(self) -> int:
        """One past the last sample's index, at the served rate."""
        return self.first + self.num_samples * self.stride


def decode_row(
    row,
    ifo_name: str,
    encoding: Encoding,
    stride: int = 1,
) -> Span:
    """
    The span of one row of fake strain.

    Refuses, by name: an empty, non-finite, zero-filled or sign-flipped row; values
    that are not integer codes (for example a row scaled by 1e21); values outside
    the fixture; a row carrying another detector's code; a reversed row; and any
    step other than ``stride`` samples (reported as a dropped, duplicated or
    displaced sample, at its first position).
    """
    values = np.asarray(row, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise DecodeRefused("empty row")
    if not np.all(np.isfinite(values)):
        raise DecodeRefused("non-finite values")
    if np.any(values == 0):
        raise DecodeRefused(f"zero samples ({int(np.sum(values == 0))}): a zero-filled fetch")
    if np.all(values < 0):
        raise DecodeRefused("negative values: the row is sign-flipped")
    if np.any(values < 0):
        raise DecodeRefused("mixed-sign values: not fake strain")
    scaled = values / encoding.unit
    codes = np.rint(scaled)
    off_integer = np.abs(scaled - codes)
    if np.max(off_integer) > INTEGER_TOLERANCE:
        raise DecodeRefused(
            f"values are not integer codes (largest distance {np.max(off_integer):.3g}); "
            "the row is rescaled or filtered"
        )
    codes = codes.astype(np.int64)
    if np.any(codes < CODE_FLOOR) or np.any(codes >= CODE_CEILING):
        raise DecodeRefused("values outside the code's range")
    detector_index = (codes - CODE_FLOOR) // CODES_PER_DETECTOR
    names = {CODE_DETECTORS[int(index)] for index in np.unique(detector_index)}
    if names != {ifo_name}:
        raise DecodeRefused(
            f"the {ifo_name} row carries the code of {sorted(names)}: detectors swapped or mixed"
        )
    samples = encoding.sample_index(codes - encoding.detector_base(ifo_name))
    steps = np.diff(samples)
    if steps.size and np.all(steps == -stride):
        raise DecodeRefused("the row is time-reversed")
    bad = np.flatnonzero(steps != stride)
    if bad.size:
        position = int(bad[0])
        step = int(steps[position])
        if step == 0:
            kind = "duplicated sample"
        elif 0 < step and step % stride == 0:
            kind = f"{step // stride - 1} dropped sample(s)"
        elif step < 0:
            kind = "samples out of order"
        else:
            kind = "displaced sample"
        raise DecodeRefused(f"{kind} after position {position} (step {step}, expected {stride})")
    return Span(ifo_name, int(samples[0]), int(samples.size), int(stride))


def decode_rows(
    rows,
    ifo_names: Sequence[str],
    encoding: Encoding,
    stride: int = 1,
    what: str = "row",
) -> np.ndarray:
    """
    Decode a (B, I, T) array: returns (B, I, 2) int64 [first, end) at the served rate.

    A row that does not decode raises ``DecodeRefused`` naming its batch and detector
    indices.
    """
    rows = np.asarray(rows)
    if rows.ndim != 3 or rows.shape[1] != len(ifo_names):
        raise DecodeRefused(f"{what}: expected (B, {len(ifo_names)}, T), got {rows.shape}")
    out = np.empty(rows.shape[:2] + (2,), dtype=np.int64)
    for b in range(rows.shape[0]):
        for d, ifo_name in enumerate(ifo_names):
            try:
                span = decode_row(rows[b, d], ifo_name, encoding, stride)
            except DecodeRefused as error:
                raise DecodeRefused(f"{what} [{b}, {d}] ({ifo_name}): {error}") from None
            out[b, d] = (span.first, span.end)
    return out


@dataclass(frozen=True)
class Window:
    """Window geometry in samples at the output rate."""

    onsource: int
    crop: int
    offsource: int

    @classmethod
    def from_seconds(cls, sample_rate_hertz, onsource, crop, offsource) -> "Window":
        rate = float(sample_rate_hertz)
        return cls(int(round(onsource * rate)), int(round(crop * rate)), int(round(offsource * rate)))


def footprints(
    onsource,
    offsource,
    ifo_names: Sequence[str],
    encoding: Encoding,
    window: Window,
    onsource_cropped: bool,
    stride: int = 1,
    adjacent: bool = True,
) -> np.ndarray:
    """
    Per-detector footprints (B, I, 2), int64 sample indices at the served rate:
    [offsource first, padded-onsource end).

    Also refuses rows of the wrong length and, when ``adjacent`` (the noise path's
    geometry), an offsource whose end is not the padded onsource's start.
    """
    on = decode_rows(onsource, ifo_names, encoding, stride, "ONSOURCE")
    off = decode_rows(offsource, ifo_names, encoding, stride, "OFFSOURCE")
    pad = window.crop * stride if onsource_cropped else 0
    expected_on = (window.onsource if onsource_cropped else window.onsource + 2 * window.crop) * stride
    on_len = on[..., 1] - on[..., 0]
    off_len = off[..., 1] - off[..., 0]
    if np.any(on_len != expected_on):
        raise DecodeRefused(f"ONSOURCE rows hold {np.unique(on_len)} samples, expected {expected_on}")
    if np.any(off_len != window.offsource * stride):
        raise DecodeRefused(
            f"OFFSOURCE rows hold {np.unique(off_len)} samples, expected {window.offsource * stride}"
        )
    padded_start = on[..., 0] - pad
    seam = off[..., 1] - padded_start
    if adjacent and np.any(seam != 0):
        b, d = np.argwhere(seam != 0)[0]
        raise DecodeRefused(
            f"row [{b}, {d}]: the offsource ends {seam[b, d]} samples from the padded onsource start"
        )
    return np.stack([off[..., 0], on[..., 1] + pad], axis=-1)


def as_seconds(samples, encoding: Encoding) -> np.ndarray:
    """Sample indices as GPS seconds (exact: the served rate is a power of two)."""
    return np.asarray(samples, dtype=np.float64) / encoding.sample_rate_hertz


def ifo_names(ifos) -> Tuple[str, ...]:
    return tuple(getattr(ifo, "name", ifo) for ifo in ifos)


def batch_arrays(batch, keys) -> Dict:
    """Arrays for ``keys`` (ReturnVariables) from an obtainer dict or a Dataset pair."""
    if isinstance(batch, tuple):
        merged = {}
        for part in batch:
            merged.update(part)
        batch = merged
    out = {}
    for key in keys:
        if key in batch:
            out[key] = np.asarray(batch[key])
        elif getattr(key, "name", None) in batch:
            out[key] = np.asarray(batch[key.name])
        else:
            raise KeyError(f"The batch has no {getattr(key, 'name', key)}; request it.")
    return out


__all__ = [
    "CODE_FLOOR", "CODE_DETECTORS", "CODES_PER_DETECTOR", "DecodeRefused", "OutsideFixture",
    "Island", "Encoding", "Span", "decode_row", "decode_rows", "Window", "footprints",
    "as_seconds", "ifo_names", "batch_arrays",
]
