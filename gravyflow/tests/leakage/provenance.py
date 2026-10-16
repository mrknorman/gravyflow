"""
The provenance code: every fake strain sample carries its own GPS time, in small
integers.

The samples of a fixture are numbered from its origin, m = n - n0, at the fixture's
served rate. Sample m sits at position j = m mod P of block b = m div P (P = 256).
Its value is the integer

    v = 1 + j + P * (digit + K * detector)

where ``digit`` is digit (j mod D) of the block number b in base K (K = 16, D = 7
digits), and ``detector`` is the detector's index in ``CODE_DETECTORS``. So every row
of at least P + D samples names, from its values alone: its detector, the block of
each sample (from the digits the block's samples carry), each sample's position, and
so each sample's absolute index. No label and no fetch log is read. Every value is at
least 1 (a zero-filled row never decodes) and at most 2^13.

Exactness. The library resamples every real-noise piece with a float32 FFT, even at
an equal rate, on pieces of any whole-second length. Its round trip moves a value by
an amount proportional to the largest value in the piece: an absolute ramp of 2^20
values moved by up to 1.9, while values below 2^13 move by about 0.01. The decoder
rounds to the nearest integer and refuses any value more than 0.25 from it, so an
error that grows refuses loudly rather than shifting a row.

The decoder refuses, never skips: a row it cannot decode raises ``DecodeRefused``
with the row and the reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

#: The detectors the code distinguishes, in the order of their indices.
CODE_DETECTORS: Tuple[str, ...] = ("H1", "L1")
#: Samples per block, digit base and digits per block number. The digit count is odd,
#: so a row thinned by any power-of-two stride still meets every digit in a block.
BLOCK_SAMPLES, DIGIT_BASE, NUM_DIGITS = 256, 16, 7
#: The values a fake sample can take: [1, CODE_CEILING).
CODE_CEILING = 1 + BLOCK_SAMPLES * DIGIT_BASE * len(CODE_DETECTORS)
#: The largest number of samples a fixture may span from its origin.
MAX_SPAN_SAMPLES = BLOCK_SAMPLES * DIGIT_BASE ** NUM_DIGITS
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

    @property
    def end(self) -> int:
        return self.first + self.num_samples


@dataclass(frozen=True)
class Encoding:
    """The code of one fixture: its served rate, its islands of data and its amplitude."""

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
        Islands covering each [start, end) interval, in seconds. Refuses a rate that
        is not a power of two (sample times must be exact in float64), overlapping
        intervals, and a span beyond the code.
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
        islands, previous_end = [], None
        for first, end in spans:
            if end <= first:
                raise ValueError(f"Empty island [{first}, {end}) samples.")
            if previous_end is not None and first < previous_end:
                raise ValueError("Fixture islands overlap; merge them first.")
            islands.append(Island(first, end - first))
            previous_end = end
        if previous_end - islands[0].first > MAX_SPAN_SAMPLES:
            raise ValueError(
                f"The fixture spans {previous_end - islands[0].first} samples at {rate:g} Hz; "
                f"the code holds {MAX_SPAN_SAMPLES}."
            )
        return cls(rate, tuple(islands), int(exponent))

    @property
    def origin(self) -> int:
        return self.islands[0].first

    @property
    def unit(self) -> float:
        return float(2.0 ** -self.exponent)

    def detector_index(self, ifo_name: str) -> int:
        if ifo_name not in CODE_DETECTORS:
            raise OutsideFixture(f"The code has no detector {ifo_name}.")
        return CODE_DETECTORS.index(ifo_name)

    def inside(self, samples) -> np.ndarray:
        """Whether each absolute sample index lies in an island."""
        samples = np.asarray(samples)
        inside = np.zeros(samples.shape, dtype=bool)
        for island in self.islands:
            inside |= (samples >= island.first) & (samples < island.end)
        return inside

    def encode(self, ifo_name: str, first: int, num_samples: int) -> np.ndarray:
        """Values of samples [first, first + num_samples) of one detector, float64."""
        detector = self.detector_index(ifo_name)
        samples = np.arange(first, first + num_samples, dtype=np.int64)
        outside = ~self.inside(samples)
        if num_samples == 0 or np.any(outside):
            missing = samples[outside]
            raise OutsideFixture(
                f"{ifo_name} samples {first}..{first + num_samples} at "
                f"{self.sample_rate_hertz:g} Hz are not in the fixture (first missing GPS "
                f"{(missing[0] if len(missing) else first) / self.sample_rate_hertz:.6f})."
            )
        m = samples - self.origin
        position, block = m % BLOCK_SAMPLES, m // BLOCK_SAMPLES
        digit = (block // DIGIT_BASE ** (position % NUM_DIGITS)) % DIGIT_BASE
        values = 1 + position + BLOCK_SAMPLES * (digit + DIGIT_BASE * detector)
        return values.astype(np.float64) * self.unit


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


def _blocks(position: np.ndarray, digit: np.ndarray) -> np.ndarray:
    """
    The block number of every sample of a forward row, from the digits its blocks
    carry. The row splits into runs at each wrap of the position; a run that carries
    every digit gives its block, and the other runs follow by one block per run, where
    their own digits must agree.
    """
    starts = np.concatenate([[0], np.flatnonzero(np.diff(position) < 0) + 1])
    ends = np.concatenate([starts[1:], [position.size]])
    runs = []
    for a, z in zip(starts, ends):
        known: Dict[int, int] = {}
        for i, d in zip((position[a:z] % NUM_DIGITS).tolist(), digit[a:z].tolist()):
            if known.setdefault(i, d) != d:
                raise DecodeRefused("a block carries two values of one digit")
        runs.append(known)
    complete = [k for k, known in enumerate(runs) if len(known) == NUM_DIGITS]
    if not complete:
        raise DecodeRefused("no block of the row carries every digit: the row is too short")
    k0 = complete[0]
    base = sum(runs[k0][i] * DIGIT_BASE ** i for i in range(NUM_DIGITS))
    blocks = np.empty(position.size, dtype=np.int64)
    for k, (a, z, known) in enumerate(zip(starts, ends, runs)):
        number = base + (k - k0)
        for i, d in known.items():
            if (number // DIGIT_BASE ** i) % DIGIT_BASE != d:
                raise DecodeRefused("blocks of the row are not consecutive")
        blocks[a:z] = number
    return blocks


def decode_row(
    row,
    ifo_name: str,
    encoding: Encoding,
    stride: int = 1,
) -> Span:
    """
    The span of one row of fake strain.

    Refuses, by name: an empty, non-finite, zero-filled or sign-flipped row; values
    that are not integer codes (for example a rescaled row) or outside the code; a
    row carrying another detector's code; samples outside the fixture; a reversed
    row; and any step other than ``stride`` samples (reported as a dropped,
    duplicated or displaced sample, at its first position).
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
    if np.any(codes < 1) or np.any(codes >= CODE_CEILING):
        raise DecodeRefused("values outside the code's range: the row is rescaled")
    c = codes.astype(np.int64) - 1
    position = c % BLOCK_SAMPLES
    digit = (c // BLOCK_SAMPLES) % DIGIT_BASE
    detectors = {CODE_DETECTORS[int(k)] for k in np.unique(c // (BLOCK_SAMPLES * DIGIT_BASE))}
    if detectors != {ifo_name}:
        raise DecodeRefused(
            f"the {ifo_name} row carries the code of {sorted(detectors)}: detectors swapped or mixed"
        )
    try:
        blocks = _blocks(position, digit)
    except DecodeRefused as forward_error:
        try:
            _blocks(position[::-1], digit[::-1])
        except DecodeRefused:
            raise forward_error from None
        raise DecodeRefused("the row is time-reversed") from None
    samples = encoding.origin + blocks * BLOCK_SAMPLES + position
    steps = np.diff(samples)
    if steps.size and np.all(steps == -stride):
        raise DecodeRefused("the row is time-reversed")
    bad = np.flatnonzero(steps != stride)
    if bad.size:
        at = int(bad[0])
        step = int(steps[at])
        if step == 0:
            kind = "duplicated sample"
        elif 0 < step and step % stride == 0:
            kind = f"{step // stride - 1} dropped sample(s)"
        elif step < 0:
            kind = "samples out of order"
        else:
            kind = "displaced sample"
        raise DecodeRefused(f"{kind} after position {at} (step {step}, expected {stride})")
    if not np.all(encoding.inside(samples)):
        raise DecodeRefused("samples outside the fixture's islands")
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
    "CODE_DETECTORS", "BLOCK_SAMPLES", "DIGIT_BASE", "NUM_DIGITS", "CODE_CEILING",
    "MAX_SPAN_SAMPLES", "DecodeRefused", "OutsideFixture", "Island", "Encoding", "Span",
    "decode_row", "decode_rows", "Window", "footprints", "as_seconds", "ifo_names",
    "batch_arrays",
]
