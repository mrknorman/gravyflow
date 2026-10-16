"""
Content fingerprints for leakage tests on synthetic data.

Synthetic noise has no GPS time to decode, so provenance is judged from the
content itself:

- **Strain windows.** Every 64-sample window of every row, at every offset, is
  a fingerprint. Two sets share data when any window of one equals, byte for
  byte, any window of the other, whatever its key (onsource, offsource,
  injections) or detector. A shifted copy, or an onsource cut from another
  example's offsource, is caught as well as an identical row.
- **Label streams.** Each label is a per-example sequence. Two sets share a
  stream when, at some lag, the rank correlation of their sequences exceeds a
  threshold corrected for every lag and pair tested; the same stream mapped to
  another range, or shifted by a few examples, has rank correlation one.
  A continuous label must also share no more exactly equal values with the
  same label of another set than chance allows, which catches a stream served
  in another order.
- **Object state.** A recursive digest of arrays, random generator states and
  plain attributes, which refuses, by name, any type whose state it cannot
  read.

Every assertion reports the counts it found, never a bare boolean, and every
helper refuses an input too small for its check to mean anything.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import itertools
import math
import pathlib
import types
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from scipy.stats import norm, poisson, rankdata

#: Samples per strain fingerprint.
WINDOW = 64
#: Family-wise false-alarm probability of one stream-independence assertion.
STREAM_ALPHA = 1e-6
#: Fewest examples a label stream may have.
MIN_STREAM_EXAMPLES = 64
#: Largest threshold a tested lag may need: a shared stream (|rho| = 1) is
#: always caught, and a lag too short to reach this is not tested.
MAX_STREAM_THRESHOLD = 0.9


# ----------------------------------------------------------------- strain

@dataclass(frozen=True)
class StrainPrints:
    """The window fingerprints of one set of strain arrays."""
    windows: np.ndarray            # void dtype, one entry per window
    rows: np.ndarray               # the row each window came from
    origins: Tuple[Tuple[str, Tuple[int, ...]], ...]   # (key, leading index) per row
    constant_windows: int          # windows excluded because every sample is equal

    @property
    def num_rows(self) -> int:
        return len(self.origins)


def strain_prints(
    arrays: Mapping[str, np.ndarray],
    absent: Optional[Mapping[str, np.ndarray]] = None,
    width: int = WINDOW,
) -> StrainPrints:
    """
    Fingerprint every ``width``-sample window of every row of ``arrays``.

    Each array's last axis is time; every leading index (example, detector,
    generator) is one row. A row that is not finite, or is all zeros, is
    refused: it carries no provenance and would hide a fetch or generation
    that failed. ``absent[key]`` (shaped like the leading axes) marks rows
    excluded on purpose, such as injections the mask says are absent.
    Windows whose samples are all equal (zero padding) are excluded and counted.
    """
    if not arrays:
        raise ValueError("strain_prints needs at least one array.")
    absent = absent or {}
    windows, rows, origins, constant = [], [], [], 0
    for key, array in arrays.items():
        array = np.asarray(array)
        if array.dtype != np.float32:
            raise TypeError(f"{key}: strain fingerprints are taken of float32 data, got {array.dtype}.")
        if array.ndim < 2 or array.shape[-1] < width:
            raise ValueError(f"{key}: need rows of at least {width} samples, got shape {array.shape}.")
        leading = array.shape[:-1]
        skip = np.zeros(leading, dtype=bool) if key not in absent else np.asarray(absent[key], dtype=bool)
        if skip.shape != leading:
            raise ValueError(f"{key}: the absent mask has shape {skip.shape}, the rows {leading}.")
        flat = array.reshape(-1, array.shape[-1])
        for r, (row, excluded) in enumerate(zip(flat, skip.reshape(-1))):
            index = np.unravel_index(r, leading)
            index = tuple(int(i) for i in index)
            if excluded:
                continue
            if not np.all(np.isfinite(row)):
                raise ValueError(f"{key}{list(index)}: the row is not finite.")
            if not np.any(row):
                raise ValueError(f"{key}{list(index)}: the row is all zeros.")
            view = np.lib.stride_tricks.sliding_window_view(row, width)
            varying = view.max(axis=1) != view.min(axis=1)
            constant += int(np.count_nonzero(~varying))
            kept = np.ascontiguousarray(view[varying])
            windows.append(kept.view(np.dtype((np.void, width * row.itemsize))).ravel())
            rows.append(np.full(len(kept), len(origins), dtype=np.int64))
            origins.append((key, index))
    if not origins:
        raise ValueError("strain_prints: every row was excluded; there is nothing to compare.")
    return StrainPrints(
        windows=np.concatenate(windows), rows=np.concatenate(rows),
        origins=tuple(origins), constant_windows=constant,
    )


def shared_rows(a: StrainPrints, b: StrainPrints) -> Dict[Tuple[str, str], int]:
    """
    For each (key of a, key of b), how many rows of ``a`` share at least one
    window with a row of ``b`` under that key. Empty when nothing is shared.
    """
    common = np.intersect1d(a.windows, b.windows)
    if common.size == 0:
        return {}
    in_a = np.isin(a.windows, common)
    in_b = np.isin(b.windows, common)
    counts: Dict[Tuple[str, str], set] = {}
    b_rows_of = {}
    for window, row in zip(b.windows[in_b], b.rows[in_b]):
        b_rows_of.setdefault(window.tobytes(), set()).add(int(row))
    for window, row in zip(a.windows[in_a], a.rows[in_a]):
        key_a = a.origins[row][0]
        for row_b in b_rows_of[window.tobytes()]:
            counts.setdefault((key_a, b.origins[row_b][0]), set()).add(int(row))
    return {pair: len(rows) for pair, rows in sorted(counts.items())}


def assert_rows_disjoint(sets: Mapping[str, StrainPrints], min_rows: int = 2) -> None:
    """
    No set shares a strain window with another.

    Raises:
        ValueError: fewer than two sets, or a set with fewer than ``min_rows`` rows.
        AssertionError: a shared window, reported as the matrix of shared rows.
    """
    if len(sets) < 2:
        raise ValueError("assert_rows_disjoint needs at least two sets.")
    for name, prints in sets.items():
        if prints.num_rows < min_rows:
            raise ValueError(f"Set {name!r} has {prints.num_rows} rows; at least {min_rows} are needed.")
    findings = []
    for (name_a, a), (name_b, b) in itertools.combinations(sets.items(), 2):
        shared = shared_rows(a, b)
        if shared:
            matrix = ", ".join(f"{ka}->{kb}: {n}" for (ka, kb), n in shared.items())
            findings.append(f"{name_a} ({a.num_rows} rows) vs {name_b} ({b.num_rows} rows): "
                            f"rows of {name_a} sharing a window [{matrix}]")
    if findings:
        raise AssertionError("Strain shared between sets:\n  " + "\n  ".join(findings))


def _unit_rows(array: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(array, dtype=np.float64).reshape(-1, np.shape(array)[-1])
    if rows.shape[0] == 0 or not np.all(np.isfinite(rows)):
        raise ValueError(f"{name}: need finite rows, got shape {np.shape(array)}.")
    rows = rows - rows.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{name}: a row is constant.")
    return rows / norms


def assert_noise_independent(sets: Mapping[str, np.ndarray], alpha: float = STREAM_ALPHA) -> float:
    """
    No row of one set shares its noise with a row of another, even when a
    signal was added to one of them, so that no window is byte-equal.

    For whitened data (approximately white noise), the normalised cross-
    correlation of two independent rows of n samples has a standard deviation
    of about 1/sqrt(n) at every lag. A row pair fails when its largest
    |correlation| over all lags exceeds z/sqrt(n), z the normal quantile at
    ``alpha`` divided by the number of (pair, lag) tests. Shared noise under a
    signal of whitened SNR rho over n samples correlates at about
    1/sqrt(1 + rho^2/n), and a shifted copy at its overlap fraction. Returns the
    threshold. Not for coloured data, whose correlations are wider.

    Raises:
        AssertionError: the row pairs over the threshold, per pair of sets.
    """
    if len(sets) < 2:
        raise ValueError("assert_noise_independent needs at least two sets.")
    rows = {name: _unit_rows(array, name) for name, array in sets.items()}
    lengths = {r.shape[1] for r in rows.values()}
    if len(lengths) != 1:
        raise ValueError(f"Rows of every set must have one length, got {sorted(lengths)}.")
    n = lengths.pop()
    size = 2 * n
    spectra = {name: np.fft.rfft(r, n=size, axis=1) for name, r in rows.items()}
    pairs = list(itertools.combinations(rows, 2))
    tests = sum(rows[a].shape[0] * rows[b].shape[0] for a, b in pairs) * (2 * n - 1)
    threshold = norm.isf(alpha / (2 * tests)) / math.sqrt(n)
    findings = []
    for a, b in pairs:
        worst = np.empty((rows[a].shape[0], rows[b].shape[0]))
        for i, spectrum in enumerate(spectra[a]):
            correlation = np.fft.irfft(spectrum[None, :] * np.conj(spectra[b]), n=size, axis=1)
            worst[i] = np.max(np.abs(correlation), axis=1)
        over = np.argwhere(worst > threshold)
        if over.size:
            i, j = over[np.argmax(worst[tuple(over.T)])]
            findings.append(f"{a} vs {b}: {len(over)} row pairs over {threshold:.3f}, "
                            f"worst {worst[i, j]:.3f} ({a} row {i}, {b} row {j})")
    if findings:
        raise AssertionError("Noise shared between sets:\n  " + "\n  ".join(findings))
    return threshold


# ----------------------------------------------------------------- labels

def label_streams(labels: Mapping[str, np.ndarray], num_examples: int) -> Dict[str, np.ndarray]:
    """
    Per-example float64 sequences from batch-first labels.

    A label shaped (examples,) is one stream; any further axes are split into
    one stream per element, named ``KEY[i, j]``.
    """
    streams = {}
    for key, value in labels.items():
        array = np.asarray(value, dtype=np.float64)
        if array.shape[:1] != (num_examples,):
            raise ValueError(f"{key}: expected {num_examples} examples first, got shape {array.shape}.")
        flat = array.reshape(num_examples, -1)
        if flat.shape[1] == 1:
            streams[key] = flat[:, 0]
        else:
            for j in range(flat.shape[1]):
                index = ", ".join(str(int(i)) for i in np.unravel_index(j, array.shape[1:]))
                streams[f"{key}[{index}]"] = flat[:, j]
    return streams


def concatenate_streams(batches: Iterable[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Join per-batch streams, in order, into one sequence per stream."""
    batches = list(batches)
    if not batches:
        raise ValueError("No batches to join.")
    keys = set(batches[0])
    for batch in batches[1:]:
        if set(batch) != keys:
            raise ValueError(f"Batches carry different streams: {sorted(keys)} against {sorted(batch)}.")
    return {key: np.concatenate([batch[key] for batch in batches]) for key in sorted(keys)}


def _is_continuous(x: np.ndarray) -> bool:
    """No two examples share a value: a draw from a continuous distribution."""
    return np.unique(x).size == x.size


def _chance_collisions(x: np.ndarray, y: np.ndarray, alpha: float) -> int:
    """
    The most exactly equal values two independent continuous streams share
    with probability above ``alpha``.

    Values stored as float32 collide by chance: with n_x and n_y values over a
    range holding R representable values, about n_x * n_y / R pairs collide.
    The estimate assumes a uniform density and takes ten times that rate, to
    cover peaked distributions; the bound is its Poisson quantile. Float64
    values essentially never collide, so the bound there is zero.
    """
    values = np.concatenate([x, y])
    single = np.array_equal(values, values.astype(np.float32).astype(np.float64))
    spacing = float(np.spacing(np.float32(np.max(np.abs(values)))) if single else np.spacing(np.max(np.abs(values))))
    span = float(np.max(values) - np.min(values))
    rate = 10.0 * x.size * y.size * spacing / span
    return int(poisson.isf(alpha, rate))


def _max_lag(n: int, z: float) -> int:
    """The largest lag whose overlap keeps the threshold at or below MAX_STREAM_THRESHOLD."""
    min_overlap = math.ceil((z / MAX_STREAM_THRESHOLD) ** 2) + 1
    return n - min_overlap


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = math.sqrt(float(np.dot(rx, rx) * np.dot(ry, ry)))
    return float(np.dot(rx, ry) / denominator) if denominator else 0.0


def lagged_rank_correlations(x: np.ndarray, y: np.ndarray, max_lag: int) -> Dict[int, Tuple[float, int]]:
    """Spearman's rho of ``x[t + lag]`` against ``y[t]``, with its overlap, for every lag in range."""
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        a, b = (x[lag:], y[:n - lag]) if lag >= 0 else (x[:n + lag], y[-lag:])
        out[lag] = (_rank_correlation(a, b), a.size)
    return out


def assert_streams_independent(
    sets: Mapping[str, Mapping[str, np.ndarray]],
    alpha: float = STREAM_ALPHA,
    min_examples: int = MIN_STREAM_EXAMPLES,
) -> Dict[str, List[str]]:
    """
    No label stream of one set is shared with any stream of another set.

    Every pair of streams across every pair of sets is tested, at every lag
    whose overlap is long enough. Two checks:

    - the lagged rank correlation: at overlap n, a lag fails when
      ``|rho| > z / sqrt(n - 1)``, where z is the two-sided normal quantile at
      ``alpha`` divided by the number of tests (Bonferroni), so independent
      streams fail with probability at most ``alpha`` and a shared stream,
      monotone-transformed or shifted, has ``|rho| = 1`` and always fails;
    - for the same label in both sets, continuous in both, more exactly equal
      values than independent draws share by chance (`_chance_collisions`):
      this catches a shared stream served in another order.

    Streams that are constant within a set are excluded, and named in the
    returned report. Returns ``{"excluded": [...], "tested": [...]}``.

    Raises:
        ValueError: fewer than two sets, a stream shorter than ``min_examples``,
            or no stream left to test.
        AssertionError: a shared stream, with its correlation, lag and threshold.
    """
    if len(sets) < 2:
        raise ValueError("assert_streams_independent needs at least two sets.")
    kept: Dict[str, Dict[str, np.ndarray]] = {}
    excluded = []
    for name, streams in sets.items():
        kept[name] = {}
        for key, x in streams.items():
            x = np.asarray(x, dtype=np.float64)
            if x.ndim != 1 or x.size < min_examples:
                raise ValueError(f"{name}/{key}: need a sequence of at least {min_examples} examples, "
                                 f"got shape {x.shape}.")
            if not np.all(np.isfinite(x)):
                raise ValueError(f"{name}/{key}: the stream is not finite.")
            if np.all(x == x[0]):
                excluded.append(f"{name}/{key} (constant {x[0]:g})")
            else:
                kept[name][key] = x
    pairs = [
        ((name_a, key_a, kept[name_a][key_a]), (name_b, key_b, kept[name_b][key_b]))
        for name_a, name_b in itertools.combinations(kept, 2)
        for key_a in kept[name_a] for key_b in kept[name_b]
    ]
    if not pairs:
        raise ValueError(f"No stream left to test; excluded: {excluded}.")

    # Bonferroni over every (pair, lag): the lag range depends on z, so iterate to a fixed point.
    z = norm.isf(alpha / 2)
    for _ in range(8):
        tests = sum(2 * max(_max_lag(min(a[2].size, b[2].size), z), 0) + 1 for a, b in pairs)
        z_next = norm.isf(alpha / (2 * tests))
        if abs(z_next - z) < 1e-9:
            break
        z = z_next

    findings, tested = [], []
    for (name_a, key_a, x), (name_b, key_b, y) in pairs:
        label = f"{name_a}/{key_a} vs {name_b}/{key_b}"
        max_lag = _max_lag(min(x.size, y.size), z)
        if max_lag < 0:
            raise ValueError(f"{label}: too few examples for a threshold of {MAX_STREAM_THRESHOLD}.")
        correlations = lagged_rank_correlations(x, y, max_lag)
        # The lag furthest beyond its own threshold (thresholds grow as the overlap shrinks).
        lag, (rho, overlap) = max(
            correlations.items(), key=lambda item: abs(item[1][0]) * math.sqrt(item[1][1] - 1))
        threshold = z / math.sqrt(overlap - 1)
        tested.append(f"{label}: rho {rho:+.3f} at lag {lag} (threshold {threshold:.3f})")
        if abs(rho) > threshold:
            findings.append(f"{label}: rho {rho:+.4f} at lag {lag} (overlap {overlap}) "
                            f"exceeds {threshold:.4f}")
        if key_a == key_b and _is_continuous(x) and _is_continuous(y):
            equal = int(np.count_nonzero(np.isin(x, y)))
            allowed = _chance_collisions(x, y, alpha / len(pairs))
            if equal > allowed:
                findings.append(f"{label}: {equal} of {x.size} values of {name_a} appear exactly in "
                                f"{name_b} (chance allows {allowed})")
    if findings:
        raise AssertionError(
            f"Label streams shared between sets ({len(pairs)} stream pairs, z = {z:.2f}):\n  "
            + "\n  ".join(findings))
    return {"excluded": excluded, "tested": tested}


# ------------------------------------------------------------ object state

class OpaqueStateError(TypeError):
    """An object holds state the digest cannot read."""


_LEAVES = (type(None), bool, int, float, complex, str, bytes)


def state_items(obj, path: str = "") -> Dict[str, bytes]:
    """
    The state of ``obj`` as ``{path: leaf bytes}``.

    Recurses through containers, dataclasses and objects with ``__dict__`` or
    ``__slots__``. Arrays contribute dtype, shape and bytes; NumPy generators,
    bit generators, seed sequences and legacy RandomStates contribute their
    full state. Functions contribute their qualified name, and closures their
    cells. An object reached twice is recorded once, then by reference, so
    cycles terminate.

    Raises:
        OpaqueStateError: a type whose state cannot be read (a lock, a thread
            pool, a Python generator, a file), named with its path.
    """
    items: Dict[str, bytes] = {}
    seen: Dict[int, str] = {}
    _walk(obj, path or "<root>", items, seen)
    return items


def state_digest(obj) -> str:
    """The sha256 of ``state_items(obj)``."""
    h = hashlib.sha256()
    for path, leaf in sorted(state_items(obj).items()):
        h.update(path.encode())
        h.update(b"\0")
        h.update(leaf)
        h.update(b"\0")
    return h.hexdigest()


def diff_states(before: Dict[str, bytes], after: Dict[str, bytes]) -> List[str]:
    """The paths whose state differs, appeared or disappeared."""
    return sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))


def assert_unchanged(before: Mapping[str, Dict[str, bytes]], objects: Mapping[str, object]) -> None:
    """
    Every object's state equals its snapshot ``before[name]`` (from `state_items`).

    Raises:
        AssertionError: the paths that changed, per object.
    """
    if set(before) != set(objects):
        raise ValueError(f"Snapshots {sorted(before)} do not match objects {sorted(objects)}.")
    changed = {}
    for name, obj in objects.items():
        if not before[name]:
            raise ValueError(f"The snapshot of {name!r} is empty.")
        paths = diff_states(before[name], state_items(obj, name))
        if paths:
            changed[name] = paths
    if changed:
        report = "\n  ".join(f"{name}: {len(p)} paths, first {p[:5]}" for name, p in changed.items())
        raise AssertionError("Caller objects changed by use:\n  " + report)


def _leaf(value) -> bytes:
    return f"{type(value).__name__}:{value!r}".encode()


def _walk(obj, path: str, items: Dict[str, bytes], seen: Dict[int, str]) -> None:
    if isinstance(obj, _LEAVES):
        items[path] = _leaf(obj)
        return
    if isinstance(obj, enum.Enum):
        items[path] = f"enum:{type(obj).__qualname__}.{obj.name}".encode()
        return
    if isinstance(obj, (np.generic,)):
        items[path] = f"{obj.dtype}:".encode() + np.asarray(obj).tobytes()
        return
    if isinstance(obj, pathlib.PurePath):
        items[path] = f"path:{obj}".encode()
        return
    if isinstance(obj, type):
        items[path] = f"type:{obj.__module__}.{obj.__qualname__}".encode()
        return
    if isinstance(obj, np.dtype):
        items[path] = f"dtype:{obj.str}".encode()
        return

    ident = id(obj)
    if ident in seen:
        items[path] = f"ref:{seen[ident]}".encode()
        return
    seen[ident] = path

    if isinstance(obj, np.ndarray) or type(obj).__module__.startswith(("jax", "jaxlib")) and hasattr(obj, "__array__"):
        array = np.asarray(obj)
        if array.dtype == object:
            items[path] = f"object-array:{array.shape}".encode()
            for i, element in enumerate(array.reshape(-1)):
                _walk(element, f"{path}[{i}]", items, seen)
        else:
            items[path] = f"array:{array.dtype.str}:{array.shape}:".encode() + np.ascontiguousarray(array).tobytes()
        return
    if isinstance(obj, np.random.Generator):
        _walk(obj.bit_generator.state, f"{path}.bit_generator.state", items, seen)
        return
    if isinstance(obj, np.random.BitGenerator):
        _walk(obj.state, f"{path}.state", items, seen)
        return
    if isinstance(obj, np.random.RandomState):
        _walk(obj.get_state(legacy=False), f"{path}.state", items, seen)
        return
    if isinstance(obj, np.random.SeedSequence):
        _walk({"entropy": obj.entropy, "spawn_key": obj.spawn_key, "pool_size": obj.pool_size,
               "n_children_spawned": obj.n_children_spawned}, f"{path}.seed_sequence", items, seen)
        return
    if isinstance(obj, dict):
        items[path] = f"dict:{len(obj)}".encode()
        for key, value in obj.items():
            _walk(value, f"{path}[{key!r}]", items, seen)
        return
    if isinstance(obj, (list, tuple)):
        items[path] = f"{type(obj).__name__}:{len(obj)}".encode()
        for i, value in enumerate(obj):
            _walk(value, f"{path}[{i}]", items, seen)
        return
    if isinstance(obj, (set, frozenset)):
        if not all(isinstance(v, _LEAVES + (enum.Enum,)) for v in obj):
            raise OpaqueStateError(f"{path}: a set of non-scalar elements has no order to digest.")
        items[path] = f"{type(obj).__name__}:".encode() + b"|".join(sorted(_leaf(v) for v in obj))
        return
    if isinstance(obj, functools.partial):
        _walk({"func": obj.func, "args": obj.args, "keywords": obj.keywords}, f"{path}.partial", items, seen)
        return
    if isinstance(obj, types.MethodType):
        _walk(obj.__func__, f"{path}.__func__", items, seen)
        _walk(obj.__self__, f"{path}.__self__", items, seen)
        return
    if isinstance(obj, (types.FunctionType, types.BuiltinFunctionType)):
        items[path] = f"function:{getattr(obj, '__module__', '')}.{obj.__qualname__}".encode()
        for i, cell in enumerate(getattr(obj, "__closure__", None) or ()):
            _walk(cell.cell_contents, f"{path}.<cell {i}>", items, seen)
        return
    if dataclasses.is_dataclass(obj) or hasattr(obj, "__dict__") or hasattr(type(obj), "__slots__"):
        if type(obj).__module__ in ("builtins", "_thread", "threading", "concurrent.futures.thread", "_io", "io"):
            raise OpaqueStateError(f"{path}: {type(obj).__module__}.{type(obj).__qualname__} has no readable state.")
        items[path] = f"object:{type(obj).__module__}.{type(obj).__qualname__}".encode()
        attributes = dict(vars(obj)) if hasattr(obj, "__dict__") else {}
        for klass in type(obj).__mro__:
            for slot in getattr(klass, "__slots__", ()):
                if slot not in ("__dict__", "__weakref__") and hasattr(obj, slot):
                    attributes[slot] = getattr(obj, slot)
        for name in sorted(attributes):
            _walk(attributes[name], f"{path}.{name}", items, seen)
        return
    raise OpaqueStateError(f"{path}: {type(obj).__module__}.{type(obj).__qualname__} has no readable state.")
