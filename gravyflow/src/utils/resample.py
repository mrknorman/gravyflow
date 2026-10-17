"""
The one resampler: every read of real strain reaches its sample rate here.

``resample`` is SciPy's polyphase FIR (``scipy.signal.resample_poly``) with an explicit
Kaiser low-pass (``scipy.signal.firwin``), designed once per reduced rate ratio. It runs
in NumPy and SciPy, never under JIT: SciPy's single-threaded C code gives the same
bytes for the same input on every call, which XLA's multi-threaded CPU kernels do not.
It computes in float64 and returns float32.

The contract, which the tests check:

- alias suppression of at least 100 dB above the new Nyquist frequency (the lower of
  the two rates' Nyquist frequencies);
- passband ripple of at most 1e-3 up to 0.9 times that frequency;
- output sample n at exactly ``t0 + n / rate_out``, where ``t0`` is the time of the
  input's first sample;
- a linear ramp kept exactly.

Near either end of the input, an output sample's filter reaches past the input, which
is extended by the line through its end samples (so a ramp is kept to the last sample).
Those outputs, within ``resample_margin`` input samples of an end, are approximations.
Every other output depends only on input samples, and is bit for bit what a longer
input gives. A read that needs exact samples extends past each end of its window by the
margin.

Changing the filter's design or the alignment changes the samples served: bump the
resampler's entry in ``DATA_VERSIONS`` (``dataset/acquisition/versions.py``).
"""

from functools import lru_cache
from math import gcd
from typing import Any, Tuple

import numpy as np
from scipy.signal import firwin, kaiserord, resample_poly

#: The passband edge, as a fraction of the new Nyquist frequency; the stopband starts at
#: the new Nyquist frequency.
PASSBAND_FRACTION = 0.9
#: The stopband attenuation the filter is designed for: the contract's 100 dB plus 2 dB,
#: because Kaiser's length estimate falls up to 0.1 dB short of its target.
DESIGN_ATTENUATION_DB = 102.0


def whole_hertz(rate_hertz: Any) -> int:
    """
    ``rate_hertz`` as an integer number of hertz.

    Raises:
        ValueError: naming the rate, if it is not a positive whole number of hertz. The
            resampler's ratios are exact integers, so it refuses any other rate.
    """
    rate = float(rate_hertz)
    if not (np.isfinite(rate) and rate > 0 and rate == np.round(rate)):
        raise ValueError(
            f"Sample rate {rate_hertz!r} Hz is not a positive whole number of hertz: "
            "gravyflow resamples only between whole-hertz rates."
        )
    return int(rate)


def resample_ratio(rate_in: Any, rate_out: Any) -> Tuple[int, int]:
    """``(up, down)``: ``rate_out / rate_in`` in lowest terms (refusing as ``whole_hertz``)."""
    rate_in, rate_out = whole_hertz(rate_in), whole_hertz(rate_out)
    divisor = gcd(rate_in, rate_out)
    return rate_out // divisor, rate_in // divisor


@lru_cache(maxsize=None)
def resample_filter(up: int, down: int) -> np.ndarray:
    """
    The low-pass FIR that resamples by ``up / down`` (in lowest terms), at the
    upsampled rate: a Kaiser window, an odd number of taps so its delay is a whole
    sample, and unit gain at zero frequency. Designed once per ratio; read-only.
    """
    widest = max(up, down)
    numtaps, beta = kaiserord(DESIGN_ATTENUATION_DB, (1.0 - PASSBAND_FRACTION) / widest)
    taps = firwin(numtaps | 1, (1.0 + PASSBAND_FRACTION) / 2.0 / widest, window=("kaiser", beta))
    taps.flags.writeable = False
    return taps


def resample_margin(rate_in: Any, rate_out: Any) -> int:
    """
    The input samples an output sample's filter reaches on each side of the output's
    own time: a read that extends this far past each end of its window resamples every
    sample of the window exactly. Zero when the rates are equal.
    """
    up, down = resample_ratio(rate_in, rate_out)
    if up == down:
        return 0
    half_length = (len(resample_filter(up, down)) - 1) // 2
    return -(-half_length // up)


def resample(x: Any, t0: Any, rate_in: Any, rate_out: Any) -> Tuple[np.ndarray, Any]:
    """
    ``x`` (samples on its last axis, at ``rate_in``) at ``rate_out``, and the time of
    its first sample, which is ``t0``, the time of ``x``'s first sample: output sample
    n is at exactly ``t0 + n / rate_out``. There are ``ceil(len * rate_out / rate_in)``
    of them. Equal rates return ``x`` unchanged, as float32.

    Raises:
        ValueError: if either rate is not a whole number of hertz (``whole_hertz``).
    """
    up, down = resample_ratio(rate_in, rate_out)
    samples = np.asarray(x)
    if up == down:
        return samples.astype(np.float32), t0
    taps = resample_filter(up, down)
    resampled = np.empty(samples.shape[:-1] + (-(-samples.shape[-1] * up // down),), dtype=np.float32)
    # One row at a time, so the float64 working copy is one row's, whatever the batch.
    for row in np.ndindex(samples.shape[:-1]):
        resampled[row] = resample_poly(
            samples[row].astype(np.float64), up, down, window=taps, padtype="line",
        )
    return resampled, t0
