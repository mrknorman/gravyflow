"""
Detection metrics over score arrays: false-alarm rate, efficiency and ROC.

Every function here is pure: it takes arrays of scores and returns numbers, with
no model, dataset or random state. One set of conventions holds throughout:

- **Scores are logits.** A score is a real number, larger meaning more
  signal-like. Use the log-odds a classifier computes before its softmax or
  sigmoid, never the probability: in float32 a softmax saturates at exactly 1.0,
  saturated scores tie, and no threshold can split a tie. `logit_scores` turns
  model outputs into scores.
- **Detected means ``score >= threshold``**, everywhere.
- **FAR is a rate in Hz**: background triggers per second of analysed livetime.
  Above-threshold windows that follow one another within ``cluster_seconds`` are
  one trigger.
- **Intervals are two-sided and central**, at ``confidence`` (0.9 by default):
  Garwood for Poisson counts, Clopper-Pearson (or Wilson) for binomial
  efficiencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Hashable, Optional, Sequence, Tuple

import numpy as np
from scipy import special, stats

DEFAULT_CONFIDENCE = 0.9


# --------------------------------------------------------------------- scores

def logit_scores(outputs) -> np.ndarray:
    """
    Detection scores from a classifier's logit outputs, as float64.

    - ``(B, 2)`` logits of a two-class softmax give ``z1 - z0``, the log-odds of
      class 1 (signal);
    - ``(B, 1)`` or ``(B,)`` logits of a sigmoid are already log-odds.

    A model with one named output gives a dict of one entry: its value is read.

    Raises:
        ValueError: on any other shape or number of outputs, or on a non-finite score.
    """
    if isinstance(outputs, dict):
        if len(outputs) != 1:
            raise ValueError(
                f"Cannot read detection scores from {len(outputs)} named outputs "
                f"{sorted(outputs)}: expected one."
            )
        (outputs,) = outputs.values()
    o = np.asarray(outputs, dtype=np.float64)
    if o.ndim == 2 and o.shape[1] == 2:
        scores = o[:, 1] - o[:, 0]
    elif o.ndim == 2 and o.shape[1] == 1:
        scores = o[:, 0]
    elif o.ndim == 1:
        scores = o
    else:
        raise ValueError(
            f"Cannot read detection scores from outputs of shape {o.shape}: expected "
            "(B, 2) two-class logits or (B,)/(B, 1) single logits."
        )
    return _finite_scores(scores, "model outputs")


def _finite_scores(scores, what: str) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(s)):
        raise ValueError(f"{what} contain {np.count_nonzero(~np.isfinite(s))} non-finite scores.")
    return s


def _count_at_or_above(sorted_ascending: np.ndarray, thresholds) -> np.ndarray:
    """Number of values >= each threshold, for values sorted ascending."""
    return sorted_ascending.size - np.searchsorted(sorted_ascending, thresholds, side="left")


# ----------------------------------------------------------------- intervals

def poisson_interval(count, confidence: float = DEFAULT_CONFIDENCE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Garwood's exact central interval on the mean of a Poisson count.

    Returns:
        ``(lower, upper)`` bounds on the expected count; divide by livetime for a
        rate. The lower bound is 0 for a count of 0.
    """
    k = np.asarray(count, dtype=np.float64)
    alpha = 1.0 - confidence
    lower = np.where(k > 0, special.gammaincinv(np.maximum(k, 1.0), alpha / 2), 0.0)
    upper = special.gammaincinv(k + 1.0, 1.0 - alpha / 2)
    return lower, upper


def binomial_interval(
    detected,
    total,
    confidence: float = DEFAULT_CONFIDENCE,
    method: str = "clopper-pearson",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-sided interval on a binomial proportion ``detected / total``.

    Args:
        method: ``"clopper-pearson"`` (exact, conservative) or ``"wilson"``.

    Returns:
        ``(lower, upper)``; NaN where ``total`` is 0.
    """
    k = np.asarray(detected, dtype=np.float64)
    n = np.asarray(total, dtype=np.float64)
    alpha = 1.0 - confidence
    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "clopper-pearson":
            lower = np.where(k > 0, stats.beta.ppf(alpha / 2, k, n - k + 1), 0.0)
            upper = np.where(k < n, stats.beta.ppf(1 - alpha / 2, k + 1, n - k), 1.0)
        elif method == "wilson":
            z = stats.norm.ppf(1 - alpha / 2)
            p = k / n
            denominator = 1 + z**2 / n
            centre = (p + z**2 / (2 * n)) / denominator
            half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
            lower, upper = centre - half_width, centre + half_width
        else:
            raise ValueError(f"Unknown binomial interval method {method!r}.")
    empty = n <= 0
    return np.where(empty, np.nan, lower), np.where(empty, np.nan, upper)


# ------------------------------------------------------------ false alarms

@dataclass(frozen=True)
class ScoreStream:
    """
    Scores of windows tiled across one stretch of contiguous data, in time order.

    Window ``i`` starts at ``i * stride_seconds`` and lasts ``window_seconds``;
    the stream's livetime is the span it covers. ``block`` labels the stretch of
    data it came from, so that an estimator can treat correlated blocks
    together (`Background.by_block`).
    """
    scores: np.ndarray
    stride_seconds: float
    window_seconds: float
    block: Optional[Hashable] = None

    def __post_init__(self):
        object.__setattr__(self, "scores", _finite_scores(self.scores, "stream scores"))
        if not 0 < self.stride_seconds <= self.window_seconds:
            raise ValueError(
                "A stream tiles its data without gaps: need 0 < stride_seconds <= window_seconds, "
                f"got stride {self.stride_seconds} and window {self.window_seconds}."
            )

    @property
    def livetime_seconds(self) -> float:
        n = self.scores.size
        return 0.0 if n == 0 else (n - 1) * self.stride_seconds + self.window_seconds


@dataclass(frozen=True)
class FarEstimate:
    """Background triggers at or above a threshold, as a rate with its interval."""
    threshold: float
    count: int
    livetime_seconds: float
    rate_hz: float
    lower_hz: float
    upper_hz: float
    confidence: float


@dataclass(frozen=True)
class FarThreshold:
    """
    The detection threshold for a target FAR.

    ``resolved`` is False when the background is too short to resolve the
    target (it allows less than one trigger in the livetime); the threshold is
    then NaN, and so is any efficiency measured at it.
    """
    target_hz: float
    threshold: float
    resolved: bool
    achieved: FarEstimate


@dataclass(frozen=True)
class FarCurve:
    """FAR as a step function of the threshold, exact at every breakpoint."""
    thresholds: np.ndarray   # descending; the FAR holds on (next threshold, threshold]
    counts: np.ndarray
    livetime_seconds: float
    rate_hz: np.ndarray
    lower_hz: np.ndarray
    upper_hz: np.ndarray
    confidence: float


class Background:
    """
    Noise-only scores, as false-alarm rates.

    Triggers: at a threshold ``t``, the windows scoring ``>= t`` are grouped in
    time; a window that starts within ``cluster_seconds`` of the previous
    above-threshold window in its stream joins that window's trigger, so a long
    excursion is one trigger, however long. Window ``i`` therefore opens a new
    trigger exactly when ``m_i < t <= s_i``, where ``s_i`` is its score and
    ``m_i`` the largest score among the windows that precede it within
    ``cluster_seconds``. Counting those intervals gives the trigger count at
    every threshold exactly, with no grid.

    Clustering makes the raw count non-monotone (a lower threshold can merge two
    triggers into one), so the FAR at ``t`` is the most triggers seen at any
    threshold ``>= t``. That envelope is monotone, and `threshold_for_far` is its
    exact inverse: an event's FAR is at most ``f`` exactly when its score is at
    least ``threshold_for_far(f).threshold``.
    """

    def __init__(self, streams: Sequence[ScoreStream], cluster_seconds: float = 0.0):
        if cluster_seconds < 0:
            raise ValueError(f"cluster_seconds must be >= 0, got {cluster_seconds}.")
        self.streams: Tuple[ScoreStream, ...] = tuple(streams)
        self.cluster_seconds = float(cluster_seconds)
        self._independent_block = None
        opens, merges = [], []
        for stream in self.streams:
            s = stream.scores
            # Windows i - k .. i - 1 start within cluster_seconds of window i.
            k = int(math.floor(self.cluster_seconds / stream.stride_seconds + 1e-9))
            k = min(k, max(s.size - 1, 0))
            if k > 0 and s.size:
                padded = np.concatenate([np.full(k, -np.inf), s])
                preceding_max = np.lib.stride_tricks.sliding_window_view(padded, k)[:-1].max(axis=1)
                merge = np.minimum(s, preceding_max)
                merges.append(merge[np.isfinite(merge)])
            opens.append(s)
        self._init_counts(
            np.concatenate(opens) if opens else np.empty(0),
            np.concatenate(merges) if merges else np.empty(0),
            sum(stream.livetime_seconds for stream in self.streams),
        )

    @classmethod
    def independent_windows(
        cls, scores, window_seconds: float, block: Optional[Hashable] = None
    ) -> "Background":
        """
        Background of windows drawn independently (no two adjacent in time).

        Each window is its own trigger when it passes, and the livetime is the
        number of windows times their duration.
        """
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds}.")
        scores = _finite_scores(scores, "background scores")
        background = cls.__new__(cls)
        background.streams = ()
        background.cluster_seconds = 0.0
        background._independent_block = block
        background._init_counts(scores, np.empty(0), scores.size * float(window_seconds))
        return background

    def _init_counts(self, opens: np.ndarray, merges: np.ndarray, livetime: float) -> None:
        self.livetime_seconds = float(livetime)
        self.num_windows = int(opens.size)
        self._opens = np.sort(opens)
        self._merges = np.sort(merges)
        # Breakpoints of the count, descending, and the running-maximum envelope.
        self._breakpoints = np.unique(np.concatenate([self._opens, self._merges]))[::-1]
        raw = (_count_at_or_above(self._opens, self._breakpoints)
               - _count_at_or_above(self._merges, self._breakpoints))
        self._envelope = np.maximum.accumulate(raw) if raw.size else raw

    # ---- counts

    def trigger_count(self, threshold) -> np.ndarray:
        """The literal number of triggers at ``threshold`` (not monotone under clustering)."""
        return (_count_at_or_above(self._opens, threshold)
                - _count_at_or_above(self._merges, threshold))

    def far_count(self, threshold) -> np.ndarray:
        """The most triggers at any threshold ``>= threshold``: the count a FAR is built on."""
        t = np.asarray(threshold, dtype=np.float64)
        # Breakpoints >= t are a prefix of the descending array.
        num_at_or_above = np.searchsorted(-self._breakpoints, -t, side="right")
        envelope = np.concatenate([[0], self._envelope]).astype(np.int64)
        return envelope[num_at_or_above]

    def _estimate(self, threshold: float, count: int, confidence: float) -> FarEstimate:
        lower, upper = poisson_interval(count, confidence)
        T = self.livetime_seconds
        return FarEstimate(
            threshold=float(threshold), count=int(count), livetime_seconds=T,
            rate_hz=count / T, lower_hz=float(lower) / T, upper_hz=float(upper) / T,
            confidence=confidence,
        )

    def windows_at_or_above(self, threshold) -> np.ndarray:
        """The number of the background's windows scoring ``>= threshold``, each counted on its own."""
        return _count_at_or_above(self._opens, threshold)

    def window_exceedance(self, threshold) -> np.ndarray:
        """
        The chance that a window of noise scores ``>= threshold``: the fraction of the
        background's windows that do, each counted on its own, clustered or not.
        """
        if self.num_windows == 0:
            raise ValueError("The background has no windows.")
        return self.windows_at_or_above(threshold) / self.num_windows

    def far(self, threshold: float, confidence: float = DEFAULT_CONFIDENCE) -> FarEstimate:
        """FAR (Hz) of detections at ``score >= threshold``, with its Poisson interval."""
        self._require_livetime()
        return self._estimate(threshold, int(self.far_count(threshold)), confidence)

    def threshold_for_far(self, far_hz: float, confidence: float = DEFAULT_CONFIDENCE) -> FarThreshold:
        """
        The lowest threshold whose FAR does not exceed ``far_hz``.

        Unresolved (threshold NaN) when ``far_hz`` allows less than one trigger
        in the livetime. A target above anything the background produces gives
        ``-inf``: every candidate passes.
        """
        self._require_livetime()
        allowed = math.floor(far_hz * self.livetime_seconds * (1 + 1e-12))
        if allowed < 1:
            nan = float("nan")
            unresolved = FarEstimate(nan, 0, self.livetime_seconds, nan, nan, nan, confidence)
            return FarThreshold(float(far_hz), nan, False, unresolved)
        threshold = self._threshold_allowing(allowed)
        return FarThreshold(float(far_hz), threshold, True, self.far(threshold, confidence))

    def _threshold_allowing(self, count: int) -> float:
        """The lowest threshold whose FAR count is at most ``count``: +inf when none is, -inf when all are."""
        if count < 0:
            return np.inf
        # The envelope rises as the threshold falls; keep the breakpoints it allows.
        num_allowed = int(np.searchsorted(self._envelope, count, side="right"))
        if num_allowed == self._breakpoints.size:
            return -np.inf
        # Just above the first breakpoint that would exceed the allowance.
        return float(np.nextafter(self._breakpoints[num_allowed], np.inf))

    def far_band_thresholds(self, far_hz: float, confidence: float = DEFAULT_CONFIDENCE) -> Tuple[float, float]:
        """
        The thresholds a true FAR of ``far_hz`` may lie between, given the background's
        finite livetime: ``(high, low)``, the lowest thresholds at which the upper and the
        lower end of the FAR's Poisson interval fall to ``far_hz``. The threshold that
        truly gives ``far_hz`` is between them wherever the interval covers the true FAR.
        ``high`` is +inf when the upper end is above ``far_hz`` even with no trigger.
        """
        self._require_livetime()
        counts = np.arange(int(self._envelope[-1] if self._envelope.size else 0) + 2)
        lower, upper = poisson_interval(counts, confidence)
        limit = far_hz * self.livetime_seconds * (1 + 1e-12)
        # Both ends rise with the count: the largest count each allows.
        high = self._threshold_allowing(int(np.searchsorted(upper, limit, side="right")) - 1)
        low = self._threshold_allowing(int(np.searchsorted(lower, limit, side="right")) - 1)
        return high, low

    def far_curve(self, confidence: float = DEFAULT_CONFIDENCE) -> FarCurve:
        """The FAR at every breakpoint, with Poisson intervals."""
        self._require_livetime()
        lower, upper = poisson_interval(self._envelope, confidence)
        T = self.livetime_seconds
        return FarCurve(
            thresholds=self._breakpoints.copy(), counts=self._envelope.copy(), livetime_seconds=T,
            rate_hz=self._envelope / T, lower_hz=lower / T, upper_hz=upper / T,
            confidence=confidence,
        )

    def by_block(self) -> Dict[Hashable, "Background"]:
        """
        One Background per ``block`` label: the hook for an uncertainty that
        treats correlated blocks of data together, where Poisson's independent
        triggers do not hold.
        """
        if not self.streams:
            return {self._independent_block: self}
        return {block: Background(streams, self.cluster_seconds) for block, streams in self._blocks().items()}

    def _blocks(self) -> Dict[Hashable, list]:
        """The streams of each ``block`` label, in order."""
        groups: Dict[Hashable, list] = {}
        for stream in self.streams:
            groups.setdefault(stream.block, []).append(stream)
        return groups

    def bootstrap(self, rng: np.random.Generator) -> "Background":
        """
        A bootstrap replicate of this background: its blocks (the ``block`` labels of
        `by_block`) drawn with replacement, as many as it has, each with all its streams;
        or, for independent windows, the windows themselves, each its own block.
        """
        if not self.streams:
            n = self.num_windows
            window_seconds = self.livetime_seconds / n
            return Background.independent_windows(self._opens[rng.integers(0, n, n)], window_seconds,
                                                  self._independent_block)
        blocks = list(self._blocks().values())
        drawn = rng.integers(0, len(blocks), len(blocks))
        return Background([stream for b in drawn for stream in blocks[b]], self.cluster_seconds)

    def _require_livetime(self) -> None:
        if self.livetime_seconds <= 0:
            raise ValueError("The background has no livetime: it holds no scored windows.")


# ---------------------------------------------------------------- efficiency

@dataclass(frozen=True)
class EfficiencyCurve:
    """Fraction of injections detected per SNR bin, with binomial intervals."""
    threshold: float
    bin_edges: np.ndarray
    snr_mean: np.ndarray
    detected: np.ndarray
    total: np.ndarray
    efficiency: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    confidence: float
    method: str


def efficiency_curve(
    snrs,
    scores,
    threshold: float,
    bin_edges: Sequence[float],
    confidence: float = DEFAULT_CONFIDENCE,
    method: str = "clopper-pearson",
) -> EfficiencyCurve:
    """
    Efficiency of injections at ``score >= threshold`` in SNR bins.

    Bins are ``[lo, hi)``, the last one closed; injections outside the edges
    are not counted. A NaN threshold (an unresolved FAR) gives NaN efficiencies.
    """
    snrs = np.asarray(snrs, dtype=np.float64).reshape(-1)
    scores = _finite_scores(scores, "injection scores")
    if snrs.shape != scores.shape:
        raise ValueError(f"{snrs.size} SNRs for {scores.size} scores.")
    edges = np.asarray(bin_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be strictly increasing, with at least two edges.")
    num_bins = edges.size - 1
    index = np.searchsorted(edges, snrs, side="right") - 1
    index[snrs == edges[-1]] = num_bins - 1
    inside = (index >= 0) & (index < num_bins)
    total = np.bincount(index[inside], minlength=num_bins)
    snr_sum = np.bincount(index[inside], weights=snrs[inside], minlength=num_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr_mean = np.where(total > 0, snr_sum / total, np.nan)
    if np.isnan(threshold):
        nan = np.full(num_bins, np.nan)
        return EfficiencyCurve(float(threshold), edges, snr_mean, nan, total, nan, nan, nan,
                               confidence, method)
    passed = inside & (scores >= threshold)
    detected = np.bincount(index[passed], minlength=num_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        efficiency = np.where(total > 0, detected / total, np.nan)
    lower, upper = binomial_interval(detected, total, confidence, method)
    return EfficiencyCurve(float(threshold), edges, snr_mean, detected, total, efficiency,
                           lower, upper, confidence, method)


# ----------------------------------------------------------------------- ROC

@dataclass(frozen=True)
class RocCurve:
    """The exact ROC: one point per distinct score, from (0, 0) to (1, 1)."""
    false_positive_rate: np.ndarray
    true_positive_rate: np.ndarray
    thresholds: np.ndarray   # descending; the first is +inf
    auc: float


def roc_curve(negative_scores, positive_scores) -> RocCurve:
    """
    The exact ROC of positives against negatives, at ``score >= threshold``.

    The AUC is the trapezoid area under the exact points. It equals the
    Mann-Whitney probability that a positive outscores a negative, with ties
    counted one half.
    """
    neg = np.sort(_finite_scores(negative_scores, "negative scores"))
    pos = np.sort(_finite_scores(positive_scores, "positive scores"))
    if neg.size == 0 or pos.size == 0:
        raise ValueError("An ROC needs at least one negative and one positive score.")
    thresholds = np.concatenate([[np.inf], np.unique(np.concatenate([neg, pos]))[::-1]])
    fpr = _count_at_or_above(neg, thresholds) / neg.size
    tpr = _count_at_or_above(pos, thresholds) / pos.size
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1])) / 2)
    return RocCurve(fpr, tpr, thresholds, auc)
