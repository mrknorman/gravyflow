"""
Detection metrics over score arrays: false-alarm rate, efficiency and ROC.

Every function here is pure: it takes arrays of scores and returns numbers, with
no model or dataset (a bootstrap takes the random generator it draws from). One
set of conventions holds throughout:

- **Scores are logits.** A score is a real number, larger meaning more
  signal-like. Use the log-odds a classifier computes before its softmax or
  sigmoid, never the probability: in float32 a softmax saturates at exactly 1.0,
  saturated scores tie, and no threshold can split a tie. `logit_scores` turns
  model outputs into scores.
- **Detected means ``score >= threshold``**, everywhere.
- **FAR is a rate in Hz**: background triggers per second of analysed livetime.
  Above-threshold windows that follow one another within ``cluster_seconds`` are
  one trigger.
- **Intervals are two-sided and central**, reported at 1 and 2 sigma
  (`DEFAULT_SIGMAS`: 68.27% and 95.45%, `sigma_level`): Garwood for Poisson
  counts, in the background's independent information (`Background.independent_count`),
  and Clopper-Pearson (or Wilson) for binomial efficiencies. A building block
  takes one ``confidence``; a result carries every level.
- **Found, and found by chance.** In window mode an injection is found when its
  window's score passes the threshold, whatever passes it: noise alone passes
  with the chance of a noise window passing (`Background.found_by_chance`, a
  `ChanceLaw`), so the fraction found by the signal is ``(f - c) / (1 - c)``
  (`subtract_chance`). It is subtracted at every threshold; at the tight
  thresholds c is negligible, at loose ones it is not. In stream mode
  (`stream`) an injection's score is the loudest trigger's within a fixed window
  of its merger (``-inf`` where no trigger lies there, so nothing finds it), and
  the chance is that a noise trigger passing the threshold lies in that window,
  about the false-alarm rate times the window (`stream.StreamBackground`): the
  estimators take the found test (a threshold on scores) and the background's
  `ChanceLaw`, the two things stream mode replaces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Hashable, Optional, Sequence, Tuple

import numpy as np
from scipy import special, stats

#: The interval levels every result reports, in sigma of a normal law: 1 sigma (68.27%) and 2 sigma (95.45%).
DEFAULT_SIGMAS = (1.0, 2.0)


def sigma_level(sigma: float) -> float:
    """The central probability within ``sigma`` standard deviations of a normal law: 0.6827 at 1, 0.9545 at 2."""
    return float(special.erf(sigma / math.sqrt(2.0)))


#: A building block's level when none is given: 1 sigma.
DEFAULT_CONFIDENCE = sigma_level(1.0)


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


def injection_scores(scores, what: str = "injection scores") -> np.ndarray:
    """
    Injection scores as float64: finite, or ``-inf`` for an injection nothing can
    find (stream mode: no trigger within its found window). NaN and ``+inf`` refuse.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    bad = np.isnan(s) | (s == np.inf)
    if np.any(bad):
        raise ValueError(f"{what} contain {np.count_nonzero(bad)} NaN or +inf scores.")
    return s


def _count_at_or_above(sorted_ascending: np.ndarray, thresholds) -> np.ndarray:
    """Number of values >= each threshold, for values sorted ascending."""
    return sorted_ascending.size - np.searchsorted(sorted_ascending, thresholds, side="left")


def chosen_count(sorted_ascending: np.ndarray, threshold) -> np.ndarray:
    """
    The count of values (sorted ascending, at least one) a threshold chosen from them
    stands on (`Background.chosen_window_chance`): those at or above it when it is one
    of them; else those at or above the highest value below it (a threshold between
    values is taken as just above the one below); all of them when none is below.
    """
    n = sorted_ascending.size
    t = np.asarray(threshold, dtype=np.float64)
    below = np.searchsorted(sorted_ascending, t, side="left")
    at_a_value = (below < n) & (sorted_ascending[np.minimum(below, n - 1)] == t)
    highest_below = sorted_ascending[np.maximum(below - 1, 0)]
    return np.where(at_a_value, n - below,
                    np.where(below > 0, _count_at_or_above(sorted_ascending, highest_below), n))


# ----------------------------------------------------------------- intervals

def poisson_interval(count, confidence: float = DEFAULT_CONFIDENCE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Garwood's exact central interval on the mean of a Poisson count (a count need
    not be whole: a time-slid background's is in its independent events).

    Returns:
        ``(lower, upper)`` bounds on the expected count; divide by livetime for a
        rate. The lower bound is 0 for a count of 0.
    """
    k = np.asarray(count, dtype=np.float64)
    alpha = 1.0 - confidence
    lower = np.where(k > 0, special.gammaincinv(np.where(k > 0, k, 1.0), alpha / 2), 0.0)
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
    together (`Background.by_block`). ``slide`` is the time slide the stream's
    windows were read at (0: unslid), so that the background knows how often it
    reuses each detector's data (`Background.data_reuse`).

    ``detector_offsets`` places each window's data on each detector's own timeline
    (s): detector j's data of window ``i`` start at ``detector_offsets[j] + i *
    stride_seconds``, taken round a ring of ``ring_seconds`` when one is given
    (wrapping slides). With them a stream background counts its independent events
    by the data they share (`disjoint_events`); without, by its slides (`data_reuse`).
    """
    scores: np.ndarray
    stride_seconds: float
    window_seconds: float
    block: Optional[Hashable] = None
    slide: int = 0
    detector_offsets: Optional[Tuple[float, ...]] = None
    ring_seconds: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "scores", _finite_scores(self.scores, "stream scores"))
        if not 0 < self.stride_seconds <= self.window_seconds:
            raise ValueError(
                "A stream tiles its data without gaps: need 0 < stride_seconds <= window_seconds, "
                f"got stride {self.stride_seconds} and window {self.window_seconds}."
            )
        if self.detector_offsets is not None:
            object.__setattr__(self, "detector_offsets", tuple(float(o) for o in self.detector_offsets))
        if self.ring_seconds is not None and not self.ring_seconds > 0:
            raise ValueError(f"ring_seconds must be > 0, got {self.ring_seconds}.")

    def data_positions(self, index) -> np.ndarray:
        """Where windows ``index``'s data start on each detector's timeline (s): (len(index), detectors)."""
        if self.detector_offsets is None:
            raise ValueError("The stream records no detector offsets.")
        times = np.asarray(index, dtype=np.float64)[:, None] * self.stride_seconds + np.asarray(self.detector_offsets)
        return np.mod(times, self.ring_seconds) if self.ring_seconds is not None else times

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


#: A rate, volume or efficiency is quoted only
#: where at least this many independent background events lie at or above its threshold
#: (`Background.independent_count`). Below it the point is still computed, plotted and marked, never quoted.
MIN_BACKGROUND_EVENTS = 10


def disjoint_events(scores, positions, footprint_seconds: float, ring_seconds: Optional[float] = None) -> np.ndarray:
    """
    The background triggers that share no data, as their scores (ascending).

    Triggers are taken loudest first (ties: in the order given), and one is accepted
    only if, in every detector, its data lie at least ``footprint_seconds`` (the data
    that set a score: a window's padded onsource; its offsource only feeds the
    spectrum) from every accepted trigger's data in that detector. ``positions``
    (triggers, detectors) is where each trigger's window starts on each detector's
    timeline (s), taken round a ring of ``ring_seconds`` if given: every window's
    onsource lies at the same offset from its start, so their distance is the same.

    Since acceptance runs loudest first, the independent events at a threshold are
    the accepted triggers at or above it. Accepted triggers share no data, so this is
    a lower bound on the background's independent events: one detector's loud
    excursion, met again in every time slide, counts once.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positions = np.asarray(positions, dtype=np.float64).reshape(scores.size, -1)
    footprint = float(footprint_seconds)
    if not footprint > 0:
        raise ValueError(f"footprint_seconds must be > 0, got {footprint_seconds}.")
    ring = None if ring_seconds is None else float(ring_seconds)
    if ring is not None:
        positions = np.mod(positions, ring)
        num_bins = max(int(math.ceil(ring / footprint)), 1)
    bins = np.floor(positions / footprint).astype(np.int64)
    if ring is not None:
        bins = bins % num_bins
    # Accepted triggers in one detector lie at least a footprint apart: at most one per bin.
    taken = [dict() for _ in range(positions.shape[1])]
    accepted = []
    for k in np.argsort(-scores, kind="stable"):
        clear = True
        for j, held in enumerate(taken):
            p, b = positions[k, j], int(bins[k, j])
            for d in (-2, -1, 0, 1, 2):  # a ring's last bin is short: look two bins either side
                q = held.get((b + d) % num_bins if ring is not None else b + d)
                if q is None:
                    continue
                gap = abs(p - q)
                if ring is not None:
                    gap = min(gap, ring - gap)
                if gap < footprint:
                    clear = False
                    break
            if not clear:
                break
        if clear:
            for j, held in enumerate(taken):
                held[int(bins[k, j])] = positions[k, j]
            accepted.append(scores[k])
    return np.sort(np.asarray(accepted, dtype=np.float64))


@dataclass(frozen=True)
class FarThreshold:
    """
    The detection threshold for a target FAR.

    ``resolved`` is False when the background is too short to resolve the
    target (it allows less than one trigger in the livetime); the threshold is
    then NaN, and so is any efficiency measured at it. ``independent_events`` is the
    background's independent events at or above the threshold (`Background.independent_count`);
    ``quotable`` holds where they are at least `MIN_BACKGROUND_EVENTS`.
    """
    target_hz: float
    threshold: float
    resolved: bool
    achieved: FarEstimate
    independent_events: int = 0

    @property
    def quotable(self) -> bool:
        """Resolved, with at least `MIN_BACKGROUND_EVENTS` independent background events at or above the threshold."""
        return bool(self.resolved and self.independent_events >= MIN_BACKGROUND_EVENTS)


@dataclass(frozen=True)
class FarCurve:
    """
    FAR as a step function of the threshold, exact at every breakpoint. ``quotable``
    marks the breakpoints with at least `MIN_BACKGROUND_EVENTS` independent events
    (``independent``, `Background.independent_count`).
    """
    thresholds: np.ndarray   # descending; the FAR holds on (next threshold, threshold]
    counts: np.ndarray
    livetime_seconds: float
    rate_hz: np.ndarray
    lower_hz: np.ndarray
    upper_hz: np.ndarray
    confidence: float
    data_reuse: int = 1
    independent: Optional[np.ndarray] = None

    @property
    def independent_events(self) -> np.ndarray:
        """The independent background events at each breakpoint."""
        if self.independent is not None:
            return np.asarray(self.independent, dtype=np.int64)
        return np.floor(np.asarray(self.counts, dtype=np.float64) / max(int(self.data_reuse), 1)).astype(np.int64)

    @property
    def quotable(self) -> np.ndarray:
        return self.independent_events >= MIN_BACKGROUND_EVENTS


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
    threshold ``>= t``. That envelope is monotone, and `threshold_for_far` inverts
    it: the threshold for a target allowing ``k`` triggers is the ``k``-th loudest
    trigger's score, where the count first reaches ``k``.
    """

    def __init__(self, streams: Sequence[ScoreStream], cluster_seconds: float = 0.0):
        if cluster_seconds < 0:
            raise ValueError(f"cluster_seconds must be >= 0, got {cluster_seconds}.")
        self.streams: Tuple[ScoreStream, ...] = tuple(streams)
        self.cluster_seconds = float(cluster_seconds)
        self._independent_block = None
        self._disjoint = None
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
        background._disjoint = None
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

    def count_interval(self, count, confidence: float = DEFAULT_CONFIDENCE) -> Tuple[np.ndarray, np.ndarray]:
        """
        Garwood's interval on the mean of the background's trigger ``count``, counted in
        its independent information: a time-slid background reads each detector's data
        in many slides, and a loud excursion in one detector's data opens a trigger in
        each, so its count is ``r`` times a Poisson count of independent events, ``r
        Poisson(c / r)``, as `bootstrap_thresholds` draws it, ``r`` the count over its
        independent events (`reuse_at`). The interval is Garwood's on ``c / r``, times
        ``r``: on a background whose triggers share no data (``r = 1``) Garwood's own.
        ``r`` bounds the count's overdispersion, so the interval is exact where one
        detector's excursions make the triggers (a statistic like the detectors' maximum)
        and conservative where both must (their sum: there the variance is nearer
        Poisson's).
        """
        count = np.asarray(count, dtype=np.float64)
        reuse = self.reuse_at(count)
        lower, upper = poisson_interval(count / reuse, confidence)
        return lower * reuse, upper * reuse

    def _estimate(self, threshold: float, count: int, confidence: float) -> FarEstimate:
        lower, upper = self.count_interval(count, confidence)
        T = self.livetime_seconds
        return FarEstimate(
            threshold=float(threshold), count=int(count), livetime_seconds=T,
            rate_hz=count / T, lower_hz=float(lower) / T, upper_hz=float(upper) / T,
            confidence=confidence,
        )

    @property
    def window_scores(self) -> np.ndarray:
        """Every window's score, each counted on its own, ascending."""
        return self._opens

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

    def chosen_window_chance(self, threshold) -> Tuple[np.ndarray, np.ndarray]:
        """
        The law of the chance that a fresh noise window passes a threshold chosen
        from these windows, at one of them or just above one (as `threshold_for_far`
        chooses): ``Beta(a, b)``, returned as ``(a, b)``.

        Of ``n`` exchangeable windows, a threshold at the ``c``-th highest is passed
        by a fresh window with a chance distributed as ``Beta(c, n - c + 1)``,
        whatever the windows' distribution: on average ``c / (n + 1)``. A threshold
        between windows is taken as just above the highest window below it, ``c``
        counting the windows at or above that one (on average ``c / (n + 1)``, not the
        ``(c - 1) / n`` of `window_exceedance`). A background that reuses its data
        (`reuse_at`, ``r``) holds the information of ``n / r`` windows, so both
        counts are taken in ``r``'s: ``Beta(c / r, (n - c) / r + 1)``. Where no window
        is below the threshold, ``c = n``.
        """
        if self.num_windows == 0:
            raise ValueError("The background has no windows.")
        n = self.num_windows
        count = chosen_count(self._opens, threshold)
        reuse = self.reuse_at(count)
        return count / reuse, (n - count) / reuse + 1.0

    def found_by_chance(self, threshold) -> "ChanceLaw":
        """
        The chance that noise alone passes each threshold, in window mode: a fresh
        noise window's chance of passing (`chosen_window_chance`'s Beta law), nothing
        past every window (an infinite threshold), NaN at NaN (an unresolved target).
        """
        t = np.atleast_1d(np.asarray(threshold, dtype=np.float64))
        a, b = self.chosen_window_chance(np.where(np.isfinite(t), t, 0.0))
        a = np.where(t == np.inf, 0.0, np.where(t == -np.inf, 1.0, a))
        b = np.where(t == -np.inf, 0.0, b)
        return ChanceLaw(np.where(np.isnan(t), np.nan, a), np.where(np.isnan(t), np.nan, b))

    def far(self, threshold: float, confidence: float = DEFAULT_CONFIDENCE) -> FarEstimate:
        """FAR (Hz) of detections at ``score >= threshold``, with its interval (`count_interval`)."""
        self._require_livetime()
        return self._estimate(threshold, int(self.far_count(threshold)), confidence)

    def threshold_for_far(self, far_hz: float, confidence: float = DEFAULT_CONFIDENCE) -> FarThreshold:
        """
        The threshold for ``far_hz``: the score of the ``k``-th loudest trigger, ``k``
        the triggers the target allows in the livetime (`_threshold_at`).

        Of the thresholds whose count is ``k`` (from just above the ``(k+1)``-th
        loudest to the ``k``-th), the ``k``-th's true FAR is ``k`` triggers on average,
        the target's: just above the ``(k+1)``-th, it would be ``k + 1``, a bias of
        ``(k + 1) / k``. Unresolved (threshold NaN) when ``far_hz`` allows less than
        one trigger in the livetime. A target above anything the background produces
        gives ``-inf``: every candidate passes. Quotable only with at least
        `MIN_BACKGROUND_EVENTS` independent events at or above it (`FarThreshold`).
        """
        self._require_livetime()
        allowed = math.floor(far_hz * self.livetime_seconds * (1 + 1e-12))
        if allowed < 1:
            nan = float("nan")
            unresolved = FarEstimate(nan, 0, self.livetime_seconds, nan, nan, nan, confidence)
            return FarThreshold(float(far_hz), nan, False, unresolved)
        threshold = self._threshold_at(allowed)
        achieved = self.far(threshold, confidence)
        return FarThreshold(float(far_hz), threshold, True, achieved, int(self.independent_count(achieved.count)))

    def _threshold_at(self, count: int) -> float:
        """
        The estimator's threshold for ``count`` triggers allowed: the score at which the FAR
        count first reaches ``count`` (the ``count``-th loudest trigger), or, where ties jump
        the count past it there, just above that score (fewer). Just above the loudest for
        0; +inf below 0; -inf when the background never reaches ``count``.
        """
        if count < 0:
            return np.inf
        # The envelope rises as the threshold falls: the first breakpoint at which it reaches the count.
        first = int(np.searchsorted(self._envelope, count - 1, side="right"))
        if first == self._breakpoints.size:
            return -np.inf
        if self._envelope[first] == count:
            return float(self._breakpoints[first])
        return float(np.nextafter(self._breakpoints[first], np.inf))

    def _threshold_allowing(self, count: int) -> float:
        """
        The lowest threshold whose FAR count is at most ``count`` (just above the ``(count +
        1)``-th loudest trigger): +inf when none is, -inf when all are. The FAR band's ends
        (`far_band_thresholds`): where the true FAR's threshold is at or above it exactly
        when the true count there is at most ``count``.
        """
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
        lower end of the FAR's interval (`count_interval`) fall to ``far_hz``. The threshold that
        truly gives ``far_hz`` is between them wherever the interval covers the true FAR.
        ``high`` is +inf when the upper end is above ``far_hz`` even with no trigger.
        """
        self._require_livetime()
        counts = np.arange(int(self._envelope[-1] if self._envelope.size else 0) + 2)
        lower, upper = self.count_interval(counts, confidence)
        limit = far_hz * self.livetime_seconds * (1 + 1e-12)
        # Both ends rise with the count: the largest count each allows.
        high = self._threshold_allowing(int(np.searchsorted(upper, limit, side="right")) - 1)
        low = self._threshold_allowing(int(np.searchsorted(lower, limit, side="right")) - 1)
        return high, low

    def far_curve(self, confidence: float = DEFAULT_CONFIDENCE) -> FarCurve:
        """The FAR at every breakpoint, with its interval (`count_interval`)."""
        self._require_livetime()
        lower, upper = self.count_interval(self._envelope, confidence)
        T = self.livetime_seconds
        return FarCurve(
            thresholds=self._breakpoints.copy(), counts=self._envelope.copy(), livetime_seconds=T,
            rate_hz=self._envelope / T, lower_hz=lower / T, upper_hz=upper / T,
            confidence=confidence, data_reuse=self.data_reuse,
            independent=self.independent_count(self._envelope),
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

    @property
    def data_reuse(self) -> int:
        """
        How many of the background's windows each detector's window of data enters:
        the number of time slides its streams were read at (each slide reads the
        same data again), 1 for unslid streams or independent windows. A stream
        background whose streams place their data (`ScoreStream.detector_offsets`)
        counts its independent events by the data its triggers share instead
        (`independent_count`).
        """
        return max(len({stream.slide for stream in self.streams}), 1)

    def _count_disjoint(self, triggers, positions, footprint_seconds: float, ring_seconds) -> None:
        """Count the independent events by the data triggers share (`disjoint_events`), at each breakpoint."""
        accepted = disjoint_events(triggers, positions, footprint_seconds, ring_seconds)
        self._disjoint = _count_at_or_above(accepted, self._breakpoints)

    def independent_count(self, count) -> np.ndarray:
        """
        The independent background events behind a FAR ``count`` (`far_count`): where the
        streams place their data, the triggers that share none (`disjoint_events`) at the
        threshold where the count is reached (past every trigger: all of them); else the
        count over the slides, ``floor(count / data_reuse)``.
        """
        c = np.asarray(count, dtype=np.float64)
        if self._disjoint is None:
            return np.floor(c / self.data_reuse).astype(np.int64)
        if not self._disjoint.size:
            return np.zeros(c.shape, dtype=np.int64)
        # The first breakpoint (thresholds falling) at which the envelope reaches the count.
        at = np.minimum(np.searchsorted(self._envelope, c, side="left"), self._envelope.size - 1)
        return np.where(c > 0, self._disjoint[at], 0).astype(np.int64)

    def reuse_at(self, count) -> np.ndarray:
        """
        The overdispersion of a FAR ``count``: the count over its independent events
        (`independent_count`), at least 1; `data_reuse` where the streams do not place
        their data.
        """
        c = np.asarray(count, dtype=np.float64)
        if self._disjoint is None:
            return np.full(c.shape, float(self.data_reuse))
        independent = self.independent_count(c).astype(np.float64)
        return np.where(independent > 0, np.maximum(c / np.maximum(independent, 1.0), 1.0), 1.0)

    def bootstrap_thresholds(self, fars_hz, rng: np.random.Generator) -> np.ndarray:
        """
        One bootstrap replicate of the thresholds for the targets ``fars_hz`` (Hz):
        NaN where the background cannot resolve a target.

        The information in a background is its count of independent events above a
        threshold, Poisson, each counted `reuse_at` times among its windows (a
        loud excursion in one detector's data opens a trigger in every slide). The
        estimator puts a target's threshold at its ``k``-th loudest trigger
        (`threshold_for_far`), ``k / reuse`` events down, where a unit-rate Poisson
        process of events has its arrival ``S ~ Gamma(k / reuse)``: the replicate's
        threshold is at the trigger the process puts there, the ``round(reuse * S)``-th
        loudest (`_threshold_at`), whose count is on average the estimator's
        ``k``. One process serves every target, its arrivals drawn as Gamma
        increments, so a replicate's thresholds fall as the target rises. Unslid
        (reuse 1), it is the bootstrap of independent windows: the true pass chance of
        the ``k``-th loudest of ``n`` is ``Gamma(k) / n``.
        """
        self._require_livetime()
        fars = np.asarray(fars_hz, dtype=np.float64).reshape(-1)
        allowed = np.floor(fars * self.livetime_seconds * (1 + 1e-12))
        thresholds = np.full(fars.size, np.nan)
        resolved = np.flatnonzero(allowed >= 1)
        if resolved.size == 0:
            return thresholds
        reuse = self.reuse_at(allowed[resolved])
        shape = allowed[resolved] / reuse
        order = np.argsort(shape, kind="stable")
        steps = np.diff(np.concatenate([[0.0], shape[order]]))
        times = np.empty(order.size)
        times[order] = np.cumsum(np.where(steps > 0, rng.gamma(np.maximum(steps, 1e-300)), 0.0))
        counts = np.rint(reuse * times)
        thresholds[resolved] = [self._threshold_at(int(count)) for count in counts]
        return thresholds

    def _require_livetime(self) -> None:
        if self.livetime_seconds <= 0:
            raise ValueError("The background has no livetime: it holds no scored windows.")


# ------------------------------------------------------------------- chance

@dataclass(frozen=True)
class ChanceLaw:
    """
    The law of the chance that noise alone passes a threshold, one per threshold:
    ``Beta(a, b)`` (`Background.found_by_chance`). ``a = 0`` is no chance (a
    threshold above every window), ``b = 0`` certainty; NaN is no threshold (an
    unresolved target).
    """
    a: np.ndarray
    b: np.ndarray

    def __post_init__(self):
        for name in ("a", "b"):
            object.__setattr__(self, name, np.atleast_1d(np.asarray(getattr(self, name), dtype=np.float64)))

    @classmethod
    def nothing(cls) -> "ChanceLaw":
        """No chance at all: a found test noise never passes."""
        return cls(np.zeros(1), np.ones(1))

    @property
    def mean(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.a == 0, 0.0, self.a / (self.a + self.b))

    def _at(self, values: np.ndarray) -> np.ndarray:
        """``values`` (drawn where the law is a Beta) with the point masses and NaNs in place."""
        point = np.where(self.a == 0, 0.0, 1.0)
        regular = (self.a > 0) & (self.b > 0)
        return np.where(np.isnan(self.a), np.nan, np.where(regular, values, point))

    def quantile(self, q: float) -> np.ndarray:
        regular = (self.a > 0) & (self.b > 0)
        return self._at(stats.beta.ppf(q, np.where(regular, self.a, 1.0), np.where(regular, self.b, 1.0)))

    def interval(self, confidence: float) -> Tuple[np.ndarray, np.ndarray]:
        """The central interval at ``confidence``: ``(lower, upper)``."""
        alpha = 1.0 - confidence
        return self.quantile(alpha / 2), self.quantile(1 - alpha / 2)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """``size`` draws of each chance: ``(size, thresholds)``."""
        regular = (self.a > 0) & (self.b > 0)
        draws = rng.beta(np.where(regular, self.a, 1.0), np.where(regular, self.b, 1.0), size=(size, self.a.size))
        return self._at(draws)


def subtract_chance(found_fraction, chance):
    """
    The fraction found by the signal, of a fraction ``found_fraction`` found where
    noise alone passes with ``chance``: ``(f - c) / (1 - c)`` (module docstring). Not
    clipped: below the chance it is negative, as an unbiased estimate may be.
    """
    f, c = np.asarray(found_fraction, dtype=np.float64), np.asarray(chance, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (f - c) / (1.0 - c)


def nested(lower: np.ndarray, upper: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Intervals at ascending levels (``(levels, ...)``) made to nest: each level's holds every
    lower level's. A term read at the ends of a range (the background's two thresholds)
    need not order its levels where the estimate is not monotone in the threshold (where
    noise alone passes as often as the signal); widening keeps each level's coverage.
    """
    return np.fmin.accumulate(lower, axis=0), np.fmax.accumulate(upper, axis=0)


def chance_subtracted_interval(detected, total, chance: "ChanceLaw", sigmas: Sequence[float] = DEFAULT_SIGMAS,
                               method: str = "clopper-pearson") -> Tuple[np.ndarray, np.ndarray]:
    """
    The interval at each of ``sigmas`` on the fraction found by the signal, of
    ``detected`` found of ``total``: each end of the binomial interval mapped by
    `subtract_chance` through the chance's own end on the far side (the lower end
    through the chance's upper end), so both errors widen it; clipped to [0, 1].
    ``(lower, upper)``, each ``(levels, ...)``.
    """
    lowers, uppers = [], []
    for sigma in sigmas:
        level = sigma_level(sigma)
        lower, upper = binomial_interval(detected, total, level, method)
        chance_lower, chance_upper = chance.interval(level)
        lowers.append(np.clip(subtract_chance(lower, chance_upper), 0.0, 1.0))
        uppers.append(np.clip(subtract_chance(upper, chance_lower), 0.0, 1.0))
    return np.array(lowers), np.array(uppers)


# ---------------------------------------------------------------- efficiency

def _injections(snrs, scores) -> Tuple[np.ndarray, np.ndarray]:
    snrs = np.asarray(snrs, dtype=np.float64).reshape(-1)
    scores = injection_scores(scores)
    if snrs.shape != scores.shape:
        raise ValueError(f"{snrs.size} SNRs for {scores.size} scores.")
    return snrs, scores


def _chance_or_nothing(chance: Optional["ChanceLaw"]) -> "ChanceLaw":
    return ChanceLaw.nothing() if chance is None else chance


@dataclass(frozen=True)
class EfficiencyCurve:
    """
    The fraction of injections found by their signal per SNR bin: ``detected`` of
    ``total`` found at ``threshold``, less the ``chance`` of a noise-only pass
    (`subtract_chance`), with intervals at each of ``sigmas`` (``(levels, bins)``).
    The binned curve is the record's estimator; the dashboard plots the rolling one.
    """
    threshold: float
    bin_edges: np.ndarray
    snr_mean: np.ndarray
    detected: np.ndarray
    total: np.ndarray
    chance: float
    efficiency: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    sigmas: Tuple[float, ...]
    method: str


def bin_index(snrs: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Each SNR's bin of ``edges``, ``[lo, hi)`` with the last closed; -1 outside them."""
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be strictly increasing, with at least two edges.")
    num_bins = edges.size - 1
    index = np.searchsorted(edges, snrs, side="right") - 1
    index[snrs == edges[-1]] = num_bins - 1
    return np.where((index >= 0) & (index < num_bins), index, -1)


def efficiency_curve(
    snrs,
    scores,
    threshold: float,
    bin_edges: Sequence[float],
    *,
    chance: Optional[ChanceLaw] = None,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
    method: str = "clopper-pearson",
) -> EfficiencyCurve:
    """
    The efficiency of injections at ``score >= threshold`` in SNR bins, with the
    chance of a noise-only pass at that threshold (``chance``, none if None)
    subtracted, and binomial intervals (`chance_subtracted_interval`).

    Bins are ``[lo, hi)``, the last one closed; injections outside the edges
    are not counted. A NaN threshold (an unresolved FAR) gives NaN efficiencies.
    Every argument after ``bin_edges`` is keyword-only: the fifth was once the
    interval's ``confidence``, and a positional argument never changes meaning.
    """
    snrs, scores = _injections(snrs, scores)
    edges = np.asarray(bin_edges, dtype=np.float64)
    index = bin_index(snrs, edges)
    num_bins, inside = edges.size - 1, index >= 0
    total = np.bincount(index[inside], minlength=num_bins)
    snr_sum = np.bincount(index[inside], weights=snrs[inside], minlength=num_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr_mean = np.where(total > 0, snr_sum / total, np.nan)
    sigmas = tuple(float(s) for s in sigmas)
    chance = _chance_or_nothing(chance)
    if np.isnan(threshold):
        nan, nans = np.full(num_bins, np.nan), np.full((len(sigmas), num_bins), np.nan)
        return EfficiencyCurve(float(threshold), edges, snr_mean, nan, total, np.nan, nan, nans, nans, sigmas, method)
    detected = np.bincount(index[inside & (scores >= threshold)], minlength=num_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        efficiency = subtract_chance(np.where(total > 0, detected / total, np.nan), chance.mean)
    lower, upper = chance_subtracted_interval(detected, total, chance, sigmas, method)
    return EfficiencyCurve(float(threshold), edges, snr_mean, detected, total, float(chance.mean[0]), efficiency,
                           lower, upper, sigmas, method)


#: The rolling efficiency curve: a window of +-0.5 in SNR, evaluated at 50 points; a
#: point with fewer injections in its window than this has none.
ROLLING_HALF_WIDTH = 0.5
ROLLING_POINTS = 50
ROLLING_MIN_INJECTIONS = 20


@dataclass(frozen=True)
class RollingEfficiency:
    """
    The rolling efficiency curve: at each ``snr``, the fraction of the
    injections within ``half_width`` of it (``[snr - w, snr + w)``) found by their
    signal (the ``chance`` subtracted), with **pointwise** bands at each of
    ``sigmas`` (``(levels, points)``): each covers its own point at its level, not
    the curve as a whole. NaN where a window holds fewer than
    `ROLLING_MIN_INJECTIONS`.
    """
    threshold: float
    snr: np.ndarray
    half_width: float
    detected: np.ndarray
    total: np.ndarray
    chance: float
    efficiency: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    sigmas: Tuple[float, ...]
    method: str

    @property
    def pointwise(self) -> bool:
        """The bands are pointwise, never simultaneous: said wherever they are shown."""
        return True


def rolling_windows(snr_range: Optional[Tuple[float, float]], snrs: np.ndarray,
                    half_width: float = ROLLING_HALF_WIDTH, num_points: int = ROLLING_POINTS) -> np.ndarray:
    """
    The rolling curve's window centres: ``num_points`` from the range's low end plus
    ``half_width`` to its high end less it (the injected SNR range; by default the
    injections' own, rounded out to whole SNRs).
    """
    lo, hi = (np.floor(snrs.min()), np.ceil(snrs.max())) if snr_range is None else snr_range
    return np.linspace(lo + half_width, hi - half_width, num_points)


def rolling_efficiency_curve(
    snrs,
    scores,
    threshold: float,
    *,
    snr_range: Optional[Tuple[float, float]] = None,
    chance: Optional[ChanceLaw] = None,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
    method: str = "clopper-pearson",
    half_width: float = ROLLING_HALF_WIDTH,
    num_points: int = ROLLING_POINTS,
) -> RollingEfficiency:
    """
    The rolling efficiency at ``score >= threshold`` (`RollingEfficiency`), its
    windows over ``snr_range`` (`rolling_windows`), the ``chance`` of a noise-only
    pass subtracted, with pointwise binomial bands (`chance_subtracted_interval`).
    """
    snrs, scores = _injections(snrs, scores)
    centres = rolling_windows(snr_range, snrs, half_width, num_points)
    order = np.argsort(snrs, kind="stable")
    sorted_snrs = snrs[order]
    first = np.searchsorted(sorted_snrs, centres - half_width, side="left")
    last = np.searchsorted(sorted_snrs, centres + half_width, side="left")
    total = last - first
    sigmas = tuple(float(s) for s in sigmas)
    chance = _chance_or_nothing(chance)
    if np.isnan(threshold):
        nan, nans = np.full(centres.size, np.nan), np.full((len(sigmas), centres.size), np.nan)
        return RollingEfficiency(float(threshold), centres, float(half_width), nan, total, np.nan, nan, nans, nans,
                                 sigmas, method)
    found = np.concatenate([[0], np.cumsum(scores[order] >= threshold)])
    detected = found[last] - found[first]
    enough = total >= ROLLING_MIN_INJECTIONS
    with np.errstate(invalid="ignore", divide="ignore"):
        efficiency = np.where(enough, subtract_chance(detected / total, chance.mean), np.nan)
    lower, upper = chance_subtracted_interval(detected, total, chance, sigmas, method)
    return RollingEfficiency(float(threshold), centres, float(half_width), detected, total, float(chance.mean[0]),
                             efficiency, np.where(enough, lower, np.nan), np.where(enough, upper, np.nan),
                             sigmas, method)


# ------------------------------------------------------------- sigmoid fit

#: The sigmoid fit groups the injections into bins this many per unit of SNR, each at its
#: injections' mean SNR: the binned likelihood is the per-injection one to the bins' width squared.
FIT_BINS_PER_SNR = 20
#: Fisher-scoring iterations of one fit, and the relative step at which it has converged.
FIT_ITERATIONS = 200
FIT_TOLERANCE = 1e-10


@dataclass(frozen=True)
class SigmoidFit:
    """
    A logistic efficiency fitted to the injections found at ``threshold``, over the
    ``chance`` of a noise-only pass: found with chance ``c + (1 - c) S(snr)``,
    ``S(snr) = 1 / (1 + exp(-slope (snr - snr_50)))``, by maximum likelihood. Its
    50% and 90% points (``snr_90 = snr_50 + ln 9 / slope``), with bootstrap
    intervals at each of ``sigmas`` (``(levels, 2)``: the 50% point, then the 90%).
    A point outside the injected SNR range, or a fit with no transition, is NaN: it
    was not measured. ``num_failed`` replicates of ``num_bootstrap`` gave no point.
    """
    threshold: float
    chance: float
    snr_50: float
    snr_90: float
    slope: float
    lower: np.ndarray
    upper: np.ndarray
    sigmas: Tuple[float, ...]
    num_bootstrap: int
    num_failed: int

    def efficiency(self, snr) -> np.ndarray:
        """The fitted efficiency of the signal, ``S(snr)``."""
        return special.expit(self.slope * (np.asarray(snr, dtype=np.float64) - self.snr_50))


def resample_groups(rng: np.random.Generator, found: np.ndarray, total: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    One bootstrap draw of the injections, in groups: each row of ``found`` of ``total``
    ((rows, groups) and (groups,)) redrawn as the injections drawn again with
    replacement would fall, each group's count multinomial over the groups and its
    found count binomial of it at the group's found fraction. (Redrawing only the
    found counts, holding each group's total, loses the variance a group too small
    to show both outcomes carries.) Returns ``(found, total)``, each (rows, groups).
    """
    found = np.atleast_2d(found)
    total = np.asarray(total, dtype=np.int64)
    n = int(total.sum())
    drawn = rng.multinomial(n, total / max(n, 1), size=found.shape[0]) if n else np.zeros(found.shape, np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(total > 0, found / np.maximum(total, 1), 0.0)
    return rng.binomial(drawn, rate), drawn


def fit_bins(snrs: np.ndarray, snr_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """The fit's bins over ``snr_range`` (`FIT_BINS_PER_SNR`): their edges, and each injection's bin (-1 outside)."""
    lo, hi = snr_range
    edges = np.linspace(lo, hi, max(int(round((hi - lo) * FIT_BINS_PER_SNR)), 1) + 1)
    return edges, bin_index(snrs, edges)


def fit_sigmoid(snr: np.ndarray, detected: np.ndarray, total: np.ndarray, chance: np.ndarray,
                snr_range: Tuple[float, float], start: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The maximum-likelihood logistic of each row of ``detected`` of ``total`` found at
    bin SNRs ``snr`` ((R, J) counts, (J,) SNRs) over chances ``chance`` (R,), by Fisher
    scoring, from ``start`` (``(snr_50, slope)``, a fit to start a bootstrap's
    replicates from) where it is finite, else from the step a uniform SNR design
    would give. Returns ``(snr_50, snr_90, slope)``, each (R,), NaN where the fit does
    not converge, does not rise, or puts a point outside ``snr_range``.
    """
    k, n = np.atleast_2d(detected).astype(np.float64), np.atleast_2d(total).astype(np.float64)
    c = np.asarray(chance, dtype=np.float64).reshape(-1, 1)
    x = np.where(np.isfinite(snr), snr, 0.0)[None, :]
    lo, hi = snr_range
    with np.errstate(invalid="ignore", divide="ignore"):
        signal = np.clip(subtract_chance(k.sum(axis=1) / n.sum(axis=1), c[:, 0]), 0.02, 0.98)
    # A step at s over SNRs uniform on [lo, hi] finds (hi - s) / (hi - lo) of them.
    slope = np.full(k.shape[0], 2.0)
    a = -slope * (hi - signal * (hi - lo))
    if start is not None:
        given = np.isfinite(start[0]) & np.isfinite(start[1])
        slope = np.where(given, start[1], slope)
        a = np.where(given, -start[1] * start[0], a)
    converged = np.zeros(k.shape[0], dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for _ in range(FIT_ITERATIONS):
            s = special.expit(a[:, None] + slope[:, None] * x)
            p = np.clip(c + (1 - c) * s, 1e-12, 1 - 1e-12)
            gradient = (1 - c) * s * (1 - s)
            weight = n * gradient**2 / (p * (1 - p))
            residual = (k - n * p) * gradient / (p * (1 - p))
            u0, u1 = residual.sum(axis=1), (residual * x).sum(axis=1)
            i00, i01, i11 = weight.sum(axis=1), (weight * x).sum(axis=1), (weight * x**2).sum(axis=1)
            det = i00 * i11 - i01**2
            da, db = (i11 * u0 - i01 * u1) / det, (i00 * u1 - i01 * u0) / det
            # A step of at most half the slope's size keeps it from overshooting through zero.
            scale = np.minimum(1.0, 0.5 * np.abs(slope) / np.maximum(np.abs(db), 1e-300))
            da, db = np.where(np.isfinite(da), da * scale, 0.0), np.where(np.isfinite(db), db * scale, 0.0)
            a, slope = a + da, slope + db
            converged = (np.abs(da) <= FIT_TOLERANCE * (1 + np.abs(a))) & (np.abs(db) <= FIT_TOLERANCE * (1 + np.abs(slope)))
            if np.all(converged | ~np.isfinite(a)):
                break
        snr_50 = -a / slope
        snr_90 = snr_50 + np.log(9.0) / slope
    good = converged & (slope > 0) & np.isfinite(snr_50)
    inside = lambda point: np.where(good & (point >= lo) & (point <= hi), point, np.nan)
    return inside(snr_50), inside(snr_90), np.where(good, slope, np.nan)


def sigmoid_fit_from_replicates(threshold: float, chance: float, points: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                sigmas: Sequence[float]) -> SigmoidFit:
    """
    A `SigmoidFit` from fits of the data (row 0 of ``points``, each (R,)) and of
    bootstrap replicates (rows 1 onward): each point's interval is the central
    range of its replicates' finite values at each level, at least wide enough to
    hold the point itself.
    """
    snr_50, snr_90, slope = points
    sigmas = tuple(float(s) for s in sigmas)
    lower, upper = np.full((len(sigmas), 2), np.nan), np.full((len(sigmas), 2), np.nan)
    replicates = np.stack([snr_50[1:], snr_90[1:]], axis=1)
    for j, point in enumerate((snr_50[0], snr_90[0])):
        values = replicates[:, j][np.isfinite(replicates[:, j])]
        if not np.isfinite(point) or values.size == 0:
            continue
        for level, sigma in enumerate(sigmas):
            alpha = 1.0 - sigma_level(sigma)
            low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
            lower[level, j], upper[level, j] = min(low, point), max(high, point)
    failed = int(np.count_nonzero(~(np.isfinite(snr_50[1:]) & np.isfinite(snr_90[1:]))))
    return SigmoidFit(float(threshold), float(chance), float(snr_50[0]), float(snr_90[0]), float(slope[0]),
                      lower, upper, sigmas, int(snr_50.size - 1), failed)


def fit_efficiency_sigmoid(
    snrs,
    scores,
    threshold: float,
    *,
    chance: Optional[ChanceLaw],
    snr_range: Tuple[float, float],
    rng: np.random.Generator,
    num_bootstrap: int = 1000,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
) -> SigmoidFit:
    """
    The sigmoid fit (`SigmoidFit`) of the injections found at ``score >=
    threshold``, over the ``chance`` of a noise-only pass (none if None), the
    injections grouped by SNR (`fit_bins`). Its bootstrap redraws the injections
    (`resample_groups`) and the chance from its law: at a fixed threshold. (At a target FAR, `efficiency.efficiency_against_snr`
    also redraws the threshold.)
    """
    snrs, scores = _injections(snrs, scores)
    chance = _chance_or_nothing(chance)
    if np.isnan(threshold):
        nan = np.full((len(tuple(sigmas)), 2), np.nan)
        return SigmoidFit(float(threshold), np.nan, np.nan, np.nan, np.nan, nan, nan.copy(),
                          tuple(float(s) for s in sigmas), int(num_bootstrap), int(num_bootstrap))
    edges, index = fit_bins(snrs, snr_range)
    inside = index >= 0
    num_bins = edges.size - 1
    total = np.bincount(index[inside], minlength=num_bins)
    detected = np.bincount(index[inside & (scores >= threshold)], minlength=num_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.bincount(index[inside], weights=snrs[inside], minlength=num_bins) / total
    point = fit_sigmoid(snr, detected[None], total[None], chance.mean, snr_range)
    counts, totals = resample_groups(rng, np.broadcast_to(detected, (num_bootstrap, num_bins)), total)
    replicates = fit_sigmoid(snr, counts, totals, chance.sample(rng, num_bootstrap)[:, 0], snr_range,
                             start=(np.repeat(point[0], num_bootstrap), np.repeat(point[2], num_bootstrap)))
    points = tuple(np.concatenate([p, r]) for p, r in zip(point, replicates))
    return sigmoid_fit_from_replicates(threshold, chance.mean[0], points, sigmas)


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
    pos = np.sort(injection_scores(positive_scores, "positive scores"))
    if neg.size == 0 or pos.size == 0:
        raise ValueError("An ROC needs at least one negative and one positive score.")
    thresholds = np.concatenate([[np.inf], np.unique(np.concatenate([neg, pos]))[::-1]])
    fpr = _count_at_or_above(neg, thresholds) / neg.size
    tpr = _count_at_or_above(pos, thresholds) / pos.size
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1])) / 2)
    return RocCurve(fpr, tpr, thresholds, auc)
