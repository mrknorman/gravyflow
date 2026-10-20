"""
Per-template background fits of single-detector triggers, as PyCBC fits and
ranks them (Nitz, Dent, Dal Canton, Fairhurst & Brown 2017, ApJ 849, 118,
section 2.1; ``pycbc.events.trigger_fits``, ``pycbc_fit_sngls_by_template``,
``pycbc_fit_sngls_over_multiparam`` and ``pycbc.events.stat.ExpFitStatistic``).

Above a threshold x_t, each template's noise triggers' ranking statistic x
(the reweighted SNR) falls exponentially: p(x) = alpha exp(-alpha (x - x_t)).
alpha's maximum-likelihood estimate from the n triggers above x_t is
1 / (mean(x) - x_t), with standard error alpha / sqrt(n). One template has few
triggers, so the fits are smoothed over templates of similar duration: here,
over the templates whose log10 duration is within `smoothing_width` of the
template's (a top-hat in log duration), pooling their triggers, so the
smoothed alpha is the count-weighted harmonic mean of theirs and the smoothed
rate their mean count above threshold per second of live time. A trigger's
log noise rate density is then

    ln r(x) = -alpha (x - x_t) + ln alpha + ln rate,

and a coincidence ranks by -sum over detectors of ln r (ExpFitStatistic's
noise term: rarer in each detector's background is louder). The duration is
the Newtonian chirp time from the band's lower edge,
tau_0 = 5 / (256 pi f) (pi G M_c f / c^3)^(-5/3).
"""

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from gravyflow.src.detection.template_grid import SOLAR_MASS_SECONDS


def exponential_fit(values, threshold: float):
    """(alpha, its standard error, count): the MLE of an exponential above `threshold` (pycbc fit_above_thresh)."""
    values = np.asarray(values, dtype=np.float64)
    above = values[values >= threshold]
    if not len(above):
        return np.nan, np.nan, 0
    alpha = 1.0 / (np.mean(above) - threshold)
    return alpha, alpha / np.sqrt(len(above)), len(above)


def chirp_time(mass_1, mass_2, lower_frequency_hertz: float) -> np.ndarray:
    """The Newtonian chirp time (s) from `lower_frequency_hertz` of each binary (detector-frame masses)."""
    m1, m2 = np.asarray(mass_1, np.float64), np.asarray(mass_2, np.float64)
    chirp = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2 * SOLAR_MASS_SECONDS
    f = float(lower_frequency_hertz)
    return 5.0 / (256.0 * np.pi * f) * (np.pi * chirp * f) ** (-5.0 / 3.0)


@dataclass(frozen=True)
class TemplateFits:
    """
    One detector's smoothed per-template fits: `alpha` and `rate` (triggers
    above threshold per second of live time) per template, the `threshold`,
    and the raw per-template `count` and `alpha_raw` (NaN without triggers).
    """
    threshold: float
    alpha: np.ndarray
    rate: np.ndarray
    count: np.ndarray
    alpha_raw: np.ndarray

    def log_noise_rate(self, template, statistic) -> np.ndarray:
        """ln r(x) of triggers of `template` with ranking statistic `statistic` (see the module docstring)."""
        template = np.asarray(template, dtype=np.int64)
        alpha, rate = self.alpha[template], self.rate[template]
        x = np.asarray(statistic, dtype=np.float64)
        return -alpha * (x - self.threshold) + np.log(alpha) + np.log(rate)


def fit_templates(triggers, bank, threshold: float = 6.0, smoothing_width: float = 0.1,
                  lower_frequency_hertz: Optional[float] = None) -> TemplateFits:
    """
    The smoothed exponential fits of one detector's background triggers
    (`DetectorTriggers`, for example from time slides or the clean data, its
    `reweighted_snr` the ranking statistic and its span the live time), per
    template of `bank`. `smoothing_width` is the half-width of the top-hat in
    log10 chirp time. The triggers must index `bank` (their `bank` its
    `bank_identity`), or they are refused by name.
    """
    from gravyflow.src.detection.stream import bank_identity
    if triggers.bank != bank_identity(bank):
        raise ValueError(f"The triggers index the bank {triggers.bank}, not the bank given ({bank_identity(bank)}).")
    num = bank.num_templates
    f_low = lower_frequency_hertz if lower_frequency_hertz is not None else (
        bank.space.lower_frequency_hertz if bank.space is not None else 20.0)
    duration = np.log10(chirp_time(bank.mass_1, bank.mass_2, f_low))
    x = np.asarray(triggers.reweighted_snr, dtype=np.float64)
    above = x >= threshold
    template = np.asarray(triggers.template)[above]
    count = np.bincount(template, minlength=num).astype(np.float64)
    excess = np.bincount(template, weights=x[above] - threshold, minlength=num)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_raw = np.where(count > 0, count / excess, np.nan)
    order = np.argsort(duration)
    sorted_duration = duration[order]
    lo = np.searchsorted(sorted_duration, sorted_duration - smoothing_width, side="left")
    hi = np.searchsorted(sorted_duration, sorted_duration + smoothing_width, side="right")
    cumulative_count = np.concatenate([[0.0], np.cumsum(count[order])])
    cumulative_excess = np.concatenate([[0.0], np.cumsum(excess[order])])
    pooled_count = cumulative_count[hi] - cumulative_count[lo]
    pooled_excess = cumulative_excess[hi] - cumulative_excess[lo]
    alpha = np.empty(num)
    rate = np.empty(num)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha[order] = pooled_count / pooled_excess
    rate[order] = pooled_count / (hi - lo) / triggers.live_seconds
    return TemplateFits(float(threshold), alpha, rate, count, alpha_raw)


def coincident_ranking(coincidences, fits: Mapping[str, TemplateFits]) -> np.ndarray:
    """-sum over detectors of ln r of each coincidence (`Coincidences`), with each detector's fits."""
    total = np.zeros(len(coincidences))
    for name, values in coincidences.reweighted_snr.items():
        total -= fits[name].log_noise_rate(coincidences.template, values)
    return total


__all__ = ["TemplateFits", "chirp_time", "coincident_ranking", "exponential_fit", "fit_templates"]
