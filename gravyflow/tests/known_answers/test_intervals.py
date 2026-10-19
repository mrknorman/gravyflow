"""
The validator's intervals cover at their stated levels: computed exactly, by
summing the count's distribution over the counts whose interval holds the
truth, with no Monte Carlo. Both levels of the standard (1 and 2 sigma), and
the validator's default.
"""

import numpy as np
import pytest
from scipy import stats

from gravyflow.src.validate import metrics
from gravyflow.tests.known_answers import truth

LEVELS = truth.LEVELS + (metrics.DEFAULT_CONFIDENCE,)


def poisson_tails(mu: float, confidence: float):
    """The chances that Garwood's interval lies wholly above and wholly below ``mu``."""
    k = np.arange(int(mu + 20 * np.sqrt(mu) + 40))
    lower, upper = metrics.poisson_interval(k, confidence)
    pmf = stats.poisson.pmf(k, mu)
    return float(np.sum(pmf[lower > mu])), float(np.sum(pmf[upper < mu]))


@pytest.mark.parametrize("confidence", LEVELS, ids=truth.LEVEL_IDS + ("default",))
def test_garwood_covers_every_mean_and_each_tail_is_at_most_half_alpha(confidence):
    # Garwood's interval is exact and central: each one-sided miss is at most (1 - CL) / 2.
    half_alpha = (1 - confidence) / 2
    for mu in np.logspace(-2, 2.5, 300):
        above, below = poisson_tails(mu, confidence)
        assert above <= half_alpha + 1e-9 and below <= half_alpha + 1e-9, (mu, above, below)


def binomial_coverage(n: int, p: float, confidence: float, method: str):
    k = np.arange(n + 1)
    lower, upper = metrics.binomial_interval(k, n, confidence, method)
    pmf = stats.binom.pmf(k, n, p)
    return float(np.sum(pmf[(lower <= p) & (p <= upper)])), float(np.sum(pmf[lower > p])), \
        float(np.sum(pmf[upper < p]))


@pytest.mark.parametrize("confidence", LEVELS, ids=truth.LEVEL_IDS + ("default",))
@pytest.mark.parametrize("n", [5, 37, 400])
def test_clopper_pearson_covers_every_proportion(confidence, n):
    half_alpha = (1 - confidence) / 2
    for p in np.linspace(0.001, 0.999, 157):
        covered, above, below = binomial_coverage(n, p, confidence, "clopper-pearson")
        assert above <= half_alpha + 1e-9 and below <= half_alpha + 1e-9, (p, above, below)
        assert covered >= confidence - 1e-9


@pytest.mark.parametrize("confidence", LEVELS, ids=truth.LEVEL_IDS + ("default",))
@pytest.mark.parametrize("n", [37, 400])
def test_wilson_covers_on_average(confidence, n):
    # Wilson's interval is approximate: its coverage oscillates about the nominal level
    # (dipping well below it near p = 0 and 1), so it is held to its average over p.
    ps = np.linspace(0.02, 0.98, 97)
    mean = np.mean([binomial_coverage(n, p, confidence, "wilson")[0] for p in ps])
    assert abs(mean - confidence) < 0.01, mean
