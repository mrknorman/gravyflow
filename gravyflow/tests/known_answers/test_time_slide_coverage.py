"""
Known answer: the FAR interval of a time-slid background covers the true rate at
its level.

Slides that share one detector's data are not independent: a loud excursion in
one detector passes in many slides, so the count's variance is several times
Poisson's, and an interval on the slid triggers as if each were independent
undercovers (with 8 slides, 2 sigma covered about 86% of the time for a sum of
the detectors' statistics, 48% for their maximum). The interval counts a
time-slid background's independent information (`Background.count_interval`).
Written pending before that landed; enabled by the branch that lands it.
"""

import numpy as np
import pytest

from gravyflow.tests.known_answers import truth
from gravyflow.tests.known_answers.test_far import slid_background


@pytest.mark.parametrize("confidence", truth.LEVELS, ids=truth.LEVEL_IDS)
@pytest.mark.parametrize("combine", ["sum", "max"])
def test_the_time_slide_far_interval_covers_the_true_rate(confidence, combine):
    rng = np.random.default_rng(51)
    replicas, num_windows, num_slides = 400, 2000, 8
    if combine == "sum":
        fn, d = (lambda h, l: (h + l) / np.sqrt(2)), truth.GaussianBank.single()
    else:
        fn, d = np.maximum, truth.GaussianBank((0.0, 0.0))
    lengths = num_windows - np.arange(1, num_slides + 1)
    t = d.threshold_for_noise_sf(20.0 / lengths.sum())
    expected = sum(truth.clustered_trigger_rate_hz(d, t, int(m), 1, 1.0, 1.0) * m for m in lengths) / lengths.sum()
    covered = []
    for _ in range(replicas):
        f = slid_background(rng, num_windows, num_slides, fn)[0].far(t, confidence)
        covered.append(f.lower_hz <= expected <= f.upper_hz)
    assert np.mean(covered) >= confidence - truth.coverage_tolerance(confidence, replicas)
