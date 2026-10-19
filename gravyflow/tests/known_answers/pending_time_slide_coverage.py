"""
PENDING known answer: the FAR interval of a time-slid background covers the
true rate at its level.

Not collected (its name does not start with ``test_``): it fails on the current
validator, whose FAR interval is Poisson's (Garwood) on the slid triggers as if
each were independent. Slides that share one detector's data are not: a loud
excursion in one detector passes in many slides, so the count's variance is
several times Poisson's and the interval undercovers (with 8 slides, 2 sigma
covers about 86% of the time for a sum of the detectors' statistics, 48% for
their maximum). It is
enabled, by renaming it ``test_time_slide_coverage.py``, on the branch that
makes the FAR interval count a time-slid background's independent information.
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
