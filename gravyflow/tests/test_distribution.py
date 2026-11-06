"""
The SINE distribution: density proportional to sin(x) on [min_, max_] within
[0, pi], the polar angle of a direction uniform on the sphere (a binary's
inclination over the sky, density sin(i) / 2 on [0, pi]).
"""

import numpy as np
import pytest
from scipy import stats

import gravyflow as gf
from gravyflow.src.dataset.features.injection import get_support_bounds


def sine(low, high, **kwargs):
    return gf.Distribution(min_=low, max_=high, type_=gf.DistributionType.SINE, **kwargs)


def sine_cdf(low, high):
    return lambda x: (np.cos(low) - np.cos(x)) / (np.cos(low) - np.cos(high))


@pytest.mark.parametrize("low, high", [(0.0, np.pi), (0.3, 1.2), (2.0, np.pi)])
def test_draws_follow_the_sine_density(low, high):
    samples = sine(low, high).sample(200_000, np.random.default_rng(5))
    assert stats.kstest(samples, sine_cdf(low, high)).pvalue > 0.01
    # A uniform draw on the same range is refused by the same test: it has the power to tell them apart.
    uniform = np.random.default_rng(5).uniform(low, high, 200_000)
    assert stats.kstest(uniform, sine_cdf(low, high)).pvalue < 1e-6


def test_the_whole_range_is_isotropic():
    # Uniform on the sphere: cos(x) uniform on [-1, 1].
    samples = sine(0.0, np.pi).sample(200_000, np.random.default_rng(6))
    assert stats.kstest(np.cos(samples), stats.uniform(loc=-1.0, scale=2.0).cdf).pvalue > 0.01


@pytest.mark.parametrize("low, high", [(0.0, np.pi), (0.3, 1.2), (1.0, 1.0 + 1e-9)])
def test_draws_lie_within_the_bounds(low, high):
    samples = sine(low, high).sample(100_000, np.random.default_rng(7))
    assert samples.shape == (100_000,) and samples.dtype == np.float64
    assert np.all(np.isfinite(samples))
    assert samples.min() >= low and samples.max() <= high


def test_bounds_given_in_either_order_mean_one_interval():
    rng = lambda: np.random.default_rng(8)  # noqa: E731
    np.testing.assert_array_equal(sine(1.2, 0.3).sample(64, rng()), sine(0.3, 1.2).sample(64, rng()))


def test_a_point_range_gives_its_point():
    np.testing.assert_array_equal(sine(0.5, 0.5).sample(4, np.random.default_rng(0)), np.full(4, 0.5))


def test_each_value_takes_one_uniform():
    # One double of the bit generator per value, as UNIFORM takes: the stream after n values is a fresh
    # stream after n doubles.
    rng, reference = np.random.default_rng(9), np.random.default_rng(9)
    sine(0.0, np.pi).sample(1000, rng)
    reference.random(1000)
    assert rng.random() == reference.random()


@pytest.mark.parametrize("low, high", [(-0.1, 1.0), (0.0, np.pi + 1e-6), (-1.0, -0.5), (3.5, 4.0)])
def test_a_range_beyond_zero_to_pi_is_refused(low, high):
    with pytest.raises(ValueError, match=r"SINE.*\[0, pi\]"):
        sine(low, high).sample(1, np.random.default_rng(0))


@pytest.mark.parametrize("bounds", [dict(min_=0.0), dict(max_=1.0), dict()])
def test_a_sine_without_both_bounds_is_refused(bounds):
    with pytest.raises(ValueError, match="min"):
        gf.Distribution(type_=gf.DistributionType.SINE, **bounds).sample(1, np.random.default_rng(0))


def test_a_sine_over_more_than_a_point_is_continuous():
    assert sine(0.0, np.pi).continuous
    assert not sine(0.5, 0.5).continuous
    assert get_support_bounds(sine(0.3, 1.2)) == (0.3, 1.2)


def test_a_binary_takes_a_sine_inclination():
    generator = gf.CBCGenerator(inclination_radians=sine(0.0, np.pi))
    assert generator.inclination_radians.type_ is gf.DistributionType.SINE
