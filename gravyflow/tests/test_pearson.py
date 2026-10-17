import pytest
import numpy as np
from keras import ops
from gravyflow.src.dataset.conditioning import pearson as gf_pearson
from gravyflow.tests._helpers import representation_tolerance


def pearson_tolerance(num_samples):
    """Representation: a float32 Pearson coefficient is sums of N products, bounded by N roundings."""
    return representation_tolerance(np.float32, 1.0, num_samples)


def test_pearson_basic():
    # Perfectly correlated
    x = ops.convert_to_tensor([[1.0, 2.0, 3.0]])
    y = ops.convert_to_tensor([[2.0, 4.0, 6.0]])
    corr = gf_pearson.pearson(x, y)
    np.testing.assert_allclose(corr, [1.0], atol=1e-5)
    
    # Anti-correlated
    y_neg = ops.convert_to_tensor([[-1.0, -2.0, -3.0]])
    corr_neg = gf_pearson.pearson(x, y_neg)
    np.testing.assert_allclose(corr_neg, [-1.0], atol=1e-5)


def test_rolling_pearson_lags_are_symmetric_and_inclusive():
    """Lags run from -M to +M samples, M = ceil(delay * rate): every allowed delay is sampled."""
    sample_rate, max_delay = 1.0, 3.5
    assert gf_pearson.num_lags(max_delay, sample_rate) == 9  # M = 4
    rng = np.random.default_rng(0)
    x = rng.normal(size=256).astype(np.float32)
    found = []
    for k in range(-4, 5):
        tensor = ops.convert_to_tensor(np.stack([x, np.roll(x, -k)])[None])
        corrs = np.asarray(gf_pearson.rolling_pearson(tensor, max_delay, sample_rate))[0, 0]
        assert corrs.shape == (9,)
        np.testing.assert_allclose(corrs.max(), 1.0, rtol=0.0, atol=pearson_tolerance(x.size))
        found.append(int(np.argmax(corrs)))
    assert sorted(found) == list(range(9))


@pytest.mark.parametrize("direction", [+1, -1])
def test_the_largest_lag_is_compared_in_both_directions(direction):
    # The lags wrapped around the window (jnp.roll): at the largest lag the samples rolled
    # in from the other end were unrelated, so a pure delay of M samples never correlated
    # fully, in either direction.
    rate, max_lag = 64.0, 0.5
    M = gf_pearson.max_lag_samples(max_lag, rate)
    N = 384
    z = np.random.default_rng(0).normal(size=N + 2 * M).astype(np.float32)
    x = z[M:M + N]
    y = z[M + direction * M:M + direction * M + N]  # y[t] = x[t + direction M]
    corrs = np.asarray(gf_pearson.rolling_pearson(np.stack([x, y])[None], max_lag, rate))[0, 0]
    assert corrs.shape == (2 * M + 1,)
    # Index i compares x[t] with y[t + i - M]: the delay is at lag -direction M.
    np.testing.assert_allclose(corrs[M - direction * M], 1.0, rtol=0.0, atol=pearson_tolerance(N))


def test_every_lag_compares_the_same_samples_of_the_first_detector():
    # Every lag correlates the first detector's centre, N - 2M samples, with the second's
    # samples l later, so every lag has the same number of samples and none wraps.
    rate, max_lag, N = 64.0, 0.25, 256
    M = gf_pearson.max_lag_samples(max_lag, rate)
    x, y = np.random.default_rng(1).normal(size=(2, N)).astype(np.float32)
    corrs = np.asarray(gf_pearson.rolling_pearson(np.stack([x, y])[None], max_lag, rate))[0, 0]
    for i, lag in enumerate(range(-M, M + 1)):
        expected = np.corrcoef(x[M:N - M], y[M + lag:N - M + lag])[0, 1]
        np.testing.assert_allclose(corrs[i], expected, rtol=0.0, atol=pearson_tolerance(N - 2 * M))
