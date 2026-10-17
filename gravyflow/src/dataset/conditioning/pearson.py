import math

from keras import ops
import jax.numpy as jnp
import jax


@jax.jit
def pearson(x, y):
    """
    The Pearson correlation of each row of x with the same row of y, (..., Time)
    -> (...), with no regulariser: proportional rows give exactly +1 or -1 at
    any scale whose sums of squares float32 holds (rolling_pearson divides by
    each row's peak first). A row with no variance correlates at 0.
    """

    mean_x = ops.mean(x, axis=-1, keepdims=True)
    mean_y = ops.mean(y, axis=-1, keepdims=True)

    numerator = ops.sum((x - mean_x) * (y - mean_y), axis=-1)

    var_x = ops.sum(ops.square(x - mean_x), axis=-1)
    var_y = ops.sum(ops.square(y - mean_y), axis=-1)

    denominator = ops.sqrt(var_x * var_y)

    defined = denominator > 0
    return ops.where(defined, numerator / ops.where(defined, denominator, 1.0), 0.0)


def max_lag_samples(max_lag_seconds: float, sample_rate_hertz: float) -> int:
    """
    The largest lag `rolling_pearson` computes, in samples: the lag range rounded up,
    so that every delay within it is sampled.
    """
    return math.ceil(float(max_lag_seconds) * float(sample_rate_hertz))


def num_lags(max_lag_seconds: float, sample_rate_hertz: float) -> int:
    """
    The number of lags `rolling_pearson` computes for each detector pair: every lag
    from -M to +M samples, ``2 M + 1``, where M is ``max_lag_samples``.
    """
    return 2 * max_lag_samples(max_lag_seconds, sample_rate_hertz) + 1


@jax.jit(static_argnames=["max_lag_seconds", "sample_rate_hertz"])
def rolling_pearson(tensor, max_lag_seconds: float, sample_rate_hertz: float):
    """
    The Pearson correlation of every detector pair at every lag from -M to +M
    samples (``num_lags``), shape (Batch, Pairs, 2 M + 1).

    Lag index i compares the first detector's centre, its samples [M, N - M), with
    the second detector's samples ``i - M`` later. Every lag therefore correlates
    the same N - 2M samples, and none wraps around the window: the largest delay is
    measured the same way in both directions.

    Scale free: the correlation does not depend on a row's scale, so each row is
    divided by its peak first, and the sums of squares of physical strain (~1e-42
    per sample) stay inside float32's range.

    Raises:
        ValueError: a window of no more than 2 M samples, which leaves no centre.
    """
    tensor = ops.convert_to_tensor(tensor)
    peak = ops.max(ops.abs(tensor), axis=-1, keepdims=True)
    tensor = tensor / ops.where(peak > 0, peak, ops.ones_like(peak))

    max_samples = max_lag_samples(max_lag_seconds, sample_rate_hertz)
    num_arrays, num_samples = ops.shape(tensor)[1], ops.shape(tensor)[2]
    centre = num_samples - 2 * max_samples
    if centre <= 0:
        raise ValueError(
            f"A lag range of {max_lag_seconds} s ({max_samples} samples each way) leaves no "
            f"samples to correlate in a window of {num_samples}."
        )

    # Pairs (i, j), i < j: (Batch, Pairs, Time).
    first, second = jnp.triu_indices(num_arrays, k=1)
    x = ops.take(tensor, first, axis=1)[..., max_samples:max_samples + centre]
    y = ops.take(tensor, second, axis=1)

    def at_lag(start):
        return pearson(x, jax.lax.dynamic_slice_in_dim(y, start, centre, axis=-1))

    # Lag l = start - M, from -M to +M: (Lags, Batch, Pairs) -> (Batch, Pairs, Lags).
    return jnp.moveaxis(jax.vmap(at_lag)(jnp.arange(2 * max_samples + 1)), 0, -1)
