"""
Stationary Gaussian noise on absolute time: one detector's noise curve as a continuous
stream, read over any span.

Coloured noise elsewhere in gravyflow (``noise.colored_noise_batcher``) colours each window
on its own. A benchmark's data is one continuous stretch per segment, read in pieces, so
here the noise is a process on GPS time: sample n (at n / rate seconds) is a function of
the seed tree, the detector and n alone, and any span reads the same samples however it is
cut.

The construction is that of PyCBC's reproducible coloured noise
(``pycbc.noise.reproduceable.colored_noise``, which MLGWSC-1's datasets 1 to 3 use), with
gravyflow's random streams in place of NumPy's legacy generator: unit white noise, drawn in
fixed blocks of absolute time, convolved with the curve's amplitude kernel truncated to
``filter_seconds`` by a Hann window. PyCBC colours each requested span in the frequency
domain and keeps its middle; here the truncated kernel is applied as an exact finite
filter (overlap-save on fixed blocks), so the samples never depend on the span read. The
two have the same spectrum, the curve's PSD smoothed over 1 / ``filter_seconds``, and no
power below ``low_frequency_hertz``.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from gravyflow.src.dataset.conditioning.detector import IFO, NoiseProfile, read_noise_curve
from gravyflow.src.rng import SeedTree

#: Filter lengths per block of white noise: each block is coloured by one transform of
#: (1 + 1 / BLOCK_FILTERS) blocks' length.
BLOCK_FILTERS = 4


@dataclass(frozen=True)
class StationaryNoise:
    """
    Stationary Gaussian noise with the one-sided PSD of ``profile``'s curve, from
    ``low_frequency_hertz`` up to the Nyquist frequency, on absolute time.

    Args:
        profile: The noise curves; each detector read must be one it covers.
        sample_rate_hertz: The sampling rate.
        low_frequency_hertz: No power below it.
        filter_seconds: The colouring kernel's length; the spectrum is the curve's,
            smoothed over its inverse. Its samples at the rate are even.
    """

    profile: NoiseProfile
    sample_rate_hertz: float = 2048.0
    low_frequency_hertz: float = 15.0
    filter_seconds: float = 128.0

    def __post_init__(self):
        samples = self.filter_seconds * self.sample_rate_hertz
        if not (samples > 0 and samples == int(samples) and int(samples) % 2 == 0):
            raise ValueError(
                f"The filter's length at the rate must be a positive even number of samples: "
                f"{self.filter_seconds} s at {self.sample_rate_hertz} Hz is {samples}.")
        if not 0.0 < self.low_frequency_hertz < self.sample_rate_hertz / 2.0:
            raise ValueError(f"The low frequency {self.low_frequency_hertz} Hz is not inside the band "
                             f"(0, {self.sample_rate_hertz / 2.0}) Hz.")

    @property
    def filter_samples(self) -> int:
        return int(self.filter_seconds * self.sample_rate_hertz)

    @property
    def block_samples(self) -> int:
        """The samples of one block of white noise, and of one coloured block."""
        return BLOCK_FILTERS * self.filter_samples

    def kernel(self, ifo: IFO) -> np.ndarray:
        """
        The colouring kernel of ``ifo``: ``filter_samples`` taps at lags
        -filter_samples / 2 to filter_samples / 2 - 1, in that order, so that the noise is
        x[n] = sum_j kernel[j] w[n - (j - filter_samples / 2)] for unit white noise w.

        As PyCBC's ``inverse_spectrum_truncation`` builds it: the curve's ASD on the grid of
        twice the filter's length, times sqrt(rate / 2) (unit-variance samples to a one-sided
        density of 1 / Hz), zero below the low frequency; its zero-phase kernel; and that
        kernel's middle ``filter_samples`` taps under a Hann window of as many.
        """
        rate, taps = self.sample_rate_hertz, self.filter_samples
        frequencies = np.fft.rfftfreq(2 * taps, 1.0 / rate)
        curve_frequencies, curve_asd = read_noise_curve(self.profile.path(ifo), up_to_hertz=rate / 2.0)
        asd = np.where(frequencies >= self.low_frequency_hertz,
                       np.interp(frequencies, curve_frequencies, curve_asd), 0.0)
        circular = np.fft.irfft(np.sqrt(rate / 2.0) * asd, n=2 * taps)
        half = taps // 2
        window = np.hanning(taps)
        # Lags -half .. -1 then 0 .. half - 1, as PyCBC windows them: the rising half of
        # the window on the negative lags and the falling half on the rest.
        return np.concatenate([circular[-half:] * window[:half], circular[:half] * window[half:]])

    def reader(self, tree: SeedTree, split: str, ifo: IFO) -> "NoiseReader":
        """A reader of ``ifo``'s noise from ``tree``'s streams of ``split``."""
        return NoiseReader(self, tree, split, ifo)


class NoiseReader:
    """
    One detector's stationary noise, read by absolute sample index. It keeps the white and
    coloured blocks it made last, so reading a stream in order makes each block once.
    """

    #: Blocks of each kind kept: a span of up to one block, read in order.
    KEPT_BLOCKS = 3

    def __init__(self, noise: StationaryNoise, tree: SeedTree, split: str, ifo: IFO):
        self.noise, self.tree, self.split, self.ifo = noise, tree, split, ifo
        self._consumer = f"stream/noise/{ifo.name}"
        taps, block = noise.filter_samples, noise.block_samples
        # The kernel on the circle of one block plus one filter, its lags wrapped.
        circle = np.zeros(block + taps)
        lags = np.arange(-(taps // 2), taps // 2)
        circle[lags % circle.size] = noise.kernel(ifo)
        self._response = np.fft.rfft(circle)
        self._white: Dict[int, np.ndarray] = {}
        self._coloured: Dict[int, np.ndarray] = {}

    def read(self, first: int, count: int) -> np.ndarray:
        """Samples ``first`` to ``first + count`` of the noise, float64; sample n is at n / rate s."""
        first, count = int(first), int(count)
        if first < 0 or count < 0:
            raise ValueError(f"A read starts at a sample at or after GPS 0 and has a length; got {first}, {count}.")
        block = self.noise.block_samples
        out = np.empty(count)
        for b in range(first // block, (first + count - 1) // block + 1) if count else ():
            lo, hi = max(first, b * block), min(first + count, (b + 1) * block)
            out[lo - first:hi - first] = self._coloured_block(b)[lo - b * block:hi - b * block]
        return out

    def _white_block(self, b: int) -> np.ndarray:
        if b not in self._white:
            self._keep(self._white)
            rng = self.tree.stream(self.split, self._consumer, k=b)
            self._white[b] = rng.standard_normal(self.noise.block_samples)
        return self._white[b]

    def _coloured_block(self, b: int) -> np.ndarray:
        if b not in self._coloured:
            if b == 0:
                raise ValueError("Noise is read from the second block of GPS time on (its filter reaches back one).")
            half = self.noise.filter_samples // 2
            # White noise from half a filter before the block to half a filter after it.
            white = np.concatenate([self._white_block(b - 1)[-half:], self._white_block(b),
                                    self._white_block(b + 1)[:half]])
            coloured = np.fft.irfft(np.fft.rfft(white) * self._response, n=white.size)
            self._keep(self._coloured)
            self._coloured[b] = coloured[half:half + self.noise.block_samples]
        return self._coloured[b]

    def _keep(self, blocks: Dict[int, np.ndarray]) -> None:
        """Make room for one block: drop the oldest beyond ``KEPT_BLOCKS``."""
        while len(blocks) >= self.KEPT_BLOCKS:
            blocks.pop(next(iter(blocks)))
