"""
The reach of a CBC population: the distance beyond which none of its sources
can have a given SNR, in noise of a given PSD.

A source's SNR in one detector is largest when it is face-on and the detector's
response to the two polarisations is largest. Every sky position and
polarisation angle gives an interferometer antenna factors with
``F+^2 + Fx^2 <= 1``, so the detector's SNR is at most the square root of the
largest eigenvalue of the polarisations' Gram matrix (``<h+|h+>``, ``<h+|hx>``,
``<hx|hx>``) in its noise, and the network SNR at most the root sum of those
eigenvalues over the detectors: a bound on every sky position, polarisation and
arrival time, exact for one detector (a source overhead), and above any real
network's best sky for several. The bound is maximised over the population's
box: each component mass over its support, each aligned spin at its bounds (and
zero, when the support holds it), the lowest band edge, face-on. Sources do not
grow louder with mass or spin monotonically (a heavy binary merges below the
band), so the masses are searched on a grid, refined around its loudest point.

Each source is generated on a window twice the scored one, its merger at the
centre, so every placement's visible part is inside it: the SNR a scored window
shows is at most the SNR computed here. Distances are Euclidean and masses in the
detector frame, as the population's generator draws them.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, Tuple

import numpy as np

import gravyflow as gf

#: The distance (Mpc) the loudest source is generated at; SNR goes as one over distance.
REFERENCE_DISTANCE_MPC = 100.0
#: Mass grid points per component, log-spaced over its support; per component around the
#: loudest point, in each of the refining rounds.
GRID_POINTS = 16
REFINE_POINTS = 9
REFINE_ROUNDS = 3
#: Waveforms generated per call, which bounds the memory of one call.
CHUNK = 128


def population_box(generator) -> Dict[str, Tuple[float, ...]]:
    """
    The values the reach searches over, per parameter of ``generator`` (a
    ``gf.CBCGenerator``): each component mass's (low, high), each aligned spin's
    bounds (and 0 between them), the lowest band edge, and each tidal
    deformability's lowest value. Refuses a population with no loudest source
    the box can find: an unbounded mass, or spins out of the orbital plane's
    normal (precession: face-on is then no bound).
    """
    from gravyflow.src.dataset.features.injection import get_support_bounds

    box = {}
    for attribute in ("mass_1_msun", "mass_2_msun"):
        low, high = get_support_bounds(getattr(generator, attribute), attribute)
        if not (np.isfinite(low) and np.isfinite(high) and low > 0):
            raise ValueError(
                f"The population's {attribute} is unbounded, so it has no loudest source: set "
                "ValidationConfig.volume_max_distance_mpc."
            )
        box[attribute] = (low, high)
    for attribute in ("spin_1_in", "spin_2_in"):
        value = getattr(generator, attribute)
        components = (0.0, 0.0, value) if isinstance(value, gf.Distribution) else tuple(value)
        bounds = [get_support_bounds(c, f"{attribute}[{j}]") for j, c in enumerate(components)]
        if any(bound != (0.0, 0.0) for bound in bounds[:2]):
            raise ValueError(
                f"The population's {attribute} has in-plane components (precession), for which a "
                "face-on source is not the loudest: set ValidationConfig.volume_max_distance_mpc."
            )
        low, high = bounds[2]
        box[attribute] = tuple(sorted({low, high} | ({0.0} if low < 0.0 < high else set())))
    box["min_frequency_hertz"] = (get_support_bounds(generator.min_frequency_hertz, "min_frequency_hertz")[0],)
    for attribute in ("lambda_1", "lambda_2"):
        box[attribute] = (get_support_bounds(getattr(generator, attribute), attribute)[0],)
    return box


def network_snr_bound(polarisations, psds, sample_rate_hertz: float, config) -> np.ndarray:
    """
    The largest network SNR of each source (``polarisations``, (P, 2, T)) over
    every sky position and polarisation angle, in detectors of one-sided PSDs
    ``psds`` ((I, F), strain^2 / Hz from DC to Nyquist), over ``config``'s band
    (a ``gf.SpectralConfig``): the root sum over detectors of the largest
    eigenvalue of the polarisations' Gram matrix (module docstring). (P,).
    """
    from gravyflow.src.dataset.tools.snr import template_sigma

    plus = np.asarray(polarisations[:, 0], dtype=np.float32)
    cross = np.asarray(polarisations[:, 1], dtype=np.float32)
    total = np.zeros(plus.shape[0])
    for psd in np.asarray(psds, dtype=np.float64):
        def power(h):
            sigma = template_sigma(h, psd, sample_rate_hertz, config.lower_frequency_hertz,
                                   config.upper_frequency_hertz)
            return np.asarray(sigma, dtype=np.float64) ** 2

        a, c = power(plus), power(cross)
        b = 0.5 * (power(plus + cross) - a - c)
        total += 0.5 * (a + c) + np.sqrt(0.25 * (a - c) ** 2 + b**2)
    return np.sqrt(total)


def _loudest(generator, box, masses, psds, sample_rate_hertz, window_seconds, config):
    """The loudest of the (m_a, m_b, chi_a, chi_b) points ``masses`` at the reference distance: (snr, point)."""
    from gravyflow.src.dataset.features.waveforms.cbc import _generate_cbc_window
    from gravyflow.src.rng import platform
    import jax

    points = np.asarray(masses, dtype=np.float64)
    snrs = np.empty(len(points))
    for start in range(0, len(points), CHUNK):
        chunk = points[start:start + CHUNK]
        n = len(chunk)
        m_a, m_b, chi_a, chi_b = chunk.T
        # Body 1 is the heavier, carrying its spin, as the generator labels it.
        swap = m_a < m_b
        m_1, m_2 = np.where(swap, m_b, m_a), np.where(swap, m_a, m_b)
        chi_1, chi_2 = np.where(swap, chi_b, chi_a), np.where(swap, chi_a, chi_b)
        spin = lambda chi: np.stack([np.zeros(n), np.zeros(n), chi], axis=-1)
        with jax.default_device(platform.cbc_device()):
            waveforms = _generate_cbc_window(
                num_waveforms=n,
                sample_rate_hertz=float(sample_rate_hertz),
                window_seconds=2.0 * float(window_seconds),
                min_frequency_hertz=np.full(n, box["min_frequency_hertz"][0]),
                approximant=generator.approximant,
                mass_1_msun=m_1, mass_2_msun=m_2,
                spin_1_in=spin(chi_1), spin_2_in=spin(chi_2),
                lambda_1=np.full(n, box["lambda_1"][0]), lambda_2=np.full(n, box["lambda_2"][0]),
                distance_mpc=np.full(n, REFERENCE_DISTANCE_MPC),
                inclination_radians=np.zeros(n),
                reference_orbital_phase_in=np.zeros(n),
            )
        waveforms = np.nan_to_num(np.asarray(waveforms), nan=0.0, posinf=0.0, neginf=0.0)
        snrs[start:start + n] = network_snr_bound(waveforms, psds, sample_rate_hertz, config)
    best = int(np.argmax(snrs))
    return float(snrs[best]), points[best]


def loudest_snr(generator, psds, sample_rate_hertz: float, window_seconds: float, config) -> float:
    """
    The largest network SNR any source of ``generator``'s population can show in
    a scored window of ``window_seconds``, at `REFERENCE_DISTANCE_MPC`, in
    detectors of PSDs ``psds``, over ``config``'s band (module docstring).
    """
    box = population_box(generator)
    grids = [np.geomspace(*box[m], GRID_POINTS) if box[m][1] > box[m][0] else np.array(box[m][:1])
             for m in ("mass_1_msun", "mass_2_msun")]
    coarse = list(product(grids[0], grids[1], box["spin_1_in"], box["spin_2_in"]))
    snr, best = _loudest(generator, box, coarse, psds, sample_rate_hertz, window_seconds, config)

    def around(grid, value):
        # The grid cells either side of the loudest point.
        j = int(np.argmin(np.abs(grid - value)))
        return np.geomspace(grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)], REFINE_POINTS) \
            if len(grid) > 1 else grid

    # Each round narrows the mass cells fourfold around the loudest point: after
    # REFINE_ROUNDS, a mass step of under 1% of the coarse grid's, where the SNR is flat.
    for _ in range(REFINE_ROUNDS):
        grids = [around(grid, value) for grid, value in zip(grids, best[:2])]
        fine = list(product(grids[0], grids[1], [best[2]], [best[3]]))
        refined, point = _loudest(generator, box, fine, psds, sample_rate_hertz, window_seconds, config)
        if refined > snr:
            snr, best = refined, point
    return snr


def horizon_distance(generator, psds, sample_rate_hertz: float, window_seconds: float, config,
                     snr: float) -> float:
    """
    The population's reach (Mpc): the distance at which its loudest possible
    source (`loudest_snr`) has network SNR ``snr``, so that every source beyond
    it has less, whatever its sky position, orientation and parameters.
    """
    return REFERENCE_DISTANCE_MPC * loudest_snr(generator, psds, sample_rate_hertz, window_seconds, config) / snr
