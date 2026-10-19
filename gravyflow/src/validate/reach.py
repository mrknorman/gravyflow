"""
The reach of a CBC population, per mass: the chirp distance beyond which no
source of it can have a given SNR, in noise of a given PSD.

A source's SNR is bounded, over every sky position, polarisation angle and
arrival time, by the network's best response: every sky position and
polarisation angle gives an interferometer antenna factors with
``F+^2 + Fx^2 <= 1``, so a detector's SNR is at most the square root of the
largest eigenvalue of the polarisations' Gram matrix (``<h+|h+>``, ``<h+|hx>``,
``<hx|hx>``) in its noise, and the network SNR at most the root sum of those
eigenvalues over the detectors (`network_snr_bound`): exact for one detector (a
source overhead), and above any real network's best sky for several.

The reach is per mass (`chirp_reach`): sources are placed by *chirp distance*,
``d_c = d (M_ref / M)^(5/6)`` (``M`` the chirp mass, ``M_ref`` two neutron stars of
1.4), the distance at which a source of the reference chirp mass would have the
same inspiral amplitude, so a reach ``R`` in chirp distance is a physical reach of
``R (M / M_ref)^(5/6)`` for chirp mass ``M``. ``R`` is the largest, over the
population's box, of the best-case SNR at a reference distance times
``(M_ref / M)^(5/6)``, over the reach SNR: every source of every mass then has at
most the reach SNR at its own mass's reach. Where a mass's best case grows more
slowly than the inspiral's ``M^(5/6)`` (a heavy binary merges below the band; a
light one's inspiral outlasts the window), its reach lies past where its best case
has the reach SNR: injections there are spent, never lost.

The box: each component mass over its support, searched on a grid refined around
its best point (sources do not grow louder with mass monotonically); each spin's
candidates, its aligned component at its bounds (and zero between them) and, for a
precessing population, its in-plane magnitude at zero and at its bound (the in-plane
spin's azimuth is a turn of the orbital phase, which the orientations search); the
lowest band edge; and the orientation. For an aligned-spin source of the dominant
(2, 2) mode, face-on is the loudest orientation, exactly. For any other (precessing,
or with higher modes) no orientation is the loudest in general, so the bound is a
search over a grid of inclinations and orbital phases (`ORIENTATIONS`), at the
`ORIENTATION_CANDIDATES` best points of the masses and spins, before the masses are
refined: a measured best case, not a proved bound.

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
from gravyflow.src.dataset.features.injection import REFERENCE_CHIRP_MASS_MSUN, chirp_mass

#: The distance (Mpc) the sources are generated at; SNR goes as one over distance.
REFERENCE_DISTANCE_MPC = 100.0
#: Mass grid points per component, log-spaced over its support; per component around the
#: loudest point, in each of the refining rounds.
GRID_POINTS = 16
REFINE_POINTS = 9
REFINE_ROUNDS = 3
#: Waveforms generated per call, which bounds the memory of one call.
CHUNK = 128
#: The orientation grid searched where face-on is not the loudest: (inclination, orbital phase) pairs.
ORIENTATIONS = np.array(list(product(np.linspace(0.0, np.pi, 7), np.linspace(0.0, np.pi, 4, endpoint=False))))
#: How many of the best mass and spin points the orientation grid is searched at.
ORIENTATION_CANDIDATES = 4
#: Approximants of aligned spins and the (2, 2) mode alone, whose loudest orientation is face-on.
FACE_ON_APPROXIMANTS = (gf.Approximant.IMRPhenomD, gf.Approximant.IMRPhenomXAS, gf.Approximant.TaylorF2,
                        gf.Approximant.IMRPhenomD_NRTidalv2)


def population_box(generator) -> Dict[str, object]:
    """
    The values the reach searches over, per parameter of ``generator`` (a
    ``gf.CBCGenerator``): each component mass's (low, high); each spin's candidate
    vectors (x, y, z); the lowest band edge; each tidal deformability's lowest value;
    and the orientations, (inclination, phase) pairs: face-on alone for an aligned
    (2, 2)-mode approximant, else `ORIENTATIONS` (module docstring). Refuses a
    population with an unbounded mass: it has no loudest source.
    """
    from gravyflow.src.dataset.features.injection import get_support_bounds

    box = {}
    for attribute in ("mass_1_msun", "mass_2_msun"):
        low, high = get_support_bounds(getattr(generator, attribute), attribute)
        if not (np.isfinite(low) and np.isfinite(high) and low > 0):
            raise ValueError(
                f"The population's {attribute} is unbounded, so it has no loudest source: set "
                "ValidationConfig.volume_reach_mpc."
            )
        box[attribute] = (low, high)
    for attribute in ("spin_1_in", "spin_2_in"):
        value = getattr(generator, attribute)
        components = (0.0, 0.0, value) if isinstance(value, gf.Distribution) else tuple(value)
        bounds = [get_support_bounds(c, f"{attribute}[{j}]") for j, c in enumerate(components)]
        low, high = bounds[2]
        aligned = sorted({low, high} | ({0.0} if low < 0.0 < high else set()))
        in_plane = np.hypot(*(max(abs(b) for b in bound) for bound in bounds[:2]))
        box[attribute] = tuple((float(x), 0.0, float(z)) for x in sorted({0.0, in_plane}) for z in aligned)
    box["min_frequency_hertz"] = (get_support_bounds(generator.min_frequency_hertz, "min_frequency_hertz")[0],)
    for attribute in ("lambda_1", "lambda_2"):
        box[attribute] = (get_support_bounds(getattr(generator, attribute), attribute)[0],)
    approximant = gf.Approximant(generator.approximant)
    precessing = any(spin[0] != 0.0 for attribute in ("spin_1_in", "spin_2_in") for spin in box[attribute])
    face_on = approximant in FACE_ON_APPROXIMANTS and not precessing
    box["orientations"] = np.zeros((1, 2)) if face_on else ORIENTATIONS
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


def _chirp_scaled_snrs(generator, box, points, psds, sample_rate_hertz, window_seconds, config) -> np.ndarray:
    """
    Each point's best-case network SNR at the reference distance times ``(M_ref / M)^(5/6)``:
    ``points`` rows are (m_a, m_b, spin_a index, spin_b index, inclination, phase).
    """
    from gravyflow.src.dataset.features.waveforms.cbc import _generate_cbc_window
    from gravyflow.src.rng import platform
    import jax

    points = np.asarray(points, dtype=np.float64)
    spins = [np.asarray(box["spin_1_in"]), np.asarray(box["spin_2_in"])]
    values = np.empty(len(points))
    for start in range(0, len(points), CHUNK):
        chunk = points[start:start + CHUNK]
        n = len(chunk)
        m_a, m_b = chunk[:, 0], chunk[:, 1]
        spin_a, spin_b = spins[0][chunk[:, 2].astype(int)], spins[1][chunk[:, 3].astype(int)]
        # Body 1 is the heavier, carrying its spin, as the generator labels it.
        swap = (m_a < m_b)[:, None]
        with jax.default_device(platform.cbc_device()):
            waveforms = _generate_cbc_window(
                num_waveforms=n,
                sample_rate_hertz=float(sample_rate_hertz),
                window_seconds=2.0 * float(window_seconds),
                min_frequency_hertz=np.full(n, box["min_frequency_hertz"][0]),
                approximant=generator.approximant,
                mass_1_msun=np.maximum(m_a, m_b), mass_2_msun=np.minimum(m_a, m_b),
                spin_1_in=np.where(swap, spin_b, spin_a), spin_2_in=np.where(swap, spin_a, spin_b),
                lambda_1=np.full(n, box["lambda_1"][0]), lambda_2=np.full(n, box["lambda_2"][0]),
                distance_mpc=np.full(n, REFERENCE_DISTANCE_MPC),
                inclination_radians=chunk[:, 4],
                reference_orbital_phase_in=chunk[:, 5],
            )
        waveforms = np.nan_to_num(np.asarray(waveforms), nan=0.0, posinf=0.0, neginf=0.0)
        values[start:start + n] = (network_snr_bound(waveforms, psds, sample_rate_hertz, config)
                                   * (REFERENCE_CHIRP_MASS_MSUN / chirp_mass(m_a, m_b)) ** (5.0 / 6.0))
    return values


def loudest_chirp_scaled_snr(generator, psds, sample_rate_hertz: float, window_seconds: float, config) -> float:
    """
    The largest, over ``generator``'s population, of a source's best-case network SNR in
    a scored window of ``window_seconds``, at `REFERENCE_DISTANCE_MPC`, in detectors of
    PSDs ``psds`` over ``config``'s band, times ``(M_ref / M)^(5/6)`` (module docstring).
    """
    box = population_box(generator)
    measure = lambda points: _chirp_scaled_snrs(generator, box, points, psds, sample_rate_hertz, window_seconds,
                                                config)
    grids = [np.geomspace(*box[m], GRID_POINTS) if box[m][1] > box[m][0] else np.array(box[m][:1])
             for m in ("mass_1_msun", "mass_2_msun")]
    spins = list(product(range(len(box["spin_1_in"])), range(len(box["spin_2_in"]))))
    face_on = box["orientations"][0]
    coarse = np.array([(m_a, m_b, s_a, s_b, *face_on) for m_a, m_b in product(*grids) for s_a, s_b in spins])
    values = measure(coarse)
    if len(box["orientations"]) > 1:
        # No orientation is the loudest in general: search them at the best mass and spin points.
        best = coarse[np.argsort(values)[::-1][:ORIENTATION_CANDIDATES]]
        oriented = np.array([(*point[:4], *orientation) for point in best for orientation in box["orientations"]])
        coarse, values = np.concatenate([coarse, oriented]), np.concatenate([values, measure(oriented)])
    value, best = float(values.max()), coarse[int(np.argmax(values))]

    def around(grid, point):
        # The grid cells either side of the best point.
        j = int(np.argmin(np.abs(grid - point)))
        return np.geomspace(grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)], REFINE_POINTS) \
            if len(grid) > 1 else grid

    # Each round narrows the mass cells fourfold around the best point: after
    # REFINE_ROUNDS, a mass step of under 1% of the coarse grid's, where the SNR is flat.
    for _ in range(REFINE_ROUNDS):
        grids = [around(grid, point) for grid, point in zip(grids, best[:2])]
        fine = np.array([(m_a, m_b, *best[2:]) for m_a, m_b in product(*grids)])
        refined = measure(fine)
        if refined.max() > value:
            value, best = float(refined.max()), fine[int(np.argmax(refined))]
    return value


def chirp_reach(generator, psds, sample_rate_hertz: float, window_seconds: float, config, snr: float) -> float:
    """
    The population's reach (Mpc, chirp distance): the chirp distance beyond which no
    source of any mass has network SNR ``snr``, whatever its sky position,
    orientation and parameters (`loudest_chirp_scaled_snr`); a source of chirp mass
    ``M`` reaches ``R (M / M_ref)^(5/6)`` of physical distance.
    """
    return REFERENCE_DISTANCE_MPC * loudest_chirp_scaled_snr(generator, psds, sample_rate_hertz, window_seconds,
                                                             config) / snr
