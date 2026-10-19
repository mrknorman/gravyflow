"""
Aligned-spin template banks: spins through the templates and the bank, the
placement's metric shortlist, and the streaming span's match against PyCBC's
overlap.
"""
import json

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.detection import TemplateBank
from gravyflow.src.detection.stream import stream_bank_placement
from gravyflow.src.detection.template_grid import max_matches

SPACE = stream_bank_placement().space


def test_each_spin_stays_with_its_mass():
    bank = TemplateBank.from_masses([10.0, 30.0], [20.0, 25.0], [0.5, -0.2], [-0.7, 0.1])
    np.testing.assert_array_equal(bank.mass_1, [20.0, 30.0])
    np.testing.assert_array_equal(bank.spin_1z, [-0.7, -0.2])
    np.testing.assert_array_equal(bank.spin_2z, [0.5, 0.1])
    assert np.all(TemplateBank.from_masses([30.0], [25.0]).spin_1z == 0.0)
    record = json.loads(json.dumps(bank.to_dict()))
    np.testing.assert_array_equal(TemplateBank.from_dict(record).spin_2z, bank.spin_2z)


def test_aligned_spins_reach_the_templates():
    # A light binary's aligned spins change its phasing enough that a
    # non-spinning template misses it; the same spins match exactly.
    spinning = SPACE.whitened_spectra([7.0], [5.0], [0.8], [0.8])
    still = SPACE.whitened_spectra([7.0], [5.0])
    assert max_matches(spinning, still)[0, 0] < 0.9
    assert max_matches(spinning, SPACE.whitened_spectra([7.0], [5.0], [0.8], [0.8]))[0, 0] > 0.9999


def test_the_metric_coordinates_predict_the_mismatch_of_near_templates():
    # For inspiral-dominated binaries the squared distance in the metric
    # coordinates is the mismatch to within a factor of two (measured: 0.58
    # to 1.3 for these pairs), which is all the placement's shortlist needs.
    base = (np.array([10.0, 10.0, 30.0, 6.0]), np.array([8.0, 8.0, 20.0, 5.5]),
            np.array([0.0, 0.0, 0.2, 0.9]), np.zeros(4))
    near = (base[0] * np.array([1.01, 1.0, 1.02, 1.003]), base[1], base[2] + np.array([0.0, 0.05, 0.0, 0.02]), base[3])
    mismatch = 1.0 - np.diag(max_matches(SPACE.whitened_spectra(*base), SPACE.whitened_spectra(*near)))
    distance = np.sum((SPACE.metric_coordinates(*base) - SPACE.metric_coordinates(*near)) ** 2, axis=1)
    ratio = distance / mismatch
    assert np.all((ratio > 0.5) & (ratio < 2.0)), (mismatch, distance)


def test_a_spinning_placement_reaches_its_minimal_match():
    # The placement in the metric it places with, with spins and the metric
    # shortlist, on the window-mode span (fast): the stopping rule leaves
    # little of the range below the minimal match. A shortlisted fitting
    # factor is a lower bound of the exact one, and here equals it.
    bank = TemplateBank.stochastic(mass_range=(20.0, 40.0), spin_range=0.5, neighbours=16,
                                   convergence_candidates=512, use_bundled=False)
    assert bank.spin_1z.min() == -0.5 and bank.spin_1z.max() == 0.5  # the spin corners are swept
    rng = np.random.default_rng(9)
    draws = bank.placement.proposal(rng, 60)
    exact = bank.fitting_factors(draws[0], draws[1], None, draws[2], draws[3])
    shortlisted = bank.fitting_factors(draws[0], draws[1], None, draws[2], draws[3], neighbours=16)
    assert np.median(exact) >= 0.97 and np.min(exact) >= 0.95, (np.median(exact), np.min(exact))
    assert np.all(shortlisted <= exact + 1e-6) and np.mean(shortlisted == exact) >= 0.95


def test_the_bank_metric_with_spins_is_pycbcs_overlap():
    # PyCBC's match (its own overlap, on the O4 high table it interpolates
    # itself) of the streaming span's spinning templates equals the bank's.
    import pycbc.filter
    import pycbc.psd
    import pycbc.types
    a = ([7.0, 30.0, 60.0], [5.0, 12.0, 40.0], [0.9, -0.5, 0.3], [0.2, 0.0, -0.8])
    b = ([7.05, 31.0, 62.0], [5.0, 11.7, 40.0], [0.85, -0.45, 0.35], [0.25, 0.1, -0.7])
    own = np.diag(max_matches(SPACE.whitened_spectra(*a), SPACE.whitened_spectra(*b)))
    n = SPACE.num_samples
    frequencies, asd = np.loadtxt(gf.NoiseProfile.O4_HIGH.path(gf.IFO.L1), delimiter=",", comments="#", unpack=True)
    psd = pycbc.psd.from_numpy_arrays(frequencies, asd ** 2, n // 2 + 1, SPACE.sample_rate_hertz / n, 10.3)
    series = lambda rows: [pycbc.types.TimeSeries(row.astype(np.float64), delta_t=1 / SPACE.sample_rate_hertz)
                           for row in SPACE.templates(*rows)]
    theirs = [pycbc.filter.match(x, y, psd=psd, low_frequency_cutoff=20.0)[0] for x, y in zip(series(a), series(b))]
    np.testing.assert_allclose(own, theirs, atol=1e-3)
    assert own.min() < 0.97 < own.max()  # either side of the minimal match


def test_the_placements_row_store_reads_like_one_array():
    # The placement keeps its spectra in fixed blocks, filled in place, so a
    # large bank is never copied whole; read by index it is one array.
    from gravyflow.src.detection.template_grid import _RowStore
    rng = np.random.default_rng(3)
    rows = (rng.standard_normal((23, 5)) + 1j * rng.standard_normal((23, 5))).astype(np.complex64)
    store = _RowStore(5, block_rows=4)
    for part in np.split(rows, [3, 4, 11, 12]):
        store.extend(part)
    assert len(store) == 23 and len(store.blocks) == 6
    index = rng.integers(0, 23, size=40)
    np.testing.assert_array_equal(store[index], rows[index])
    np.testing.assert_array_equal(store[:], rows)
    with pytest.raises(IndexError):
        store[np.array([23])]


def test_fitting_factors_by_bank_chunk_equal_the_whole_bank():
    # Fitting factors read the bank a chunk at a time (a large bank's spectra
    # do not fit in memory); with every template, or a metric shortlist, they
    # equal the matches with the whole bank at once.
    rng = np.random.default_rng(4)
    bank = TemplateBank.from_masses(*(values for values in (
        rng.uniform(20, 40, 21), rng.uniform(20, 40, 21), rng.uniform(-0.5, 0.5, 21), rng.uniform(-0.5, 0.5, 21))))
    bank = TemplateBank(*bank.parameters, placement=stream_bank_placement(mass_range=(20.0, 40.0), spin_range=0.5))
    draws = (rng.uniform(20, 40, 5), rng.uniform(20, 40, 5), rng.uniform(-0.5, 0.5, 5), rng.uniform(-0.5, 0.5, 5))
    whole = np.max(max_matches(SPACE.whitened_spectra(*draws), SPACE.whitened_spectra(*bank.parameters)), axis=1)
    np.testing.assert_allclose(bank.fitting_factors(draws[0], draws[1], None, draws[2], draws[3], chunk=2),
                               whole, atol=1e-6)
    near = bank.fitting_factors(draws[0], draws[1], None, draws[2], draws[3], neighbours=21, chunk=2)
    np.testing.assert_allclose(near, whole, atol=1e-6)
