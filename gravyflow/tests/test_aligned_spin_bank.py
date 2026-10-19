"""
Aligned-spin template banks: spins through the templates and the bank, the
placement's metric shortlist, and the streaming filter's bundled bank (its
key, its range, its coverage by its own metric and by PyCBC's overlap).
"""
import json

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.detection import TemplateBank
from gravyflow.src.detection.stream import STREAM_BANK_MAX_TEMPLATES, StreamingMatchedFilter, stream_bank_placement
from gravyflow.src.detection.template_grid import MatchSpace, max_matches

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


# ---------------------------------------------------------------------------
# The streaming filter's bundled bank
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bundled():
    return TemplateBank.place(stream_bank_placement(), max_templates=STREAM_BANK_MAX_TEMPLATES, use_bundled=True)


def test_the_streaming_filters_default_bank_is_bundled_and_names_what_it_depends_on(bundled):
    from gravyflow.src.detection.bundle_banks import BUNDLED
    from gravyflow.src.detection.template_grid import BUNDLED_BANK_DIRECTORY
    from gravyflow.src.resources import PACKAGE
    name = "bbh_aligned_5_95_s0.99_mm0.97_2048hz_24s_20hz_o4high.json"
    assert BUNDLED[name] is stream_bank_placement
    record = json.loads((PACKAGE / BUNDLED_BANK_DIRECTORY / name).read_text())
    key = json.loads(json.dumps(stream_bank_placement().key()))
    assert record["arguments"] == key, \
        "a stale bundled bank: place it again with python -m gravyflow.src.detection.bundle_banks"
    assert key["space"]["noise_profile_name"] == "O4_HIGH" and key["space"]["visible_seconds"] == 24.0
    assert key["spin_range"] == 0.99 and key["minimal_match"] == 0.97 and key["mass_range"] == [5.0, 95.0]
    assert bundled.to_dict() == record
    assert StreamingMatchedFilter().bank.to_dict() == record


def test_the_bundled_bank_spans_its_range(bundled):
    # Boundary sweeps put templates on every edge and spin corner.
    m1, m2, s1, s2 = bundled.parameters
    assert np.all(m1 >= m2) and m2.min() == 5.0 and m1.max() == 95.0
    for spin in (s1, s2):
        assert spin.min() == -0.99 and spin.max() == 0.99


def test_the_bundled_bank_covers_its_boundary(bundled):
    # The mass edges (m2 = 5, m1 = 95) at each spin corner are swept in
    # order, so between the placement's own boundary candidates the fitting
    # factor stays near the minimal match (the window-mode banks' bound).
    edge = np.exp(np.linspace(np.log(5.0), np.log(95.0), 25))
    m1, m2, s1, s2 = [], [], [], []
    for a in (0.99, -0.99):
        for b in (0.99, -0.99):
            m1 += [edge, np.full(25, 95.0)]
            m2 += [np.full(25, 5.0), edge]
            s1 += [np.full(50, a)]
            s2 += [np.full(50, b)]
    m1, m2, s1, s2 = (np.concatenate(v) for v in (m1, m2, s1, s2))
    ff = bundled.fitting_factors(m1, m2, None, s1, s2, neighbours=32)
    worst = np.argsort(ff)[:3]
    assert ff.min() >= 0.965, [(m1[i], m2[i].round(2), s1[i], s2[i], ff[i].round(4)) for i in worst]


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
