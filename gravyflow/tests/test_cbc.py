"""
CBC windows from ``_generate_cbc_window``: batching, input forms, refusal, and
where the merger sits.

How the waveforms match LALSimulation's is ``test_cbc_lal_parity.py``'s, and
the chirp-time helpers are ``test_cbc_units.py``'s.
"""
import lal
import lalsimulation
import numpy as np
import pytest
from keras import ops

from gravyflow.src.dataset.features.waveforms.cbc import _generate_cbc_window

SAMPLE_RATE = 2048.0
F_LOW = 20.0
F_REF = 20.0


def test_each_row_of_a_batch_is_its_source_generated_alone():
    """
    Row i of a batch is source i generated on its own, whatever form each input
    takes: per-row arrays, a flat spin list, a (num_waveforms, 3) spin array, a
    tensor, a Python float.
    """
    mass_1 = np.array([30.0, 12.0, 45.0])
    mass_2 = np.array([25.0, 8.0, 20.0])
    spin_1 = np.array([[0.1, 0.0, 0.3], [0.0, 0.2, -0.4], [0.3, -0.1, 0.5]])
    spin_2 = np.array([[0.0, 0.0, 0.2], [0.1, 0.0, 0.0], [-0.2, 0.2, 0.1]])
    inclination = np.array([0.3, 1.1, 2.0])
    distance = np.array([300.0, 500.0, 900.0])
    phase = np.array([0.1, 2.0, 4.0])

    batch = np.asarray(_generate_cbc_window(
        num_waveforms=3,
        sample_rate_hertz=SAMPLE_RATE,
        window_seconds=2.0,
        min_frequency_hertz=None,
        approximant="IMRPhenomPv2",
        mass_1_msun=mass_1,
        mass_2_msun=mass_2,
        spin_1_in=spin_1.ravel().tolist(),
        spin_2_in=spin_2,
        inclination_radians=inclination,
        distance_mpc=distance,
        reference_orbital_phase_in=phase,
    ))
    assert batch.shape == (3, 2, int(2.0 * SAMPLE_RATE))

    for i in range(3):
        alone = np.asarray(_generate_cbc_window(
            num_waveforms=1,
            sample_rate_hertz=SAMPLE_RATE,
            window_seconds=2.0,
            min_frequency_hertz=None,
            approximant="IMRPhenomPv2",
            mass_1_msun=ops.convert_to_tensor(mass_1[i]),
            mass_2_msun=float(mass_2[i]),
            spin_1_in=spin_1[i].tolist(),
            spin_2_in=spin_2[i],
            inclination_radians=float(inclination[i]),
            distance_mpc=float(distance[i]),
            reference_orbital_phase_in=float(phase[i]),
        ))[0]
        # Representation: one float32 result through an inverse FFT of
        # 2 ** 12 bins, so a few ulp per log2 N of the peak. Rows mixed up, or a
        # spin or scalar broadcast wrongly, differ at order the peak itself.
        tolerance = 4 * np.log2(alone.shape[-1]) * np.finfo(np.float32).eps * np.abs(alone).max()
        np.testing.assert_allclose(batch[i], alone, rtol=0.0, atol=tolerance, err_msg=f"row {i}")


def test_an_unknown_approximant_is_refused():
    with pytest.raises(ValueError, match="not supported"):
        _generate_cbc_window(
            num_waveforms=1,
            sample_rate_hertz=SAMPLE_RATE,
            window_seconds=1.0,
            min_frequency_hertz=None,
            approximant="InvalidApproximant",
        )


# (m1, m2, spin_1, spin_2, inclination); precessing spins for IMRPhenomPv2.
ALIGNED_SOURCES = [
    (10.0, 10.0, (0, 0, 0.0), (0, 0, 0.0), 0.0),
    (20.0, 20.0, (0, 0, 0.5), (0, 0, 0.5), 0.4),
    (30.0, 30.0, (0, 0, 0.0), (0, 0, 0.0), 1.0),
    (40.0, 21.0, (0, 0, -0.6), (0, 0, 0.3), 0.8),
    (60.0, 50.0, (0, 0, 0.2), (0, 0, -0.4), 1.2),
]
PRECESSING_SOURCES = [
    (10.0, 10.0, (0.5, 0.2, 0.4), (0.1, -0.6, 0.3), 1.0),
    (30.0, 30.0, (0.3, 0.1, 0.5), (-0.2, 0.3, 0.4), 0.4),
    (40.0, 21.0, (0.7, 0.2, 0.3), (0.1, -0.3, 0.2), 0.8),
]


def _lal_peak_after_merger_samples(approximant, source):
    """
    Where the peak of h+^2 + hx^2 sits relative to LAL's merger time, t = 0 of
    SimInspiralChooseTDWaveform, in samples at SAMPLE_RATE.
    """
    m1, m2, spin_1, spin_2, inclination = source
    hp, hc = lalsimulation.SimInspiralChooseTDWaveform(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *map(float, spin_1), *map(float, spin_2),
        400e6 * lal.PC_SI, inclination, 0.0, 0.0, 0.0, 0.0,
        1.0 / SAMPLE_RATE, F_LOW, F_REF, None,
        lalsimulation.GetApproximantFromString(approximant),
    )
    power = hp.data.data ** 2 + hc.data.data ** 2
    # The epoch is a whole number of samples before t = 0, to LAL's rounding.
    return int(np.argmax(power)) + int(round(float(hp.epoch) * SAMPLE_RATE))


@pytest.mark.parametrize(
    "approximant, sources",
    [
        ("IMRPhenomD", ALIGNED_SOURCES),
        ("IMRPhenomXAS", ALIGNED_SOURCES),
        ("IMRPhenomPv2", PRECESSING_SOURCES),
    ],
)
def test_the_merger_is_at_the_window_centre_as_lal_places_it(approximant, sources):
    """
    The merger is sample K // 2 of the K-sample window, whatever the window: the
    peak of h+^2 + hx^2 sits as far from it as LAL's peak sits from LAL's
    merger time.
    """
    failures = []
    for source in sources:
        expected = _lal_peak_after_merger_samples(approximant, source)
        m1, m2, spin_1, spin_2, inclination = source
        for window_seconds in (1.0, 1.5, 3.0):
            waveform = np.asarray(_generate_cbc_window(
                num_waveforms=1,
                sample_rate_hertz=SAMPLE_RATE,
                window_seconds=window_seconds,
                min_frequency_hertz=None,
                approximant=approximant,
                f_ref=F_REF,
                mass_1_msun=m1,
                mass_2_msun=m2,
                spin_1_in=np.asarray(spin_1, dtype=float),
                spin_2_in=np.asarray(spin_2, dtype=float),
                inclination_radians=inclination,
            ))[0]
            num_samples = int(SAMPLE_RATE * window_seconds)
            assert waveform.shape == (2, num_samples)
            peak = int(np.argmax(np.sum(waveform.astype(np.float64) ** 2, axis=0))) - num_samples // 2
            # Exact to one sample: each peak is the nearest sample to the
            # same continuous peak, and the two grids may round it apart by one.
            if abs(peak - expected) > 1:
                failures.append(f"{m1}+{m2}, {window_seconds} s: peak {peak:+d}, LAL {expected:+d} samples")
    assert not failures, f"{approximant}: merger not where LAL places it: {failures}"
