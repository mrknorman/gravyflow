"""
CBC windows from ``_generate_cbc_window``: batching, input forms, refusal, and
where the merger sits.

How the waveforms match LALSimulation's is ``test_cbc_lal_parity.py``'s, and
the chirp-time helpers are ``test_cbc_units.py``'s.
"""
import numpy as np
import pytest
from keras import ops

from gravyflow.src.dataset.features.waveforms.cbc import _generate_cbc_window

SAMPLE_RATE = 2048.0


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


def test_generate_cbc_window_puts_the_merger_at_the_centre():
    """The merger is sample K // 2 of the K-sample window, whatever the window."""
    for window_seconds in (1.0, 1.5, 3.0):
        waveforms = np.asarray(_generate_cbc_window(
            num_waveforms=1,
            sample_rate_hertz=2048.0,
            window_seconds=window_seconds,
            min_frequency_hertz=None,
            mass_1_msun=30.0,
            mass_2_msun=30.0,
            approximant="IMRPhenomD"
        ))
        num_samples = int(2048.0 * window_seconds)
        assert waveforms.shape == (1, 2, num_samples)
        peak = int(np.argmax(np.sum(waveforms[0] ** 2, axis=0)))
        assert abs(peak - num_samples // 2) <= 10, (window_seconds, peak, num_samples // 2)
