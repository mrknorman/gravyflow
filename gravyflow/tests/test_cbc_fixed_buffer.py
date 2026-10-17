"""
CBC waveforms are generated on a buffer sized from the requested window alone.

- The buffer does not depend on the masses: one window length, one generator shape.
- Each waveform is whole from ``min_frequency_hertz``, tapered below it, and
  begins no earlier than the buffer, so the kept window matches a long-buffer
  reference.
- The dataset path asks CBC generation for the padded window, not for the whole
  inspiral from ``min_frequency_hertz``.
- The merger stays at the centre of the injection window, moved by its random shift.
"""
import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.features.waveforms import cbc

SAMPLE_RATE = 2048.0
ONSOURCE = 1.0
CROP = 0.5
F_MIN = 20.0


def _uniform(low, high):
    return gf.Distribution(min_=low, max_=high, type_=gf.DistributionType.UNIFORM)


def _record_buffers(monkeypatch):
    """Record the buffer length of every call to the private CBC generator."""
    buffers = []
    generate = cbc._generate_cbc_waveform_jit

    def recording(**kwargs):
        buffers.append(float(kwargs["duration_seconds"]))
        return generate(**kwargs)

    monkeypatch.setattr(cbc, "_generate_cbc_waveform_jit", recording)
    return buffers


def _record_windows(monkeypatch):
    """Record the window every CBCGenerator.generate call is asked for."""
    windows = []
    generate = gf.CBCGenerator.generate

    def recording(self, num_waveforms, sample_rate_hertz, duration_seconds, seed=None):
        windows.append(float(duration_seconds))
        return generate(self, num_waveforms, sample_rate_hertz, duration_seconds, seed)

    monkeypatch.setattr(gf.CBCGenerator, "generate", recording)
    return windows


def _injection_batch(generator, num_examples=2):
    """One batch of the dataset path's injections: 1 s onsource, 0.5 s crop, 2048 Hz."""
    injection_generator = gf.InjectionGenerator(waveform_generators=[generator], seed=42)
    return next(injection_generator(
        sample_rate_hertz=SAMPLE_RATE,
        onsource_duration_seconds=ONSOURCE,
        crop_duration_seconds=CROP,
        num_examples_per_batch=num_examples,
    ))


def test_the_buffer_does_not_depend_on_the_masses(monkeypatch):
    buffers = _record_buffers(monkeypatch)
    for mass in (1.4, 8.0, 60.0):
        generator = gf.CBCGenerator(mass_1_msun=mass, mass_2_msun=mass, network=[gf.IFO.L1])
        waveforms, _ = generator.generate(4, SAMPLE_RATE, 1.0, seed=1)
        assert waveforms.shape == (4, 2, int(SAMPLE_RATE))
        assert np.isfinite(np.asarray(waveforms)).all()
    assert buffers == [2.0, 2.0, 2.0]


def test_the_dataset_path_asks_for_the_padded_window(monkeypatch):
    windows = _record_windows(monkeypatch)
    buffers = _record_buffers(monkeypatch)
    network = [gf.IFO.H1, gf.IFO.L1]
    for low, high in ((1.2, 2.5), (10.0, 60.0)):
        generator = gf.CBCGenerator(
            mass_1_msun=_uniform(low, high), mass_2_msun=_uniform(low, high), network=network
        )
        _injection_batch(generator)

    # The window the injection generator keeps, its shift and delay room, and the
    # room for the random shift: nothing depends on the masses.
    # The placement range's widest reach (the whole window by default).
    padding = max(abs(bound) for bound in generator.placement_range_seconds(ONSOURCE))
    delay = float(gf.Network(network).max_arrival_time_difference_seconds)
    shift_samples = int(padding * SAMPLE_RATE)
    window = (ONSOURCE + 2.0 * CROP) + (2.0 * (padding + delay) + 1.0) + 2.0 * shift_samples / SAMPLE_RATE

    assert windows == pytest.approx([window, window])
    assert buffers == [16.0, 16.0]


def _sources(mass_range, spins, seed, count):
    rng = np.random.default_rng(seed)
    masses = rng.uniform(*mass_range, (2, count))
    if spins == "aligned":
        spin_1 = np.zeros((count, 3))
        spin_2 = np.zeros((count, 3))
        spin_1[:, 2] = rng.uniform(-0.8, 0.8, count)
        spin_2[:, 2] = rng.uniform(-0.8, 0.8, count)
    else:
        def isotropic():
            direction = rng.normal(size=(count, 3))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            return direction * rng.uniform(0.0, 0.8, (count, 1))
        spin_1, spin_2 = isotropic(), isotropic()
    return dict(
        mass_1_msun=masses.max(0),
        mass_2_msun=masses.min(0),
        spin_1_in=spin_1,
        spin_2_in=spin_2,
        inclination_radians=rng.uniform(0.0, np.pi, count),
        distance_mpc=rng.uniform(100.0, 1000.0, count),
        reference_orbital_phase_in=rng.uniform(0.0, 2.0 * np.pi, count),
    )


# A reference bound. The buffer is invisible beside gravyflow's agreement with
# LALSimulation: the time-domain parity test (test_cbc_lal_parity.py) finds a
# worst match of 0.99992 against SimInspiralTD and ChooseTDWaveform over 248
# sources, so the buffer may move a kept window by no more than that residual,
# 8e-5, and never decides parity.
MINIMUM_BUFFER_OVERLAP = 1.0 - 8e-5


def _overlap(a, b):
    """Normalised time-domain inner product per polarisation; the worse of the two."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    inner = np.sum(a * b, axis=-1) / np.sqrt(np.sum(a * a, axis=-1) * np.sum(b * b, axis=-1))
    return inner.min(axis=-1)


@pytest.mark.parametrize(
    "approximant, mass_range, spins",
    [
        ("IMRPhenomD", (1.2, 2.5), "aligned"),
        ("IMRPhenomD", (3.0, 10.0), "aligned"),
        ("IMRPhenomD", (10.0, 100.0), "aligned"),
        ("IMRPhenomXAS", (3.0, 10.0), "aligned"),
        ("IMRPhenomPv2", (3.0, 10.0), "precessing"),
    ],
)
def test_the_kept_window_matches_a_long_buffer(monkeypatch, approximant, mass_range, spins):
    # The window the dataset path keeps from CBC generation.
    windows = _record_windows(monkeypatch)
    _injection_batch(gf.CBCGenerator(network=[gf.IFO.H1, gf.IFO.L1]), num_examples=1)
    monkeypatch.undo()
    window = windows[0]

    count = 8
    source = _sources(mass_range, spins, seed=int(10 * mass_range[0]), count=count)
    f_min = np.full(count, F_MIN, dtype=np.float32)
    kept = cbc._generate_cbc_window(
        count, SAMPLE_RATE, window, f_min, approximant=cbc.Approximant(approximant), **source
    )

    # The reference: the same placement and start rule on a buffer long enough to
    # hold every source's whole inspiral from F_MIN.
    inspiral = np.asarray(cbc.calc_duration_from_f_min(source["mass_1_msun"], source["mass_2_msun"], F_MIN))
    long_buffer = float(2.0 ** np.ceil(np.log2(max(4.0 * window, inspiral.max() + window / 2.0 + 2.0))))
    kept_samples = int(window * SAMPLE_RATE)
    reference = cbc._generate_cbc_waveform_jit(
        num_waveforms=count,
        sample_rate_hertz=SAMPLE_RATE,
        duration_seconds=long_buffer,
        approximant=cbc.Approximant(approximant),
        # The merger on sample K // 2 of the K kept samples.
        coalescence_time=long_buffer - (kept_samples - kept_samples // 2) / SAMPLE_RATE,
        min_frequency_hertz=f_min,
        **source,
    )[..., -int(window * SAMPLE_RATE):]

    assert kept.shape == reference.shape
    assert _overlap(kept, reference).min() >= MINIMUM_BUFFER_OVERLAP


def test_the_merger_is_at_the_window_centre_plus_its_shift():
    """
    Each injection is its source's own window moved by its shift: the peak sits
    ``shift`` samples from where it sits in the source generated alone, centred.
    Where that is relative to the merger is test_cbc.py's.
    """
    generator = gf.CBCGenerator(
        mass_1_msun=_uniform(20.0, 40.0), mass_2_msun=_uniform(20.0, 40.0), network=[gf.IFO.L1]
    )
    injections, _, parameters = _injection_batch(generator, num_examples=8)
    injections = np.asarray(injections[0], dtype=np.float64)
    shifts = np.asarray(parameters["shift"][0])
    assert np.any(shifts != 0)
    assert np.all(shifts == np.round(shifts))

    def source(parameter):
        return np.asarray(parameters[parameter][0], dtype=np.float64)

    P = gf.WaveformParameters
    window = injections.shape[-1]
    alone = np.asarray(cbc._generate_cbc_window(
        len(shifts), SAMPLE_RATE, window / SAMPLE_RATE, source(P.MIN_FREQUENCY_HERTZ),
        approximant=generator.approximant,
        mass_1_msun=source(P.MASS_1_MSUN),
        mass_2_msun=source(P.MASS_2_MSUN),
        spin_1_in=source(P.SPIN_1_IN),
        spin_2_in=source(P.SPIN_2_IN),
        inclination_radians=source(P.INCLINATION_RADIANS),
        distance_mpc=source(P.DISTANCE_MPC),
        reference_orbital_phase_in=source(P.REFERENCE_ORBITAL_PHASE_IN),
    ), dtype=np.float64)

    peaks = np.argmax(np.sum(injections ** 2, axis=1), axis=-1)
    peaks_alone = np.argmax(np.sum(alone ** 2, axis=1), axis=-1)
    # Exact: a shift is a whole number of samples, and the window does not
    # depend on the buffer it was cut from.
    np.testing.assert_array_equal(peaks - shifts.astype(int), peaks_alone)
