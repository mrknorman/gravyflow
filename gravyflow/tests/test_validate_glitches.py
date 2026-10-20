"""
Glitch robustness, known answers: the false-positive rate per glitch class, each
background trigger's nearest glitch, a model run over every labelled glitch, the
real data's volume against its Gaussian twin, and the record and dashboard.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gravyflow.src.dataset.config import WindowSpec
from gravyflow.src.validate import glitches as gl
from gravyflow.src.validate.metrics import binomial_interval, sigma_level
from gravyflow.src.validate.stream import StreamGeometry


def make_glitches():
    return gl.Glitches(
        ifos=np.array(["H1", "H1", "L1", "L1", "H1"]),
        gps=np.array([100.0, 200.0, 150.0, 300.0, 50.0]),
        labels=np.array(["Blip", "Scattered_Light", "Blip", "Koi_Fish", "Tomte"]),
    )


# ------------------------------------------------------------ the glitch list

def test_glitches_sorted_by_detector_and_time():
    g = make_glitches()
    gps, labels = g.of("H1")
    np.testing.assert_array_equal(gps, [50.0, 100.0, 200.0])
    np.testing.assert_array_equal(labels, ["Tomte", "Blip", "Scattered_Light"])
    assert g.classes == ("Blip", "Koi_Fish", "Scattered_Light", "Tomte")
    assert len(g) == 5


def test_glitches_refuse_ragged_columns():
    with pytest.raises(ValueError, match="same length"):
        gl.Glitches(ifos=np.array(["H1"]), gps=np.array([1.0, 2.0]), labels=np.array(["Blip"]))


def test_glitches_from_gravity_spy_reads_the_catalogue(monkeypatch):
    calls = []

    def query(ifo, start, end, min_ml_confidence):
        calls.append((ifo, start, end, min_ml_confidence))
        return {"event_time": np.array([10.0, 5.0]) + (0 if ifo == "H1" else 1),
                "ml_label": np.array(["Blip", "Whistle"]), "snr": np.array([8.0, 9.0])}

    monkeypatch.setattr(gl.gravity_spy, "query", query)
    g = gl.Glitches.from_gravity_spy(["H1", "L1"], 0.0, 20.0, min_ml_confidence=0.5)
    assert calls == [("H1", 0.0, 20.0, 0.5), ("L1", 0.0, 20.0, 0.5)]
    np.testing.assert_array_equal(g.of("L1")[0], [6.0, 11.0])
    np.testing.assert_array_equal(g.of("L1")[1], ["Whistle", "Blip"])
    assert "zenodo" in g.source.lower()


# --------------------------------------------------------- nearest glitch

def test_nearest_glitch_known_answers():
    g = make_glitches()
    # Triggers' per-detector GPS (H1, L1): L1 slid by +1000 s, as in a time slide.
    times = np.array([
        [102.0, 1102.0],    # H1 glitch at 100: 2 s after it
        [140.0, 1149.5],    # L1 has nothing near 1149.5; H1's nearest is 100 (40 s) vs 200 (60 s)
        [10.0, 1290.0],     # H1 50: 40 s; L1 300: 990 s off -> H1 wins
        [151.0, 151.0],     # L1 glitch at 150: 1 s before (zero lag)
    ])
    near = gl.nearest_glitch(times, ["H1", "L1"], g)
    np.testing.assert_allclose(near.distance_seconds, [2.0, 40.0, 40.0, 1.0])
    np.testing.assert_allclose(near.offset_seconds, [2.0, 40.0, -40.0, 1.0])
    np.testing.assert_array_equal(near.label, ["Blip", "Blip", "Tomte", "Blip"])
    np.testing.assert_array_equal(near.ifo, ["H1", "H1", "H1", "L1"])
    np.testing.assert_allclose(near.glitch_gps, [100.0, 100.0, 50.0, 150.0])


def test_nearest_glitch_ties_take_the_earlier_glitch_and_none_is_empty():
    g = gl.Glitches(ifos=np.array(["H1", "H1"]), gps=np.array([10.0, 20.0]), labels=np.array(["A", "B"]))
    near = gl.nearest_glitch(np.array([[15.0, 0.0]]), ["H1", "V1"], g)
    assert near.label[0] == "A" and near.distance_seconds[0] == 5.0
    none = gl.nearest_glitch(np.array([[15.0]]), ["V1"], g)
    assert np.isnan(none.distance_seconds[0]) and none.label[0] == "" and none.ifo[0] == ""


# ------------------------------------------------------- per-class rates

def test_per_class_false_positive_rates_known_answer():
    labels = np.array(["Blip"] * 4 + ["Koi_Fish"] * 2 + ["Tomte"] * 3)
    scores = np.array([5.0, 1.0, 3.0, -np.inf, 9.0, 2.0, 0.0, 0.0, 0.0])
    rates = gl.glitch_false_positive_rates(scores, labels, thresholds=[2.0, 4.0], far_hz=[1e-3, 1e-5])
    assert rates.labels == ("Blip", "Koi_Fish", "Tomte")
    np.testing.assert_array_equal(rates.counts, [4, 2, 3])
    np.testing.assert_array_equal(rates.passed, [[2, 2, 0], [1, 1, 0]])
    np.testing.assert_allclose(rates.rate, [[0.5, 1.0, 0.0], [0.25, 0.5, 0.0]])
    for s, sigma in enumerate(rates.sigmas):
        lo, hi = binomial_interval(rates.passed, rates.counts[None, :], sigma_level(sigma))
        np.testing.assert_allclose(rates.lower[s], lo)
        np.testing.assert_allclose(rates.upper[s], hi)
    assert rates.sigmas == (1.0, 2.0)
    # The 2 sigma band contains the 1 sigma band.
    assert np.all(rates.lower[1] <= rates.lower[0]) and np.all(rates.upper[1] >= rates.upper[0])


def test_unresolved_threshold_gives_nan_rates():
    rates = gl.glitch_false_positive_rates(np.array([1.0, 2.0]), np.array(["A", "B"]), thresholds=[np.nan])
    assert np.all(np.isnan(rates.rate)) and np.all(rates.passed == 0)


def test_rates_refuse_mismatched_inputs():
    with pytest.raises(ValueError, match="one label per score"):
        gl.glitch_false_positive_rates(np.array([1.0]), np.array(["A", "B"]), thresholds=[0.0])


# ----------------------------------------------- the model over every glitch

def spike_model(onsource, offsource):
    """A fake model: each window's score is the largest absolute strain within 8 samples of its centre."""
    centre = onsource.shape[-1] // 2
    return np.max(np.abs(onsource[..., centre - 8:centre + 8]), axis=(1, 2))


def test_score_glitches_runs_the_model_over_every_glitch():
    window = WindowSpec(sample_rate_hertz=64.0, onsource_duration_seconds=1.0, offsource_duration_seconds=1.0,
                        crop_duration_seconds=0.25)
    geometry = StreamGeometry.of(window, 0.25)
    g = gl.Glitches(ifos=np.array(["H1", "L1", "H1"]), gps=np.array([10.0, 20.0, 30.0]),
                    labels=np.array(["Blip", "Blip", "Tomte"]))
    loud = {10.0: 7.0, 20.0: 3.0, 30.0: 0.5}
    seen = []

    def strain_around(ifo, gps, half_samples):
        seen.append((ifo, gps, half_samples))
        data = np.zeros((2, 2 * half_samples))
        data[0 if ifo == "H1" else 1, half_samples] = loud[gps]
        return data

    scores = gl.score_glitches(spike_model, strain_around, geometry, g, cluster_seconds=1.0,
                               found_window_seconds=0.2, batch_size=2)
    # In the list's order: by detector, then time.
    np.testing.assert_allclose(scores, [7.0, 0.5, 3.0])
    half = geometry.local_half_samples(1.2)
    assert [s[2] for s in seen] == [half] * 3
    assert [s[:2] for s in seen] == [("H1", 10.0), ("H1", 30.0), ("L1", 20.0)]


def test_score_glitches_refuses_a_short_stretch():
    window = WindowSpec(sample_rate_hertz=64.0, onsource_duration_seconds=1.0, offsource_duration_seconds=1.0,
                        crop_duration_seconds=0.25)
    geometry = StreamGeometry.of(window, 0.25)
    g = gl.Glitches(ifos=np.array(["H1"]), gps=np.array([1.0]), labels=np.array(["Blip"]))
    with pytest.raises(ValueError, match="samples"):
        gl.score_glitches(spike_model, lambda ifo, gps, h: np.zeros((1, 10)), geometry, g, 1.0, 0.1)


# ------------------------------------------------------- the background tail

def stream_validator(scores, gps, lengths, stride=0.25, input_seconds=1.0):
    config = SimpleNamespace(stream=True, stream_cluster_seconds=1.0, cluster_seconds=1.0)
    return SimpleNamespace(far_scores=np.asarray(scores, dtype=float), noise_gps_times=np.asarray(gps, dtype=float),
                           noise_streams={"lengths": np.asarray(lengths), "stride_seconds": np.float64(stride)},
                           config=config, input_duration_seconds=input_seconds,
                           dataset_args={"ifos": None}, ifo_names=["H1", "L1"])


def test_background_triggers_and_their_nearest_glitches():
    # One stream of 12 windows at 0.25 s: triggers are the loudest within 1 s.
    scores = [0, 1, 5, 1, 0, 0, 0, 0, 0, 0, 4, 0]
    starts = np.arange(12) * 0.25 + 99.5   # each window's first cropped onsource sample, H1
    gps = np.stack([starts, starts + 1000.0], axis=1)
    v = stream_validator(scores, gps, [12])
    trig_scores, trig_gps = gl.background_triggers(v)
    np.testing.assert_allclose(trig_scores, [5.0, 4.0])
    np.testing.assert_allclose(trig_gps[:, 0], [99.5 + 0.5 + 0.5, 99.5 + 2.5 + 0.5])
    record = gl.trigger_glitch_record(v, make_glitches(), ["H1", "L1"])
    np.testing.assert_allclose(record["distance_seconds"], [0.5, 2.5])
    np.testing.assert_array_equal(record["label"], ["Blip", "Blip"])
    tail = gl.tail_by_class(record, threshold=4.5, near_seconds=1.0)
    assert tail == {"Blip": 1}
    tail = gl.tail_by_class(record, threshold=0.0, near_seconds=1.0)
    assert tail == {"Blip": 1, "none": 1}


def test_background_triggers_need_gps_times():
    v = stream_validator([0, 1], np.zeros((2, 2)), [2])
    v.noise_gps_times = None
    with pytest.raises(ValueError, match="GPS"):
        gl.background_triggers(v)


# ------------------------------------------------- real against its twin

def curve(volume, lower, upper, quotable=True):
    v = np.array([volume])
    return SimpleNamespace(far_hz=np.array([1e-3]), volume_within_reach_gpc3=v,
                           joint_lower_gpc3=np.array([[lower[0]], [lower[1]]]),
                           joint_upper_gpc3=np.array([[upper[0]], [upper[1]]]), sigmas=(1.0, 2.0),
                           quotable=np.array([quotable]))


def test_twin_comparison_known_answer():
    real = curve(2.0, (2.0 / np.e ** 0.3, 2.0 / np.e ** 0.6), (2.0 * np.e ** 0.4, 2.0 * np.e ** 0.8))
    twin = curve(4.0, (4.0 / np.e ** 0.3, 4.0 / np.e ** 0.6), (4.0 * np.e ** 0.4, 4.0 * np.e ** 0.8))
    cmp = gl.twin_comparison(real, twin)
    np.testing.assert_allclose(cmp.ratio, [0.5])
    # Lower: the real's lower and the twin's upper log-widths in quadrature.
    np.testing.assert_allclose(cmp.ratio_lower[0], [0.5 * np.exp(-np.hypot(0.3, 0.4))])
    np.testing.assert_allclose(cmp.ratio_upper[1], [0.5 * np.exp(np.hypot(0.8, 0.6))])
    assert cmp.quotable.tolist() == [True]
    cmp = gl.twin_comparison(real, curve(4.0, (3.0, 2.0), (5.0, 6.0), quotable=False))
    assert cmp.quotable.tolist() == [False]


def test_twin_comparison_refuses_other_levels():
    real = curve(2.0, (1.0, 0.5), (3.0, 4.0))
    twin = curve(2.0, (1.0, 0.5), (3.0, 4.0))
    twin.far_hz = np.array([1e-4])
    with pytest.raises(ValueError, match="same false-alarm"):
        gl.twin_comparison(real, twin)
