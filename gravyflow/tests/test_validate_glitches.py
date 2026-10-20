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


# ------------------------------------------------- the record and the dashboard

def built_validator(num=4000, seed=5):
    """A stream-mode result built by hand: one stream of real-noise windows at 0.25 s, H1 and L1 at zero lag."""
    from gravyflow.src.validate import ValidationConfig, Validator

    v = Validator("model", ValidationConfig(num_examples=num, far_thresholds=[1e-1, 1e-2]),
                  {"onsource_duration_seconds": 1.0, "offsource_duration_seconds": 16.0})
    v.far_scores = np.random.default_rng(seed).standard_normal(num)
    starts = 1_000_000.0 + np.arange(num) * 0.25
    v.noise_gps_times = np.stack([starts, starts], axis=1)
    v.noise_streams = {"lengths": np.array([num]), "slides": np.array([0]), "blocks": np.array([0]),
                       "stride_seconds": np.float64(0.25)}
    rng = np.random.default_rng(seed + 1)
    g = gl.Glitches(ifos=rng.choice(["H1", "L1"], 60), gps=1_000_000.0 + rng.uniform(0, num * 0.25, 60),
                    labels=rng.choice(["Blip", "Koi_Fish", "Scattered_Light"], 60))
    v.glitch_data = gl.glitch_data(g, rng.normal(1.0, 1.5, len(g)))
    v.trigger_glitches = gl.trigger_glitch_record(v, g, ["H1", "L1"])
    return v


def test_the_glitch_records_round_trip(tmp_path):
    from gravyflow.src.validate import Validator

    v = built_validator()
    v.timing = {"threads": 4, "machine": {"hardware": "Test CPU"}, "throughput": None, "latency": None, "abba": None}
    v.training_cost = {"kind": "quoted", "value": 12.0, "units": "GPU-hours", "hardware": None,
                       "source": "Author et al. 2021", "epochs": None, "notes": None}
    v.save(tmp_path / "v.h5")
    w = Validator.load(tmp_path / "v.h5")
    for key in ("glitch_data", "trigger_glitches"):
        assert set(getattr(w, key)) == set(getattr(v, key))
        for k, value in getattr(v, key).items():
            np.testing.assert_array_equal(getattr(w, key)[k], value)
    assert w.glitch_data["labels"].dtype.kind == "U" and w.trigger_glitches["label"].dtype.kind == "U"
    assert w.timing == v.timing and w.training_cost == v.training_cost
    # A record without them reads them as absent.
    bare = Validator("bare", v.config)
    bare.save(tmp_path / "bare.h5")
    b = Validator.load(tmp_path / "bare.h5")
    assert b.glitch_data is None and b.trigger_glitches is None and b.timing is None and b.training_cost is None


def test_the_glitch_tab_shows_rates_twin_and_tail():
    from gravyflow.src.validate import tabs

    v = built_validator()
    rates = gl.validator_glitch_rates(v)
    assert rates.labels == ("Blip", "Koi_Fish", "Scattered_Light") and rates.rate.shape == (2, 3)
    plot = tabs.glitch_rate_plot([v])
    resolved = int(np.isfinite(rates.threshold).sum())
    assert resolved >= 1
    # Per resolved level: a 2σ bar, a 1σ bar and the points.
    assert len(plot.renderers) == 3 * resolved
    by_class, loudest = tabs.tail_tables(v)
    assert set(by_class["Nearest glitch class"]) <= {"Blip", "Koi_Fish", "Scattered_Light", "none"}
    assert len(loudest) == tabs.TAIL_ROWS and loudest["Score"].is_monotonic_decreasing
    real = SimpleNamespace(name="model", sensitive_volume_curve=lambda: curve(2.0, (1.5, 1.0), (2.5, 3.0)))
    twin = SimpleNamespace(name="model twin", sensitive_volume_curve=lambda: curve(4.0, (3.0, 2.0), (5.0, 6.0)))
    plot, table = tabs.twin_plot([real], {"model": twin})
    assert table["Real / twin"].tolist() == [0.5]
    tab = tabs.generate_glitch_tab([v], {})
    text = " ".join(str(getattr(o, "object", "")) for o in tab.objects)
    assert "No Gaussian twin given" in text and "No glitch record" not in text


def test_the_dashboard_keeps_every_tab_and_adds_glitches_and_timing(tmp_path):
    v = built_validator()
    template = v.plot(tmp_path / "report.html")
    tabs_ = template.main[1]
    names = list(tabs_._names)
    # Every tab the dashboard had is kept; Glitches and Timing are added after them.
    for kept in ("Efficiency", "ROC Curves", "FAR Curves", "Parameter Space"):
        assert kept in names
    assert names[-2:] == ["Glitches", "Timing"]
    assert (tmp_path / "report.html").stat().st_size > 0


class FixtureObtainer:
    """A real-noise obtainer serving fixture strain: zeros at 64 Hz with each glitch a spike in its own detector."""

    def __init__(self, readable, glitches, loudness, vetoes=np.empty((0, 2))):
        self.readable, self.glitches, self.loudness, self.vetoes = readable, glitches, loudness, vetoes
        self.reads = []

    def get_all_segment_times(self, ifo):
        return self.readable

    def get_event_veto_segments(self):
        return self.vetoes

    def acquire(self, sample_rate_hertz, rows, ifos):
        for row in rows:
            data, starts = [], []
            for j, ifo in enumerate(ifos):
                start, end = row[j]
                self.reads.append((ifo.name, float(start), float(end)))
                first = np.ceil(start * sample_rate_hertz) / sample_rate_hertz
                times = first + np.arange(int((end - first) * sample_rate_hertz)) / sample_rate_hertz
                strain = np.zeros(times.size)
                for name, gps in zip(self.glitches.ifos, self.glitches.gps):
                    if name == ifo.name and times[0] <= gps <= times[-1]:
                        strain[int(round((gps - first) * sample_rate_hertz))] = self.loudness[float(gps)]
                data.append(strain)
                starts.append(first)
            yield SimpleNamespace(data=data, start_gps_time=np.array(starts))


def fixture_stage(per_class=None, vetoes=np.empty((0, 2))):
    import gravyflow as gf

    spec = gf.SplitSpec()
    a = 1_256_000_000.0
    readable = np.array([[a, a + 200_000.0]])
    test = spec.admissible(readable, "test")
    t0 = float(test[0, 0])
    gps = [t0 + 100.25, t0 + 200.5, t0 + 300.75, t0 + 400.0, t0 + 1.0, test[0, 1] + 5000.0]
    glitches = gl.Glitches(ifos=np.array(["H1", "L1", "H1", "L1", "H1", "H1"]), gps=np.array(gps),
                           labels=np.array(["Blip", "Blip", "Koi_Fish", "Blip", "Blip", "Blip"]))
    loudness = dict(zip(map(float, gps), [7.0, 3.0, 5.0, 2.0, 9.0, 9.0]))
    obtainer = FixtureObtainer(readable, glitches, loudness, vetoes)
    window = WindowSpec(sample_rate_hertz=64.0, onsource_duration_seconds=1.0, offsource_duration_seconds=1.0,
                        crop_duration_seconds=0.25)
    stage = gl.glitch_stage(obtainer, [gf.IFO.H1, gf.IFO.L1], spec, "test", StreamGeometry.of(window, 0.25),
                            spike_model, cluster_seconds=1.0, found_window_seconds=0.2, glitches=glitches,
                            per_class=per_class, rng=np.random.default_rng(0))
    return stage, obtainer, t0


def test_the_glitch_stage_scores_every_glitch_of_the_held_out_split():
    stage, obtainer, t0 = fixture_stage()
    # The glitch 1 s into the test time has no room for its stretch; the one in the train split is not read.
    assert len(stage.all) == 6
    np.testing.assert_allclose(stage.scored.gps - t0, [100.25, 300.75, 200.5, 400.0])
    np.testing.assert_allclose(stage.scores, [7.0, 5.0, 3.0, 2.0])
    np.testing.assert_array_equal(stage.scored.labels, ["Blip", "Koi_Fish", "Blip", "Blip"])
    # Every read is both detectors at zero lag, within the test time.
    assert {name for name, _, _ in obtainer.reads} == {"H1", "L1"}
    assert all(t0 < start and end < t0 + 9000 for _, start, end in obtainer.reads)


def test_the_glitch_stage_keeps_out_events_and_subsamples_per_class():
    _, _, t0 = fixture_stage()
    stage, _, _ = fixture_stage(vetoes=np.array([[t0 + 290.0, t0 + 310.0]]))
    assert "Koi_Fish" not in stage.scored.labels
    stage, _, _ = fixture_stage(per_class=1)
    assert sorted(stage.scored.labels) == ["Blip", "Koi_Fish"]


def test_the_bank_runs_the_glitch_stage_when_asked(monkeypatch):
    import gravyflow as gf
    from gravyflow.src.validate import ValidationConfig
    from gravyflow.src.validate.bank import ValidationBank

    assert ValidationConfig().score_glitches is False   # no default run reads glitches
    with pytest.raises(ValueError, match="score_glitches"):
        ValidationConfig(score_glitches=0)
    seen = {}

    def stage(obtainer, ifos, split_spec, group, geometry, score, cluster_seconds, found_window_seconds,
              per_class=None, rng=None, glitches=None):
        seen.update(ifos=[i.name for i in ifos], group=group, per_class=per_class, cluster=cluster_seconds)
        g = gl.Glitches(ifos=np.array(["H1"]), gps=np.array([1.0]), labels=np.array(["Blip"]))
        return gl.GlitchStage(g, g, np.array([2.0]))

    monkeypatch.setattr(gl, "glitch_stage", stage)
    bank = ValidationBank.__new__(ValidationBank)
    bank.config = ValidationConfig(score_glitches=50)
    bank.dataset_args = {"ifos": [gf.IFO.H1, gf.IFO.L1],
                         "noise_obtainer": SimpleNamespace(noise_type=gf.NoiseType.REAL)}
    bank.__dict__["_noise_dataset"] = SimpleNamespace(
        noise_obtainer=SimpleNamespace(ifo_data_obtainer=object()),
        window_spec=WindowSpec(sample_rate_hertz=64.0, onsource_duration_seconds=1.0,
                               offsource_duration_seconds=1.0, crop_duration_seconds=0.25))
    monkeypatch.setattr(ValidationBank, "split_spec", property(lambda self: gf.SplitSpec()))
    bank.generate_glitches()
    assert seen == {"ifos": ["H1", "L1"], "group": "test", "per_class": 50, "cluster": 1.0}
    assert bank.glitch_stage.scores.tolist() == [2.0] and bank.glitch_ifos == ["H1", "L1"]
    bank.config = ValidationConfig(score_glitches=True, mode="window")
    with pytest.raises(ValueError, match="stream mode"):
        bank.generate_glitches()


def test_the_gps_timeline_takes_per_detector_times():
    # Stream-mode real-noise records hold one GPS time per window and detector (windows, detectors).
    from gravyflow.src.validate.plotting import generate_gps_distribution_plot

    starts = 1_000_000.0 + np.concatenate([np.arange(400) * 0.25, 5000.0 + np.arange(400) * 0.25])
    slid = SimpleNamespace(noise_gps_times=np.stack([starts, starts + 3.5], axis=1))
    flat = SimpleNamespace(noise_gps_times=starts)
    layout = generate_gps_distribution_plot([slid])
    title = layout.children[0].title.text
    assert "2 segments" in title and "detector 0 of 2" in title
    # Detector 0's times are the timeline, as a one-detector record's are.
    flat_title = generate_gps_distribution_plot([flat]).children[0].title.text
    assert flat_title.startswith(title.split(";")[0].rstrip(")")) and "detector" not in flat_title
