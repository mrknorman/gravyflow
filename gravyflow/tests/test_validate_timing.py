"""
The validator's timing, known answers on a fake model of known cost and a fake
clock: throughput per second of strain (batch), data-to-trigger latency
(streaming), the A-B-B-A order and its statistics, no ratio across hardware, and
the training-cost record.
"""

import numpy as np
import pytest

from gravyflow.src.dataset.config import WindowSpec
from gravyflow.src.utils.timing import HostLoad, MIN_STEPS
from gravyflow.src.validate import timing as vt
from gravyflow.src.validate.stream import StreamGeometry

WINDOW = WindowSpec(sample_rate_hertz=64.0, onsource_duration_seconds=1.0, offsource_duration_seconds=1.0,
                    crop_duration_seconds=0.25)
MACHINE = {"hardware": "Test CPU, 4 cores (CPU)"}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def costed_model(clock, per_window, log=None, name="A"):
    """A fake model whose cost is ``per_window`` fake seconds per window scored."""
    def score(onsource, offsource):
        clock.now += per_window * onsource.shape[0]
        if log is not None:
            log.append(name)
        return np.zeros(onsource.shape[0])
    return score


# ------------------------------------------------------------ throughput

def test_throughput_known_cost():
    clock = FakeClock()
    geometry = StreamGeometry.of(WINDOW, 0.25)
    strain = np.zeros((2, 64 * 32))  # 32 s of strain
    windows = geometry.num_windows(strain.shape[1])
    timing = vt.stream_throughput(costed_model(clock, 0.01), geometry, strain, threads=4, repeats=5,
                                  batch_size=16, clock=clock, hardware=MACHINE["hardware"])
    assert timing.mode == "batch" and timing.threads == 4 and timing.strain_seconds == 32.0
    np.testing.assert_allclose(timing.walls, [0.01 * windows] * 5)
    np.testing.assert_allclose(timing.cold_seconds, 0.01 * windows)
    summary = vt.throughput_summary(timing)
    for key in ("median", "p90", "p99", "max"):
        assert summary[key] == pytest.approx(0.01 * windows / 32.0)


# ------------------------------------------------------------- latency

def test_latency_known_cost_and_algorithmic_wait():
    clock = FakeClock()
    geometry = StreamGeometry.of(WINDOW, 0.25)
    steps = MIN_STEPS + 1
    strain = np.zeros((2, geometry.footprint_samples + (steps - 1) * geometry.stride_samples))
    timing = vt.stream_latency(costed_model(clock, 0.002), geometry, strain, cluster_seconds=1.0, threads=2,
                               clock=clock, hardware=MACHINE["hardware"])
    assert timing.mode == "streaming" and len(timing.walls) == MIN_STEPS
    np.testing.assert_allclose(timing.walls, 0.002)
    # Data after a window's time: half its padded onsource (0.75 s), then the 1 s the clustering waits.
    assert timing.algorithmic_wait_seconds == pytest.approx(0.75 + 1.0)
    latency = {r["statistic"]: r["value"] for r in timing.records()}
    assert latency["median"] == pytest.approx(1.752) and latency["p99"] == pytest.approx(1.752)
    assert timing.strain_seconds == pytest.approx(MIN_STEPS * 0.25)


def test_latency_refuses_a_short_replay():
    clock = FakeClock()
    geometry = StreamGeometry.of(WINDOW, 0.25)
    with pytest.raises(ValueError, match="steps"):
        vt.stream_latency(costed_model(clock, 0.001), geometry, np.zeros((1, geometry.footprint_samples * 3)),
                          cluster_seconds=1.0, threads=1, clock=clock, hardware="x")


# ---------------------------------------------------------------- A-B-B-A

def test_abba_order_and_statistics_known_cost():
    clock, log = FakeClock(), []

    def a():
        clock.now += 1.0
        log.append("A")

    def b():
        clock.now += 3.0
        log.append("B")

    result = vt.abba_rounds(a, b, rounds=3, threads=4, clock=clock, hardware=MACHINE["hardware"])
    # One cold call of each first, then A B B A three times.
    assert log == ["A", "B"] + ["A", "B", "B", "A"] * 3
    assert result.order == ("A", "B", "B", "A") * 3
    np.testing.assert_allclose(result.a_walls, [1.0] * 6)
    np.testing.assert_allclose(result.b_walls, [3.0] * 6)
    assert result.cold == (1.0, 3.0)
    assert result.summary("a")["median"] == 1.0 and result.summary("b")["p90"] == 3.0
    assert result.ratio == pytest.approx(3.0)
    assert result.spread_a == 0.0 and result.distinguishable
    d = result.to_dict()
    assert d["order"][:4] == ["A", "B", "B", "A"] and d["ratio_b_to_a"] == pytest.approx(3.0)


def test_abba_drift_cancels_in_the_pair_means():
    clock = FakeClock()
    calls = {"n": 0}

    def drifting(cost):
        def run():
            calls["n"] += 1
            clock.now += cost * (1.0 + 0.1 * calls["n"])   # the host slows steadily
        return run

    result = vt.abba_rounds(drifting(1.0), drifting(1.0), rounds=1, threads=1, clock=clock, hardware="x")
    # Same cost: A at calls 3 and 6, B at 4 and 5; the means match under linear drift.
    assert result.mean("a") == pytest.approx(result.mean("b"))
    assert not result.distinguishable


def test_no_ratio_across_hardware():
    load = HostLoad(0.1, 0.1, 4, 1)
    a = vt.TimingSummary("batch", {"median": 1.0}, 1, {"hardware": "Chip A"}, load)
    b = vt.TimingSummary("batch", {"median": 2.0}, 1, {"hardware": "Chip B"}, load)
    with pytest.raises(ValueError, match="different hardware"):
        vt.timing_ratio(a, b)
    c = vt.TimingSummary("batch", {"median": 2.0}, 1, {"hardware": "Chip A"}, load)
    assert vt.timing_ratio(a, c) == pytest.approx(2.0)
    d = vt.TimingSummary("batch", {"median": 2.0}, 2, {"hardware": "Chip A"}, load)
    with pytest.raises(ValueError, match="thread"):
        vt.timing_ratio(a, d)


# ----------------------------------------------------- the timing record

def test_timing_record_carries_machine_load_and_threads():
    clock = FakeClock()
    geometry = StreamGeometry.of(WINDOW, 0.25)
    strain = np.zeros((2, 64 * 16))
    throughput = vt.stream_throughput(costed_model(clock, 0.01), geometry, strain, threads=4, repeats=3,
                                      clock=clock, hardware=MACHINE["hardware"])
    record = vt.timing_record(throughput=throughput)
    assert record["threads"] == 4 and record["machine"]["hardware"] == MACHINE["hardware"]
    assert "host_load" in record["throughput"] and record["latency"] is None
    assert record["throughput"]["summary"]["median"] == pytest.approx(
        0.01 * geometry.num_windows(strain.shape[1]) / 16.0)


def test_the_timing_tab_shows_median_tail_machine_and_cost():
    from types import SimpleNamespace

    from gravyflow.src.validate import tabs

    clock = FakeClock()
    geometry = StreamGeometry.of(WINDOW, 0.25)
    throughput = vt.stream_throughput(costed_model(clock, 0.01), geometry, np.zeros((2, 64 * 16)), threads=4,
                                      repeats=3, clock=clock, hardware=MACHINE["hardware"])
    steps = np.zeros((2, geometry.footprint_samples + MIN_STEPS * geometry.stride_samples))
    latency = vt.stream_latency(costed_model(clock, 0.002), geometry, steps, cluster_seconds=1.0, threads=4,
                                clock=clock, hardware=MACHINE["hardware"])
    abba = vt.abba_rounds(lambda: None, lambda: None, rounds=1, threads=4, clock=clock, hardware="x")
    v = SimpleNamespace(name="model", timing=vt.timing_record(throughput, latency, abba),
                        training_cost=vt.TrainingCost.quoted(12.0, "GPU-hours", "Author et al. 2021").to_dict())
    table = tabs.timing_tables(v)
    assert list(table["Statistic"][:4]) == ["median", "p90", "p99", "max"]
    latency_rows = table[table["Quantity"].str.startswith("Data-to-trigger")]
    np.testing.assert_allclose(latency_rows["Value"], 1.752)
    assert table["Quantity"].iloc[-1] == "Training cost (quoted)" and "Author" in table["Units"].iloc[-1]
    tab = tabs.generate_timing_tab([v, SimpleNamespace(name="other", timing=None, training_cost=None)])
    text = " ".join(str(getattr(o, "object", "")) for o in tab.objects)
    assert MACHINE["hardware"] in text and "**Threads:** 4" in text and "Not timed" in text


# ------------------------------------------------------------ training cost

def test_training_cost_measured_and_quoted():
    clock = FakeClock()

    def train():
        clock.now += 3600.0
        return "weights"

    result, cost = vt.measure_training(train, clock=clock, hardware=MACHINE["hardware"], epochs=10)
    assert result == "weights"
    assert cost.kind == "measured" and cost.value == 3600.0 and cost.units == "s"
    assert cost.hardware == MACHINE["hardware"] and cost.epochs == 10
    assert vt.TrainingCost.from_dict(cost.to_dict()) == cost

    quoted = vt.TrainingCost.quoted(12.0, "GPU-hours", source="Author et al. 2021, section 3")
    assert quoted.kind == "quoted" and quoted.source.startswith("Author")
    with pytest.raises(ValueError, match="source"):
        vt.TrainingCost.quoted(12.0, "GPU-hours", source="")
    with pytest.raises(ValueError, match="kind"):
        vt.TrainingCost("guessed", 1.0, "s")
