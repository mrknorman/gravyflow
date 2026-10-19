"""
The timing harness: throughput, latency and A-B-B-A comparisons.

The callables timed here have a known cost (a sleep, which takes at least its duration, or fixed
NumPy work), so the tests assert on structure and on ordering: a call that sleeps for T takes
at least T, one doing ten times the work measures more. No test asserts an exact time.
"""

import json
import os
import time

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.utils import timing
from gravyflow.src.utils.timing import (
    MIN_STEPS, AbbaComparison, HostLoad, TooFewSteps, Timing, compare_abba, percentile, summarise,
    time_batch, time_streaming)

SCHEMA_FIELDS = {"quantity", "statistic", "value", "units", "normalised", "hardware", "scope", "mode",
                 "grade", "notes"}


def sleeper(seconds):
    return lambda *_: time.sleep(seconds)


def work(size):
    """Fixed NumPy work whose cost grows with ``size``."""
    matrix = np.random.default_rng(0).standard_normal((size, size))
    return lambda *_: matrix @ matrix


class TestStatistics:
    def test_percentile_is_nearest_rank(self):
        values = list(range(1, 101))
        assert percentile(values, 50) == 50
        assert percentile(values, 90) == 90
        assert percentile(values, 99) == 99
        assert percentile(values, 100) == 100
        assert percentile([3.0], 99) == 3.0
        assert percentile([5, 1, 3], 1) == 1

    def test_percentile_refuses(self):
        with pytest.raises(ValueError, match="no values"):
            percentile([], 50)
        for q in (0, -1, 100.5):
            with pytest.raises(ValueError, match="q must be"):
                percentile([1.0], q)

    def test_summary(self):
        values = [float(v) for v in range(1, 1001)]
        assert summarise(values) == {"median": 500.5, "p90": 900.0, "p99": 990.0, "max": 1000.0}


class TestHostLoad:
    def test_busy_beyond_the_timed_work_s_own_threads(self):
        # 8 cores, 2 threads, 0.25 per core: busy above 2 + 2 = 4.
        quiet = HostLoad(start=1.0, end=3.9, cpu_count=8, threads=2)
        busy_at_start = HostLoad(start=4.5, end=1.0, cpu_count=8, threads=2)
        busy_at_end = HostLoad(start=1.0, end=4.5, cpu_count=8, threads=2)
        assert quiet.loaded is False and quiet.note() is None
        assert busy_at_start.loaded is True and busy_at_end.loaded is True
        assert "host load" in busy_at_end.note()

    def test_unmeasured_is_not_quiet(self):
        unknown = HostLoad(start=None, end=None, cpu_count=8, threads=1)
        assert unknown.loaded is None
        assert "could not be measured" in unknown.note()

    def test_an_unmeasurable_host_flags_the_timing(self, monkeypatch):
        monkeypatch.setattr(timing, "_load_average", lambda: None)
        result = time_batch(sleeper(0.001), strain_seconds=1.0, threads=1, scope="network", repeats=2,
                            hardware="test")
        assert result.load.loaded is None
        assert all("could not be measured" in record["notes"] for record in result.records())

    def test_a_timing_under_load_is_flagged(self, monkeypatch):
        monkeypatch.setattr(timing, "_load_average", lambda: 10.0 * (os.cpu_count() or 1))
        result = time_batch(sleeper(0.001), strain_seconds=1.0, threads=1, scope="network", repeats=2,
                            hardware="test")
        assert result.load.loaded is True
        assert all("host load" in record["notes"] for record in result.records())

    def test_a_quiet_host_is_recorded_quiet(self, monkeypatch):
        monkeypatch.setattr(timing, "_load_average", lambda: 0.0)
        result = time_batch(sleeper(0.001), strain_seconds=1.0, threads=1, scope="network", repeats=2,
                            hardware="test")
        assert result.load.loaded is False
        assert result.load.start == 0.0 and result.load.end == 0.0
        assert all(record["notes"] is None for record in result.records())


class TestMachine:
    def test_description(self):
        machine = timing.machine_description()
        assert machine["hardware"] and machine["cpu_count"] >= 1
        assert machine["platform"]["backend"] == "cpu"
        assert machine["platform"]["xla_flags"] == os.environ.get("XLA_FLAGS", "")
        assert machine["cpu_model"] in machine["hardware"]
        json.dumps(machine)


class TestBatch:
    def test_walls_are_whole_calls(self):
        result = time_batch(sleeper(0.02), strain_seconds=4.0, threads=1, scope="network", repeats=3)
        assert isinstance(result, Timing)
        assert len(result.walls) == 3
        assert all(wall >= 0.02 for wall in result.walls)
        assert result.cold_seconds >= 0.02
        assert result.strain_seconds == 4.0 and result.threads == 1
        assert result.mode == "batch" and result.algorithmic_wait_seconds is None

    def test_the_cold_call_is_the_first_and_apart(self):
        calls = []

        def run():
            calls.append(len(calls))
            time.sleep(0.03 if len(calls) == 1 else 0.001)

        result = time_batch(run, strain_seconds=1.0, threads=1, scope="network", repeats=4)
        assert len(calls) == 5
        assert result.cold_seconds >= 0.03
        assert len(result.walls) == 4
        assert result.cold_seconds > max(result.walls)

    def test_the_cold_start_does_not_enter_the_record(self):
        state = {"first": True}

        def run():
            time.sleep(0.2 if state.pop("first", False) else 0.001)

        result = time_batch(run, strain_seconds=1.0, threads=1, scope="network", repeats=3)
        (record,) = result.records()
        assert record["value"] < 0.2 < result.cold_seconds

    def test_throughput_is_wall_per_second_of_strain(self):
        result = time_batch(sleeper(0.02), strain_seconds=4.0, threads=1, scope="network", repeats=3)
        (record,) = result.records()
        assert record["quantity"] == "inference_cost" and record["statistic"] == "median"
        assert record["value"] == record["normalised"]
        assert record["value"] >= 0.02 / 4.0
        assert record["units"] == "compute-seconds per second of strain"
        assert record["mode"] == "batch" and record["scope"] == "network" and record["grade"] == "measured"
        # The median of the walls, over the strain.
        assert record["value"] * 4.0 == pytest.approx(sorted(result.walls)[1])

    def test_a_longer_segment_for_the_same_wall_costs_less_per_second(self):
        short = time_batch(sleeper(0.01), strain_seconds=1.0, threads=1, scope="network", repeats=2)
        long = time_batch(sleeper(0.01), strain_seconds=100.0, threads=1, scope="network", repeats=2)
        assert long.records()[0]["value"] < short.records()[0]["value"]

    def test_more_work_measures_more(self):
        light = time_batch(work(32), strain_seconds=1.0, threads=1, scope="network", repeats=5)
        heavy = time_batch(work(640), strain_seconds=1.0, threads=1, scope="network", repeats=5)
        assert heavy.records()[0]["value"] > light.records()[0]["value"]

    def test_one_repeat_is_a_single(self):
        result = time_batch(sleeper(0.001), strain_seconds=1.0, threads=1, scope="end_to_end", repeats=1)
        (record,) = result.records()
        assert record["statistic"] == "single" and record["scope"] == "end_to_end"

    def test_an_asynchronous_result_is_waited_for(self):
        class Pending:
            def block_until_ready(self):
                time.sleep(0.05)

        for result in (Pending(), [Pending()], {"scores": (Pending(),)}):
            timed = time_batch(lambda: result, strain_seconds=1.0, threads=1, scope="network", repeats=1)
            assert timed.walls[0] >= 0.05

    def test_a_jax_computation_is_waited_for(self):
        import jax
        import jax.numpy as jnp

        matrix = jnp.asarray(np.random.default_rng(0).standard_normal((512, 512)), dtype=jnp.float32)
        step = jax.jit(lambda m: jnp.linalg.matrix_power(m, 8))
        timed = time_batch(lambda: step(matrix), strain_seconds=1.0, threads=1, scope="network", repeats=2)
        assert timed.walls[0] > 0

    @pytest.mark.parametrize("kwargs, message", [
        (dict(strain_seconds=0.0), "strain_seconds must be positive"),
        (dict(threads=0), "threads is the thread count"),
        (dict(threads=1.5), "threads is the thread count"),
        (dict(scope="model"), "scope is one of"),
        (dict(repeats=0), "repeats must be"),
    ])
    def test_refuses(self, kwargs, message):
        arguments = dict(strain_seconds=1.0, threads=1, scope="network", repeats=1)
        arguments.update(kwargs)
        with pytest.raises(ValueError, match=message):
            time_batch(sleeper(0), **arguments)


class TestStreaming:
    def replay(self, step, steps=MIN_STEPS, **overrides):
        arguments = dict(algorithmic_wait_seconds=0.25, threads=1, scope="end_to_end", hardware="test machine")
        arguments.update(overrides)
        return time_streaming(step, range(steps + 1), **arguments)

    def test_steps_and_cold_start(self):
        seen = []

        def step(chunk):
            seen.append(chunk)

        result = self.replay(step, step_strain_seconds=0.5)
        assert seen == list(range(MIN_STEPS + 1))
        assert len(result.walls) == MIN_STEPS
        assert result.cold_seconds >= 0
        assert result.mode == "streaming" and result.threads == 1
        assert result.strain_seconds == 0.5 * MIN_STEPS
        assert result.algorithmic_wait_seconds == 0.25

    def test_latency_is_step_compute_plus_the_declared_wait(self):
        result = self.replay(sleeper(0.001), steps=MIN_STEPS)
        records = {record["statistic"]: record for record in result.records()}
        assert list(records) == ["median", "p90", "p99", "max"]
        summary = summarise(result.walls)
        for statistic, record in records.items():
            assert record["quantity"] == "latency" and record["units"] == "s"
            assert record["value"] == pytest.approx(summary[statistic] + 0.25)
            assert record["value"] == record["normalised"]
            assert record["value"] >= 0.001 + 0.25
            assert record["mode"] == "streaming" and record["grade"] == "measured"
        values = [records[statistic]["value"] for statistic in ("median", "p90", "p99", "max")]
        assert values == sorted(values)

    def test_the_tail_is_a_step_that_happened(self):
        slow = {500}
        count = [-1]

        def step(chunk):
            count[0] += 1
            if count[0] in slow:
                time.sleep(0.05)

        result = self.replay(step)
        summary = summarise(result.walls)
        assert summary["max"] >= 0.05
        assert summary["median"] < 0.05
        assert summary["p99"] < 0.05, "one slow step in a thousand is the max, not the p99"

    def test_the_streams_own_cost_is_not_timed(self):
        def stream():
            for chunk in range(MIN_STEPS + 1):
                time.sleep(0.001)
                yield chunk

        result = time_streaming(lambda chunk: None, stream(), algorithmic_wait_seconds=0.0, threads=1,
                                scope="network", hardware="test")
        # The stream took at least a second to produce; its steps, doing nothing, took far less.
        assert sum(result.walls) < 0.5

    def test_more_work_per_step_measures_more(self):
        light = self.replay(work(8)).records()
        heavy = self.replay(work(160)).records()
        assert heavy[0]["value"] > light[0]["value"]

    def test_a_replay_that_is_too_short_is_refused(self):
        with pytest.raises(TooFewSteps, match="needs at least 1000"):
            self.replay(sleeper(0), steps=MIN_STEPS - 1)
        with pytest.raises(TooFewSteps, match="empty"):
            time_streaming(sleeper(0), [], algorithmic_wait_seconds=0.0, threads=1, scope="network")

    @pytest.mark.parametrize("kwargs, message", [
        (dict(algorithmic_wait_seconds=-1.0), "algorithmic_wait_seconds must be"),
        (dict(step_strain_seconds=0.0), "step_strain_seconds must be positive"),
        (dict(scope="network or so"), "scope is one of"),
        (dict(threads=0), "threads is the thread count"),
    ])
    def test_refuses(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            self.replay(sleeper(0), **kwargs)


class TestAbba:
    def test_the_order_is_a_b_b_a_each_its_own_call(self):
        log = []
        comparison = compare_abba(lambda: log.append("A"), lambda: log.append("B"), threads=1, hardware="test")
        # One cold call of each, then A, B, B, A.
        assert log == ["A", "B", "A", "B", "B", "A"]
        assert isinstance(comparison, AbbaComparison)
        assert len(comparison.walls_in_order) == 4

    def test_the_numbers_are_the_rules(self):
        comparison = AbbaComparison(a_walls=(1.0, 1.25), b_walls=(2.0, 3.0), cold_a=9.0, cold_b=9.0,
                                    threads=1, machine={"hardware": "test"},
                                    load=HostLoad(0.0, 0.0, 4, 1))
        assert comparison.walls_in_order == (1.0, 2.0, 3.0, 1.25)
        assert comparison.mean_a == 1.125 and comparison.mean_b == 2.5
        assert comparison.spread_a == 0.25
        assert comparison.difference == 1.375
        assert comparison.distinguishable is True

    def test_a_difference_inside_the_spread_is_no_difference(self):
        inside = AbbaComparison((1.0, 1.5), (1.1, 1.3), 0.0, 0.0, 1, {"hardware": "t"}, HostLoad(0, 0, 4, 1))
        assert abs(inside.difference) < inside.spread_a and inside.distinguishable is False
        faster = AbbaComparison((1.0, 1.1), (0.5, 0.6), 0.0, 0.0, 1, {"hardware": "t"}, HostLoad(0, 0, 4, 1))
        assert faster.difference < 0 and faster.distinguishable is True

    def test_a_tenfold_cost_is_distinguished(self):
        comparison = compare_abba(sleeper(0.01), sleeper(0.1), threads=1, hardware="test")
        assert all(wall >= 0.01 for wall in comparison.a_walls)
        assert all(wall >= 0.1 for wall in comparison.b_walls)
        assert comparison.mean_b > comparison.mean_a
        assert comparison.difference > 0.05
        assert comparison.cold_a >= 0.01 and comparison.cold_b >= 0.1

    def test_the_record(self):
        comparison = compare_abba(sleeper(0.001), sleeper(0.001), threads=2, hardware="test")
        record = comparison.to_dict()
        json.dumps(record)
        assert record["order"] == ["A", "B", "B", "A"]
        assert len(record["walls"]) == 4
        assert record["walls"][0] == record["a_walls"][0] and record["walls"][3] == record["a_walls"][1]
        assert record["walls"][1:3] == record["b_walls"]
        assert {"mean_a", "mean_b", "spread_a", "difference", "distinguishable", "cold_seconds",
                "threads", "machine", "host_load"} <= set(record)
        assert record["threads"] == 2 and record["host_load"]["loaded"] in (True, False, None)

    def test_the_load_is_checked(self, monkeypatch):
        monkeypatch.setattr(timing, "_load_average", lambda: 10.0 * (os.cpu_count() or 1))
        comparison = compare_abba(sleeper(0), sleeper(0), threads=1, hardware="test")
        assert comparison.load.loaded is True

    def test_refuses_a_missing_thread_count(self):
        with pytest.raises(ValueError, match="threads is the thread count"):
            compare_abba(sleeper(0), sleeper(0), threads=0)


class TestRecord:
    def test_runtime_records_have_the_catalogues_fields(self):
        batch = time_batch(sleeper(0.001), strain_seconds=2.0, threads=1, scope="network", repeats=2)
        streaming = time_streaming(sleeper(0), range(MIN_STEPS + 1), algorithmic_wait_seconds=0.1, threads=1,
                                   scope="end_to_end")
        for result in (batch, streaming):
            for record in result.records():
                assert set(record) == SCHEMA_FIELDS
                assert record["grade"] == "measured"
                assert record["hardware"] == result.machine["hardware"]
                assert record["quantity"] in ("inference_cost", "latency")
                assert record["mode"] in ("batch", "streaming")
                assert record["scope"] in ("network", "end_to_end")

    def test_the_timing_record_keeps_the_walls_and_the_machine(self):
        result = time_batch(sleeper(0.001), strain_seconds=2.0, threads=3, scope="network", repeats=2)
        record = json.loads(json.dumps(result.to_dict()))
        assert record["walls"] == pytest.approx(list(result.walls))
        assert record["cold_seconds"] == pytest.approx(result.cold_seconds)
        assert record["strain_seconds"] == 2.0 and record["threads"] == 3
        assert record["machine"]["hardware"] and record["machine"]["cpu_count"] >= 1
        assert record["host_load"]["cpu_count"] >= 1 and "loaded" in record["host_load"]
        assert record["runtime"] == json.loads(json.dumps(result.records()))

    def test_a_caller_may_name_the_hardware(self):
        result = time_batch(sleeper(0), strain_seconds=1.0, threads=1, scope="network", repeats=1,
                            hardware="1x test device")
        assert result.records()[0]["hardware"] == "1x test device"


class TestPublicApi:
    def test_the_names_are_public(self):
        assert gf.time_batch is time_batch
        assert gf.time_streaming is time_streaming
        assert gf.compare_abba is compare_abba
        assert gf.Timing is Timing
        assert gf.AbbaComparison is AbbaComparison
