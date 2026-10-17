"""
TRANSIENT mode, offline: the event list as GWOSC's event API serves it (the committed
snapshot), and datasets of real events read from the catalogue fake ``offline_events``
(``conftest``), whose strain is unit Gaussian noise seeded by each event's time.
"""

import csv

import numpy as np
import pytest

import gravyflow as gf
# Imported before ``offline_events`` patches the event module: a first import inside that
# patch would bind the patched function here as the original, outliving the fixture.
from gravyflow.src.dataset.features import segment_builders  # noqa: F401
from gravyflow.tests.test_public_sources import FIXTURES, offline_event_api  # noqa: F401  (pytest fixture)

RV = gf.ReturnVariables
RATE, ONSOURCE, OFFSOURCE = 2048.0, 1.0, 16.0


class TestTransientSegmentRetrieval:
    """Every row of every catalogue the event API lists is an event time."""

    @pytest.fixture(autouse=True)
    def _events(self, offline_event_api):
        return offline_event_api

    @staticmethod
    def snapshot_times():
        with open(FIXTURES / "gwosc_events.csv", newline="") as handle:
            return np.array([float(row["gps"]) for row in csv.DictReader(handle) if row["gps"]])

    def test_every_listed_row_is_an_event_time(self):
        np.testing.assert_array_equal(np.sort(gf.get_all_event_times()), np.sort(self.snapshot_times()))

    def test_o3_events_are_those_in_its_bounds(self):
        run = gf.ObservingRun.O3.value
        times = gf.get_all_event_times()
        in_o3 = lambda t: t[(t >= run.start_gps_time) & (t <= run.end_gps_time)]
        expected = in_o3(self.snapshot_times())
        assert len(expected) > 0
        np.testing.assert_array_equal(np.sort(in_o3(times)), np.sort(expected))


def events_dataset(ifos, batch, input_variable=RV.WHITENED_ONSOURCE):
    """A test-split dataset of real events, one batch of ``batch`` examples."""
    obtainer = gf.IFODataObtainer(observing_runs=gf.ObservingRun.O3, data_quality=gf.DataQuality.BEST,
                                  data_labels=[gf.DataLabel.EVENTS])
    obtainer.augmentations = []
    noise = gf.NoiseObtainer(ifo_data_obtainer=obtainer, ifos=ifos, noise_type=gf.NoiseType.REAL)
    return gf.Dataset(
        noise_obtainer=noise, sample_rate_hertz=RATE, onsource_duration_seconds=ONSOURCE,
        offsource_duration_seconds=OFFSOURCE, num_examples_per_batch=batch, steps_per_epoch=1,
        group="test", input_variables=[input_variable],
        output_variables=[RV.START_GPS_TIME],
    )


class TestTransientDatasetOutput:

    def test_a_batch_holds_each_event_once_centred_in_its_window(self, offline_events):
        x, y = events_dataset([gf.IFO.L1], batch=len(offline_events))[0]
        whitened = np.asarray(x[RV.WHITENED_ONSOURCE.name])
        assert whitened.shape == (len(offline_events), 1, int(ONSOURCE * RATE))
        assert np.all(np.isfinite(whitened)) and np.any(whitened != 0)
        starts = np.sort(np.asarray(y[RV.START_GPS_TIME.name], dtype=np.float64)[:, 0])
        # Each window is centred on its event: it starts at the sample at or before the
        # event less half the window. GPS near 1.3e9 holds 2.4e-7 s steps in float64: a
        # few of them, never a sample (4.9e-4 s), separate the label from the formula.
        events = np.array(sorted(e["gps"] for e in offline_events))
        expected = np.floor((events - ONSOURCE / 2) * RATE) / RATE
        np.testing.assert_allclose(starts, expected, rtol=0, atol=4 * np.spacing(events.max()))

    def test_an_events_detectors_are_read_at_one_time(self, offline_events):
        """A real event is the stated exception to separated detectors: every detector at its time."""
        x, y = events_dataset([gf.IFO.H1, gf.IFO.L1], batch=len(offline_events), input_variable=RV.ONSOURCE)[0]
        assert np.shape(x[RV.ONSOURCE.name]) == (len(offline_events), 2, int(ONSOURCE * RATE))
        gps = np.asarray(y[RV.START_GPS_TIME.name], dtype=np.float64)
        np.testing.assert_array_equal(gps[:, 0], gps[:, 1])


@pytest.fixture
def listed_events():
    """
    The offline events as the one event catalogue lists them (``conftest.catalogued_events``),
    so the library's own selection (by run, confidence and name) chooses among them; every
    record passes data quality and reads Gaussian strain seeded by its time.
    """
    from unittest import mock

    from gravyflow.tests.conftest import OFFLINE_EVENTS, catalogued_events, usable_over_its_runs

    def gaussian_strain(self, record, reads, window_spec):
        rng = np.random.default_rng(int(record.transient_gps_time))
        shape = (len(reads),)
        return (rng.standard_normal(shape + (window_spec.num_onsource_samples,)).astype(np.float32),
                rng.standard_normal(shape + (window_spec.num_offsource_samples,)).astype(np.float32))

    with mock.patch.object(gf.TransientDataObtainer, "_read_example", gaussian_strain), \
         mock.patch.object(gf.TransientDataObtainer, "get_all_segment_times", usable_over_its_runs), \
         catalogued_events(OFFLINE_EVENTS):
        yield [dict(e) for e in OFFLINE_EVENTS]


class TestSpecificEventAcquisition:

    def test_named_events_are_the_events_served(self, listed_events):
        names = [listed_events[0]["name"], listed_events[2]["name"]]
        obtainer = gf.TransientDataObtainer(
            observing_runs=gf.ObservingRun.O3, data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.EVENTS], event_names=names,
        )
        obtainer.augmentations = []
        batch = next(obtainer.get_onsource_offsource_chunks(
            sample_rate_hertz=RATE, onsource_duration_seconds=ONSOURCE, crop_duration_seconds=0.5,
            offsource_duration_seconds=OFFSOURCE, num_examples_per_batch=2,
            ifos=[gf.IFO.H1, gf.IFO.L1], group="test",
        ))
        assert np.shape(batch[RV.ONSOURCE])[:2] == np.shape(batch[RV.OFFSOURCE])[:2] == (2, 2)
        assert {segment.name for segment in obtainer.transient_segments} == set(names)
        served = np.asarray(batch[RV.TRANSIENT_GPS_TIME], dtype=np.float64)
        np.testing.assert_array_equal(served[:, 0], served[:, 1])
        assert set(served[:, 0]) == {e["gps"] for e in listed_events if e["name"] in names}
