"""
Transient windows: their geometry, their labels and their data quality.

A transient's window is read from one stretch of strain: the stored offsource
followed by the stored onsource, the transient on its stored onsource's centre
sample. Every request, from the download or from any cache read path, takes its
padded onsource centred on that sample and its offsource ending where the padded
onsource begins, as for noise. ``START_GPS_TIME`` is the GPS time of the first
onsource sample returned after cropping, as for noise, never the transient's time.

The strain is a fake GWOSC fetch: the samples of [start, end) on the absolute grid
at the requested rate, each valued its GPS time minus ``REF``, so a returned sample
decodes to its GPS time.
"""

from contextlib import contextmanager
from unittest import mock

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

import gravyflow as gf
from gravyflow.src.dataset.acquisition.base import BaseDataObtainer
from gravyflow.src.dataset.acquisition.transient_segment import TransientSegment
from gravyflow.src.dataset.config import TransientDefaults, WindowSpec
from gravyflow.src.dataset.features.glitch import GlitchType
from gravyflow.src.dataset.features.glitch_cache import TransientCache
from gravyflow.src.utils.gps import gps_to_key

RV = gf.ReturnVariables
REF = 1240000000.0  # inside O3; every sample is valued its GPS time minus REF
STORED_RATE = TransientDefaults.CACHE_SAMPLE_RATE_HERTZ
RATE = 2048.0
WINDOW = dict(
    sample_rate_hertz=RATE,
    onsource_duration_seconds=1.0,
    crop_duration_seconds=0.5,
    offsource_duration_seconds=16.0,
    num_examples_per_batch=3,
    scale_factor=1.0,
)
SPEC = WindowSpec.from_params(
    sample_rate_hertz=RATE, onsource_duration_seconds=1.0,
    crop_duration_seconds=0.5, offsource_duration_seconds=16.0,
)
#: Three glitches, off the sample grid at every rate.
TIMES = REF + np.array([10.3701, 150.01234, 290.90007])
#: Times inside one data-quality segment covering every record.
SCIENCE = np.array([[REF - 1000.0, REF + 1000.0]])


def records(times=TIMES, seen_in=(gf.IFO.L1,)):
    return [
        TransientSegment(
            gps_key=gps_to_key(float(t)),
            transient_gps_time=float(t),
            start_gps_time=float(t) - 48.2,
            end_gps_time=float(t) + 48.2,
            label=gf.DataLabel.GLITCHES,
            kind=GlitchType.BLIP,
            observing_run=gf.ObservingRun.O3,
            seen_in=list(seen_in),
        )
        for t in times
    ]


class GridFetch:
    """
    A fake GWOSC fetch: the samples of [start, end) on the absolute grid at the
    requested rate, from the first grid sample at or after ``start``, each valued
    its GPS time minus REF. ``bad`` maps a detector name to the times whose
    download fails (raises) or, with ``nan``, returns NaN around them.
    """

    def __init__(self, bad=None, nan=False):
        self.calls = []
        self.bad = bad or {}
        self.nan = nan

    def __call__(self, start, end, ifo, frame_type, channel, sample_rate_hertz=None):
        self.calls.append((ifo.name, float(start), float(end)))
        rate = float(sample_rate_hertz)
        first, stop = int(np.ceil(start * rate - 1e-6)), int(np.ceil(end * rate - 1e-6))
        times = np.arange(first, stop, dtype=np.float64) / rate
        values = times - REF
        for t in self.bad.get(ifo.name, ()):
            if start <= t <= end:
                if not self.nan:
                    raise ValueError(f"no {ifo.name} data at {t}")
                values[np.abs(times - t) < 2.0] = np.nan
        return TimeSeries(values, t0=first / rate, sample_rate=rate, channel=f"{ifo.name}:FAKE-GRID")


@contextmanager
def offline(fetch=None, catalogue=None, science=SCIENCE):
    """The glitch catalogue, its data quality and GWOSC strain, offline."""
    fetch = fetch or GridFetch()
    with mock.patch("gravyflow.src.dataset.features.segment_builders.build_glitch_segments",
                    return_value=catalogue if catalogue is not None else records()), \
         mock.patch.object(gf.TransientDataObtainer, "get_all_segment_times",
                           lambda self, ifo: science), \
         mock.patch.object(BaseDataObtainer, "get_segment_data_static", staticmethod(fetch)):
        yield fetch


def obtainer(directory):
    obtainer = gf.TransientDataObtainer(
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.GLITCHES],
        observing_runs=gf.ObservingRun.O3,
        augmentations=[],
    )
    obtainer.data_directory = directory
    return obtainer


def first_batch(directory, ifos, **window):
    return next(obtainer(directory)(ifos=ifos, group="all", seed=1, **{**WINDOW, **window}))


def transient_times(batch):
    """Each example's transient time, from the decoded centre of its padded onsource."""
    onsource = np.asarray(batch[RV.ONSOURCE], dtype=np.float64)
    return REF + onsource[..., onsource.shape[-1] // 2]


IFO_SETS = [[gf.IFO.L1], [gf.IFO.H1, gf.IFO.L1]]


class TestStartGpsTime:
    """START_GPS_TIME is the first returned onsource sample's time (the owner's ruling)."""

    @pytest.mark.parametrize("ifos", IFO_SETS, ids=lambda ifos: "".join(i.name for i in ifos))
    def test_obtainer_labels_the_first_cropped_sample(self, tmp_path, ifos):
        with offline():
            batch = first_batch(tmp_path, ifos)
        onsource = np.asarray(batch[RV.ONSOURCE], dtype=np.float64)
        first = REF + onsource[..., SPEC.num_crop_samples]
        label = np.asarray(batch[RV.START_GPS_TIME])
        assert label.dtype == np.float64 and label.shape == first.shape
        error = label - first
        assert np.max(np.abs(error)) <= 1.0 / RATE, (
            f"START_GPS_TIME minus the first cropped onsource sample: "
            f"{error.min():.6f} to {error.max():.6f} s"
        )

    def test_dataset_labels_the_first_cropped_sample(self, tmp_path):
        with offline():
            dataset = gf.Dataset(
                noise_obtainer=obtainer(tmp_path),
                ifos=[gf.IFO.L1],
                sample_rate_hertz=RATE,
                onsource_duration_seconds=1.0,
                crop_duration_seconds=0.5,
                offsource_duration_seconds=16.0,
                num_examples_per_batch=3,
                steps_per_epoch=1,
                scale_factor=1.0,
                group="all",
                input_variables=[RV.ONSOURCE],
                output_variables=[RV.START_GPS_TIME],
            )
            x, y = dataset[0]
        first = REF + np.asarray(x[RV.ONSOURCE.name], dtype=np.float64)[..., 0]
        error = np.asarray(y[RV.START_GPS_TIME.name]) - first.reshape(np.shape(y[RV.START_GPS_TIME.name]))
        assert np.max(np.abs(error)) <= 1.0 / RATE, (
            f"Dataset START_GPS_TIME minus the first returned onsource sample: "
            f"{error.min():.6f} to {error.max():.6f} s"
        )


class TestGeometry:
    """One geometry for every transient window (B204)."""

    @pytest.mark.parametrize("ifos", IFO_SETS, ids=lambda ifos: "".join(i.name for i in ifos))
    def test_offsource_ends_where_the_padded_onsource_begins(self, tmp_path, ifos):
        with offline():
            batch = first_batch(tmp_path, ifos)
        onsource = np.asarray(batch[RV.ONSOURCE], dtype=np.float64)
        offsource = np.asarray(batch[RV.OFFSOURCE], dtype=np.float64)
        assert offsource.shape[-1] == SPEC.num_offsource_samples
        gap = onsource[..., 0] - offsource[..., -1]
        assert np.allclose(gap, 1.0 / RATE, atol=1e-4), f"offsource to onsource: {gap.ravel()} s"
        assert np.allclose(np.diff(offsource, axis=-1), 1.0 / RATE, atol=1e-4), "offsource not contiguous"

    def test_the_transient_is_the_padded_onsource_centre(self, tmp_path):
        with offline():
            batch = first_batch(tmp_path, [gf.IFO.L1])
        served = np.sort(np.unique(np.round(transient_times(batch), 3)))
        error = np.abs(np.sort(transient_times(batch)[:, 0]) - TIMES)
        assert np.max(error) <= 1.0 / RATE, f"centre minus transient time: {error} s ({served})"

    def test_warm_equals_cold(self, tmp_path):
        with offline() as fetch:
            cold = first_batch(tmp_path, [gf.IFO.H1, gf.IFO.L1])
            fetched = len(fetch.calls)
            warm = first_batch(tmp_path, [gf.IFO.H1, gf.IFO.L1])
        assert fetched > 0 and len(fetch.calls) == fetched, "the second read was not a cache hit"
        order = np.argsort(np.asarray(cold[RV.TRANSIENT_GPS_TIME])[:, 0])
        again = np.argsort(np.asarray(warm[RV.TRANSIENT_GPS_TIME])[:, 0])
        for key in (RV.ONSOURCE, RV.OFFSOURCE, RV.START_GPS_TIME):
            np.testing.assert_array_equal(
                np.asarray(cold[key])[order], np.asarray(warm[key])[again], err_msg=key.name
            )

    def test_every_cache_read_path_reads_one_geometry(self, tmp_path):
        with offline():
            expected = first_batch(tmp_path, [gf.IFO.H1, gf.IFO.L1])
        order = np.argsort(np.asarray(expected[RV.TRANSIENT_GPS_TIME])[:, 0])
        on_expected = np.asarray(expected[RV.ONSOURCE])[order]
        off_expected = np.asarray(expected[RV.OFFSOURCE])[order]
        cache = TransientCache(tmp_path / "glitch_cache.h5", mode="r")
        keys = [gps_to_key(float(t)) for t in TIMES]
        indices = np.array([cache._build_gps_index()[key] for key in keys])
        request = dict(sample_rate_hertz=RATE, onsource_duration=SPEC.total_onsource_duration_seconds,
                       offsource_duration=SPEC.offsource_duration_seconds)
        reads = {"get_batch": cache.get_batch(indices, target_ifos=["H1", "L1"], **request)[:2]}
        by_key = [cache.get_by_key(key, **request) for key in keys]
        reads["get_by_key"] = (np.stack([r[0] for r in by_key]), np.stack([r[1] for r in by_key]))
        cache.enable_chunked_mode(chunk_size=2, **request)
        reads["get_batch_from_chunk"] = cache.get_batch_from_chunk(indices, target_ifos=["H1", "L1"])[:2]
        chunked = [cache.get_from_chunk(int(i), target_ifos=["H1", "L1"]) for i in indices]
        reads["get_from_chunk"] = (np.stack([r[0] for r in chunked]), np.stack([r[1] for r in chunked]))
        for path, (onsource, offsource) in reads.items():
            np.testing.assert_array_equal(onsource, on_expected, err_msg=f"{path} onsource")
            np.testing.assert_array_equal(offsource, off_expected, err_msg=f"{path} offsource")
