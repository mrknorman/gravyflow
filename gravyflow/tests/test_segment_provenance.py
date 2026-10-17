"""
Real-noise labels and the provenance of the strain read through the store.

``START_GPS_TIME`` is the GPS time of the first sample of the onsource window
returned after cropping, counted from the first sample read (never the nominal
segment start). Strain is stored as the source's own tiles, keyed by the source, the
detector, the rate and the time alone, and each tile's record carries its first
sample's time, rate and source, so warm reads equal cold ones and every
configuration of one source shares its tiles. Content fingerprints read every
parameter that determines the data.

The strain here is a fake GWOSC fetch: the samples of an interval on the absolute
grid at the requested rate, each valued its index from a fixed grid sample, so a
window's first sample decodes to its GPS time.
"""

import inspect
import logging

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.acquisition.sources import SegmentStrain
from gravyflow.src.dataset.store import Store
from gravyflow.src.dataset.store.tile import key_from_path
from gravyflow.src.validate.provenance import UnfingerprintableError, canonical, config_fingerprint
from gravyflow.tests.conftest import gwosc_serving

RATE = 256.0
ORIGIN = 1238175000.0  # inside O3; the fake's values count from each segment's start
ONSOURCE, CROP, OFFSOURCE, BATCH = 1.0, 0.5, 4.0, 8
#: One row, a piece per detector, starting off the sample grid and off whole seconds.
ROW = [[ORIGIN + 0.3, ORIGIN + 64.3], [ORIGIN + 100.7, ORIGIN + 164.7]]
#: The first sample of GWOSC's 4096 s tile holding ROW: the fake's values count from it.
TILE = 1238171648.0
IFOS = [gf.IFO.H1, gf.IFO.L1]


class GridFetch:
    """
    A fake GWOSC fetch, publishing RATE: the samples of [start, end) on the absolute
    grid, from the first grid sample at or after ``start`` (``first_grid_gps``), each
    valued its index from TILE's first sample. Records its calls.
    """

    rates = (RATE,)
    source_id = "gwosc-fake-grid"

    def __init__(self):
        self.calls = []

    def __call__(self, ifo, sample_rate_hertz, interval):
        start, end = interval
        self.calls.append((ifo.name, float(start), float(end)))
        rate = float(sample_rate_hertz)
        first, stop = int(np.ceil(start * rate - 1e-6)), int(np.ceil(end * rate - 1e-6))
        values = np.arange(first, stop, dtype=np.float64) - TILE * rate
        return SegmentStrain(values.astype(np.float32), first / rate, rate, self.source_id)


def first_grid_gps(start, rate=RATE):
    return np.ceil(start * rate - 1e-6) / rate


def refuse_fetch(*args, **kwargs):
    raise AssertionError("fetched although the segment was cached")


refuse_fetch.rates = (RATE,)
refuse_fetch.source_id = GridFetch.source_id


def fetching(fetch):
    """Every GWOSC() source serves ``fetch`` (``conftest.gwosc_serving``)."""
    return gwosc_serving(fetch)


def obtainer(**kwargs):
    arguments = dict(data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE],
                     observing_runs=gf.ObservingRun.O3)
    arguments.update(kwargs)
    built = gf.IFODataObtainer(**arguments)
    built.augmentations = []
    return built


def batches(source, mode, num_batches=2, seed=3):
    """``num_batches`` batches of the one row ROW, as the obtainer serves them."""
    source.valid_segments = np.asarray([ROW], dtype=np.float64)
    generator = source.get_onsource_offsource_chunks(
        sample_rate_hertz=RATE, onsource_duration_seconds=ONSOURCE, crop_duration_seconds=CROP,
        offsource_duration_seconds=OFFSOURCE, num_examples_per_batch=BATCH, ifos=IFOS,
        scale_factor=1.0, seed=seed, sampling_mode=mode, crop=True,
    )
    return [next(generator) for _ in range(num_batches)]


def decoded_start_gps(onsource):
    """The GPS time of each (example, detector)'s first returned sample, decoded."""
    first = np.rint(np.asarray(onsource, dtype=np.float64)[..., 0])
    return TILE + first / RATE


class TestStartGpsTime:

    @pytest.mark.parametrize("mode", [gf.SamplingMode.RANDOM, gf.SamplingMode.GRID], ids=lambda m: m.name)
    def test_is_the_first_returned_onsource_sample(self, mode):
        with fetching(GridFetch()):
            served = batches(obtainer(), mode)
        for batch in served:
            onsource = batch[gf.ReturnVariables.ONSOURCE]
            assert onsource.shape == (BATCH, len(IFOS), int(ONSOURCE * RATE))
            labels = np.asarray(batch[gf.ReturnVariables.START_GPS_TIME])
            assert labels.dtype == np.float64
            # Each window is consecutive samples of its own piece.
            steps = np.diff(np.asarray(onsource, dtype=np.float64), axis=-1)
            assert np.all(np.abs(steps - 1.0) < 0.25)
            error = labels - decoded_start_gps(onsource)
            assert np.max(np.abs(error)) <= 1e-6, f"label minus first sample: {error.min()} to {error.max()} s"

    def test_is_one_definition_with_the_crop(self):
        spec = gf.WindowSpec.from_params(sample_rate_hertz=RATE, onsource_duration_seconds=ONSOURCE,
                                         offsource_duration_seconds=OFFSOURCE, crop_duration_seconds=CROP)
        padded = np.arange(spec.num_onsource_samples, dtype=np.float32)[None, :]
        first_kept = float(np.asarray(gf.crop_samples(padded, ONSOURCE, RATE))[0, 0])
        assert spec.num_crop_samples == first_kept == CROP * RATE
        assert spec.onsource_start_gps(np.array([ORIGIN])) == pytest.approx(ORIGIN + CROP, abs=1e-9)


def stored_records():
    """Every committed tile's record in the store under the cache root."""
    store = Store(gf.src.dataset.acquisition.gwosc_data.user_cache_dir("store"))
    return [store.open(key_from_path(store.root, path)).record
            for path in sorted(store.root.glob("v*/*/*/*/*/*.json"))]


class TestStoreReads:

    def test_warm_build_equals_cold(self):
        with fetching(GridFetch()) as fetch:
            cold_batches = batches(obtainer(), gf.SamplingMode.GRID)
        # One whole tile per detector, fetched once.
        assert sorted(fetch.calls) == [(ifo.name, TILE, TILE + 4096.0) for ifo in IFOS]
        records = stored_records()
        assert [(r.key.detector, r.key.t0, r.key.rate, r.key.source_id) for r in records] == [
            (ifo.name, TILE, int(RATE), GridFetch.source_id) for ifo in IFOS]

        with fetching(refuse_fetch):
            warm_batches = batches(obtainer(), gf.SamplingMode.GRID)
        for cold_batch, warm_batch in zip(cold_batches, warm_batches):
            for key in (gf.ReturnVariables.ONSOURCE, gf.ReturnVariables.OFFSOURCE,
                        gf.ReturnVariables.START_GPS_TIME):
                np.testing.assert_array_equal(np.asarray(cold_batch[key]), np.asarray(warm_batch[key]))

    def test_configurations_of_one_source_share_its_tiles(self):
        # A tile's content depends only on its source, detector, rate and time, so
        # the selection (flags, splits, bins) chooses which tiles are read, never what
        # they hold, and every configuration reads the tiles once fetched.
        with fetching(GridFetch()) as fetch:
            batches(obtainer(), gf.SamplingMode.GRID)
        with fetching(refuse_fetch):
            for other in (obtainer(gwosc_flags=gf.GWOSC_DATA_FLAGS),
                          obtainer(max_segment_duration_seconds=1024.0),
                          obtainer(split_spec=gf.SplitSpec(version="fold-1"))):
                batches(other, gf.SamplingMode.GRID)
        assert len(fetch.calls) == len(IFOS)


#: Each data-determining parameter of the real-noise obtainers, and another value.
OBTAINER_PERTURBATIONS = {
    "data_labels": [gf.DataLabel.GLITCHES],
    "observing_runs": gf.ObservingRun.O2,
    "segment_order": gf.SegmentOrder.CHRONOLOGICAL,
    "max_segment_duration_seconds": 1024.0,
    "saturation": 4.0,
    "overrides": {"end_gps_times": ORIGIN + 1.0e6},
    "event_types": [gf.EventConfidence.MARGINAL],
    "augmentations": [gf.SignReversal()],
    "balanced_glitch_types": True,
    "event_names": ["GW150914"],
    "gwosc_flags": gf.GWOSC_DATA_FLAGS,
    "split_spec": gf.SplitSpec(version="fold-1"),
    "source": gf.LocalFrames(),
    "train_on_real_events": True,
    "coincidence": gf.Coincidence.COINCIDENT,
    "exclusions": [("H1", (ORIGIN, ORIGIN + 1.0), "a test exclusion")],
}
#: Data-determining parameters with one buildable value (public runs publish BEST only).
SINGLE_VALUED = {"data_quality"}
#: Parameters that change how data is fetched or logged, never the data.
NOT_DATA = {"force_acquisition", "logging_level"}
#: Parameters a NOISE obtainer ignores (they select transients).
TRANSIENT_ONLY = {"event_types", "balanced_glitch_types", "event_names", "train_on_real_events"}


class TestFingerprint:

    def base_arguments(self):
        return dict(data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE],
                    observing_runs=gf.ObservingRun.O3)

    def test_every_obtainer_parameter_is_classified(self):
        parameters = set(inspect.signature(gf.IFODataObtainer).parameters)
        assert parameters == set(OBTAINER_PERTURBATIONS) | SINGLE_VALUED | NOT_DATA
        transient = {**self.base_arguments(), "data_labels": [gf.DataLabel.GLITCHES]}
        assert SINGLE_VALUED <= set(gf.IFODataObtainer(**transient).specification())

    @pytest.mark.parametrize("name", sorted(OBTAINER_PERTURBATIONS))
    def test_changes_with_every_data_parameter(self, name):
        arguments = self.base_arguments()
        if name in TRANSIENT_ONLY:
            arguments["data_labels"] = [gf.DataLabel.GLITCHES]
        if name == "train_on_real_events":
            # Training on real events needs the coincident opt-in.
            arguments["coincidence"] = gf.Coincidence.COINCIDENT
        reference = config_fingerprint(gf.IFODataObtainer(**arguments))
        arguments[name] = OBTAINER_PERTURBATIONS[name]
        assert config_fingerprint(gf.IFODataObtainer(**arguments)) != reference

    @pytest.mark.parametrize("name", sorted(NOT_DATA))
    def test_ignores_how_data_is_fetched(self, name):
        changed = {"force_acquisition": True, "logging_level": logging.DEBUG}
        reference = config_fingerprint(gf.IFODataObtainer(**self.base_arguments()))
        arguments = {**self.base_arguments(), name: changed[name]}
        assert config_fingerprint(gf.IFODataObtainer(**arguments)) == reference

    def test_noise_obtainer_reads_its_data_obtainer(self):
        # O1 with the default flags and O3 with DATA alone once shared a fingerprint:
        # the data obtainer was read as its type's name.
        def noise(**kwargs):
            return gf.NoiseObtainer(noise_type=gf.NoiseType.REAL, ifos=[gf.IFO.H1],
                                    ifo_data_obtainer=gf.IFODataObtainer(**{**self.base_arguments(), **kwargs}))
        assert config_fingerprint(noise(observing_runs=gf.ObservingRun.O1)) \
            != config_fingerprint(noise(gwosc_flags=gf.GWOSC_DATA_FLAGS))

        fields = set(inspect.signature(gf.NoiseObtainer).parameters)
        reference = config_fingerprint(gf.NoiseObtainer(noise_type=gf.NoiseType.REAL, ifos=[gf.IFO.H1]))
        changed = {"ifo_data_obtainer": noise(observing_runs=gf.ObservingRun.O1).ifo_data_obtainer,
                   "ifos": [gf.IFO.L1], "noise_type": gf.NoiseType.PSEUDO_REAL,
                   "noise_profile": gf.NoiseProfile.O3_LOW}
        # The split is the data obtainer's (its split_spec), read with it.
        assert fields == set(changed)
        for name, value in changed.items():
            other = gf.NoiseObtainer(**{"noise_type": gf.NoiseType.REAL, "ifos": [gf.IFO.H1], name: value})
            assert config_fingerprint(other) != reference, name

    def test_refuses_what_it_cannot_read(self):
        with pytest.raises(UnfingerprintableError, match="builtins.object"):
            canonical(object())
