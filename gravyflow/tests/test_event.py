"""
The event list: event times by confidence, and the events with their parameter
estimates, all read from the one catalogue.
"""

from pathlib import Path

import pytest
import numpy as np
from gwpy.table import EventTable

import gravyflow as gf
from gravyflow.src.resources import PACKAGE
from gravyflow.src.dataset.features import event as event_module
from gravyflow.tests.test_public_sources import FIXTURES, offline_event_api  # noqa: F401  (pytest fixture)


_CATALOGUE = {
    "gps": np.array([1126259462.4, 1187008882.4, 1200000000.0, 1250000000.0]),
    "chirp_mass_source": np.array([28.6, 1.186, np.nan, np.nan]),
    "name": np.array(["GW150914", "GW170817", "marginal", "ias"]),
    "catalog": np.array(["GWTC-1-confident", "GWTC-1-confident", "GWTC-1-marginal",
                         "IAS-O3a"]),
}


class TestEventTimeFunctions:
    """The event-time functions read the one event list, ``get_event_catalogue``."""

    @pytest.fixture(autouse=True)
    def catalogue(self, monkeypatch):
        monkeypatch.setattr(event_module, "get_event_catalogue",
                            lambda use_cache=True: _CATALOGUE)

    def test_get_confident_event_times(self):
        np.testing.assert_array_equal(gf.get_confident_event_times(use_cache=False),
                                      [1126259462.4, 1187008882.4])

    def test_get_marginal_event_times(self):
        np.testing.assert_array_equal(gf.get_marginal_event_times(use_cache=False),
                                      [1200000000.0])

    def test_get_all_event_times_is_every_vetoed_event(self):
        np.testing.assert_array_equal(gf.get_all_event_times(use_cache=False),
                                      _CATALOGUE["gps"])

    def test_get_event_times_by_type_confident_only(self):
        times = gf.get_event_times_by_type([gf.EventConfidence.CONFIDENT], use_cache=False)
        np.testing.assert_array_equal(times, [1126259462.4, 1187008882.4])

    def test_get_event_times_by_type_empty_list(self):
        times = gf.get_event_times_by_type([], use_cache=False)
        assert len(times) == 0


_NAN = np.nan

#: The one event list with its parameter-estimation columns: GW150914 twice (the
#: cumulative GWTC table and GWTC-1), a confident O3 event without masses, a marginal
#: and an IAS candidate.
_PE_CATALOGUE = {
    "gps": np.array([1126259462.4, 1126259462.43, 1187008882.4, 1240000000.2,
                     1200000000.0, 1250000000.0]),
    "chirp_mass_source": np.array([28.6, 28.6, 1.186, _NAN, _NAN, _NAN]),
    "name": np.array(["GW150914", "GW150914", "GW170817", "GW-NO-MASSES", "marginal", "ias"]),
    "catalog": np.array(["GWTC", "GWTC-1-confident", "GWTC-1-confident", "GWTC-3-confident",
                         "GWTC-1-marginal", "IAS-O3a"]),
    "mass_1_source": np.array([35.6, 36.2, 1.46, _NAN, 20.0, 30.0]),
    "mass_2_source": np.array([30.6, 29.1, 1.27, _NAN, 10.0, 25.0]),
    "luminosity_distance": np.array([440.0, 430.0, 40.0, _NAN, 1000.0, 2000.0]),
    "p_astro": np.array([1.0, 1.0, 1.0, _NAN, 0.5, 0.9]),
}


def _refuse_table_fetch(*args, **kwargs):
    raise AssertionError("get_events_with_params fetched a catalogue table itself")


class TestEventsWithParams:
    """
    ``get_events_with_params`` reads the one event list, ``get_event_catalogue``, with
    its parameter-estimation columns: no second list, no fetch of its own and no
    file inside the package.
    """

    @pytest.fixture(autouse=True)
    def catalogue(self, monkeypatch):
        monkeypatch.setattr(event_module, "get_event_catalogue",
                            lambda use_cache=True: _PE_CATALOGUE)
        monkeypatch.setattr(EventTable, "fetch_open_data", _refuse_table_fetch)

    def test_confident_events_from_the_one_list(self):
        events = event_module.get_events_with_params()
        assert [e["name"] for e in events] == ["GW150914", "GW170817", "GW-NO-MASSES"]
        assert [e["gps"] for e in events] == sorted(e["gps"] for e in events)
        # A duplicate keeps its first catalogue's row: the cumulative GWTC table.
        gw150914 = events[0]
        assert (gw150914["catalog"], gw150914["mass1"], gw150914["mass2"],
                gw150914["distance"], gw150914["p_astro"]) == ("GWTC", 35.6, 30.6, 440.0, 1.0)
        assert [e["observing_run"] for e in events] == ["O1", "O2", "O3"]

    def test_one_event_under_three_catalogues_is_one_record(self, monkeypatch):
        # One name, its GPS times across a 0.1 s rounding edge: rounding kept two.
        listed = dict(_PE_CATALOGUE)
        listed.update(
            gps=np.array([1126259462.44, 1126259462.46, 1126259462.49]),
            chirp_mass_source=np.array([28.6, 28.6, 28.6]),
            name=np.array(["GW150914"] * 3),
            catalog=np.array(["GWTC", "GWTC-1-confident", "GWTC-2.1-confident"]),
            **{key: _PE_CATALOGUE[key][:3] for key in
               ("mass_1_source", "mass_2_source", "luminosity_distance", "p_astro")},
        )
        monkeypatch.setattr(event_module, "get_event_catalogue", lambda use_cache=True: listed)
        events = event_module.get_events_with_params()
        assert [(e["name"], e["catalog"]) for e in events] == [("GW150914", "GWTC")]

    def test_source_types_from_catalogue_masses(self):
        """Median source masses against 3 Msun; no masses is UNKNOWN, never BBH."""
        from gravyflow.src.dataset.features.segment_builders import build_event_segments

        kinds = {s.name: s.kind for s in build_event_segments(observing_runs=None)}
        assert kinds == {"GW150914": gf.SourceType.BBH, "GW170817": gf.SourceType.BNS,
                         "GW-NO-MASSES": gf.SourceType.UNKNOWN}
        source_type = event_module.source_type_from_masses
        assert source_type(1.4, 10.0) == source_type(10.0, 1.4) == gf.SourceType.NSBH
        assert source_type(23.2, 2.6) == gf.SourceType.NSBH  # GW190814's secondary
        assert source_type(3.0, 3.0) == gf.SourceType.BBH

    def test_marginal_events(self):
        events = event_module.get_events_with_params(event_types=[gf.EventConfidence.MARGINAL])
        assert [e["name"] for e in events] == ["marginal"]

    def test_runs_and_names_select(self):
        assert [e["name"] for e in event_module.get_events_with_params([gf.ObservingRun.O2])] == ["GW170817"]
        named = event_module.get_events_with_params(event_names=["GW-NO-MASSES"])
        assert [e["name"] for e in named] == ["GW-NO-MASSES"]

    def test_missing_parameters_read_as_nan(self, monkeypatch):
        (event,) = event_module.get_events_with_params(event_names=["GW-NO-MASSES"])
        assert all(np.isnan(event[key]) for key in ("mass1", "mass2", "distance", "p_astro"))
        # A table without the columns at all (an older list) reads the same way.
        bare = {key: _PE_CATALOGUE[key] for key in ("gps", "chirp_mass_source", "name", "catalog")}
        monkeypatch.setattr(event_module, "get_event_catalogue", lambda use_cache=True: bare)
        (event,) = event_module.get_events_with_params(event_names=["GW170817"])
        assert np.isnan(event["mass1"]) and event["name"] == "GW170817"

    def test_search_reads_the_parameters(self):
        assert gf.search_events(source_type=gf.SourceType.BNS) == ["GW170817"]
        assert gf.search_events(source_type="bns") == ["GW170817"]
        assert gf.search_events(distance_range=(None, 100.0)) == ["GW170817"]

    def test_nothing_is_written_inside_the_package(self):
        package = Path(PACKAGE)
        before = sorted(str(p) for p in package.rglob("*") if "__pycache__" not in p.parts)
        event_module.get_events_with_params()
        event_module.get_events_with_params(event_types=[gf.EventConfidence.MARGINAL])
        after = sorted(str(p) for p in package.rglob("*") if "__pycache__" not in p.parts)
        assert after == before


class TestEventCatalogueParameters:
    """The cached event list carries the parameter-estimation columns and mass bounds (format 4)."""

    def test_columns_are_fetched_cached_and_read_back(self, monkeypatch):
        from astropy.table import Table
        from gwosc import api as gwosc_api

        table = Table(rows=[("GW150914", 1126259462.4, 28.6, 35.6, 30.6, 440.0, 1.0)],
                      names=("commonName", "GPS", "chirp_mass_source", "mass_1_source",
                             "mass_2_source", "luminosity_distance", "p_astro"))
        monkeypatch.setattr(gwosc_api, "fetch_cataloglist_json",
                            lambda *args, **kwargs: {"GWTC-1-confident": {}})
        monkeypatch.setattr(EventTable, "fetch_open_data",
                            lambda catalog, *args, **kwargs: table)
        fetched = event_module.get_event_catalogue(use_cache=False)
        monkeypatch.setattr(EventTable, "fetch_open_data", _refuse_table_fetch)
        cached = event_module.get_event_catalogue()
        assert event_module.event_catalogue_cache_path().name == "event_catalogue_v4.json"
        for catalogue in (fetched, cached):
            assert catalogue["mass_1_source"][0] == 35.6 and catalogue["mass_2_source"][0] == 30.6
            assert catalogue["luminosity_distance"][0] == 440.0 and catalogue["p_astro"][0] == 1.0

    def test_an_event_whose_mass_interval_crosses_the_threshold_is_ambiguous(self, monkeypatch):
        """
        The 90% bounds of each component's source mass are fetched and cached, and an
        event is ``mass_ambiguous`` where either component's interval crosses 3 solar
        masses (GW200210_092254: m2 2.83 -0.42 +0.47); its type still follows the
        medians. Without bounds an event is not called ambiguous.
        """
        from astropy.table import Table
        from gwosc import api as gwosc_api

        names = ("commonName", "GPS", "chirp_mass_source", "mass_1_source", "mass_1_source_lower",
                 "mass_1_source_upper", "mass_2_source", "mass_2_source_lower", "mass_2_source_upper")
        table = Table(rows=[
            ("GW150914", 1126259462.4, 28.6, 35.6, -3.1, 4.7, 30.6, -4.4, 3.0),
            ("GW200210_092254", 1265361792.9, 6.56, 24.1, -4.6, 7.5, 2.83, -0.42, 0.47),
        ], names=names)
        monkeypatch.setattr(gwosc_api, "fetch_cataloglist_json",
                            lambda *args, **kwargs: {"GWTC-3-confident": {}})
        monkeypatch.setattr(EventTable, "fetch_open_data", lambda catalog, *args, **kwargs: table)
        event_module.get_event_catalogue(use_cache=False)
        monkeypatch.setattr(EventTable, "fetch_open_data", _refuse_table_fetch)
        events = {e["name"]: e for e in event_module.get_events_with_params()}
        assert events["GW150914"]["mass_ambiguous"] is False
        assert events["GW200210_092254"]["mass_ambiguous"] is True
        assert event_module.source_type_from_masses(events["GW200210_092254"]["mass1"],
                                                    events["GW200210_092254"]["mass2"]) == gf.SourceType.NSBH
        assert event_module.mass_ambiguous(35.6, np.nan, np.nan, 2.9, np.nan, np.nan) is False


class TestCatalogMappings:

    def test_no_catalog_overlap(self):
        """Test confident and marginal catalogs don't overlap."""
        confident = set(event_module.CONFIDENT_CATALOGS)
        marginal = set(event_module.MARGINAL_CATALOGS)
        assert len(confident & marginal) == 0

    def test_every_gwtc_release_gwosc_lists_has_a_confidence(self):
        """
        Each GWTC catalogue of the event-API snapshot is read as confident or marginal,
        O4's GWTC-4.0, GWTC-4.1 and GWTC-5.0 among them: a release left out is served
        by no event-time or event-example query (its events stay vetoed from noise).
        """
        listed = {row["catalog"] for row in _snapshot_rows() if row["catalog"].startswith("GWTC")}
        assert {"GWTC-4.0", "GWTC-4.1", "GWTC-5.0"} <= listed
        classified = set(event_module.CONFIDENT_CATALOGS) | set(event_module.MARGINAL_CATALOGS)
        assert listed - classified == set()


def _snapshot_rows():
    import csv

    with open(FIXTURES / "gwosc_events.csv", newline="") as handle:
        return list(csv.DictReader(handle))


class TestObservingRunO4Events:
    """O4's events, from the event-API snapshot: GWTC-4.x for O4a, GWTC-5.0 for O4b."""

    def test_o4_events_are_confident_and_named_once(self, offline_event_api):
        o4 = gf.ObservingRun.O4.value
        expected = {row["name"] for row in _snapshot_rows()
                    if row["catalog"] in event_module.CONFIDENT_CATALOGS
                    and o4.start_gps_time <= float(row["gps"]) <= o4.end_gps_time}
        events = event_module.get_events_with_params(observing_runs=[gf.ObservingRun.O4])
        # 140 in O4a (GWTC-4.1, a superset of GWTC-4.0's 129) and 161 in O4b (GWTC-5.0).
        assert len(expected) == 301 and {"GW230529_181500", "GW250114_082203"} <= expected
        assert [e["name"] for e in events if e["name"] in expected] == [e["name"] for e in events]
        assert len({e["name"] for e in events}) == len(events) == len(expected)
        assert [e["gps"] for e in events] == sorted(e["gps"] for e in events)
        assert {e["observing_run"] for e in events} == {"O4"}
        # Confident times include every O4 release's, and a marginal query none of them.
        confident = gf.get_confident_event_times()
        for catalog in ("GWTC-4.0", "GWTC-4.1", "GWTC-5.0"):
            times = [float(row["gps"]) for row in _snapshot_rows() if row["catalog"] == catalog]
            assert len(times) and np.all(np.isin(times, confident)), catalog
        marginal = gf.get_marginal_event_times()
        assert not np.any((marginal >= o4.start_gps_time) & (marginal <= o4.end_gps_time))
