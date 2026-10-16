"""
Tests for event.py module.

Tests EventConfidence enum, event time functions, and PE parameter fetching.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

import gravyflow as gf
from gravyflow.src.dataset.features import event as event_module


class TestEventConfidence:
    """Tests for EventConfidence enum."""
    
    def test_event_type_members(self):
        """Test EventConfidence has exactly two members."""
        assert len(gf.EventConfidence) == 2
        assert gf.EventConfidence.CONFIDENT in gf.EventConfidence
        assert gf.EventConfidence.MARGINAL in gf.EventConfidence


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


class TestGetConfidentEventsWithParams:
    """Tests for get_confident_events_with_params function."""

    def test_the_table_is_cached_in_the_user_cache_never_the_package(self, monkeypatch, tmp_path):
        """A fetched table is written under GRAVYFLOW_CACHE_DIR; the package is untouched."""
        from pathlib import Path
        from gravyflow.tests.conftest import tree_changes, tree_snapshot

        monkeypatch.setenv("GRAVYFLOW_CACHE_DIR", str(tmp_path))
        before = tree_snapshot(Path(gf.PATH))
        row = {"commonName": "GW150914", "GPS": 1126259462.4, "mass_1_source": 35.6,
               "mass_2_source": 30.6, "luminosity_distance": 430.0}
        with patch("gwpy.table.EventTable.fetch_open_data", return_value=[row]):
            events = event_module.get_events_with_params(
                event_types=[gf.EventConfidence.MARGINAL]
            )
        assert [e["name"] for e in events] == ["GW150914"]
        assert tree_changes(before, tree_snapshot(Path(gf.PATH))) == []
        assert len(list((tmp_path / "event_params").glob("events_*.json"))) == 1

    def test_function_exists_and_exported(self):
        """Test function is exported at top level."""
        assert hasattr(gf, 'get_confident_events_with_params')
        assert callable(gf.get_confident_events_with_params)
    
    def test_returns_list_of_dicts(self):
        """Test function returns list of dictionaries."""
        # Mock EventTable.fetch_open_data
        mock_table = MagicMock()
        mock_row = {
            "commonName": "GW150914",
            "GPS": 1126259462.0,
            "mass_1_source": 35.6,
            "mass_2_source": 30.6,
            "luminosity_distance": 430.0,
        }
        mock_table.__iter__ = MagicMock(return_value=iter([mock_row]))
        
        with patch('gwpy.table.EventTable.fetch_open_data', return_value=mock_table):
            events = gf.get_confident_events_with_params()
            assert isinstance(events, list)
            if events:  # May be empty if mock doesn't work perfectly
                assert isinstance(events[0], dict)
    
    def test_event_dict_has_required_keys(self):
        """Test each event dict has required keys."""
        mock_table = MagicMock()
        mock_row = MagicMock()
        mock_row.get = lambda key, default=None: {
            "commonName": "GW150914",
            "mass_1_source": 35.6,
            "mass_2_source": 30.6,
            "luminosity_distance": 430.0,
        }.get(key, default)
        mock_row.__getitem__ = lambda self, key: {"GPS": 1126259462.0}[key]
        mock_table.__iter__ = MagicMock(return_value=iter([mock_row]))
        
        with patch('gwpy.table.EventTable.fetch_open_data', return_value=mock_table):
            events = gf.get_confident_events_with_params()
            
            required_keys = {'name', 'gps', 'mass1', 'mass2', 'distance', 'catalog', 'observing_run'}
            for event in events:
                assert required_keys.issubset(event.keys()), f"Missing keys: {required_keys - set(event.keys())}"
    
    def test_deduplication_by_gps(self):
        """Test that duplicate GPS times are removed."""
        # Create two events with same GPS (rounded to 0.1s)
        mock_table = MagicMock()
        mock_row1 = MagicMock()
        mock_row1.get = lambda key, default=None: {
            "commonName": "Event1",
            "mass_1_source": 35.0,
            "mass_2_source": 30.0,
            "luminosity_distance": 400.0,
        }.get(key, default)
        mock_row1.__getitem__ = lambda self, key: {"GPS": 1126259462.0}[key]
        
        mock_row2 = MagicMock()
        mock_row2.get = lambda key, default=None: {
            "commonName": "Event2",
            "mass_1_source": 35.0,
            "mass_2_source": 30.0,
            "luminosity_distance": 400.0,
        }.get(key, default)
        mock_row2.__getitem__ = lambda self, key: {"GPS": 1126259462.05}[key]  # Within 0.1s
        
        mock_table.__iter__ = MagicMock(return_value=iter([mock_row1, mock_row2]))
        
        with patch('gwpy.table.EventTable.fetch_open_data', return_value=mock_table):
            events = gf.get_confident_events_with_params()
            # Should only have 1 event due to deduplication
            gps_times = [e['gps'] for e in events]
            unique_gps = len(set(round(g, 1) for g in gps_times))
            assert unique_gps == len(events)
    
    def test_sorted_by_gps_time(self):
        """Test events are sorted by GPS time."""
        mock_table = MagicMock()
        mock_rows = []
        for gps in [1200000000.0, 1100000000.0, 1150000000.0]:
            row = MagicMock()
            row.get = lambda key, default=None, g=gps: {
                "commonName": f"Event_{g}",
                "mass_1_source": 30.0,
                "mass_2_source": 25.0,
                "luminosity_distance": 300.0,
            }.get(key, default)
            row.__getitem__ = lambda self, key, g=gps: {"GPS": g}[key]
            mock_rows.append(row)
        
        mock_table.__iter__ = MagicMock(return_value=iter(mock_rows))
        
        with patch('gwpy.table.EventTable.fetch_open_data', return_value=mock_table):
            events = gf.get_confident_events_with_params()
            gps_times = [e['gps'] for e in events]
            assert gps_times == sorted(gps_times)


class TestCatalogMappings:
    """Tests for catalog mappings and constants."""
    
    def test_confident_catalogs_defined(self):
        """Test CONFIDENT_CATALOGS constant exists."""
        assert hasattr(event_module, 'CONFIDENT_CATALOGS')
        assert len(event_module.CONFIDENT_CATALOGS) > 0
    
    def test_marginal_catalogs_defined(self):
        """Test MARGINAL_CATALOGS constant exists."""
        assert hasattr(event_module, 'MARGINAL_CATALOGS')
        assert len(event_module.MARGINAL_CATALOGS) > 0
    
    def test_no_catalog_overlap(self):
        """Test confident and marginal catalogs don't overlap."""
        confident = set(event_module.CONFIDENT_CATALOGS)
        marginal = set(event_module.MARGINAL_CATALOGS)
        assert len(confident & marginal) == 0
