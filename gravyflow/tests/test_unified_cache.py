"""
Test Suite for Unified Cache Architecture

Tests the cache-first acquisition where all glitches are stored in the same HDF5 
cache whether acquired during precaching or lazy loading.
"""

import pytest
import numpy as np
import shutil
from pathlib import Path
import tempfile

import gravyflow as gf
from gravyflow.src.dataset.features.glitch_cache import (
    GlitchCache, 
    CACHE_SAMPLE_RATE_HERTZ, 
    CACHE_ONSOURCE_DURATION, 
    CACHE_OFFSOURCE_DURATION
)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache files."""
    tmpdir = tempfile.mkdtemp(prefix="test_cache_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_cache(temp_cache_dir):
    """Create a sample cache with test data."""
    cache_path = temp_cache_dir / "test_cache.h5"
    cache = GlitchCache(cache_path, mode='w')
    
    N = 10
    num_ons = int(CACHE_SAMPLE_RATE_HERTZ * CACHE_ONSOURCE_DURATION)
    num_offs = int(CACHE_SAMPLE_RATE_HERTZ * CACHE_OFFSOURCE_DURATION)
    
    cache.initialize_file(
        sample_rate_hertz=CACHE_SAMPLE_RATE_HERTZ,
        onsource_duration=CACHE_ONSOURCE_DURATION,
        offsource_duration=CACHE_OFFSOURCE_DURATION,
        ifo_names=['L1'],
        num_ifos=1,
        onsource_samples=num_ons,
        offsource_samples=num_offs
    )
    
    # Create test data with identifiable values
    onsource = np.arange(N * 1 * num_ons, dtype=np.float32).reshape(N, 1, num_ons)
    offsource = np.arange(N * 1 * num_offs, dtype=np.float32).reshape(N, 1, num_offs) + 1000000
    gps_times = np.array([1000000.0 + i * 10 for i in range(N)])  # 1000000.0, 1000010.0, ...
    labels = np.arange(N, dtype=np.int32)
    
    cache.append(onsource, offsource, gps_times, labels)
    
    return cache_path, gps_times


class TestGPSLookup:
    """Tests for GPS-based cache lookup."""
    
    def test_has_gps_returns_true_for_existing(self, sample_cache):
        cache_path, gps_times = sample_cache
        cache = GlitchCache(cache_path, mode='r')
        
        for gps in gps_times:
            assert cache.has_gps(gps), f"Expected GPS {gps} to be in cache"
    
    def test_has_gps_returns_false_for_missing(self, sample_cache):
        cache_path, _ = sample_cache
        cache = GlitchCache(cache_path, mode='r')
        
        assert not cache.has_gps(999999.0), "Expected missing GPS to return False"
        assert not cache.has_gps(2000000.0), "Expected missing GPS to return False"
    
    def test_get_by_gps_returns_data(self, sample_cache):
        cache_path, gps_times = sample_cache
        cache = GlitchCache(cache_path, mode='r')
        
        result = cache.get_by_gps(gps_times[0])
        assert result is not None, "Expected get_by_gps to return data"
        
        ons, offs, gps, label = result
        assert gps == gps_times[0]
        assert label == 0
    
    def test_get_by_gps_returns_none_for_missing(self, sample_cache):
        cache_path, _ = sample_cache
        cache = GlitchCache(cache_path, mode='r')
        
        result = cache.get_by_gps(999999.0)
        assert result is None, "Expected missing GPS to return None"
    
    def test_get_by_gps_crops_and_resamples(self, sample_cache):
        cache_path, gps_times = sample_cache
        cache = GlitchCache(cache_path, mode='r')
        
        target_rate = 1024.0
        target_ons_dur = 1.0
        target_off_dur = 4.0
        
        result = cache.get_by_gps(
            gps_times[0],
            sample_rate_hertz=target_rate,
            onsource_duration=target_ons_dur,
            offsource_duration=target_off_dur
        )
        
        assert result is not None
        ons, offs, _, _ = result
        
        expected_ons_samples = int(target_rate * target_ons_dur)
        expected_off_samples = int(target_rate * target_off_dur)
        
        assert ons.shape[-1] == expected_ons_samples, f"Expected {expected_ons_samples} onsource samples"
        assert offs.shape[-1] == expected_off_samples, f"Expected {expected_off_samples} offsource samples"


class TestAppendSingle:
    """Tests for single-item cache append."""
    
    def test_append_single_adds_to_cache(self, temp_cache_dir):
        cache_path = temp_cache_dir / "append_test.h5"
        cache = GlitchCache(cache_path, mode='w')
        
        num_ons = int(CACHE_SAMPLE_RATE_HERTZ * CACHE_ONSOURCE_DURATION)
        num_offs = int(CACHE_SAMPLE_RATE_HERTZ * CACHE_OFFSOURCE_DURATION)
        
        cache.initialize_file(
            sample_rate_hertz=CACHE_SAMPLE_RATE_HERTZ,
            onsource_duration=CACHE_ONSOURCE_DURATION,
            offsource_duration=CACHE_OFFSOURCE_DURATION,
            ifo_names=['L1'],
            num_ifos=1,
            onsource_samples=num_ons,
            offsource_samples=num_offs
        )
        
        # Append single item
        ons = np.ones((1, num_ons), dtype=np.float32)
        offs = np.ones((1, num_offs), dtype=np.float32) * 2
        gps = 1234567.890
        label = 5
        
        cache.append_single(ons, offs, gps, label)
        
        # Verify it exists
        assert cache.has_gps(gps), "Appended GPS should exist in cache"
        
        # Verify data
        result = cache.get_by_gps(gps)
        assert result is not None
        ons_r, offs_r, gps_r, label_r = result
        
        assert gps_r == gps
        assert label_r == label
        np.testing.assert_array_equal(ons_r, ons)
    
    def test_append_single_updates_index(self, temp_cache_dir):
        cache_path = temp_cache_dir / "index_test.h5"
        cache = GlitchCache(cache_path, mode='w')
        
        num_ons = int(CACHE_SAMPLE_RATE_HERTZ * CACHE_ONSOURCE_DURATION)
        num_offs = int(CACHE_SAMPLE_RATE_HERTZ * CACHE_OFFSOURCE_DURATION)
        
        cache.initialize_file(
            sample_rate_hertz=CACHE_SAMPLE_RATE_HERTZ,
            onsource_duration=CACHE_ONSOURCE_DURATION,
            offsource_duration=CACHE_OFFSOURCE_DURATION,
            ifo_names=['L1'],
            num_ifos=1,
            onsource_samples=num_ons,
            offsource_samples=num_offs
        )
        
        # Append multiple items one by one
        for i in range(5):
            ons = np.ones((1, num_ons), dtype=np.float32) * i
            offs = np.ones((1, num_offs), dtype=np.float32) * i
            gps = 1000000.0 + i
            cache.append_single(ons, offs, gps, i)
        
        # All should be findable
        for i in range(5):
            assert cache.has_gps(1000000.0 + i), f"GPS {1000000.0 + i} should be in cache"


class TestCacheConstants:
    """Tests for cache constants."""
    
    def test_onsource_duration_is_32s(self):
        assert CACHE_ONSOURCE_DURATION == 32.0
    
    def test_offsource_duration_is_32s(self):
        assert CACHE_OFFSOURCE_DURATION == 32.0
    
    def test_sample_rate_is_4096hz(self):
        assert CACHE_SAMPLE_RATE_HERTZ == 4096.0


class TestCachePersistence:
    """Tests for cache file persistence across reopens."""
    
    def test_cache_persists_after_close(self, temp_cache_dir):
        cache_path = temp_cache_dir / "persist_test.h5"
        
        num_ons = int(CACHE_SAMPLE_RATE_HERTZ * CACHE_ONSOURCE_DURATION)
        num_offs = int(CACHE_SAMPLE_RATE_HERTZ * CACHE_OFFSOURCE_DURATION)
        
        # Create and populate
        cache1 = GlitchCache(cache_path, mode='w')
        cache1.initialize_file(
            sample_rate_hertz=CACHE_SAMPLE_RATE_HERTZ,
            onsource_duration=CACHE_ONSOURCE_DURATION,
            offsource_duration=CACHE_OFFSOURCE_DURATION,
            ifo_names=['L1'],
            num_ifos=1,
            onsource_samples=num_ons,
            offsource_samples=num_offs
        )
        
        gps = 1234567.0
        cache1.append_single(
            np.zeros((1, num_ons), dtype=np.float32),
            np.zeros((1, num_offs), dtype=np.float32),
            gps, 0
        )
        del cache1  # Close
        
        # Reopen and verify
        cache2 = GlitchCache(cache_path, mode='r')
        assert cache2.has_gps(gps), "Data should persist after cache is closed and reopened"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
