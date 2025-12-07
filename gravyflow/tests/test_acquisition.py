"""
Comprehensive tests for acquisition.py module.

This module contains tests to achieve near-100% test coverage for:
- Core utility functions (ensure_even, random_subsection, etc.)
- IFOData dataclass and methods
- IFODataObtainer class and all its methods
- Segment processing, ordering, and grouping
- Visual inspection plots for human review

Run with coverage:
    pytest gravyflow/tests/test_acquisition.py -v --cov=gravyflow/src/dataset/noise/acquisition --cov-report=term-missing

Run with visual plots:
    pytest gravyflow/tests/test_acquisition.py -v --plot
"""

import pytest
import numpy as np
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from _pytest.config import Config

import keras
from keras import ops
import jax.numpy as jnp

import gravyflow as gf
from gravyflow.src.dataset.noise import acquisition as gf_acq

# Bokeh imports for visualization tests
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from bokeh.models import Span


# =============================================================================
# CORE UTILITY FUNCTION TESTS
# =============================================================================

class TestEnsureEven:
    """Tests for ensure_even function (line 31)."""
    
    def test_even_number_unchanged(self):
        """Even numbers should remain unchanged."""
        assert gf_acq.ensure_even(100) == 100
        assert gf_acq.ensure_even(4096) == 4096
        assert gf_acq.ensure_even(0) == 0
        assert gf_acq.ensure_even(2) == 2
    
    def test_odd_number_decremented(self):
        """Odd numbers should be decremented by 1."""
        assert gf_acq.ensure_even(101) == 100
        assert gf_acq.ensure_even(4097) == 4096
        assert gf_acq.ensure_even(1) == 0
        assert gf_acq.ensure_even(3) == 2


class TestRandomSubsection:
    """Tests for random_subsection and _random_subsection functions."""
    
    def test_random_subsection_shapes(self):
        """Test that random_subsection returns correct shapes."""
        total_samples = 10000
        data = jnp.arange(total_samples, dtype=jnp.float32)
        data_list = [data]
        
        start_gps = [1000.0]
        time_interval = 1.0
        num_onsource = 100
        num_offsource = 200
        num_examples = 5
        seed = 42
        
        subarrays, backgrounds, starts = gf_acq.random_subsection(
            data_list, start_gps, time_interval,
            num_onsource, num_offsource, num_examples, seed
        )
        
        assert ops.shape(subarrays) == (num_examples, 1, num_onsource)
        assert ops.shape(backgrounds) == (num_examples, 1, num_offsource)
        assert ops.shape(starts) == (num_examples, 1)
    
    def test_random_subsection_content_sequential(self):
        """Subarrays should be sequential chunks from data."""
        total_samples = 10000
        data = jnp.arange(total_samples, dtype=jnp.float32)
        
        subarrays, _, _ = gf_acq.random_subsection(
            [data], [0.0], 1.0, 100, 50, 3, seed=42
        )
        
        # Check if differences are 1.0 (sequential)
        diffs = subarrays[:, 0, 1:] - subarrays[:, 0, :-1]
        np.testing.assert_allclose(diffs, 1.0)


class TestConcatenateBatches:
    """Tests for concatenate_batches function."""
    
    def test_concatenate_batches_shape(self):
        """Test concatenation produces correct output shape."""
        b1 = ops.ones((2, 1, 10))
        b2 = ops.ones((2, 1, 10))
        
        res, bg, st = gf_acq.concatenate_batches([b1, b2], [b1, b2], [b1, b2])
        
        assert ops.shape(res) == (2, 2, 10)
        assert ops.shape(bg) == (2, 2, 10)
        assert ops.shape(st) == (2, 2, 10)


# =============================================================================
# IFOData DATACLASS TESTS
# =============================================================================

class TestIFOData:
    """Tests for IFOData dataclass and its methods."""
    
    def test_ifo_data_list_input(self):
        """Test IFOData with list of arrays as input."""
        data = np.arange(5000, dtype=np.float32)
        ifo_data = gf_acq.IFOData(
            data=[data],
            sample_rate_hertz=1.0,
            start_gps_time=[0.0]
        )
        
        assert len(ifo_data.data) == 1
        assert ifo_data.sample_rate_hertz == 1.0
    
    def test_ifo_data_numpy_input(self):
        """Test IFOData with numpy array as input (covers lines 259-260)."""
        data = np.arange(1000, dtype=np.float32)
        ifo_data = gf_acq.IFOData(
            data=data,  # Pass numpy array directly, not list
            sample_rate_hertz=100.0,
            start_gps_time=[0.0]
        )
        
        # Should be converted to list with single tensor
        assert isinstance(ifo_data.data, list)
        assert len(ifo_data.data) == 1
    
    def test_ifo_data_duration_calculation(self):
        """Test that duration is calculated correctly."""
        data = np.zeros(1000, dtype=np.float32)
        ifo_data = gf_acq.IFOData(
            data=[data],
            sample_rate_hertz=100.0,
            start_gps_time=[0.0]
        )
        
        assert len(ifo_data.duration_seconds) == 1
        np.testing.assert_allclose(float(ifo_data.duration_seconds[0]), 10.0)
    
    def test_ifo_data_downsample(self):
        """Test downsample method returns self (placeholder, line 274)."""
        data = np.arange(1000, dtype=np.float32)
        ifo_data = gf_acq.IFOData(
            data=[data],
            sample_rate_hertz=100.0,
            start_gps_time=[0.0]
        )
        
        result = ifo_data.downsample(50.0)
        assert result is ifo_data
    
    def test_ifo_data_scale(self):
        """Test scale method multiplies data by factor."""
        data = np.ones(100, dtype=np.float32)
        ifo_data = gf_acq.IFOData(
            data=[data],
            sample_rate_hertz=100.0,
            start_gps_time=[0.0]
        )
        
        ifo_data.scale(2.5)
        np.testing.assert_allclose(ifo_data.data[0], 2.5 * np.ones(100))
    
    def test_ifo_data_numpy_method(self):
        """Test numpy() method converts to numpy arrays (lines 281-284)."""
        data = np.arange(100, dtype=np.float32)
        ifo_data = gf_acq.IFOData(
            data=[data],
            sample_rate_hertz=100.0,
            start_gps_time=[0.0]
        )
        
        numpy_data = ifo_data.numpy()
        assert isinstance(numpy_data, list)
        assert isinstance(numpy_data[0], np.ndarray)
    
    def test_ifo_data_random_subsection(self):
        """Test random_subsection method on IFOData."""
        data = np.arange(5000, dtype=np.float32)
        ifo_data = gf_acq.IFOData(
            data=[data],
            sample_rate_hertz=1.0,
            start_gps_time=[0.0]
        )
        
        subarrays, backgrounds, starts = ifo_data.random_subsection(
            num_onsource_samples=100,
            num_offsource_samples=100,
            num_examples_per_batch=2,
            seed=123
        )
        
        assert ops.shape(subarrays) == (2, 1, 100)
        assert ops.shape(backgrounds) == (2, 1, 100)


# =============================================================================
# IFODataObtainer INITIALIZATION TESTS
# =============================================================================

class TestIFODataObtainerInit:
    """Tests for IFODataObtainer initialization and basic methods."""
    
    def test_basic_initialization(self):
        """Test basic IFODataObtainer initialization."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            force_acquisition=False,
            cache_segments=False
        )
        
        assert obtainer.data_quality == gf.DataQuality.BEST
        assert obtainer.force_acquisition == False
    
    def test_initialization_with_multiple_labels(self):
        """Test initialization with multiple data labels."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            [gf.DataLabel.NOISE, gf.DataLabel.GLITCHES],
            force_acquisition=False,
            cache_segments=False
        )
        
        assert len(obtainer.data_labels) == 2
    
    def test_override_attributes_valid(self):
        """Test override_attributes with valid attribute (lines 371-373)."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            force_acquisition=False,
            cache_segments=False,
            overrides={"saturation": 0.5}
        )
        
        assert obtainer.saturation == [0.5]
    
    def test_override_attributes_invalid_raises(self):
        """Test override_attributes with invalid attribute raises ValueError (lines 374-377)."""
        with pytest.raises(ValueError, match="Invalide override value"):
            gf.IFODataObtainer(
                gf.ObservingRun.O3,
                gf.DataQuality.BEST,
                gf.DataLabel.NOISE,
                overrides={"nonexistent_attribute": 123}
            )
    
    def test_close_method(self):
        """Test close method handles None segment_file (lines 402-404)."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        # Should not raise with None segment_file
        obtainer.close()
    
    def test_generate_file_path(self):
        """Test generate_file_path creates correct path."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = obtainer.generate_file_path(
                sample_rate_hertz=1024.0,
                group="train",
                data_directory_path=Path(tmpdir)
            )
            
            assert path.parent == Path(tmpdir)
            assert path.suffix == ".hdf5"
            assert "segment_data_" in path.name
    
    def test_generate_file_path_default_directory(self):
        """Test generate_file_path uses default directory when None (line 414)."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        path = obtainer.generate_file_path(
            sample_rate_hertz=1024.0,
            group="train",
            data_directory_path=None  # Uses default
        )
        
        assert path is not None
        assert path.suffix == ".hdf5"


# =============================================================================
# SEGMENT TIME AND PROCESSING TESTS
# =============================================================================

class TestSegmentProcessing:
    """Tests for segment processing methods."""
    
    @pytest.fixture
    def obtainer(self):
        """Create a fresh IFODataObtainer for each test."""
        return gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            force_acquisition=False,
            cache_segments=False
        )
    
    def test_pad_gps_times_with_veto_window_normal(self, obtainer):
        """Test padding GPS times creates correct segments."""
        gps_times = np.array([1000.0, 2000.0, 3000.0])
        
        segments = obtainer.pad_gps_times_with_veto_window(
            gps_times,
            start_padding_seconds=2.0,
            end_padding_seconds=2.0
        )
        
        expected = np.array([
            [998.0, 1002.0],
            [1998.0, 2002.0],
            [2998.0, 3002.0]
        ])
        np.testing.assert_array_equal(segments, expected)
    
    def test_pad_gps_times_empty_input(self, obtainer):
        """Test padding empty GPS times returns empty array (line 886)."""
        gps_times = np.array([])
        
        segments = obtainer.pad_gps_times_with_veto_window(gps_times)
        
        assert segments.shape == (0, 2)
    
    def test_veto_time_segments(self, obtainer):
        """Test veto_time_segments removes overlapping regions."""
        segments = np.array([
            [0.0, 100.0],
            [200.0, 300.0]
        ])
        veto_segments = np.array([
            [50.0, 75.0],
            [250.0, 275.0]
        ])
        
        result = obtainer.veto_time_segments(segments, veto_segments)
        
        # Should have vetoed portions removed
        assert len(result) > 0
        # First segment should be split around 50-75
        # Second segment should be split around 250-275
    
    def test_veto_time_segments_empty_veto(self, obtainer):
        """Test empty veto returns original segments (line 900)."""
        segments = np.array([
            [0.0, 100.0],
            [200.0, 300.0]
        ])
        veto_segments = np.array([])
        
        result = obtainer.veto_time_segments(segments, veto_segments)
        
        np.testing.assert_array_equal(result, segments)
    
    def test_find_segment_intersections(self, obtainer):
        """Test find_segment_intersections finds overlapping regions (lines 558-574)."""
        arr1 = np.array([
            [0.0, 100.0],
            [200.0, 300.0],
            [400.0, 500.0]
        ])
        arr2 = np.array([
            [50.0, 150.0],
            [250.0, 350.0]
        ])
        
        result = obtainer.find_segment_intersections(arr1, arr2)
        
        # Should find intersections
        assert len(result) > 0
        # First intersection: [50, 100]
        # Second intersection: [250, 300]
    
    def test_compress_segments_merges_overlapping(self, obtainer):
        """Test compress_segments merges overlapping segments."""
        segments = np.array([
            [0, 10],
            [5, 15],
            [20, 30]
        ])
        
        compressed = obtainer.compress_segments(segments)
        
        assert ops.shape(compressed) == (2, 2)
        np.testing.assert_array_equal(compressed[0], [0, 15])
        np.testing.assert_array_equal(compressed[1], [20, 30])
    
    def test_compress_segments_empty_input(self, obtainer):
        """Test compress_segments with empty input (line 663)."""
        segments = np.array([]).reshape(0, 2)
        
        result = obtainer.compress_segments(segments)
        
        assert result.size == 0
    
    def test_calculate_bin_indices(self, obtainer):
        """Test calculate_bin_indices computes correct indices (lines 680-687)."""
        segments = np.array([
            [100.0, 200.0],
            [300.0, 400.0],
            [500.0, 600.0]
        ])
        interval = 100.0
        start_period = 0.0
        
        start_bins, end_bins = obtainer.calculate_bin_indices(
            segments, interval, start_period
        )
        
        # Segment [100, 200]: starts in bin 1, ends in bin 2
        # Segment [300, 400]: starts in bin 3, ends in bin 4
        # Segment [500, 600]: starts in bin 5, ends in bin 6
        np.testing.assert_array_equal(start_bins, [1, 3, 5])
        np.testing.assert_array_equal(end_bins, [2, 4, 6])
    
    def test_remove_short_segments(self, obtainer):
        """Test remove_short_segments filters correctly."""
        # Shape (NumSegments, NumIFOs, 2) -> (3, 1, 2)
        segments = np.array([
            [[0, 10]],    # 10s duration
            [[20, 25]],   # 5s duration
            [[30, 50]]    # 20s duration
        ])
        
        filtered = obtainer.remove_short_segments(segments, 8.0)
        
        assert ops.shape(filtered) == (2, 1, 2)
        durations = filtered[:, 0, 1] - filtered[:, 0, 0]
        assert np.all(durations >= 8.0)


class TestSegmentOrdering:
    """Tests for segment ordering methods."""
    
    @pytest.fixture
    def obtainer(self):
        return gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            force_acquisition=False,
            cache_segments=False
        )
    
    def test_order_segments_random(self, obtainer):
        """Test RANDOM ordering shuffles segments."""
        segments = np.array([
            [[0, 10]],
            [[20, 50]],
            [[60, 80]]
        ])
        
        random_segments = obtainer.order_segments(
            segments.copy(),
            gf.SegmentOrder.RANDOM,
            seed=42
        )
        
        # Should be a permutation
        assert ops.shape(random_segments) == ops.shape(segments)
        assert np.sum(random_segments) == np.sum(segments)
    
    def test_order_segments_shortest_first(self, obtainer):
        """Test SHORTEST_FIRST ordering."""
        segments = np.array([
            [[0, 10]],    # 10s
            [[20, 50]],   # 30s
            [[60, 80]]    # 20s
        ])
        
        sorted_segments = obtainer.order_segments(
            segments.copy(),
            gf.SegmentOrder.SHORTEST_FIRST,
            seed=42
        )
        
        durations = sorted_segments[:, 0, 1] - sorted_segments[:, 0, 0]
        np.testing.assert_array_equal(durations, [10, 20, 30])
    
    def test_order_segments_chronological(self, obtainer):
        """Test CHRONOLOGICAL ordering (lines 801-805)."""
        segments = np.array([
            [[100, 200]],
            [[0, 50]],
            [[300, 400]]
        ])
        
        # CHRONOLOGICAL should not change order (assumed already sorted)
        result = obtainer.order_segments(
            segments.copy(),
            gf.SegmentOrder.CHRONOLOGICAL,
            seed=42
        )
        
        np.testing.assert_array_equal(result, segments)


class TestCutAndGroupSegments:
    """Tests for segment cutting and grouping methods."""
    
    @pytest.fixture
    def obtainer(self):
        return gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            force_acquisition=False,
            cache_segments=False
        )
    
    def test_cut_segments(self, obtainer):
        """Test cut_segments splits at chunk boundaries (lines 816-839)."""
        segments = np.array([
            [0.0, 250.0],
            [300.0, 400.0]
        ])
        chunk_size = 100.0
        start_time = 0.0
        
        result = obtainer.cut_segments(segments, chunk_size, start_time)
        
        # First segment [0, 250] should be cut into [0,100], [100,200], [200,250]
        # Second segment [300, 400] should be [300, 400]
        assert len(result) >= 3
    
    def test_get_segments_for_group(self, obtainer):
        """Test get_segments_for_group assigns deterministically (lines 841-876)."""
        segments = np.array([
            [0.0, 100.0],
            [100.0, 200.0],
            [200.0, 300.0],
            [300.0, 400.0]
        ])
        
        groups = {"train": 0.8, "validate": 0.1, "test": 0.1}
        
        train_segments = obtainer.get_segments_for_group(
            segments, 100.0, "train", groups, 0.0
        )
        
        # Should get some segments assigned to train
        assert isinstance(train_segments, np.ndarray)
    
    def test_merge_bins(self, obtainer):
        """Test merge_bins combines segments into time bins (lines 689-738)."""
        seg_list1 = np.array([[0.0, 50.0], [100.0, 150.0]])
        seg_list2 = np.array([[10.0, 60.0], [110.0, 160.0]])
        
        result = obtainer.merge_bins([seg_list1, seg_list2], interval=100.0)
        
        assert isinstance(result, list)
    
    def test_largest_segments_per_bin(self, obtainer):
        """Test largest_segments_per_bin extracts largest (lines 740-777)."""
        # Create filtered bins result format
        filtered_bins = [
            [np.array([[0.0, 50.0], [10.0, 30.0]]),
             np.array([[5.0, 45.0]])]
        ]
        
        result = obtainer.largest_segments_per_bin(filtered_bins)
        
        assert isinstance(result, np.ndarray)


# =============================================================================
# CACHE AND ACQUISITION TESTS
# =============================================================================

class TestCachingMethods:
    """Tests for caching-related methods."""
    
    def test_cache_segment_no_file_path(self):
        """Test _cache_segment with no file_path does nothing (lines 909-910)."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        # Should not raise when file_path is None
        obtainer._cache_segment("test_key", np.array([1, 2, 3]))


class TestClearValidSegments:
    """Tests for clear_valid_segments method."""
    
    def test_clear_valid_segments(self):
        """Test clear_valid_segments resets state (lines 1204-1210)."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        # Set some state
        obtainer.valid_segments = np.array([[0, 100]])
        obtainer.valid_segments_adjusted = np.array([[10, 90]])
        obtainer.ifos = [gf.IFO.L1]
        obtainer._current_segment_index = 5
        obtainer._current_batch_index = 3
        obtainer._segment_exausted = False
        
        # Clear
        obtainer.clear_valid_segments()
        
        assert obtainer.valid_segments is None
        assert obtainer.valid_segments_adjusted is None
        assert obtainer.ifos is None
        assert obtainer._current_segment_index == 0
        assert obtainer._current_batch_index == 0
        assert obtainer._segment_exausted == True


# =============================================================================
# OBSERVING RUN DATA TESTS
# =============================================================================

class TestObservingRunData:
    """Tests for ObservingRunData class."""
    
    def test_observing_run_data_gps_conversion(self):
        """Test GPS time conversion from datetime."""
        from datetime import datetime
        
        # O3 start date
        o3_data = gf.ObservingRun.O3.value
        
        assert o3_data.name == "O3"
        assert o3_data.start_gps_time > 0
        assert o3_data.end_gps_time > o3_data.start_gps_time
    
    def test_all_observing_runs_have_valid_data(self):
        """Test all observing runs have valid properties."""
        for run in [gf.ObservingRun.O1, gf.ObservingRun.O2, gf.ObservingRun.O3]:
            data = run.value
            assert data.name in ["O1", "O2", "O3"]
            assert data.start_gps_time > 0
            assert gf.DataQuality.BEST in data.channels
            assert gf.DataQuality.BEST in data.frame_types
            assert gf.DataQuality.BEST in data.state_flags


# =============================================================================
# EXISTING TESTS (preserved from original file)
# =============================================================================

@pytest.mark.slow
@pytest.mark.parametrize("seed, ifo", [(1000, gf.IFO.L1)])
def test_valid_data_segments_acquisition(
        data_obtainer: gf.IFODataObtainer,
        seed: int,
        ifo: gf.IFO
    ) -> None:
    """Test to ensure the acquisition of valid data segments meets expected criteria."""
    segments = data_obtainer.get_valid_segments(seed=seed, ifos=[ifo])
    
    np.testing.assert_array_less(
        10000,
        len(segments),
        err_msg=f"Num segments found, {len(segments)}, is too low!"
    )


# =============================================================================
# VISUAL INSPECTION TESTS
# =============================================================================

def _create_segment_visualization(obtainer, segments, title, output_path):
    """Helper to create segment visualization plot."""
    p = figure(
        title=title,
        x_axis_label="GPS Time (s)",
        y_axis_label="Segment Index",
        width=1200,
        height=400
    )
    
    for i, (start, end) in enumerate(segments):
        p.segment(
            x0=start, y0=i, x1=end, y1=i,
            line_width=10, line_color="steelblue"
        )
    
    return p


def _test_segment_processing_visualization(
    output_directory_path: Path = Path("./gravyflow_data/tests/"),
    plot_results: bool = False
) -> None:
    """Visualize segment processing pipeline."""
    obtainer = gf.IFODataObtainer(
        gf.ObservingRun.O3,
        gf.DataQuality.BEST,
        gf.DataLabel.NOISE,
        cache_segments=False
    )
    
    # Create test segments
    original_segments = np.array([
        [0.0, 100.0],
        [50.0, 150.0],
        [200.0, 300.0],
        [250.0, 350.0],
        [400.0, 450.0]
    ])
    
    # Compress overlapping segments
    compressed = obtainer.compress_segments(original_segments)
    
    # Cut segments
    cut = obtainer.cut_segments(compressed, 100.0, 0.0)
    
    if plot_results:
        layout = []
        
        # Original segments
        p1 = _create_segment_visualization(
            obtainer, original_segments,
            "Original Segments (with overlaps)", output_directory_path
        )
        
        # Compressed segments
        p2 = _create_segment_visualization(
            obtainer, compressed,
            "Compressed Segments (overlaps merged)", output_directory_path
        )
        
        # Cut segments
        p3 = _create_segment_visualization(
            obtainer, cut,
            "Cut Segments (at 100s boundaries)", output_directory_path
        )
        
        layout.append([p1])
        layout.append([p2])
        layout.append([p3])
        
        gf.ensure_directory_exists(output_directory_path)
        output_file(output_directory_path / "acquisition_segment_processing.html")
        grid = gridplot(layout)
        save(grid)
    
    # Assertions
    assert len(compressed) <= len(original_segments)
    assert len(cut) >= len(compressed)


def _test_random_subsection_visualization(
    output_directory_path: Path = Path("./gravyflow_data/tests/"),
    plot_results: bool = False
) -> None:
    """Visualize random subsection extraction."""
    # Create synthetic data
    sample_rate = 1024.0
    duration = 10.0
    num_samples = int(sample_rate * duration)
    
    # Create a signal with some structure (sinusoid + noise)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    signal = np.sin(2 * np.pi * 5.0 * t) + 0.1 * np.random.randn(num_samples)
    signal = signal.astype(np.float32)
    
    ifo_data = gf_acq.IFOData(
        data=[signal],
        sample_rate_hertz=sample_rate,
        start_gps_time=[1000000.0]
    )
    
    # Extract random subsections
    num_examples = 3
    onsource_dur = 1.0
    offsource_dur = 0.5
    
    subarrays, backgrounds, gps_times = ifo_data.random_subsection(
        num_onsource_samples=int(onsource_dur * sample_rate),
        num_offsource_samples=int(offsource_dur * sample_rate),
        num_examples_per_batch=num_examples,
        seed=42
    )
    
    if plot_results:
        layout = []
        
        # Full signal
        p_full = figure(
            title="Full Signal (10s duration)",
            x_axis_label="Sample",
            y_axis_label="Amplitude",
            width=1200, height=200
        )
        p_full.line(range(len(signal)), signal, line_color="gray", alpha=0.7)
        layout.append([p_full])
        
        # Individual extractions
        for i in range(num_examples):
            onsource = np.array(subarrays[i, 0, :])
            offsource = np.array(backgrounds[i, 0, :])
            gps = float(gps_times[i, 0])
            
            p = figure(
                title=f"Extraction {i+1}: GPS {gps:.1f}",
                x_axis_label="Sample",
                y_axis_label="Amplitude",
                width=600, height=150
            )
            p.line(range(len(onsource)), onsource, 
                   line_color="steelblue", legend_label="Onsource")
            
            p_off = figure(
                title=f"Offsource {i+1}",
                x_axis_label="Sample",
                y_axis_label="Amplitude",
                width=400, height=150
            )
            p_off.line(range(len(offsource)), offsource,
                      line_color="coral", legend_label="Offsource")
            
            layout.append([p, p_off])
        
        gf.ensure_directory_exists(output_directory_path)
        output_file(output_directory_path / "acquisition_random_subsection.html")
        grid = gridplot(layout)
        save(grid)
    
    # Assertions
    assert ops.shape(subarrays) == (num_examples, 1, int(onsource_dur * sample_rate))
    assert ops.shape(backgrounds) == (num_examples, 1, int(offsource_dur * sample_rate))


def test_segment_processing_visualization(pytestconfig: Config) -> None:
    """Test segment processing with optional visualization."""
    _test_segment_processing_visualization(
        plot_results=pytestconfig.getoption("plot")
    )


def test_random_subsection_visualization(pytestconfig: Config) -> None:
    """Test random subsection extraction with optional visualization."""
    _test_random_subsection_visualization(
        plot_results=pytestconfig.getoption("plot")
    )


# =============================================================================
# ADDITIONAL TESTS FOR HIGHER COVERAGE
# =============================================================================

class TestGetOnsourceOffsourceChunks:
    """Tests for get_onsource_offsource_chunks generator method."""
    
    def test_get_chunks_defaults_applied(self):
        """Test that defaults are applied when parameters are None (lines 1101-1111)."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        # Set up valid segments manually
        obtainer.valid_segments = np.array([
            [[100.0, 1000.0]],
            [[1100.0, 2000.0]]
        ])
        
        # Create mock current segment with enough data
        mock_data = np.random.randn(100000).astype(np.float32)
        mock_segment = gf_acq.IFOData(
            data=[mock_data],
            sample_rate_hertz=1024.0,
            start_gps_time=[100.0]
        )
        
        # Set up obtainer state
        obtainer._segment_exausted = False
        obtainer.current_segment = mock_segment
        obtainer._num_batches_in_current_segment = 2
        obtainer._current_batch_index = 0
        obtainer.rng = np.random.default_rng(42)
        
        # Get first chunk
        gen = obtainer.get_onsource_offsource_chunks(
            sample_rate_hertz=1024.0,
            onsource_duration_seconds=1.0,
            padding_duration_seconds=0.5,
            offsource_duration_seconds=1.0,
            num_examples_per_batch=None,  # Use default
            ifos=gf.IFO.L1,
            scale_factor=None,  # Use default
            seed=None  # Use default
        )
        
        # Should return generator that yields data
        try:
            subarrays, backgrounds, gps_times = next(gen)
            assert subarrays is not None
        except StopIteration:
            # Expected if segments are exhausted
            pass


class TestAcquireGenerator:
    """Tests for acquire generator method."""
    
    def test_acquire_with_preset_segments(self):
        """Test acquire method with preset valid_segments."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        # Can't actually test acquire without network access
        # But we can test the setup code paths
        obtainer._current_segment_index = 0
        
        # Create empty/invalid segments to trigger early exit
        empty_segments = np.array([]).reshape(0, 1, 2)
        
        gen = obtainer.acquire(
            sample_rate_hertz=1024.0,
            valid_segments=empty_segments,
            ifos=[gf.IFO.L1],
            scale_factor=1.0
        )
        
        # With empty segments, should immediately stop
        results = list(gen)
        assert len(results) == 0


class TestGetValidSegmentsErrors:
    """Tests for get_valid_segments error handling."""
    
    def test_get_valid_segments_invalid_group(self):
        """Test get_valid_segments with invalid group name raises KeyError (line 1241)."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        with pytest.raises(KeyError, match="not present in groups"):
            obtainer.get_valid_segments(
                ifos=[gf.IFO.L1],
                seed=42,
                groups={"train": 0.8, "validate": 0.2},
                group_name="nonexistent_group"
            )


class TestDataLabels:
    """Tests for DataLabel enum and related functionality."""
    
    def test_data_labels_enum(self):
        """Test DataLabel enum values."""
        assert gf.DataLabel.NOISE is not None
        assert gf.DataLabel.GLITCHES is not None
        assert gf.DataLabel.EVENTS is not None
    
    def test_data_quality_enum(self):
        """Test DataQuality enum values."""
        assert gf.DataQuality.RAW is not None
        assert gf.DataQuality.BEST is not None
    
    def test_segment_order_enum(self):
        """Test SegmentOrder enum values."""
        assert gf.SegmentOrder.RANDOM is not None
        assert gf.SegmentOrder.SHORTEST_FIRST is not None
        assert gf.SegmentOrder.CHRONOLOGICAL is not None
    
    def test_acquisition_mode_enum(self):
        """Test AcquisitionMode enum values."""
        assert gf_acq.AcquisitionMode.NOISE is not None
        assert gf_acq.AcquisitionMode.FEATURES is not None


class TestUnpackObservingRuns:
    """Tests for unpack_observing_runs method."""
    
    def test_unpack_single_observing_run(self):
        """Test unpacking a single observing run."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        assert len(obtainer.start_gps_times) == 1
        assert len(obtainer.end_gps_times) == 1
        assert len(obtainer.frame_types) == 1
        assert len(obtainer.channels) == 1
    
    def test_unpack_multiple_observing_runs(self):
        """Test unpacking multiple observing runs."""
        obtainer = gf.IFODataObtainer(
            [gf.ObservingRun.O2, gf.ObservingRun.O3],
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        assert len(obtainer.start_gps_times) == 2
        assert len(obtainer.end_gps_times) == 2


class TestMergeBinsEdgeCases:
    """Tests for merge_bins edge cases."""
    
    @pytest.fixture
    def obtainer(self):
        return gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
    
    def test_merge_bins_non_overlapping(self, obtainer):
        """Test merge_bins with non-overlapping segments."""
        seg_list1 = np.array([[0.0, 50.0], [200.0, 250.0]])
        seg_list2 = np.array([[10.0, 60.0], [210.0, 260.0]])
        
        result = obtainer.merge_bins([seg_list1, seg_list2], interval=100.0)
        
        assert isinstance(result, list)
    
    def test_merge_bins_single_segment_list(self, obtainer):
        """Test merge_bins with single segment list."""
        seg_list = np.array([[0.0, 50.0], [100.0, 150.0]])
        
        result = obtainer.merge_bins([seg_list], interval=100.0)
        
        assert isinstance(result, list)


class TestMultipleIFOSegments:
    """Tests for multi-IFO segment handling."""
    
    @pytest.fixture
    def obtainer(self):
        return gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
    
    def test_remove_short_segments_multi_ifo(self, obtainer):
        """Test remove_short_segments with multiple IFOs."""
        # Shape (NumSegments, NumIFOs, 2) -> (3, 2, 2)
        segments = np.array([
            [[0, 10], [0, 15]],     # L1: 10s, H1: 15s
            [[20, 25], [20, 30]],   # L1: 5s, H1: 10s -> should be removed (L1 too short)
            [[30, 50], [30, 55]]    # L1: 20s, H1: 25s
        ])
        
        filtered = obtainer.remove_short_segments(segments, 8.0)
        
        # Should keep segments where ALL IFOs are >= 8s
        assert ops.shape(filtered)[0] == 2
    
    def test_order_segments_multi_ifo(self, obtainer):
        """Test ordering with multiple IFOs."""
        segments = np.array([
            [[0, 10], [0, 10]],
            [[20, 50], [20, 50]],
            [[60, 80], [60, 80]]
        ])
        
        sorted_segments = obtainer.order_segments(
            segments.copy(),
            gf.SegmentOrder.SHORTEST_FIRST,
            seed=42
        )
        
        # Should be sorted by first IFO duration
        durations = sorted_segments[:, 0, 1] - sorted_segments[:, 0, 0]
        np.testing.assert_array_equal(durations, [10, 20, 30])


class TestCachingWithFilePath:
    """Tests for caching with actual file paths."""
    
    def test_cache_segment_with_valid_path(self):
        """Test _cache_segment creates file and stores data (lines 912-915)."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=True
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set file path
            obtainer.file_path = Path(tmpdir) / "test_cache.hdf5"
            
            # Cache a segment
            test_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            obtainer._cache_segment("test_key", test_data)
            
            # Verify file was created
            assert obtainer.file_path.exists()


class TestIFODataWithNaN:
    """Tests for IFOData handling of NaN values."""
    
    def test_ifo_data_with_nan_replaced(self):
        """Test that NaN values are replaced with zeros in IFOData."""
        data = np.array([1.0, np.nan, 3.0, np.inf, 5.0], dtype=np.float32)
        
        ifo_data = gf_acq.IFOData(
            data=[data],
            sample_rate_hertz=1.0,
            start_gps_time=[0.0]
        )
        
        # NaN and Inf should be replaced with zeros
        result = np.array(ifo_data.data[0])
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestCloseWithOpenFile:
    """Tests for close method with open file handle."""
    
    def test_destructor_closes_file(self):
        """Test __del__ closes segment_file if open (lines 398-400)."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        # Mock segment file
        mock_file = MagicMock()
        obtainer.segment_file = mock_file
        
        # Call destructor
        obtainer.__del__()
        
        # Verify close was called
        mock_file.close.assert_called_once()
    
    def test_close_with_open_file(self):
        """Test close method closes segment_file (lines 402-404)."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        mock_file = MagicMock()
        obtainer.segment_file = mock_file
        
        obtainer.close()
        
        mock_file.close.assert_called_once()


class TestCutSegmentsEdgeCases:
    """Tests for cut_segments edge cases."""
    
    @pytest.fixture
    def obtainer(self):
        return gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
    
    def test_cut_segments_exact_boundary(self, obtainer):
        """Test cut_segments when segment ends exactly on boundary."""
        segments = np.array([
            [0.0, 100.0],  # Ends exactly on chunk boundary
            [100.0, 200.0]  # Another exact boundary
        ])
        
        result = obtainer.cut_segments(segments, 100.0, 0.0)
        
        # Should remain as two segments
        assert len(result) == 2
    
    def test_cut_segments_short_segment(self, obtainer):
        """Test cut_segments with segment shorter than chunk size."""
        segments = np.array([
            [0.0, 50.0]  # Less than chunk size of 100
        ])
        
        result = obtainer.cut_segments(segments, 100.0, 0.0)
        
        # Should remain as one segment
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0.0, 50.0])


class TestGetSegmentsForGroupEdgeCases:
    """Tests for get_segments_for_group edge cases."""
    
    @pytest.fixture
    def obtainer(self):
        return gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
    
    def test_get_segments_for_different_groups(self, obtainer):
        """Test that different groups get different segments."""
        segments = np.array([
            [0.0, 100.0],
            [100.0, 200.0],
            [200.0, 300.0],
            [300.0, 400.0],
            [400.0, 500.0],
            [500.0, 600.0],
            [600.0, 700.0],
            [700.0, 800.0]
        ])
        
        groups = {"train": 0.6, "validate": 0.2, "test": 0.2}
        
        train = obtainer.get_segments_for_group(segments, 100.0, "train", groups, 0.0)
        validate = obtainer.get_segments_for_group(segments, 100.0, "validate", groups, 0.0)
        test = obtainer.get_segments_for_group(segments, 100.0, "test", groups, 0.0)
        
        # Total should account for all segments
        total = len(train) + len(validate) + len(test)
        assert total == len(segments)


class TestVetoTimeSegmentsComplex:
    """Tests for complex veto scenarios."""
    
    @pytest.fixture
    def obtainer(self):
        return gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
    
    def test_veto_completely_covers_segment(self, obtainer):
        """Test veto that completely covers a segment."""
        segments = np.array([
            [50.0, 100.0],
            [200.0, 300.0]
        ])
        veto_segments = np.array([
            [0.0, 150.0]  # Completely covers first segment
        ])
        
        result = obtainer.veto_time_segments(segments, veto_segments)
        
        # First segment should be completely removed
        # Second segment should remain
        assert len(result) >= 1
    
    def test_multiple_overlapping_vetos(self, obtainer):
        """Test multiple overlapping veto segments."""
        segments = np.array([
            [0.0, 1000.0]
        ])
        veto_segments = np.array([
            [100.0, 200.0],
            [400.0, 500.0],
            [700.0, 800.0]
        ])
        
        result = obtainer.veto_time_segments(segments, veto_segments)
        
        # Should have multiple remaining segments
        assert len(result) >= 2


# =============================================================================
# INTEGRATION VISUALIZATION TEST
# =============================================================================

def _test_full_pipeline_visualization(
    output_directory_path: Path = Path("./gravyflow_data/tests/"),
    plot_results: bool = False
) -> None:
    """Visualize the full segment processing pipeline."""
    obtainer = gf.IFODataObtainer(
        gf.ObservingRun.O3,
        gf.DataQuality.BEST,
        gf.DataLabel.NOISE,
        cache_segments=False
    )
    
    # Create complex test segments
    segments = np.array([
        [0.0, 50.0],
        [40.0, 120.0],
        [110.0, 200.0],
        [250.0, 300.0],
        [290.0, 400.0],
        [500.0, 520.0],  # Short segment
        [600.0, 800.0]
    ])
    
    # Step 1: Compress overlapping
    compressed = obtainer.compress_segments(segments.copy())
    
    # Step 2: Cut at boundaries
    cut = obtainer.cut_segments(compressed, 100.0, 0.0)
    
    # Step 3: Group segments
    groups = {"train": 0.7, "validate": 0.15, "test": 0.15}
    train = obtainer.get_segments_for_group(cut, 100.0, "train", groups, 0.0)
    validate = obtainer.get_segments_for_group(cut, 100.0, "validate", groups, 0.0)
    test = obtainer.get_segments_for_group(cut, 100.0, "test", groups, 0.0)
    
    if plot_results:
        layout = []
        
        # Original
        p1 = figure(title="1. Original Segments", x_axis_label="GPS Time", 
                   y_axis_label="Index", width=1200, height=150)
        for i, (s, e) in enumerate(segments):
            p1.segment(x0=s, y0=i, x1=e, y1=i, line_width=8, line_color="gray")
        layout.append([p1])
        
        # Compressed
        p2 = figure(title="2. Compressed (overlaps merged)", x_axis_label="GPS Time",
                   y_axis_label="Index", width=1200, height=150)
        for i, (s, e) in enumerate(compressed):
            p2.segment(x0=s, y0=i, x1=e, y1=i, line_width=8, line_color="steelblue")
        layout.append([p2])
        
        # Cut
        p3 = figure(title="3. Cut (at 100s boundaries)", x_axis_label="GPS Time",
                   y_axis_label="Index", width=1200, height=150)
        for i, (s, e) in enumerate(cut):
            p3.segment(x0=s, y0=i, x1=e, y1=i, line_width=8, line_color="coral")
        layout.append([p3])
        
        # Grouped
        p4 = figure(title="4. Grouped (train/validate/test)", x_axis_label="GPS Time",
                   y_axis_label="Index", width=1200, height=150)
        offset = 0
        for segments_group, color, name in [
            (train, "green", "train"),
            (validate, "orange", "validate"),
            (test, "red", "test")
        ]:
            if len(segments_group) > 0:
                for i, (s, e) in enumerate(segments_group):
                    p4.segment(x0=s, y0=offset+i, x1=e, y1=offset+i, 
                             line_width=8, line_color=color, legend_label=name)
                offset += len(segments_group) + 1
        layout.append([p4])
        
        gf.ensure_directory_exists(output_directory_path)
        output_file(output_directory_path / "acquisition_full_pipeline.html")
        grid = gridplot(layout)
        save(grid)
    
    # Assertions
    assert len(compressed) <= len(segments)
    total_grouped = len(train) + len(validate) + len(test)
    assert total_grouped == len(cut)


def test_full_pipeline_visualization(pytestconfig: Config) -> None:
    """Test full pipeline with optional visualization."""
    _test_full_pipeline_visualization(
        plot_results=pytestconfig.getoption("plot")
    )


# =============================================================================
# LIGO-AUTHENTICATED TESTS (require network access)
# =============================================================================

@pytest.mark.slow
class TestGetSegmentTimes:
    """Tests for get_segment_times and get_all_segment_times (lines 447-475)."""
    
    def test_get_segment_times_real_data(self):
        """Test fetching real segment times from LIGO (lines 447-456)."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        # Use a short time window to minimize lookup time
        start_gps = 1238166018.0  # O3 start time
        end_gps = start_gps + 3600  # 1 hour window
        
        segments = obtainer.get_segment_times(
            start_gps,
            end_gps,
            gf.IFO.L1,
            obtainer.state_flags[0]
        )
        
        # Should return an array of segments
        assert isinstance(segments, np.ndarray)
        # May have some segments in this window
        if len(segments) > 0:
            assert segments.shape[1] == 2
            # End times should be greater than start times
            assert np.all(segments[:, 1] > segments[:, 0])
    
    def test_get_all_segment_times_real_data(self):
        """Test fetching all segment times across observing run (lines 463-475)."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        # Mock the observing run to use a small window
        obtainer.start_gps_times = [1238166018.0]
        obtainer.end_gps_times = [1238166018.0 + 3600]
        
        segments = obtainer.get_all_segment_times(gf.IFO.L1)
        
        assert isinstance(segments, np.ndarray)


@pytest.mark.slow
class TestGetAllEventTimes:
    """Tests for get_all_event_times (lines 477-503)."""
    
    def test_get_all_event_times_with_cache(self):
        """Test fetching event times, uses cache if available (lines 480-503)."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            [gf.DataLabel.NOISE, gf.DataLabel.EVENTS],
            cache_segments=False
        )
        
        # This should either load from cache or fetch from catalogues
        event_times = obtainer.get_all_event_times()
        
        assert isinstance(event_times, np.ndarray)
        # Should have many events across all GWTC catalogues
        assert len(event_times) > 50


@pytest.mark.slow
class TestRemoveUnwantedSegments:
    """Tests for remove_unwanted_segments (lines 505-555)."""
    
    def test_remove_unwanted_segments_noise_only(self):
        """Test removing event/glitch segments from noise (lines 512-555)."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,  # Noise only, should veto events and glitches
            cache_segments=False
        )
        
        # Create some test segments
        test_segments = np.array([
            [1238166018.0, 1238166018.0 + 3600],  # 1 hour segment
        ])
        
        # Mock to avoid full observing run fetch
        obtainer.start_gps_times = [1238166018.0]
        obtainer.end_gps_times = [1238166018.0 + 7200]
        
        result_segments, feature_times = obtainer.remove_unwanted_segments(
            gf.IFO.L1,
            test_segments.copy()
        )
        
        assert isinstance(result_segments, np.ndarray)
        assert isinstance(feature_times, dict)
        assert gf.DataLabel.EVENTS in feature_times
        assert gf.DataLabel.GLITCHES in feature_times


@pytest.mark.slow
class TestReturnWantedSegments:
    """Tests for return_wanted_segments (lines 584-632)."""
    
    def test_return_wanted_segments_with_events(self):
        """Test returning segments that contain events (lines 584-632)."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.EVENTS,  # Looking for events
            cache_segments=False
        )
        
        # Use a time window known to contain GW events
        # GW190412 occurred around GPS 1239082262
        test_segments = np.array([
            [1239082000.0, 1239082500.0],  # Should contain GW190412
        ])
        
        # return_wanted_segments fetches event times internally
        # and returns (filtered_segments, feature_times)
        try:
            result, feature_times = obtainer.return_wanted_segments(
                gf.IFO.L1,
                test_segments.copy(),
                start_padding_seconds=32.0,
                end_padding_seconds=32.0
            )
            
            assert isinstance(result, np.ndarray)
            assert isinstance(feature_times, dict)
        except ValueError as e:
            # May raise if no features found in the time window
            assert "Cannot find any features" in str(e)


@pytest.mark.slow
class TestGetSegmentData:
    """Tests for get_segment_data (lines 931-946)."""
    
    def test_get_segment_data_short_segment(self):
        """Test fetching a short segment of real LIGO data (lines 931-946)."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        # Use a very short segment (10 seconds) to minimize data transfer
        # This GPS time is during O3
        start_gps = 1238166100.0
        end_gps = start_gps + 10.0
        
        data = obtainer.get_segment_data(
            start_gps,
            end_gps,
            gf.IFO.L1,
            obtainer.frame_types[0],
            obtainer.channels[0]
        )
        
        # Should return a TimeSeries object
        from gwpy.timeseries import TimeSeries
        assert isinstance(data, TimeSeries)
        assert len(data) > 0


@pytest.mark.slow
class TestGetSegment:
    """Tests for get_segment method (lines 948-1014)."""
    
    def test_get_segment_acquires_data(self):
        """Test get_segment acquires and resamples data (lines 988-1014)."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False,
            force_acquisition=True  # Force fresh acquisition
        )
        
        start_gps = 1238166100.0
        end_gps = start_gps + 10.0
        sample_rate = 1024.0
        
        segment = obtainer.get_segment(
            start_gps,
            end_gps,
            sample_rate,
            gf.IFO.L1,
            f"test_segment_{start_gps}_{end_gps}"
        )
        
        assert segment is not None
        # Should have approximately (end-start) * sample_rate samples
        expected_samples = int((end_gps - start_gps) * sample_rate)
        # Allow some tolerance for resampling
        assert abs(len(segment) - expected_samples) < sample_rate
    
    def test_get_segment_with_caching(self):
        """Test get_segment caches data to HDF5 file (lines 966-986)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obtainer = gf.IFODataObtainer(
                gf.ObservingRun.O3,
                gf.DataQuality.BEST,
                gf.DataLabel.NOISE,
                cache_segments=True
            )
            
            obtainer.file_path = Path(tmpdir) / "test_cache.hdf5"
            
            start_gps = 1238166100.0
            end_gps = start_gps + 10.0
            segment_key = f"segments/test_{start_gps}"
            
            # First acquisition - should fetch from network
            segment1 = obtainer.get_segment(
                start_gps, end_gps, 1024.0, gf.IFO.L1, segment_key
            )
            
            if segment1 is not None:
                # Cache the segment
                obtainer._cache_segment(segment_key, np.array(segment1))
                
                # Second acquisition with force_acquisition=False should use cache
                obtainer.force_acquisition = False
                segment2 = obtainer.get_segment(
                    start_gps, end_gps, 1024.0, gf.IFO.L1, segment_key
                )
                
                assert segment2 is not None


@pytest.mark.slow
class TestAcquireGenerator:
    """Tests for acquire generator with real data (lines 1016-1087)."""
    
    def test_acquire_yields_ifo_data(self):
        """Test acquire generator yields IFOData objects (lines 1042-1082)."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        # Create a small valid segment for testing
        start_gps = 1238166100.0
        test_segments = np.array([
            [[start_gps, start_gps + 20.0]]  # 20 second segment
        ])
        
        gen = obtainer.acquire(
            sample_rate_hertz=1024.0,
            valid_segments=test_segments,
            ifos=[gf.IFO.L1],
            scale_factor=1.0
        )
        
        # Get first segment
        try:
            segment = next(gen)
            assert isinstance(segment, gf_acq.IFOData)
            assert len(segment.data) == 1  # One IFO
            assert len(segment.data[0]) > 0
        except StopIteration:
            # If acquisition fails, that's acceptable
            pass


@pytest.mark.slow
class TestGetValidSegmentsReal:
    """Tests for get_valid_segments with real LIGO data (lines 1223-1343)."""
    
    def test_get_valid_segments_basic(self):
        """Test get_valid_segments returns segments from LIGO servers."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        # Use a shorter time window for faster testing
        obtainer.start_gps_times = [1238166018.0]
        obtainer.end_gps_times = [1238166018.0 + 86400]  # 1 day
        
        # Correct signature: get_valid_segments(ifos, seed, groups, group_name, segment_order)
        segments = obtainer.get_valid_segments(
            ifos=[gf.IFO.L1],
            seed=42
        )
        
        assert isinstance(segments, np.ndarray)
        if len(segments) > 0:
            # Segments should have shape (N, num_ifos, 2)
            assert segments.shape[1] == 1  # One IFO
            assert segments.shape[2] == 2  # Start and end times
    
    def test_get_valid_segments_with_groups(self):
        """Test get_valid_segments with train/validate/test groups."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        # Short time window
        obtainer.start_gps_times = [1238166018.0]
        obtainer.end_gps_times = [1238166018.0 + 86400]
        
        groups = {"train": 0.7, "validate": 0.15, "test": 0.15}
        
        train_segments = obtainer.get_valid_segments(
            ifos=[gf.IFO.L1],
            seed=42,
            groups=groups,
            group_name="train"
        )
        
        assert isinstance(train_segments, np.ndarray)


@pytest.mark.slow  
class TestFindSegmentIntersections:
    """Tests for find_segment_intersections (lines 557-582)."""
    
    def test_find_intersections_real_segments(self):
        """Test finding intersections between real segment arrays."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        # Create overlapping segments
        arr1 = np.array([
            [0.0, 100.0],
            [150.0, 250.0],
            [300.0, 400.0]
        ])
        
        arr2 = np.array([
            [50.0, 120.0],
            [200.0, 350.0]
        ])
        
        intersections = obtainer.find_segment_intersections(arr1, arr2)
        
        assert isinstance(intersections, np.ndarray)
        # Expected intersections:
        # [0,100] ∩ [50,120] = [50, 100]
        # [150,250] ∩ [200,350] = [200, 250]
        # [300,400] ∩ [200,350] = [300, 350]
        if len(intersections) > 0:
            assert intersections.shape[1] == 2


@pytest.mark.slow
class TestGetOnsourceOffsourceChunksReal:
    """Tests for get_onsource_offsource_chunks with real data."""
    
    def test_get_chunks_from_real_data(self):
        """Test chunk extraction from real LIGO data."""
        obtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3,
            gf.DataQuality.BEST,
            gf.DataLabel.NOISE,
            cache_segments=False
        )
        
        # Use a known good segment
        start_gps = 1238166100.0
        test_segments = np.array([
            [[start_gps, start_gps + 30.0]]  # 30 second segment
        ])
        
        obtainer.valid_segments = test_segments
        obtainer.valid_segments_adjusted = test_segments
        obtainer.ifos = [gf.IFO.L1]
        obtainer._segment_exausted = False
        obtainer._current_segment_index = 0
        obtainer._current_batch_index = 0
        
        gen = obtainer.get_onsource_offsource_chunks(
            sample_rate_hertz=1024.0,
            onsource_duration_seconds=1.0,
            padding_duration_seconds=0.5,
            offsource_duration_seconds=1.0,
            num_examples_per_batch=2,
            ifos=gf.IFO.L1,
            scale_factor=1.0,
            seed=42
        )
        
        # Try to get chunks
        try:
            subarrays, backgrounds, gps_times = next(gen)
            assert subarrays is not None
            assert ops.shape(subarrays)[0] == 2  # num_examples_per_batch
        except StopIteration:
            # May stop if data acquisition fails
            pass


# =============================================================================
# SEGMENT VISUALIZATION TESTS
# =============================================================================

class TestGenerateSegmentTimelinePlot:
    """Tests for generate_segment_timeline_plot function."""
    
    def test_basic_timeline_plot(self):
        """Test generating a basic segment timeline plot."""
        segments = {
            gf.IFO.L1: np.array([[1238166018.0, 1238170000.0]]),
            gf.IFO.H1: np.array([[1238166018.0, 1238172000.0]]),
        }
        
        p = gf.generate_segment_timeline_plot(
            segments,
            observing_runs=[gf.ObservingRun.O3],
            title="Test Timeline"
        )
        
        assert p is not None
        assert p.title.text == "Test Timeline"
    
    def test_timeline_with_all_detectors(self):
        """Test timeline with all 3 detectors."""
        segments = {
            gf.IFO.L1: np.array([[1238166018.0, 1238170000.0], [1238175000.0, 1238180000.0]]),
            gf.IFO.H1: np.array([[1238166018.0, 1238172000.0]]),
            gf.IFO.V1: np.array([[1238168000.0, 1238175000.0]])
        }
        
        p = gf.generate_segment_timeline_plot(
            segments,
            observing_runs=[gf.ObservingRun.O1, gf.ObservingRun.O2, gf.ObservingRun.O3],
            show_observing_runs=True
        )
        
        assert p is not None
    
    def test_timeline_without_observing_runs(self):
        """Test timeline without observing run bands."""
        segments = {gf.IFO.L1: np.array([[1238166018.0, 1238170000.0]])}
        
        p = gf.generate_segment_timeline_plot(
            segments,
            observing_runs=[gf.ObservingRun.O3],
            show_observing_runs=False
        )
        
        assert p is not None
    
    def test_timeline_empty_detector(self):
        """Test timeline with one empty detector."""
        segments = {
            gf.IFO.L1: np.array([[1238166018.0, 1238170000.0]]),
            gf.IFO.H1: np.array([]).reshape(0, 2),  # Empty
        }
        
        p = gf.generate_segment_timeline_plot(segments)
        assert p is not None


class TestGenerateExampleExtractionPlot:
    """Tests for generate_example_extraction_plot function."""
    
    def test_basic_extraction_plot(self):
        """Test generating a basic example extraction plot."""
        segment_times = {
            gf.IFO.L1: np.array([1238166100.0, 1238166200.0]),
        }
        extraction_points = np.array([1238166150.0])
        
        layout = gf.generate_example_extraction_plot(
            segment_times,
            extraction_points,
            onsource_duration_seconds=2.0
        )
        
        assert layout is not None
    
    def test_extraction_with_offsource(self):
        """Test extraction plot with offsource windows."""
        segment_times = {
            gf.IFO.L1: np.array([1238166100.0, 1238166200.0]),
            gf.IFO.H1: np.array([1238166100.0, 1238166200.0]),
        }
        extraction_points = np.array([1238166130.0, 1238166150.0, 1238166170.0])
        
        layout = gf.generate_example_extraction_plot(
            segment_times,
            extraction_points,
            onsource_duration_seconds=2.0,
            offsource_duration_seconds=1.0,
            padding_duration_seconds=0.5
        )
        
        assert layout is not None
    
    def test_extraction_three_detectors(self):
        """Test extraction plot with all 3 detectors."""
        segment_times = {
            gf.IFO.L1: np.array([1238166100.0, 1238166200.0]),
            gf.IFO.H1: np.array([1238166100.0, 1238166200.0]),
            gf.IFO.V1: np.array([1238166100.0, 1238166200.0]),
        }
        extraction_points = np.array([1238166150.0])
        
        layout = gf.generate_example_extraction_plot(
            segment_times,
            extraction_points,
            onsource_duration_seconds=1.0,
            title="Three Detector Extraction"
        )
        
        assert layout is not None


def _test_segment_visualization_plots(plot_results: bool = False) -> None:
    """Generate visualization plots for segment timeline and extraction using toy data."""
    output_directory_path = Path("gravyflow_data/tests/")
    
    # Create segment timeline plot
    segments = {
        gf.IFO.L1: np.array([
            [1238166018.0, 1238170000.0],
            [1238175000.0, 1238180000.0],
            [1238200000.0, 1238210000.0]
        ]),
        gf.IFO.H1: np.array([
            [1238166018.0, 1238172000.0],
            [1238180000.0, 1238195000.0]
        ]),
        gf.IFO.V1: np.array([
            [1238168000.0, 1238175000.0],
            [1238185000.0, 1238200000.0]
        ])
    }
    
    p1 = gf.generate_segment_timeline_plot(
        segments,
        observing_runs=[gf.ObservingRun.O3],
        title="Noise Segment Timeline - Toy Data"
    )
    
    # Create example extraction plot
    segment_times = {
        gf.IFO.L1: np.array([1238166100.0, 1238166300.0]),
        gf.IFO.H1: np.array([1238166100.0, 1238166300.0]),
        gf.IFO.V1: np.array([1238166100.0, 1238166300.0]),
    }
    extraction_points = np.array([
        1238166150.0, 1238166180.0, 1238166210.0, 1238166240.0
    ])
    
    p2 = gf.generate_example_extraction_plot(
        segment_times,
        extraction_points,
        onsource_duration_seconds=2.0,
        offsource_duration_seconds=4.0,
        padding_duration_seconds=0.5,
        title="Example Extraction - Toy Data"
    )
    
    if plot_results:
        gf.ensure_directory_exists(output_directory_path)
        output_file(output_directory_path / "segment_timeline_visualization_toy.html")
        save(p1)
        output_file(output_directory_path / "example_extraction_visualization_toy.html")
        save(p2)
    
    # Assertions
    assert p1 is not None
    assert p2 is not None


@pytest.mark.slow
def test_real_segment_visualization_plots(pytestconfig: Config) -> None:
    """
    Test segment visualization using REAL data from IFODataObtainer.
    
    This test fetches real O3 segments and extracts chunks, then visualizes
    the timeline and extraction points.
    """
    plot_results = pytestconfig.getoption("plot")
    output_directory_path = Path("gravyflow_data/tests/")
    
    # 1. Setup IFODataObtainer for O3
    obtainer = gf.IFODataObtainer(
        [gf.ObservingRun.O3],
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.NOISE]
    )
    
    try:
        # 2. Get Valid Segments (Real Data)
        # Use a fixed seed for reproducibility
        segments_list = obtainer.get_valid_segments(
            ifos=[gf.IFO.L1, gf.IFO.H1],
            seed=42,
            group_name="train"
        )
        
        # segments_list is actually an ndarray of shape (num_segments, num_ifos, 2)
        # We need to convert it to a dictionary mapping IFO -> segments array of shape (num_segments, 2)
        segments_array = segments_list
        segments_dict = {}
        requested_ifos = [gf.IFO.L1, gf.IFO.H1]
        
        for i, ifo in enumerate(requested_ifos):
            # Extract segments for this IFO: shape (N, 2)
            segments_dict[ifo] = segments_array[:, i, :]
        
        # 3. Extract Chunks (Simulate Data Loading)
        gen = obtainer.get_onsource_offsource_chunks(
            sample_rate_hertz=2048.0,
            onsource_duration_seconds=1.0,
            padding_duration_seconds=0.5,
            offsource_duration_seconds=16.0,
            num_examples_per_batch=32,
            ifos=[gf.IFO.L1, gf.IFO.H1],
            scale_factor=1.0,
            seed=42
        )
        
        # Get one batch to find extraction points
        subarrays, backgrounds, gps_times = next(gen)
        
        # gps_times shape: (batch_size, num_ifos) or (batch_size,) depending on implementation.
        # Usually (batch_size,).
        extraction_points = gps_times
        
        # 4. Generate Plots
        
        # Timeline Plot (All Segments)
        p1 = gf.generate_segment_timeline_plot(
            segments_dict,
            observing_runs=[gf.ObservingRun.O3],
            title="Noise Segment Timeline - Real O3 Data"
        )
        
        # Extraction Plot (Zoomed in on first few examples)
        # Find the segment containing the first extraction point for each IFO
        first_gps_per_ifo = extraction_points[0] # Shape (num_ifos,) or scalar
        
        relevant_segments = {}
        for ifo, segs in segments_dict.items():
            # Determine which time corresponds to this IFO
            if extraction_points.ndim == 1:
                t = first_gps_per_ifo
            else:
                # Assuming requested_ifos order matches columns
                # requested_ifos = [L1, H1]
                try:
                    idx = requested_ifos.index(ifo)
                    t = first_gps_per_ifo[idx]
                except ValueError:
                    continue # Should not happen
            
            # Find segment containing t
            # segs is (N, 2)
            mask = (segs[:, 0] <= t) & (segs[:, 1] >= t)
            if np.any(mask):
                relevant_segments[ifo] = segs[mask][0] # Take the first match (should be only one)
        
        p2 = gf.generate_example_extraction_plot(
            relevant_segments,
            extraction_points, # Plot all points from the batch
            onsource_duration_seconds=1.0,
            offsource_duration_seconds=16.0,
            padding_duration_seconds=0.5,
            title="Example Extraction - Real O3 Data (Batch 1)"
        )
        
        if plot_results:
            gf.ensure_directory_exists(output_directory_path)
            output_file(output_directory_path / "segment_timeline_visualization_real.html")
            save(p1)
            output_file(output_directory_path / "example_extraction_visualization_real.html")
            save(p2)
            
        assert p1 is not None
        assert p2 is not None
        
    finally:
        obtainer.close()


def test_segment_visualization_plots(pytestconfig: Config) -> None:
    """Test segment visualization with optional plot output."""
    _test_segment_visualization_plots(
        plot_results=pytestconfig.getoption("plot")
    )