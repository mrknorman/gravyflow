"""
Tests for noise sampling modes (RANDOM vs GRID) and segment cycling.

These tests verify:
1. Segment cycling works correctly (regression test for bug fix)
2. RANDOM mode produces varied GPS times with possible overlap
3. GRID mode produces sequential non-overlapping chunks
4. Multi-detector mode produces different GPS times per detector
5. Segment metadata caching works without auth
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

import gravyflow as gf


class TestSamplingModeEnum:
    """Test SamplingMode enum is properly exported."""
    
    def test_sampling_mode_exists(self):
        """Verify SamplingMode enum is accessible."""
        assert hasattr(gf, 'SamplingMode')
        assert hasattr(gf.SamplingMode, 'RANDOM')
        assert hasattr(gf.SamplingMode, 'GRID')
    
    def test_sampling_mode_values_distinct(self):
        """Verify RANDOM and GRID are distinct values."""
        assert gf.SamplingMode.RANDOM != gf.SamplingMode.GRID


class TestDefaultSamplingModes:
    """Test default sampling mode selection."""
    
    def test_train_defaults_to_random(self):
        """Training group should default to RANDOM mode."""
        noise_obtainer = gf.NoiseObtainer(
            noise_type=gf.NoiseType.WHITE,
            ifos=[gf.IFO.L1]
        )
        
        dataset = gf.GravyflowDataset(
            noise_obtainer=noise_obtainer,
            group="train",
            input_variables=[gf.ReturnVariables.ONSOURCE],
            output_variables=[gf.ReturnVariables.INJECTION_MASKS],
            steps_per_epoch=1
        )
        assert dataset.sampling_mode == gf.SamplingMode.RANDOM
    
    def test_validate_defaults_to_random(self):
        """Validation group should also default to RANDOM mode."""
        noise_obtainer = gf.NoiseObtainer(
            noise_type=gf.NoiseType.WHITE,
            ifos=[gf.IFO.L1]
        )
        
        dataset = gf.GravyflowDataset(
            noise_obtainer=noise_obtainer,
            group="validate",
            input_variables=[gf.ReturnVariables.ONSOURCE],
            output_variables=[gf.ReturnVariables.INJECTION_MASKS],
            steps_per_epoch=1
        )
        assert dataset.sampling_mode == gf.SamplingMode.RANDOM
    
    def test_test_defaults_to_random(self):
        """Test group should also default to RANDOM mode."""
        noise_obtainer = gf.NoiseObtainer(
            noise_type=gf.NoiseType.WHITE,
            ifos=[gf.IFO.L1]
        )
        
        dataset = gf.GravyflowDataset(
            noise_obtainer=noise_obtainer,
            group="test",
            input_variables=[gf.ReturnVariables.ONSOURCE],
            output_variables=[gf.ReturnVariables.INJECTION_MASKS],
            steps_per_epoch=1
        )
        assert dataset.sampling_mode == gf.SamplingMode.RANDOM
    
    def test_explicit_grid_mode_works(self):
        """Explicit GRID mode should work when specified."""
        noise_obtainer = gf.NoiseObtainer(
            noise_type=gf.NoiseType.WHITE,
            ifos=[gf.IFO.L1]
        )
        
        # Explicit GRID
        dataset = gf.GravyflowDataset(
            noise_obtainer=noise_obtainer,
            group="train",
            sampling_mode=gf.SamplingMode.GRID,
            input_variables=[gf.ReturnVariables.ONSOURCE],
            output_variables=[gf.ReturnVariables.INJECTION_MASKS],
            steps_per_epoch=1
        )
        assert dataset.sampling_mode == gf.SamplingMode.GRID


class TestDefaultSaturation:
    """Test default saturation value for efficiency."""
    
    def test_default_saturation_is_one_eighth(self):
        """Default saturation should be 8.0 (higher = more samples, 8x oversampling)."""
        obtainer = gf.IFODataObtainer(
            observing_runs=gf.ObservingRun.O3,
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.NOISE]
        )
        assert obtainer.saturation == 8.0


class TestIFODataGridWindows:
    """IFOData.grid_windows: each slot once a pass, detectors kept apart."""

    SAMPLE_RATE = 64.0
    ONSOURCE, OFFSOURCE = 64, 256  # 1 s and 4 s

    def _row(self, durations, starts):
        from keras import ops

        return gf.IFOData(
            data=[ops.zeros((int(self.SAMPLE_RATE * d),), dtype="float32") for d in durations],
            sample_rate_hertz=self.SAMPLE_RATE,
            start_gps_time=list(starts),
        )

    def test_every_slot_once_then_none(self):
        """A pass serves each detector's slot at most once; a second call on the marks serves nothing."""
        row = self._row([100.0, 100.0], [1e9, 1e9])
        onsource, offsource, gps, used = row.grid_windows(self.ONSOURCE, self.OFFSOURCE, 10.0, [None, None])
        assert onsource.shape[:2] == offsource.shape[:2] == gps.shape == (len(gps), 2)
        assert len(gps) > 0
        for d in range(2):
            assert len(np.unique(gps[:, d])) == len(gps)
        assert np.min(np.abs(gps[:, 0] - gps[:, 1])) >= 10.0
        again = row.grid_windows(self.ONSOURCE, self.OFFSOURCE, 10.0, used)[2]
        assert len(again) == 0

    @pytest.mark.parametrize("durations, starts", [
        ([40.0, 40.0], [1e9, 1e9]),            # equal short pieces
        ([1024.0, 2048.0], [1e9 + 1024, 1e9]),  # one piece inside the other
    ])
    def test_adversarial_layouts_stay_apart(self, durations, starts):
        """Equal short pieces and a piece inside another: no example's detectors closer than the separation."""
        row = self._row(durations, starts)
        gps = row.grid_windows(self.ONSOURCE, self.OFFSOURCE, 12.0, [None, None])[2]
        assert len(gps) > 0
        assert np.min(np.abs(gps[:, 0] - gps[:, 1])) >= 12.0

    def test_inseparable_row_serves_nothing(self):
        """Pieces too short to hold two windows 2S apart give no example, never a coincident one."""
        row = self._row([8.0, 8.0], [1e9, 1e9])
        assert len(row.grid_windows(self.ONSOURCE, self.OFFSOURCE, 12.0, [None, None])[2]) == 0


class TestSegmentCyclingRegression:
    """Regression tests for segment cycling bug fix.
    
    Prior to the fix, the generator would get stuck on a single segment
    because the inner while loop condition didn't check _segment_exausted.
    """
    
    def test_segment_exhausted_flag_checked_in_loop(self):
        """Verify _segment_exausted is checked in the inner while condition."""
        # This is a code structure test - verify the fix is in place
        # Note: The public get_onsource_offsource_chunks is now a wrapper,
        # check the private implementation method instead.
        import inspect
        source = inspect.getsource(gf.NoiseDataObtainer._yield_onsource_offsource_chunks)
        
        # The fix added "and not self._segment_exhausted" to the inner while
        assert "and not self._segment_exhausted" in source
    
    def test_outer_while_true_wraps_segment_acquisition(self):
        """Verify outer while True loop wraps both segment acquisition and batch yielding."""
        # Note: The public get_onsource_offsource_chunks is now a wrapper,
        # check the private implementation method instead.
        import inspect
        source = inspect.getsource(gf.NoiseDataObtainer._yield_onsource_offsource_chunks)
        
        # The fix wrapped everything in "while True"
        assert "while True:" in source
        # And the segment acquisition is inside it
        assert "while self._segment_exhausted:" in source
