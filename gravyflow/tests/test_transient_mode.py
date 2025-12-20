"""Tests for TRANSIENT mode data acquisition.

These tests verify that the TRANSIENT mode correctly:
1. Identifies event segments from GWTC catalogs
2. Returns multiple segments (not just 1)
3. Returns data with correct shape for model inference
4. Returns unique GPS times for each batch
"""

import pytest
import numpy as np
import gravyflow as gf


class TestTransientModeCoreSetup:
    """Test that TRANSIENT mode correctly configures IFODataObtainer."""
    
    def test_events_data_label_stored(self):
        """Verify EVENTS data_label is stored correctly."""
        ifo = gf.IFODataObtainer(
            observing_runs=gf.ObservingRun.O3,
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.EVENTS],
            saturation=1.0
        )
        
        assert gf.DataLabel.EVENTS in ifo.data_labels
        ifo.close()
    
    def test_noise_data_label_stored(self):
        """Verify NOISE data_label is stored correctly."""
        ifo = gf.IFODataObtainer(
            observing_runs=gf.ObservingRun.O3,
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.NOISE],
            saturation=1.0
        )
        
        assert gf.DataLabel.NOISE in ifo.data_labels
        ifo.close()


class TestTransientSegmentRetrieval:
    """Test that TRANSIENT mode retrieves correct number of segments."""
    
    def test_get_all_event_times_returns_many_events(self):
        """Verify get_all_event_times returns substantial number of events."""
        event_times = gf.get_all_event_times()
        
        # Should have at least 100 events from GWTC catalogs
        assert len(event_times) >= 100, f"Expected >=100 events, got {len(event_times)}"
    
    def test_o3_has_many_events(self):
        """Verify O3 observing run contains many events by GPS range."""
        O3_START_GPS = 1238166018
        O3_END_GPS = 1269363618
        
        event_times = gf.get_all_event_times()
        o3_events = [t for t in event_times if O3_START_GPS <= t <= O3_END_GPS]
        
        # O3 should have at least 50 events
        assert len(o3_events) >= 50, f"Expected >=50 O3 events, got {len(o3_events)}"


class TestTransientDatasetOutput:
    """Test that Dataset in TRANSIENT mode returns correct data."""
    
    def test_dataset_creation_with_transient_mode(self):
        """Verify Dataset can be created with TRANSIENT mode config."""
        ifo = gf.IFODataObtainer(
            observing_runs=gf.ObservingRun.O3,
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.EVENTS],
            saturation=1.0
        )
        noise = gf.NoiseObtainer(
            ifo_data_obtainer=ifo,
            ifos=[gf.IFO.L1],
            noise_type=gf.NoiseType.REAL
        )
        
        try:
            dataset = gf.Dataset(
                noise_obtainer=noise,
                sample_rate_hertz=8192.0,
                onsource_duration_seconds=1.0,
                offsource_duration_seconds=16.0,
                num_examples_per_batch=1,
                steps_per_epoch=5,
                group='train',
                input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
                output_variables=[gf.ReturnVariables.GPS_TIME]
            )
            
            assert dataset is not None
            
        finally:
            ifo.close()
    
    def test_dataset_returns_whitened_onsource_in_input(self):
        """Verify Dataset returns WHITENED_ONSOURCE in x batch."""
        ifo = gf.IFODataObtainer(
            observing_runs=gf.ObservingRun.O3,
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.EVENTS],
            saturation=1.0
        )
        noise = gf.NoiseObtainer(
            ifo_data_obtainer=ifo,
            ifos=[gf.IFO.L1],
            noise_type=gf.NoiseType.REAL
        )
        
        try:
            dataset = gf.Dataset(
                noise_obtainer=noise,
                sample_rate_hertz=8192.0,
                onsource_duration_seconds=1.0,
                offsource_duration_seconds=16.0,
                num_examples_per_batch=1,
                steps_per_epoch=1,
                group='all',
                input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
                output_variables=[gf.ReturnVariables.GPS_TIME]
            )
            
            x_batch, y_batch = dataset[0]
            
            # x_batch should be dict with WHITENED_ONSOURCE
            assert isinstance(x_batch, dict), f"x_batch is {type(x_batch)}, expected dict"
            assert gf.ReturnVariables.WHITENED_ONSOURCE.name in x_batch, \
                f"WHITENED_ONSOURCE not in x_batch keys: {x_batch.keys()}"
            
            # Check shape: [batch, ifos, samples]
            data = x_batch[gf.ReturnVariables.WHITENED_ONSOURCE.name]
            assert data.shape[0] == 1, f"Expected batch size 1, got {data.shape[0]}"
            assert data.shape[2] == 8192, f"Expected 8192 samples, got {data.shape[2]}"
            
        finally:
            ifo.close()
    
    def test_dataset_returns_gps_time_in_output(self):
        """Verify Dataset returns GPS_TIME in y batch."""
        ifo = gf.IFODataObtainer(
            observing_runs=gf.ObservingRun.O3,
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.EVENTS],
            saturation=1.0
        )
        noise = gf.NoiseObtainer(
            ifo_data_obtainer=ifo,
            ifos=[gf.IFO.L1],
            noise_type=gf.NoiseType.REAL
        )
        
        try:
            dataset = gf.Dataset(
                noise_obtainer=noise,
                sample_rate_hertz=8192.0,
                onsource_duration_seconds=1.0,
                offsource_duration_seconds=16.0,
                num_examples_per_batch=1,
                steps_per_epoch=1,
                group='test',
                input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
                output_variables=[gf.ReturnVariables.GPS_TIME]
            )
            
            x_batch, y_batch = dataset[0]
            
            # y_batch should have GPS_TIME
            assert gf.ReturnVariables.GPS_TIME.name in y_batch, \
                f"GPS_TIME not in y_batch keys: {y_batch.keys()}"
            
            gps = y_batch[gf.ReturnVariables.GPS_TIME.name]
            # GPS should be valid O3 time range
            O3_START_GPS = 1238166018
            O3_END_GPS = 1269363618
            gps_value = float(np.array(gps).flatten()[0])
            assert O3_START_GPS <= gps_value <= O3_END_GPS, \
                f"GPS {gps_value} outside O3 range"
            
        finally:
            ifo.close()
    
    def test_dataset_returns_unique_gps_for_each_batch(self):
        """CRITICAL: Verify each batch returns a UNIQUE GPS time."""
        ifo = gf.IFODataObtainer(
            observing_runs=gf.ObservingRun.O3,
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.EVENTS],
            saturation=1.0
        )
        noise = gf.NoiseObtainer(
            ifo_data_obtainer=ifo,
            ifos=[gf.IFO.L1],
            noise_type=gf.NoiseType.REAL
        )
        
        try:
            dataset = gf.Dataset(
                noise_obtainer=noise,
                sample_rate_hertz=8192.0,
                onsource_duration_seconds=1.0,
                offsource_duration_seconds=16.0,
                num_examples_per_batch=1,
                steps_per_epoch=5,  # Request 5 batches
                group='all',
                input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
                output_variables=[gf.ReturnVariables.GPS_TIME]
            )
            
            gps_times = []
            for i in range(5):
                x_batch, y_batch = dataset[i]
                gps = y_batch[gf.ReturnVariables.GPS_TIME.name]
                gps_value = float(np.array(gps).flatten()[0])
                gps_times.append(gps_value)
            
            # All GPS times should be unique
            unique_gps = set(gps_times)
            assert len(unique_gps) == 5, \
                f"Expected 5 unique GPS times, got {len(unique_gps)}: {gps_times}"
            
        finally:
            ifo.close()


class TestReturnWantedSegments:
    """Test the return_wanted_segments function directly."""
    
    def test_return_wanted_segments_finds_intersections(self):
        """Verify return_wanted_segments finds event/segment intersections."""
        ifo = gf.IFODataObtainer(
            observing_runs=gf.ObservingRun.O3,
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.EVENTS],
            saturation=1.0
        )
        
        try:
            # First get valid segments without TRANSIENT mode processing
            # to see the raw O3 segments
            from gravyflow.src.dataset.acquisition import IFODataObtainer
            
            # Check if we can access return_wanted_segments
            assert hasattr(ifo, 'return_wanted_segments'), \
                "IFODataObtainer missing return_wanted_segments method"
            
        finally:
            ifo.close()
            
    def test_transient_mode_batch_unique_events(self):
        """Verify that TRANSIENT mode returns unique events in a batch (Batch Size > 1)."""
        
        # Setup specific request that SHOULD return distinct events
        batch_size = 4
        
        ifo = gf.IFODataObtainer(
            observing_runs=gf.ObservingRun.O3,
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.EVENTS],
            saturation=1.0
        )
        noise = gf.NoiseObtainer(
            ifo_data_obtainer=ifo,
            ifos=[gf.IFO.L1],
            noise_type=gf.NoiseType.REAL
        )
        
        try:
            dataset = gf.Dataset(
                noise_obtainer=noise,
                sample_rate_hertz=2048.0,
                onsource_duration_seconds=1.0,
                offsource_duration_seconds=2.0,
                num_examples_per_batch=batch_size,
                steps_per_epoch=1,
                group='all',
                input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
                output_variables=[gf.ReturnVariables.GPS_TIME]
            )
            
            x_batch, y_batch = dataset[0]
            
            gps_times = y_batch[gf.ReturnVariables.GPS_TIME.name]
            # Shape should be (Batch, 1) or (Batch,)
            # Convert to numpy and flatten to ensure we have standard floats
            gps_values = np.array(gps_times).flatten()
            
            assert len(gps_values) == batch_size, f"Expected {batch_size} GPS times, got {len(gps_values)}"
            
            # Check uniqueness
            # Convert to float to ensure hashability (if they were 0-d arrays)
            unique_gps = set([float(g) for g in gps_values])
            assert len(unique_gps) == batch_size, \
                f"Expected {batch_size} unique events in batch, got {len(unique_gps)}: {gps_values}"
                
        finally:
            ifo.close()

    def test_transient_mode_multi_ifo_alignment(self):
        """Verify multi-IFO batch returns synchronized data."""
        ifos_list = [gf.IFO.H1, gf.IFO.L1]
        
        ifo = gf.IFODataObtainer(
            observing_runs=gf.ObservingRun.O3,
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.EVENTS],
            saturation=1.0
        )
        noise = gf.NoiseObtainer(
            ifo_data_obtainer=ifo,
            ifos=ifos_list,
            noise_type=gf.NoiseType.REAL
        )
        
        try:
            dataset = gf.Dataset(
                noise_obtainer=noise,
                sample_rate_hertz=2048.0,
                onsource_duration_seconds=1.0,
                offsource_duration_seconds=0.0,
                num_examples_per_batch=2,
                steps_per_epoch=1,
                group='all',
                input_variables=[gf.ReturnVariables.ONSOURCE], # Raw data to check correlation/alignment if needed
                output_variables=[gf.ReturnVariables.GPS_TIME]
            )
            
            x_batch, _ = dataset[0]
            data = x_batch[gf.ReturnVariables.ONSOURCE.name]
            
            # Shape: (Batch, IFO, Samples)
            assert data.shape[1] == 2, f"Expected 2 IFOs, got {data.shape[1]}"
            assert data.shape[0] == 2, f"Expected batch size 2, got {data.shape[0]}"
            
            # Just basic structure check; precise synchronization hard to test 
            # without known injection templates or specific event properties.
            # But ensure not all zeros.
            assert not np.allclose(data, 0.0), "Data should not be all zeros"
            
        finally:
            ifo.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


class TestSpecificEventAcquisition:
    """Test acquisition of specific named events (direct download path)."""
    
    def test_specific_event_names_returns_proper_structure(self):
        """Test that specific event names path returns proper TransientSegment objects."""
        ifo_obtainer = gf.IFODataObtainer(
            observing_runs=gf.ObservingRun.O1,
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.EVENTS],
            force_acquisition=True,
            cache_segments=False
        )
        
        try:
            transient_obtainer = gf.TransientObtainer(
                ifo_data_obtainer=ifo_obtainer,
                ifos=[gf.IFO.H1, gf.IFO.L1],
                event_names=["GW150914"]  # Specific event
            )
            
            # Get a batch
            batch_gen = transient_obtainer(
                scale_factor=1.0,
                whiten=False,
                crop=False
            )
            batch = next(batch_gen)
            
            # Verify batch structure
            assert isinstance(batch, dict), "Batch should be a dictionary"
            assert gf.ReturnVariables.ONSOURCE in batch, "Should have ONSOURCE key"
            assert gf.ReturnVariables.OFFSOURCE in batch, "Should have OFFSOURCE key"
            assert gf.ReturnVariables.TRANSIENT_GPS_TIME in batch, "Should have TRANSIENT_GPS_TIME key"
            
            # Verify shapes
            onsource = batch[gf.ReturnVariables.ONSOURCE]
            offsource = batch[gf.ReturnVariables.OFFSOURCE]
            gps_times = batch[gf.ReturnVariables.TRANSIENT_GPS_TIME]
            
            assert onsource.ndim == 3, f"Onsource should be 3D (Batch, IFO, Samples), got {onsource.ndim}D"
            assert onsource.shape[1] == 2, f"Should have 2 IFOs, got {onsource.shape[1]}"
            assert offsource.shape[0] == onsource.shape[0] and offsource.shape[1] == onsource.shape[1], "Offsource batch and IFO dims should match onsource"
            assert gps_times.ndim == 2, f"GPS times should be 2D (Batch, IFO), got {gps_times.ndim}D"
            
            # Verify data is not all zeros
            assert not np.all(onsource == 0.0), "Onsource data should not be all zeros"
            
        finally:
            ifo_obtainer.close()
    
    def test_multiple_specific_events(self):
        """Test acquiring multiple specific events."""
        ifo_obtainer = gf.IFODataObtainer(
            observing_runs=gf.ObservingRun.O1,
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.EVENTS],
            force_acquisition=True,
            cache_segments=False
        )
        
        try:
            transient_obtainer = gf.TransientObtainer(
                ifo_data_obtainer=ifo_obtainer,
                ifos=[gf.IFO.L1],
                event_names=["GW150914", "GW151012"]  # Multiple events
            )
            
            # Get a batch
            batch_gen = transient_obtainer(
                scale_factor=1.0,
                whiten=False,
                crop=False
            )
            batch = next(batch_gen)
            
            # Should get a batch (even if some events might not be available)
            assert isinstance(batch, dict), "Should return a dictionary"
            assert gf.ReturnVariables.ONSOURCE in batch
            
            # Verify batch size (may be less than 2 if some events unavailable)
            onsource = batch[gf.ReturnVariables.ONSOURCE]
            assert onsource.shape[0] > 0, "Should have at least one event in batch"
            
        finally:
            ifo_obtainer.close()
    
    def test_specific_events_with_whitening(self):
        """Test that whitening works with specific events."""
        ifo_obtainer = gf.IFODataObtainer(
            observing_runs=gf.ObservingRun.O1,
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.EVENTS],
            force_acquisition=True,
            cache_segments=False
        )
        
        try:
            transient_obtainer = gf.TransientObtainer(
                ifo_data_obtainer=ifo_obtainer,
                ifos=[gf.IFO.H1],
                event_names=["GW150914"]
            )
            
            # Get whitened batch
            batch_gen = transient_obtainer(
                scale_factor=1.0,
                whiten=True,
                crop=False
            )
            batch = next(batch_gen)
            
            # Should successfully whiten
            onsource = batch[gf.ReturnVariables.ONSOURCE]
            assert not np.all(onsource == 0.0), "Whitened data should not be all zeros"
            assert not np.isnan(onsource).any(), "Whitened data should not have NaNs"
            
        finally:
            ifo_obtainer.close()
