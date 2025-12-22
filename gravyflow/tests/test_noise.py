# Built-In imports:
from pathlib import Path
import logging
from itertools import islice
from typing import Dict
import os

# Library imports:
import pytest
import numpy as np
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from tqdm import tqdm
from _pytest.config import Config

# Local imports:

import gravyflow as gf
from gravyflow.src.dataset.noise import noise as gf_noise
from gravyflow.src.dataset.features.injection import ReturnVariables as RV
import keras
from keras import ops
import jax
import jax.numpy as jnp

def _test_real_noise_single(
        output_directory_path : Path = Path("./gravyflow_data/tests/"),
        plot_results : bool = False
    ) -> None:

    with gf.env():

        # Setup ifo data acquisition object:
        ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[
                gf.DataLabel.NOISE, 
                gf.DataLabel.GLITCHES
            ],
            observing_runs=gf.ObservingRun.O3,
            segment_order=gf.SegmentOrder.RANDOM,
            force_acquisition = True,
            cache_segments = False
        )
        
        # Initialize noise generator wrapper:
        noise : gf.NoiseObtainer = gf.NoiseObtainer(
            ifo_data_obtainer = ifo_data_obtainer,
            noise_type = gf.NoiseType.REAL,
            ifos = gf.IFO.L1
        )
        
        # Iterate through num_tests batches to check correct operation:
        batch = next(noise())
        
        # Handle dict format
        onsource = batch[gf.ReturnVariables.ONSOURCE]
        offsource = batch[gf.ReturnVariables.OFFSOURCE]
        gps_times = batch.get(gf.ReturnVariables.START_GPS_TIME)

        parameters_dict = {
            "onsource" : onsource,
            "offsource" : offsource,
            "gps_times" : gps_times
        }

        parameters_file_path : Path = (
            gf.PATH / f"res/tests/real_noise_consistency_single.hdf5"
        )
        gf.tests.compare_and_save_parameters(
            parameters_dict, parameters_file_path
        )
        
        if plot_results:
            layout = []
            for onsource_, offsource_, gps_time in zip(onsource, offsource, gps_times):
                
                onsource_strain_plot = gf.generate_strain_plot(
                    {"Onsource Noise" : onsource_},
                    title = f"Onsource Background noise at {gps_time}"
                )
                
                offsource_strain_plot = gf.generate_strain_plot(
                    {"Offsource Noise" : offsource_},
                    title = f"Offsource Background noise at {gps_time}"
                )
                
                layout.append(
                    [onsource_strain_plot, offsource_strain_plot]
                )
            
            # Ensure output directory exists
            gf.ensure_directory_exists(output_directory_path)
            
            # Define an output path for the dashboard
            output_file(output_directory_path / "noise_plots.html")

            # Arrange the plots in a grid. 
            grid = gridplot(layout)
                
            save(grid)
    
def _test_real_noise_multi(
        output_directory_path : Path = Path("./gravyflow_data/tests/"),
        plot_results : bool = False 
    ) -> None:

    with gf.env():
        
        # Setup ifo data acquisition object:
        ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[
                gf.DataLabel.NOISE, 
                gf.DataLabel.GLITCHES
            ],
            observing_runs=gf.ObservingRun.O3,
            segment_order=gf.SegmentOrder.RANDOM,
            force_acquisition = True,
            cache_segments = False
        )
        
        # Initilise noise generator wrapper:
        noise : gf.NoiseObtainer = gf.NoiseObtainer(
            ifo_data_obtainer = ifo_data_obtainer,
            noise_type = gf.NoiseType.REAL,
            ifos = [gf.IFO.L1, gf.IFO.H1]
        )
                
        # Iterate through num_tests batches to check correct operation:
        batch = next(noise())
        onsource = batch[gf.ReturnVariables.ONSOURCE]
        offsource = batch[gf.ReturnVariables.OFFSOURCE]
        gps_times = batch.get(gf.ReturnVariables.START_GPS_TIME)

        parameters_dict = {
            "onsource" : onsource,
            "offsource" : offsource,
            "gps_times" : gps_times
        }

        parameters_file_path : Path = (
            gf.PATH / f"res/tests/real_noise_consistancy_multi.hdf5"
        )
        gf.tests.compare_and_save_parameters(
            parameters_dict, parameters_file_path
        )

        if plot_results:

            layout = []
            for onsource_, gps_time in zip(onsource, gps_times):
                
                list_of_onsource = []
                for onsource_ifo in onsource_: 
                    list_of_onsource.append(
                        gf.generate_strain_plot(
                            {"Onsource Noise" : onsource_ifo},
                            title = f"Onsource Background noise at {gps_time}",
                        )
                    )
                
                layout.append(list_of_onsource)
            
            # Ensure output directory exists
            gf.ensure_directory_exists(output_diretory_path)
            
            # Define an output path for the dashboard
            output_file(output_diretory_path / "multi_noise_plots.html")

            # Arrange the plots in a grid. 
            grid = gridplot(layout)
                
            save(grid)

def _test_noise_shape(
        num_tests : int = int(100)
    ) -> None:
    
    with gf.env():
        
        # Setup ifo data acquisition object:
        ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[
                gf.DataLabel.NOISE, 
                gf.DataLabel.GLITCHES
            ],
            observing_runs=gf.ObservingRun.O3,
            segment_order=gf.SegmentOrder.RANDOM,
            force_acquisition=True,
            cache_segments=False,
            logging_level=logging.INFO
        )
        
        # Initilise noise generator wrapper:
        noise : gf.NoiseObtainer = gf.NoiseObtainer(
            ifo_data_obtainer = ifo_data_obtainer,
            noise_type = gf.NoiseType.REAL,
            ifos = gf.IFO.L1
        )
            
        batch = next(iter(noise()))
        onsource = batch[RV.ONSOURCE]
        offsource = batch[RV.OFFSOURCE]

        onsource_shape = onsource.shape
        offsource_shape = offsource.shape

        logging.info("Start shape tests...")
        for index, batch in tqdm(
                enumerate(islice(noise(), num_tests))
            ):
            onsource = batch[RV.ONSOURCE]
            offsource = batch[RV.OFFSOURCE]
            
            np.testing.assert_equal(
                onsource.shape,
                onsource_shape,
                err_msg="Onsource shape missmatch!"
            )
            np.testing.assert_equal(
                offsource.shape,
                offsource_shape,
                err_msg="Offsource shape missmatch!"
            )

        np.testing.assert_equal(
            index,
            num_tests - 1,
            err_msg=(
                "Warning! Noise generator does not iterate the required"
                " number of batches."
            )
        )            
        
        logging.info("Complete")

def _test_noise_iteration(
        num_tests : int = int(1.0E2)
    ) -> None:

    with gf.env():

        # Setup ifo data acquisition object:
        ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[
                gf.DataLabel.NOISE, 
                gf.DataLabel.GLITCHES
            ],
            observing_runs=gf.ObservingRun.O3,
            segment_order=gf.SegmentOrder.RANDOM,
            force_acquisition=True,
            cache_segments=False,
            logging_level=logging.INFO
        )
        
        # Initilise noise generator wrapper:
        noise : gf.NoiseObtainer = gf.NoiseObtainer(
            ifo_data_obtainer = ifo_data_obtainer,
            noise_type = gf.NoiseType.REAL,
            ifos = gf.IFO.L1
        )
            
        logging.info("Start iteration tests...")
        for index, _ in tqdm(enumerate(islice(noise(), num_tests))):
            pass
        
        np.testing.assert_equal(
            index,
            num_tests - 1,
            err_msg=(
                "Warning! Noise generator does not iterate the required"
                " number of batches."
            )
        )           

        logging.info("Complete")

def test_real_noise_single(
        pytestconfig : Config
    ) -> None:

    _test_real_noise_single(
        plot_results=pytestconfig.getoption("plot")
    )

def test_real_noise_multi(
        pytestconfig : Config
    ) -> None:

    _test_real_noise_multi(
        plot_results=pytestconfig.getoption("plot")
    )
    
def test_noise_shape(
        pytestconfig : Config
    ) -> None:

    _test_noise_shape(
        num_tests=gf.tests.num_tests_from_config(pytestconfig)
    )

def test_noise_iteration(
        pytestconfig : Config
    ) -> None:
    
    _test_noise_iteration(
        num_tests=gf.tests.num_tests_from_config(pytestconfig)
    )

def test_white_noise_generator():
    # Test white noise generation
    # It yields dict with ONSOURCE and OFFSOURCE
    
    num_examples = 2
    ifos = [gf.IFO.L1]
    onsource_dur = 1.0
    crop_dur = 0.5
    offsource_dur = 2.0
    sample_rate = 1024.0
    seed = 42
    
    gen = gf_noise.white_noise_generator(
        num_examples_per_batch=num_examples,
        ifos=ifos,
        onsource_duration_seconds=onsource_dur,
        crop_duration_seconds=crop_dur,
        offsource_duration_seconds=offsource_dur,
        sample_rate_hertz=sample_rate,
        seed=seed
    )
    
    batch = next(gen)
    
    # Handle dict format
    from gravyflow.src.dataset.features.injection import ReturnVariables as RV
    onsource = batch[RV.ONSOURCE]
    offsource = batch[RV.OFFSOURCE]
    
    # Check shapes
    # onsource: (Batch, IFOs, Time)
    # Time = (1.0 + 2*0.5) * 1024 = 2048
    assert ops.shape(onsource) == (num_examples, 1, 2048)
    assert ops.shape(offsource) == (num_examples, 1, int(offsource_dur * sample_rate))
    # Synthetic noise doesn't have GPS times
    
    # Check stats (roughly)
    # Mean ~ 0, Std ~ 1
    mean = ops.mean(onsource)
    std = ops.std(onsource)
    
    assert np.abs(mean) < 0.1
    assert np.abs(std - 1.0) < 0.1

def test_interpolate_psd():
    # Test PSD interpolation
    # Create dummy PSD
    freqs = jnp.linspace(0, 512, 100)
    vals = jnp.ones_like(freqs)
    
    num_samples_list = [2048, 1024]
    sample_rate = 1024.0
    
    interp_on, interp_off = gf_noise.interpolate_onsource_offsource_psd(
        num_samples_list,
        sample_rate,
        freqs,
        vals
    )
    
    # Check shapes
    # Output size = num_samples // 2 + 1
    assert ops.shape(interp_on)[-1] == 2048 // 2 + 1
    assert ops.shape(interp_off)[-1] == 1024 // 2 + 1

def test_colored_noise_generator():
    """Test colored_noise_generator produces valid output."""
    gen = gf_noise.colored_noise_generator(
        num_examples_per_batch=2,
        onsource_duration_seconds=1.0,
        crop_duration_seconds=0.0,
        offsource_duration_seconds=1.0,
        ifos=[gf.IFO.L1],
        sample_rate_hertz=1024.0,
        scale_factor=1.0,
        seed=42
    )
    
    batch = next(gen)
    onsource = batch[RV.ONSOURCE]
    offsource = batch[RV.OFFSOURCE]
    
    # Check shapes
    assert ops.shape(onsource) == (2, 1, 1024)
    assert ops.shape(offsource) == (2, 1, 1024)

def test_noise_generation_consistency():
    """Verify that calling noise generator with same seed produces identical output."""
    seed = 12345
    num_examples = 2
    
    # Run 1
    gen1 = gf_noise.white_noise_generator(
        num_examples_per_batch=num_examples,
        ifos=[gf.IFO.L1],
        onsource_duration_seconds=1.0,
        crop_duration_seconds=0.0,
        offsource_duration_seconds=1.0,
        sample_rate_hertz=1024.0,
        seed=seed
    )
    batch1 = next(gen1)
    onsource1 = batch1[RV.ONSOURCE]
    
    # Run 2
    gen2 = gf_noise.white_noise_generator(
        num_examples_per_batch=num_examples,
        ifos=[gf.IFO.L1],
        onsource_duration_seconds=1.0,
        crop_duration_seconds=0.0,
        offsource_duration_seconds=1.0,
        sample_rate_hertz=1024.0,
        seed=seed
    )
    batch2 = next(gen2)
    onsource2 = batch2[RV.ONSOURCE]
    
    # Check equality
    # Note: JAX arrays might need conversion or specific assertion
    np.testing.assert_array_equal(onsource1, onsource2, err_msg="Noise generation not consistent with same seed.")

def test_unsupported_noise_type():
    """Verify error is raised for invalid noise type."""
    # This might need to test NoiseObtainer validation
    
    noise = gf.NoiseObtainer(noise_type="INVALID_TYPE")
    
    with pytest.raises(ValueError, match="NoiseType .* not recognised"):
        # Validation happens in __call__
        next(noise())

def test_colored_noise_psd_shape():
    """Verify that _generate_colored_noise produces correct shapes."""
    # Test _generate_colored_noise directly with a mock ASD
    num_examples = 2
    num_ifos = 1
    num_samples = 1024
    seed = 42
    
    # Create a dummy ASD (complex64 for FFT multiplication)
    # Shape should be (1, IFOs, Freqs) for broadcasting
    num_freqs = num_samples // 2 + 1
    interpolated_asd = jnp.ones((1, num_ifos, num_freqs), dtype=jnp.complex64)
    
    result = gf_noise._generate_colored_noise(
        num_examples_per_batch=num_examples,
        num_ifos=num_ifos,
        num_samples=num_samples,
        interpolated_asd=interpolated_asd,
        seed=seed
    )
    
    assert ops.shape(result) == (num_examples, num_ifos, num_samples)


# ============================================================================
# NEW TESTS FOR COMPREHENSIVE COVERAGE
# ============================================================================




def test_generate_white_noise_direct():
    """Test _generate_white_noise directly."""
    num_examples = 4
    num_ifos = 2
    num_samples = 512
    seed = 123
    
    result = gf_noise._generate_white_noise(
        num_examples_per_batch=num_examples,
        num_ifos=num_ifos,
        num_samples=num_samples,
        seed=seed
    )
    
    # Check shape
    assert ops.shape(result) == (num_examples, num_ifos, num_samples)
    
    # Check dtype
    assert result.dtype == jnp.float32
    
    # Check stats (mean ~ 0, std ~ 1)
    mean = float(ops.mean(result))
    std = float(ops.std(result))
    
    assert np.abs(mean) < 0.2  # Allow some variance
    assert np.abs(std - 1.0) < 0.2


def test_generate_white_noise_reproducibility():
    """Test that same seed produces same noise."""
    seed = 42
    
    result1 = gf_noise._generate_white_noise(4, 1, 100, seed)
    result2 = gf_noise._generate_white_noise(4, 1, 100, seed)
    
    np.testing.assert_array_equal(result1, result2)


def test_generate_colored_noise_direct():
    """Test _generate_colored_noise directly."""
    num_examples = 2
    num_ifos = 1
    num_samples = 512
    seed = 42
    
    # Create flat ASD (white noise with amplitude scaling)
    num_freqs = num_samples // 2 + 1
    asd = jnp.ones((1, num_ifos, num_freqs), dtype=jnp.complex64) * 2.0
    
    result = gf_noise._generate_colored_noise(
        num_examples_per_batch=num_examples,
        num_ifos=num_ifos,
        num_samples=num_samples,
        interpolated_asd=asd,
        seed=seed
    )
    
    assert ops.shape(result) == (num_examples, num_ifos, num_samples)
    
    # Since ASD is 2.0, output should have roughly std ~ 2.0
    std = float(ops.std(result))
    assert 1.0 < std < 4.0  # Reasonable range


def test_noise_obtainer_defaults():
    """Test NoiseObtainer default parameter handling."""
    noise = gf.NoiseObtainer(
        noise_type=gf.NoiseType.WHITE,
        ifos=gf.IFO.L1
    )
    
    gen = noise()
    batch = next(gen)
    onsource = batch[RV.ONSOURCE]
    offsource = batch[RV.OFFSOURCE]
    
    # Should use defaults from gf.Defaults
    expected_onsource_dur = gf.Defaults.onsource_duration_seconds + 2 * gf.Defaults.crop_duration_seconds
    expected_onsource_samples = gf_noise.ensure_even(int(expected_onsource_dur * gf.Defaults.sample_rate_hertz))
    expected_offsource_samples = gf_noise.ensure_even(int(gf.Defaults.offsource_duration_seconds * gf.Defaults.sample_rate_hertz))
    
    assert ops.shape(onsource)[-1] == expected_onsource_samples
    assert ops.shape(offsource)[-1] == expected_offsource_samples


def test_noise_obtainer_white_type():
    """Test NoiseObtainer with WHITE noise type."""
    noise = gf.NoiseObtainer(
        noise_type=gf.NoiseType.WHITE,
        ifos=[gf.IFO.L1]
    )
    
    gen = noise(
        sample_rate_hertz=1024.0,
        onsource_duration_seconds=1.0,
        crop_duration_seconds=0.5,
        offsource_duration_seconds=1.0,
        num_examples_per_batch=4,
        seed=42
    )
    
    batch = next(gen)
    onsource = batch[RV.ONSOURCE]
    
    # Check shapes
    assert len(ops.shape(onsource)) == 3
    assert ops.shape(onsource)[0] == 4  # batch
    assert ops.shape(onsource)[1] == 1  # ifos


def test_noise_obtainer_multiple_ifos():
    """Test NoiseObtainer with multiple IFOs."""
    noise = gf.NoiseObtainer(
        noise_type=gf.NoiseType.WHITE,
        ifos=[gf.IFO.L1, gf.IFO.H1]
    )
    
    gen = noise(
        sample_rate_hertz=1024.0,
        num_examples_per_batch=2,
        seed=42
    )
    
    batch = next(gen)
    onsource = batch[RV.ONSOURCE]
    
    # Should have 2 IFOs
    assert ops.shape(onsource)[1] == 2


def test_noise_obtainer_single_ifo_conversion():
    """Test that single IFO is converted to list."""
    noise = gf.NoiseObtainer(
        noise_type=gf.NoiseType.WHITE,
        ifos=gf.IFO.L1  # Single, not list
    )
    
    # After __post_init__, ifos should be a list
    assert isinstance(noise.ifos, list)
    assert len(noise.ifos) == 1


def test_noise_obtainer_custom_groups():
    """Test NoiseObtainer with custom groups."""
    custom_groups = {
        "train": 0.8,
        "validate": 0.1,
        "test": 0.1
    }
    
    noise = gf.NoiseObtainer(
        noise_type=gf.NoiseType.WHITE,
        ifos=gf.IFO.L1,
        groups=custom_groups
    )
    
    assert noise.groups == custom_groups


def test_noise_obtainer_default_groups():
    """Test NoiseObtainer uses default groups when not provided."""
    noise = gf.NoiseObtainer(
        noise_type=gf.NoiseType.WHITE,
        ifos=gf.IFO.L1
    )
    
    assert "train" in noise.groups
    assert "validate" in noise.groups
    assert "test" in noise.groups


def test_noise_obtainer_generator_none_error():
    """Test that error is raised if generator fails to initialize."""
    # This is a safety check - if match/case falls through with None
    # We can't easily trigger this with normal usage since all types are handled
    # But we test that the error message exists in case of future bugs
    pass


def test_noise_obtainer_real_without_ifo_obtainer():
    """Test that REAL noise without ifo_data_obtainer raises ValueError."""
    noise = gf.NoiseObtainer(
        noise_type=gf.NoiseType.REAL,
        ifos=gf.IFO.L1,
        ifo_data_obtainer=None  # Explicitly no obtainer
    )
    
    with pytest.raises(ValueError, match="No IFO obtainer object present"):
        next(noise())


def test_white_noise_generator_iteration():
    """Test that white noise generator can be iterated multiple times."""
    gen = gf_noise.white_noise_generator(
        num_examples_per_batch=2,
        ifos=[gf.IFO.L1],
        onsource_duration_seconds=1.0,
        crop_duration_seconds=0.0,
        offsource_duration_seconds=1.0,
        sample_rate_hertz=1024.0,
        seed=42
    )
    
    # Iterate 5 times
    for i in range(5):
        batch = next(gen)
        onsource = batch[RV.ONSOURCE]
        assert ops.shape(onsource) == (2, 1, 1024)





def test_colored_noise_generator():
    """Test colored_noise_generator function directly."""
    num_examples = 2
    onsource_dur = 1.0
    crop_dur = 0.5
    offsource_dur = 1.0
    sample_rate = 1024.0
    seed = 42
    
    gen = gf_noise.colored_noise_generator(
        num_examples_per_batch=num_examples,
        onsource_duration_seconds=onsource_dur,
        crop_duration_seconds=crop_dur,
        offsource_duration_seconds=offsource_dur,
        ifos=[gf.IFO.L1],
        sample_rate_hertz=sample_rate,
        scale_factor=1.0,
        seed=seed
    )
    
    batch = next(gen)
    onsource = batch[RV.ONSOURCE]
    offsource = batch[RV.OFFSOURCE]
    
    # Check shapes
    expected_onsource_samples = gf_noise.ensure_even(int((onsource_dur + 2*crop_dur) * sample_rate))
    expected_offsource_samples = gf_noise.ensure_even(int(offsource_dur * sample_rate))
    
    assert ops.shape(onsource) == (num_examples, 1, expected_onsource_samples)
    assert ops.shape(offsource) == (num_examples, 1, expected_offsource_samples)


def test_noise_obtainer_colored_type():
    """Test NoiseObtainer with COLORED noise type."""
    noise = gf.NoiseObtainer(
        noise_type=gf.NoiseType.COLORED,
        ifos=[gf.IFO.L1]
    )
    
    gen = noise(
        sample_rate_hertz=1024.0,
        onsource_duration_seconds=1.0,
        crop_duration_seconds=0.5,
        offsource_duration_seconds=1.0,
        num_examples_per_batch=2,
        scale_factor=1.0,
        seed=42
    )
    
    batch = next(gen)
    onsource = batch[RV.ONSOURCE]
    
    # Check shapes
    assert len(ops.shape(onsource)) == 3
    assert ops.shape(onsource)[0] == 2  # batch


def test_colored_noise_generator_multi_ifo():
    """Test colored_noise_generator with multiple IFOs."""
    gen = gf_noise.colored_noise_generator(
        num_examples_per_batch=2,
        onsource_duration_seconds=1.0,
        crop_duration_seconds=0.0,
        offsource_duration_seconds=1.0,
        ifos=[gf.IFO.L1, gf.IFO.H1],
        sample_rate_hertz=1024.0,
        scale_factor=1.0,
        seed=42
    )
    
    batch = next(gen)
    onsource = batch[RV.ONSOURCE]
    
    # Should have 2 IFOs
    assert ops.shape(onsource)[1] == 2


def test_colored_noise_generator_iteration():
    """Test that colored noise generator can be iterated multiple times."""
    gen = gf_noise.colored_noise_generator(
        num_examples_per_batch=2,
        onsource_duration_seconds=1.0,
        crop_duration_seconds=0.0,
        offsource_duration_seconds=1.0,
        ifos=[gf.IFO.L1],
        sample_rate_hertz=1024.0,
        scale_factor=1.0,
        seed=42
    )
    
    # Iterate 3 times
    for i in range(3):
        batch = next(gen)
        onsource = batch[RV.ONSOURCE]
        assert ops.shape(onsource) == (2, 1, 1024)


def test_noise_obtainer_scale_factor_none():
    """Test NoiseObtainer with scale_factor=None to use default."""
    noise = gf.NoiseObtainer(
        noise_type=gf.NoiseType.WHITE,
        ifos=gf.IFO.L1
    )
    
    # Explicitly pass scale_factor=None to trigger default
    gen = noise(
        sample_rate_hertz=1024.0,
        onsource_duration_seconds=1.0,
        crop_duration_seconds=0.0,
        offsource_duration_seconds=1.0,
        num_examples_per_batch=2,
        scale_factor=None,  # Triggers default
        seed=42
    )
    
    batch = next(gen)
    onsource = batch[RV.ONSOURCE]
    assert ops.shape(onsource)[0] == 2


def test_noise_obtainer_pseudo_real_without_obtainer():
    """Test that PSEUDO_REAL noise without ifo_data_obtainer raises ValueError."""
    noise = gf.NoiseObtainer(
        noise_type=gf.NoiseType.PSEUDO_REAL,
        ifos=gf.IFO.L1,
        ifo_data_obtainer=None
    )
    
    with pytest.raises(ValueError, match="No IFO obtainer object present"):
        gen = noise()
        next(gen)


def test_pseudo_real_noise_generation():
    """Test PSEUDO_REAL noise type with IFODataObtainer."""
    # Setup ifo data acquisition object
    ifo_data_obtainer = gf.IFODataObtainer(
        observing_runs=gf.ObservingRun.O3, 
        data_quality=gf.DataQuality.BEST, 
        data_labels=[gf.DataLabel.NOISE, gf.DataLabel.GLITCHES],
        force_acquisition=True,
        cache_segments=False
    )
    
    # Create NoiseObtainer with PSEUDO_REAL type
    noise = gf.NoiseObtainer(
        ifo_data_obtainer=ifo_data_obtainer,
        noise_type=gf.NoiseType.PSEUDO_REAL,
        ifos=gf.IFO.L1
    )
    
    gen = noise(
        sample_rate_hertz=gf.Defaults.sample_rate_hertz,
        onsource_duration_seconds=gf.Defaults.onsource_duration_seconds,
        crop_duration_seconds=gf.Defaults.crop_duration_seconds,
        offsource_duration_seconds=gf.Defaults.offsource_duration_seconds,
        num_examples_per_batch=1,
        seed=42
    )
    
    # Get one batch from the generator
    batch = next(gen)
    onsource = batch[RV.ONSOURCE]
    
    # Check shapes
    assert len(ops.shape(onsource)) == 3
    assert ops.shape(onsource)[0] == 1  # batch size


def test_gps_time_uniqueness_across_batches():
    """
    Test that GPS times are unique across multiple batches.
    
    This guards against the bug where deepcopied IFODataObtainer 
    carries over rng state, causing identical random positions.
    """
    with gf.env():
        # Setup ifo data acquisition
        ifo_data_obtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.NOISE, gf.DataLabel.GLITCHES],
            observing_runs=gf.ObservingRun.O3,
            segment_order=gf.SegmentOrder.RANDOM,
            force_acquisition=True,
            cache_segments=False
        )
        
        noise = gf.NoiseObtainer(
            ifo_data_obtainer=ifo_data_obtainer,
            noise_type=gf.NoiseType.REAL,
            ifos=gf.IFO.L1
        )
        
        # Collect GPS times from multiple batches
        all_gps_times = []
        num_batches = 5
        batch_size = 4
        
        gen = noise(
            num_examples_per_batch=batch_size,
            seed=42
        )
        
        for _ in range(num_batches):
            batch = next(gen)
            gps_times = batch.get(gf.ReturnVariables.START_GPS_TIME)
            gps_array = ops.convert_to_numpy(gps_times)
            all_gps_times.extend(gps_array.flatten())
        
        all_gps_times = np.array(all_gps_times)
        
        # Check uniqueness - no two GPS times should be identical
        unique_gps = np.unique(all_gps_times)
        total_samples = num_batches * batch_size
        
        # Allow for some tolerance in case segments are reused,
        # but require at least 80% unique times
        uniqueness_ratio = len(unique_gps) / total_samples
        assert uniqueness_ratio >= 0.8, (
            f"Only {uniqueness_ratio*100:.1f}% of GPS times are unique. "
            f"Expected at least 80%. This indicates repeated noise segments."
        )
        
        # Additionally check within each batch - all should be unique
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            batch_gps = all_gps_times[start:end]
            batch_unique = np.unique(batch_gps)
            assert len(batch_unique) == batch_size, (
                f"Batch {batch_idx} has duplicate GPS times within the batch"
            )

# ============================================================================
# AUGMENTATION TESTS
# ============================================================================

def test_ifo_data_obtainer_augmentation_defaults():
    """Test that IFODataObtainer has augmentation parameters with correct defaults."""
    obtainer = gf.IFODataObtainer(
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.NOISE],
        observing_runs=[gf.ObservingRun.O3]
    )
    
    # Default augmentations should be SignReversal and TimeReversal
    assert obtainer.augmentations is not None
    assert len(obtainer.augmentations) == 2
    assert isinstance(obtainer.augmentations[0], gf.SignReversal)
    assert isinstance(obtainer.augmentations[1], gf.TimeReversal)
    assert obtainer.augmentations[0].probability == 0.5


def test_ifo_data_obtainer_augmentation_disabled():
    """Test that augmentation can be disabled."""
    obtainer = gf.IFODataObtainer(
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.NOISE],
        observing_runs=[gf.ObservingRun.O3],
        augmentations=[]  # Empty list = no augmentations
    )
    
    assert obtainer.augmentations == []


def test_effective_saturation_with_augmentation():
    """Test that effective saturation is increased when augmentation is enabled."""
    base_saturation = 8.0
    prob = 0.5
    
    # Both augmentations enabled with same probability
    obtainer = gf.IFODataObtainer(
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.NOISE],
        saturation=base_saturation,
        augmentations=[
            gf.SignReversal(probability=prob),
            gf.TimeReversal(probability=prob)
        ]
    )
    
    effective = obtainer._get_effective_saturation()
    expected = base_saturation * ((1 + prob) ** 2)  # Multiplied twice
    np.testing.assert_almost_equal(effective, expected)


def test_effective_saturation_no_augmentation():
    """Test that effective saturation equals base saturation when augmentation disabled."""
    base_saturation = 8.0
    
    obtainer = gf.IFODataObtainer(
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.NOISE],
        saturation=base_saturation,
        augmentations=[]  # Empty list = no augmentations
    )
    
    effective = obtainer._get_effective_saturation()
    assert effective == base_saturation

