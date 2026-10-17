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
from gravyflow.src.resources import PACKAGE
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
            batch = next(noise(group="train"))
            
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
                PACKAGE / f"res/tests/real_noise_consistency_single.hdf5"
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
            batch = next(noise(group="train"))
            onsource = batch[gf.ReturnVariables.ONSOURCE]
            offsource = batch[gf.ReturnVariables.OFFSOURCE]
            gps_times = batch.get(gf.ReturnVariables.START_GPS_TIME)

            parameters_dict = {
                "onsource" : onsource,
                "offsource" : offsource,
                "gps_times" : gps_times
            }

            parameters_file_path : Path = (
                PACKAGE / f"res/tests/real_noise_consistancy_multi.hdf5"
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
                gf.ensure_directory_exists(output_directory_path)
                
                # Define an output path for the dashboard
                output_file(output_directory_path / "multi_noise_plots.html")

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
                
            batch = next(iter(noise(group="train")))
            onsource = batch[RV.ONSOURCE]
            offsource = batch[RV.OFFSOURCE]

            onsource_shape = onsource.shape
            offsource_shape = offsource.shape

            logging.info("Start shape tests...")
            for index, batch in tqdm(
                    enumerate(islice(noise(group="train"), num_tests))
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
            for index, _ in tqdm(enumerate(islice(noise(group="train"), num_tests))):
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

def test_unsupported_noise_type():
    """Verify error is raised for invalid noise type."""
    # This might need to test NoiseObtainer validation
    
    noise = gf.NoiseObtainer(noise_type="INVALID_TYPE")
    
    with pytest.raises(ValueError, match="NoiseType .* not recognised"):
        # Validation happens in __call__
        next(noise(group="train"))

def test_noise_obtainer_defaults():
    """Test NoiseObtainer default parameter handling."""
    noise = gf.NoiseObtainer(
        noise_type=gf.NoiseType.WHITE,
        ifos=gf.IFO.L1
    )
    
    gen = noise(group="train")
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
        group="train",
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
        group="train",
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


def test_noise_obtainer_has_no_split_of_its_own():
    """The split is the data obtainer's (its SplitSpec); NoiseObtainer takes no fractions."""
    with pytest.raises(TypeError):
        gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=gf.IFO.L1, groups={"train": 1.0})
    assert not hasattr(gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=gf.IFO.L1), "groups")


def test_noise_obtainer_requires_a_named_split():
    noise = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=gf.IFO.L1)
    with pytest.raises(ValueError, match="Name the split"):
        next(noise())
    with pytest.raises(KeyError, match="a split is one of"):
        next(noise(group="all"))


# Removed: test_noise_obtainer_generator_none_error
# This was an empty test that couldn't be triggered through normal usage.
# The error path it was meant to test is covered by type checking.


def test_noise_obtainer_real_without_ifo_obtainer():
    """REAL noise without an ifo_data_obtainer defaults to public GWOSC data;
    removing the obtainer afterwards raises ValueError."""
    noise = gf.NoiseObtainer(
        noise_type=gf.NoiseType.REAL,
        ifos=gf.IFO.L1,
        ifo_data_obtainer=None  # No obtainer: the public-data default is built
    )
    assert noise.ifo_data_obtainer is not None
    assert noise.ifo_data_obtainer.data_labels == [gf.DataLabel.NOISE]

    noise.ifo_data_obtainer = None
    with pytest.raises(ValueError, match="No IFO obtainer object present"):
        next(noise(group="train"))


def test_noise_obtainer_colored_type():
    """Test NoiseObtainer with COLORED noise type."""
    noise = gf.NoiseObtainer(
        noise_type=gf.NoiseType.COLORED,
        ifos=[gf.IFO.L1]
    )
    
    gen = noise(
        group="train",
        sample_rate_hertz=1024.0,
        onsource_duration_seconds=1.0,
        crop_duration_seconds=0.5,
        offsource_duration_seconds=1.0,
        num_examples_per_batch=2,
        seed=42
    )
    
    batch = next(gen)
    onsource = batch[RV.ONSOURCE]
    
    # Check shapes
    assert len(ops.shape(onsource)) == 3
    assert ops.shape(onsource)[0] == 2  # batch


def test_noise_obtainer_scale_factor_is_deprecated_and_ignored():
    """A scale_factor passed to a NoiseObtainer warns and changes nothing: noise is strain."""
    noise = gf.NoiseObtainer(
        noise_type=gf.NoiseType.COLORED,
        ifos=gf.IFO.L1
    )
    window = dict(
        group="train",
        sample_rate_hertz=1024.0,
        onsource_duration_seconds=1.0,
        crop_duration_seconds=0.5,
        offsource_duration_seconds=2.0,
        num_examples_per_batch=2,
        seed=42
    )
    expected = next(noise(**window))[RV.ONSOURCE]
    with pytest.warns(DeprecationWarning, match="scale_factor is deprecated.*physical strain"):
        served = next(noise(scale_factor=3.0, **window))[RV.ONSOURCE]
    np.testing.assert_array_equal(np.asarray(served), np.asarray(expected))
    # Strain: the O3 curves give a raw onsource std of order 1e-18 to 1e-21.
    assert 1e-23 < float(np.std(np.asarray(expected, np.float64))) < 1e-16


def test_noise_obtainer_pseudo_real_without_obtainer():
    """Test that PSEUDO_REAL noise without ifo_data_obtainer raises ValueError."""
    noise = gf.NoiseObtainer(
        noise_type=gf.NoiseType.PSEUDO_REAL,
        ifos=gf.IFO.L1,
        ifo_data_obtainer=None
    )
    
    with pytest.raises(ValueError, match="No IFO obtainer object present"):
        gen = noise(group="train")
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
        group="train",
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
            group="train",
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

# --- Synthetic noise batchers: one batch per address -------------------------

def _streams(seed=7, split="train", epoch=0, k=0):
    return gf.SeedTree(seed).batch(split, epoch, k)


@pytest.mark.parametrize("kind", ["white", "colored"])
def test_noise_batchers_shapes_and_addresses(kind):
    ifos = [gf.IFO.L1, gf.IFO.H1]
    if kind == "white":
        batch = gf_noise.white_noise_batcher(4, ifos, 1.0, 0.5, 4.0, 1024.0)
    else:
        batch = gf_noise.colored_noise_batcher(4, 1.0, 0.5, 4.0, ifos, 1024.0)
    first = batch(_streams())
    onsource = np.asarray(first[gf.ReturnVariables.ONSOURCE])
    offsource = np.asarray(first[gf.ReturnVariables.OFFSOURCE])
    assert onsource.shape == (4, 2, 2048) and offsource.shape == (4, 2, 4096)
    assert np.all(np.isfinite(onsource)) and np.all(np.isfinite(offsource))
    # The same address gives the same batch; any other address another.
    np.testing.assert_array_equal(np.asarray(batch(_streams())[gf.ReturnVariables.ONSOURCE]), onsource)
    for other in (_streams(k=1), _streams(epoch=1), _streams(split="validate"), _streams(seed=8)):
        assert not np.array_equal(np.asarray(batch(other)[gf.ReturnVariables.ONSOURCE]), onsource)


def test_each_detector_has_its_own_noise_stream():
    one = gf_noise.white_noise_batcher(2, [gf.IFO.L1], 1.0, 0.5, 4.0, 512.0)(_streams())
    two = gf_noise.white_noise_batcher(2, [gf.IFO.L1, gf.IFO.H1], 1.0, 0.5, 4.0, 512.0)(_streams())
    # Adding H1 leaves L1's noise as it was.
    np.testing.assert_array_equal(np.asarray(two[gf.ReturnVariables.ONSOURCE])[:, 0],
                                  np.asarray(one[gf.ReturnVariables.ONSOURCE])[:, 0])


def test_white_noise_is_unit_normal():
    batch = gf_noise.white_noise_batcher(64, [gf.IFO.L1], 1.0, 0.0, 1.0, 1024.0)(_streams())
    x = np.asarray(batch[gf.ReturnVariables.ONSOURCE])
    assert abs(x.mean()) < 0.01 and abs(x.std() - 1.0) < 0.01


def test_color_noise_shapes():
    white = jnp.ones((2, 3, 64), dtype=jnp.float32)
    asd = jnp.ones((1, 3, 33), dtype=jnp.complex64)
    assert gf_noise.color_noise(white, asd, sample_rate_hertz=64.0).shape == (2, 3, 64)


def test_obtainer_iterator_walks_the_batches_of_epoch_zero():
    obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1])
    kwargs = dict(sample_rate_hertz=512.0, onsource_duration_seconds=1.0, crop_duration_seconds=0.5,
                  offsource_duration_seconds=4.0, num_examples_per_batch=2)
    iterator = obtainer(seed=3, group="validate", **kwargs)
    batch = obtainer.batcher(gf.SeedTree(3), "validate", **kwargs)
    for k in range(3):
        np.testing.assert_array_equal(
            np.asarray(next(iterator)[gf.ReturnVariables.ONSOURCE]),
            np.asarray(batch(gf.SeedTree(3).batch("validate", 0, k))[gf.ReturnVariables.ONSOURCE]),
        )
