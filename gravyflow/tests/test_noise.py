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


# --- Real noise addressed by index, offline: GRID and PSEUDO_REAL --------------
#
# The strain is a fake GWOSC fetch of hashed uniform noise: each sample's value is a
# function of its absolute index alone, scaled by the amplitude of the science
# segment it lies in, so an example's level names the kind of segment it came from.

from gravyflow.src.dataset.acquisition.sources import SegmentStrain  # noqa: E402
from gravyflow.src.dataset.acquisition import noise as acquisition_noise  # noqa: E402
from gravyflow.tests.conftest import gwosc_serving  # noqa: E402

OFFLINE_RATE = 256.0
OFFLINE_WINDOW = dict(sample_rate_hertz=OFFLINE_RATE, onsource_duration_seconds=1.0,
                      crop_duration_seconds=0.5, offsource_duration_seconds=4.0)
#: A window's footprint: its padded onsource and its offsource (s).
FOOTPRINT_SECONDS = 6.0
#: The first split block (8192 s) inside O3.
BLOCK_SECONDS = 8192.0
FIRST_BLOCK = float(np.ceil(1238170000.0 / BLOCK_SECONDS) * BLOCK_SECONDS)
#: Equal splits, so a few blocks give every split some time.
THIRDS = gf.SplitSpec(fractions={"test": 1 / 3, "validate": 1 / 3, "train": 1 / 3})


def science_layout(lengths, amplitudes, blocks, offset=200.0):
    """
    One science segment a block, ``offset`` s into it: block i's lasts
    ``lengths[i % len]`` seconds at amplitude ``amplitudes[i % len]``. Returns
    (start, end, amplitude) rows.
    """
    return [(FIRST_BLOCK + BLOCK_SECONDS * i + offset,
             FIRST_BLOCK + BLOCK_SECONDS * i + offset + lengths[i % len(lengths)],
             amplitudes[i % len(amplitudes)]) for i in range(blocks)]


class HashedFetch:
    """A fake GWOSC fetch at OFFLINE_RATE: hashed uniform noise, scaled by each sample's segment's amplitude."""

    rates = (OFFLINE_RATE,)
    source_id = "gwosc-fake-hashed"

    def __init__(self, layout):
        self.layout = layout

    def __call__(self, ifo, sample_rate_hertz, interval):
        start, end = interval
        rate = float(sample_rate_hertz)
        first, stop = int(np.ceil(start * rate - 1e-6)), int(np.ceil(end * rate - 1e-6))
        x = np.arange(first, stop, dtype=np.uint64) + np.uint64(sum(map(ord, ifo.name)) << 40)
        with np.errstate(over="ignore"):  # splitmix64
            x = x * np.uint64(0x9E3779B97F4A7C15)
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            x = x ^ (x >> np.uint64(31))
        values = (x >> np.uint64(11)).astype(np.float64) / float(2 ** 53) - 0.5
        gps = np.arange(first, stop, dtype=np.float64) / rate
        for segment_start, segment_end, amplitude in self.layout:
            values[(gps >= segment_start) & (gps < segment_end)] *= amplitude
        return SegmentStrain(values.astype(np.float32), first / rate, rate, self.source_id)


def offline_noise(layout, **kwargs):
    """A noise data obtainer whose science time is ``layout``'s segments, with nothing vetoed."""

    class OfflineNoise(gf.NoiseDataObtainer):
        def get_segment_times(self, start, end, ifo, state_flags):
            rows = [(s, e) for s, e, _ in layout if s >= start and e <= end]
            return np.asarray(rows, dtype=np.float64).reshape(-1, 2)

        def remove_unwanted_segments(self, ifo, valid_segments):
            return valid_segments

    arguments = dict(observing_runs=gf.ObservingRun.O3, data_quality=gf.DataQuality.BEST,
                     data_labels=[gf.DataLabel.NOISE], augmentations=[])
    arguments.update(kwargs)
    return OfflineNoise(**arguments)


#: Thirty blocks of 40 s and 30 s pieces, in three equal splits: every split has some.
GRID_LAYOUT = science_layout([40.0, 30.0], [1.0], blocks=30)
GRID_BATCH = 4


def grid_dataset(group):
    return gf.Dataset(
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.REAL, ifos=[gf.IFO.L1],
                                        ifo_data_obtainer=offline_noise(GRID_LAYOUT, split_spec=THIRDS)),
        **OFFLINE_WINDOW, num_examples_per_batch=GRID_BATCH, group=group, seed=11, ifos=[gf.IFO.L1],
        sampling_mode=gf.SamplingMode.GRID,
        input_variables=[gf.ReturnVariables.ONSOURCE], output_variables=[gf.ReturnVariables.START_GPS_TIME],
    )


def grid_starts_of(ds, indices):
    """START_GPS_TIME of the batches ``indices``, in that order: (n, B, I)."""
    return np.stack([np.asarray(ds[k][1][gf.ReturnVariables.START_GPS_TIME.name]) for k in indices])


class TestGridByIndex:
    """GRID is a finite set addressed by index: floor(N / B) batches, each window once, the same every epoch."""

    @pytest.mark.parametrize("group", ["train", "validate", "test"])
    def test_length_is_the_whole_batches_of_the_set(self, group, caplog):
        with gwosc_serving(HashedFetch(GRID_LAYOUT)), caplog.at_level(logging.INFO):
            ds = grid_dataset(group)
        plan = ds.noise.plan
        assert isinstance(plan, acquisition_noise.GridPlan)
        assert len(plan.starts) > 2 * GRID_BATCH, "the split serves too few GRID windows to test"
        assert len(ds) == len(plan.starts) // GRID_BATCH
        assert plan.remainder == len(plan.starts) % GRID_BATCH
        assert f"remainder of {plan.remainder} example" in caplog.text, "the remainder is not reported"
        with pytest.raises(IndexError):
            ds.streams(len(ds))

    def test_every_window_once(self):
        with gwosc_serving(HashedFetch(GRID_LAYOUT)):
            ds = grid_dataset("train")
            starts = grid_starts_of(ds, range(len(ds)))
        assert starts.shape[:2] == (len(ds), GRID_BATCH)
        assert len(np.unique(starts[..., 0])) == len(ds) * GRID_BATCH, "a GRID window was served twice"

    def test_a_batch_is_a_function_of_its_index(self):
        with gwosc_serving(HashedFetch(GRID_LAYOUT)):
            ds = grid_dataset("test")
            backwards = grid_starts_of(ds, reversed(range(len(ds))))[::-1]
            fresh = grid_starts_of(grid_dataset("test"), range(len(ds)))
        np.testing.assert_array_equal(backwards, fresh)

    @pytest.mark.parametrize("group", ["train", "validate", "test"])
    def test_every_epoch_serves_the_same_set(self, group):
        with gwosc_serving(HashedFetch(GRID_LAYOUT)):
            ds = grid_dataset(group)
            first = grid_starts_of(ds, range(len(ds)))
            ds.on_epoch_end()
            ds.epoch = 3
            again = grid_starts_of(ds, range(len(ds)))
        np.testing.assert_array_equal(first, again)

    def test_the_obtainer_iterator_ends_with_the_set(self):
        with gwosc_serving(HashedFetch(GRID_LAYOUT)):
            length = len(grid_dataset("train"))
            noise = gf.NoiseObtainer(noise_type=gf.NoiseType.REAL, ifos=[gf.IFO.L1],
                                     ifo_data_obtainer=offline_noise(GRID_LAYOUT, split_spec=THIRDS))
            served = list(islice(noise(**OFFLINE_WINDOW, num_examples_per_batch=GRID_BATCH, group="train", seed=11,
                                       sampling_mode=gf.SamplingMode.GRID), 3 * length))
        assert len(served) == length


class TestGridPairs:
    """GRID's pairs: slots one stride apart, each once a pass; detector k's the next free one from its middle, 2S away."""

    ON, OFF = 64, 256  # samples

    def starts(self, counts, firsts, separation, coincident=False):
        return acquisition_noise.grid_examples(np.asarray([firsts], np.int64), np.asarray([counts], np.int64),
                             np.zeros((1, len(counts), 2)), self.ON, self.OFF, separation, coincident)

    def test_each_slot_once_and_separated(self):
        starts = self.starts([6400, 6400], [0, 0], 640)
        assert len(starts) > 0
        for d in range(2):
            assert len(np.unique(starts[:, d])) == len(starts)
            assert np.all((starts[:, d] - self.OFF) % self.ON == 0), "a window off its slot"
        assert np.min(np.abs(starts[:, 0] - starts[:, 1])) >= 640

    @pytest.mark.parametrize("counts, firsts", [
        ([2560, 2560], [0, 0]),          # equal short pieces
        ([65536, 131072], [65536, 0]),   # one piece inside the other
    ])
    def test_adversarial_layouts_stay_apart(self, counts, firsts):
        starts = self.starts(counts, firsts, 768)
        assert len(starts) > 0
        assert np.min(np.abs(starts[:, 0] - starts[:, 1])) >= 768

    def test_an_inseparable_row_serves_nothing(self):
        assert len(self.starts([512, 512], [0, 0], 768)) == 0

    def test_coincident_slots_start_together(self):
        starts = self.starts([6400, 6400], [0, 0], 0, coincident=True)
        assert len(starts) == len(np.unique(starts[:, 0])) > 0
        np.testing.assert_array_equal(starts[:, 0], starts[:, 1])

    def test_a_piece_in_two_rows_serves_each_slot_once(self):
        firsts = np.array([[0, 100000], [0, 300000]], np.int64)
        counts = np.array([[6400, 6400], [6400, 6400]], np.int64)
        rows = np.array([[[0, 25], [390, 415]], [[0, 25], [1171, 1196]]], np.float64)
        starts = acquisition_noise.grid_examples(firsts, counts, rows, self.ON, self.OFF, 640, False)
        assert len(starts) > 0
        assert len(np.unique(starts[:, 0])) == len(starts), "detector 0's piece served a slot twice"

    def test_a_plan_has_floor_n_over_b_batches(self):
        starts = np.arange(30, dtype=np.int64).reshape(15, 2)
        plan = acquisition_noise.GridPlan("test", starts, 4)
        assert (plan.batches, plan.remainder) == (3, 3)
        np.testing.assert_array_equal(plan.examples(2), starts[8:12])
        with pytest.raises(IndexError):
            plan.examples(3)


SHORT, LONG = 200.0, 600.0
QUIET, LOUD = 1.0, 100.0
#: Short quiet pieces and long loud ones, three times as long, one a block.
PSEUDO_LAYOUT = science_layout([SHORT, LONG], [QUIET, LOUD], blocks=24)


def pseudo_real(layout, **kwargs):
    return gf.NoiseObtainer(noise_type=gf.NoiseType.PSEUDO_REAL, ifos=[gf.IFO.L1],
                            ifo_data_obtainer=offline_noise(layout, **kwargs))


def loud_share(batches):
    """The share of examples coloured by a loud segment: their level is far above the quietest's."""
    levels = np.concatenate([np.std(np.asarray(b[RV.ONSOURCE]), axis=-1).ravel() for b in batches])
    return float(np.mean(levels > np.sqrt(LOUD / QUIET) * np.min(levels)))


class TestPseudoRealByIndex:
    """PSEUDO_REAL colours each example by the PSD of a row of the RANDOM plan: endless, each row by its length."""

    BATCH = 64

    def batches(self, noise, n, group="train"):
        return list(islice(noise(**OFFLINE_WINDOW, num_examples_per_batch=self.BATCH, group=group, seed=5), n))

    def test_rows_serve_in_proportion_to_their_length(self):
        with gwosc_serving(HashedFetch(PSEUDO_LAYOUT)):
            served = self.batches(pseudo_real(PSEUDO_LAYOUT, saturation=0.25), 120)
        expected = (LONG - FOOTPRINT_SECONDS) / (LONG + SHORT - 2 * FOOTPRINT_SECONDS)
        assert abs(loud_share(served) - expected) < 0.08, \
            f"loud share {loud_share(served):.3f}, where the rows' lengths give {expected:.3f}"

    def test_serves_past_one_pass(self):
        with gwosc_serving(HashedFetch(PSEUDO_LAYOUT)):
            served = self.batches(pseudo_real(PSEUDO_LAYOUT, saturation=0.25), 400)
        assert len(served) == 400, f"the reader ended after {len(served)} batches"

    def test_a_batch_is_a_function_of_its_address(self):
        kwargs = dict(**OFFLINE_WINDOW, num_examples_per_batch=8, steps_per_epoch=10)
        tree = gf.SeedTree(5)
        with gwosc_serving(HashedFetch(PSEUDO_LAYOUT)):
            noise = pseudo_real(PSEUDO_LAYOUT, saturation=0.25)
            batch = noise.batcher(tree, "train", **kwargs)
            late = np.asarray(batch(tree.batch("train", 3, 7))[RV.ONSOURCE])
            early = np.asarray(batch(tree.batch("train", 0, 0))[RV.ONSOURCE])
            again = noise.batcher(tree, "train", **kwargs)
            np.testing.assert_array_equal(np.asarray(again(tree.batch("train", 0, 0))[RV.ONSOURCE]), early)
            np.testing.assert_array_equal(np.asarray(again(tree.batch("train", 3, 7))[RV.ONSOURCE]), late)
        assert not np.array_equal(early, late)

    def test_a_short_piece_is_left_out_not_read(self):
        # Every block also holds a 1 s science segment, whose interior is shorter than a window.
        layout = science_layout([1.0], [QUIET], blocks=24, offset=150.0) + science_layout([LONG], [QUIET], blocks=24)
        with gwosc_serving(HashedFetch(layout)):
            served = self.batches(pseudo_real(layout, segment_order=gf.SegmentOrder.SHORTEST_FIRST), 3)
        assert len(served) == 3
        assert all(np.all(np.isfinite(np.asarray(b[RV.ONSOURCE]))) for b in served)

    def test_a_row_without_power_is_refused(self):
        layout = science_layout([LONG], [0.0, QUIET, QUIET], blocks=24)
        with gwosc_serving(HashedFetch(layout)):
            with pytest.raises(ValueError, match="no power"):
                self.batches(pseudo_real(layout, saturation=0.25), 200)

    def test_precomputed_segments_are_served(self):
        with gwosc_serving(HashedFetch(PSEUDO_LAYOUT)):
            obtainer = offline_noise(PSEUDO_LAYOUT, saturation=0.25)
            obtainer.get_valid_segments([gf.IFO.L1], 5, "train")
            noise = gf.NoiseObtainer(noise_type=gf.NoiseType.PSEUDO_REAL, ifos=[gf.IFO.L1],
                                     ifo_data_obtainer=obtainer)
            served = self.batches(noise, 2)
        assert len(served) == 2

    def test_both_are_index_addressed(self):
        with gwosc_serving(HashedFetch(PSEUDO_LAYOUT)):
            pseudo = gf.Dataset(noise_obtainer=pseudo_real(PSEUDO_LAYOUT, saturation=0.25), **OFFLINE_WINDOW,
                                num_examples_per_batch=4, group="train", seed=5, steps_per_epoch=3,
                                input_variables=[gf.ReturnVariables.ONSOURCE],
                                output_variables=[gf.ReturnVariables.INJECTION_MASKS])
        with gwosc_serving(HashedFetch(GRID_LAYOUT)):
            grid = grid_dataset("test")
        assert pseudo.seed_metadata["index_addressed"] and grid.seed_metadata["index_addressed"]
