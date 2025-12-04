# Built-In imports:
import logging
from pathlib import Path
from itertools import islice
from typing import Dict
import os

# Library imports:
import numpy as np
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from tqdm import tqdm
import pytest
from _pytest.config import Config

# Local imports:
# Local imports:
import gravyflow as gf
from gravyflow.src.dataset.features import injection as gf_inj
import keras
from keras import ops

def _test_injection_iteration(
        num_tests : int = int(1.0E2)
    ) -> None:

    with gf.env():

        # Define injection directory path:
        injection_directory_path : Path = Path(
            gf.tests.PATH / "example_injection_parameters"
        )
        
        phenom_d_generator : gf.RippleGenerator = gf.WaveformGenerator.load(
            injection_directory_path / "phenom_d_parameters.json"
        )
        
        injection_generator : gf.InjectionGenerator = gf.InjectionGenerator(
            waveform_generators=phenom_d_generator,
            parameters_to_return=[
                gf.WaveformParameters.MASS_1_MSUN, 
                gf.WaveformParameters.MASS_2_MSUN
            ]
        )
        
        logging.info("Start iteration tests...")
        for index, _ in tqdm(enumerate(islice(injection_generator(), num_tests))):
            print(index)
            pass
        
        assert index == num_tests - 1, "Warning! Injection generator does not iterate the required number of batches"
        
        logging.info("Compete!")
    
def _test_phenom_d_injection(
        num_tests : int = 10,
        plot_results : bool = False
    ) -> None:
    
    output_diretory_path : Path = gf.PATH.parent.parent / "gravyflow_data/tests/"

    with gf.env():

        # Define injection directory path:
        injection_directory_path : Path = Path(
            gf.tests.PATH / "example_injection_parameters"
        )
        
        phenom_d_generator_high_mass : gf.RippleGenerator = gf.WaveformGenerator.load(
            injection_directory_path / "phenom_d_parameters_high_mass.json"
        )
        
        phenom_d_generator_low_mass : gf.RippleGenerator = gf.WaveformGenerator.load(
            injection_directory_path / "phenom_d_parameters_low_mass.json"
        )
        
        injection_generator : gf.InjectionGenerator = gf.InjectionGenerator(
                waveform_generators=[
                    phenom_d_generator_high_mass, 
                    phenom_d_generator_low_mass
                ],
                parameters_to_return = [
                    gf.WaveformParameters.MASS_1_MSUN, 
                    gf.WaveformParameters.MASS_2_MSUN
                ]
            )
                
        injections, _, parameters = next(injection_generator(num_examples_per_batch=num_tests))

        current_parameters = {
            'high_mass_injections': injections[0],
            'mass_1_msun_high_mass' : parameters[
                gf.WaveformParameters.MASS_1_MSUN
            ][0],
            'mass_2_msun_high_mass' : parameters[
                gf.WaveformParameters.MASS_2_MSUN
            ][0],
            'low_mass_injections': injections[1],
            'mass_1_msun_low_mass' : parameters[
                gf.WaveformParameters.MASS_1_MSUN
            ][1],
            'mass_2_msun_low_mass' : parameters[
                gf.WaveformParameters.MASS_2_MSUN
            ][1]
        }
        parameters_file_path : Path = (
            gf.PATH / f"res/tests/phenom_d_consistancy.hdf5"
        )
        gf.tests.compare_and_save_parameters(
            current_parameters, parameters_file_path
        )

        if plot_results:
            
            high_mass = [
                gf.generate_strain_plot(
                    {"Plus": injection[0], "Cross": injection[1]},
                    title=f"PhenomD injection example: mass_1 {m1} msun; mass_2 {m2} msun"
                )
                for injection, m1, m2 in zip(
                    current_parameters["high_mass_injections"], 
                    current_parameters["mass_1_msun_high_mass"], 
                    current_parameters["mass_2_msun_high_mass"]
                )
            ]

            low_mass = [
                gf.generate_strain_plot(
                    {"Plus": injection[0], "Cross": injection[1]},
                    title=f"PhenomD injection example: mass_1 {m1} msun; mass_2 {m2} msun"
                )
                for injection, m1, m2 in zip(
                    current_parameters["high_mass_injections"], 
                    current_parameters["mass_1_msun_low_mass"], 
                    current_parameters["mass_2_msun_low_mass"]
                )
            ]
                
            layout = [list(item) for item in zip(low_mass, high_mass)]
            
            # Ensure output directory exists
            gf.ensure_directory_exists(output_diretory_path)
            
            # Define an output path for the dashboard
            output_file(output_diretory_path / "injection_plots.html")

            # Arrange the plots in a grid. 
            grid = gridplot(layout)
                
            save(grid)

def _test_wnb_injection(
        num_tests : int = 10,
        plot_results : bool = False
    ) -> None:

    output_diretory_path : Path = gf.PATH.parent.parent / "gravyflow_data/tests/"
    
    with gf.env():
        # Define injection directory path:
        injection_directory_path : Path = (
            gf.tests.PATH / "example_injection_parameters"
        )
        
        wnb_generator : gf.WNBGenerator = gf.WaveformGenerator.load(
            injection_directory_path / "wnb_parameters.json"
        )
        
        wnb_generator.injection_chance = 1.0
        
        injection_generator : gf.InjectionGenerator = gf.InjectionGenerator(
            waveform_generators=wnb_generator,
            parameters_to_return=[
                gf.WaveformParameters.DURATION_SECONDS,
                gf.WaveformParameters.MIN_FREQUENCY_HERTZ, 
                gf.WaveformParameters.MAX_FREQUENCY_HERTZ
            ]
        )
                
        injections, _, parameters = next(
            injection_generator(num_examples_per_batch=num_tests)
        )
        current_parameters = {
            'injections': injections[0],
            'duration_seconds' : parameters[
                gf.WaveformParameters.DURATION_SECONDS
            ][0],
            'min_frequency_hertz' : parameters[
                gf.WaveformParameters.MIN_FREQUENCY_HERTZ
            ][0],
            'max_frequency_hertz' : parameters[
                gf.WaveformParameters.MAX_FREQUENCY_HERTZ
            ][0]
        }
        parameters_file_path : Path = (
            gf.PATH / f"res/tests/wnb_consistancy.hdf5"
        )
        gf.tests.compare_and_save_parameters(
            current_parameters, parameters_file_path
        )

        if plot_results:

            layout = [
                [gf.generate_strain_plot(
                    {"Plus": injection[0], "Cross": injection[1]},
                    title=f"WNB injection example: min frequency {min_frequency_hertz} "
                    f"hertz; min frequency {max_frequency_hertz} hertz; duration "
                    f"{duration} seconds."
                )]
                for injection, duration, min_frequency_hertz, max_frequency_hertz in zip(
                    injections[0], 
                    parameters[gf.WaveformParameters.DURATION_SECONDS][0],
                    parameters[gf.WaveformParameters.MIN_FREQUENCY_HERTZ][0], 
                    parameters[gf.WaveformParameters.MAX_FREQUENCY_HERTZ][0],
                )
            ]
                    
            # Ensure output directory exists
            gf.ensure_directory_exists(output_diretory_path)
            
            # Define an output path for the dashboard
            output_file(output_diretory_path / "wnb_plots.html")

            # Arrange the plots in a grid. 
            grid = gridplot(layout)
                
            save(grid)

@pytest.mark.slow
def test_phenom_d_injection(
        pytestconfig : Config
    ) -> None:

    _test_phenom_d_injection(
        plot_results=pytestconfig.getoption("plot")
    )

@pytest.mark.slow
def test_wnb_injection(
        pytestconfig : Config
    ) -> None:

    _test_wnb_injection(
        plot_results=pytestconfig.getoption("plot")
    )

def test_injection_iteration(
        pytestconfig : Config
    ) -> None:
    
    _test_injection_iteration(
        num_tests=gf.tests.num_tests_from_config(pytestconfig)
    )



def test_calculate_hrss():
    # Batch=2, Channels=1, Time=100
    # Signal 1: Amplitude 1.0
    # Signal 2: Amplitude 2.0
    
    t = np.ones((100,), dtype=np.float32)
    s1 = t
    s2 = 2.0 * t
    
    injection = np.stack([s1, s2], axis=0) # (2, 100)
    injection = injection[:, np.newaxis, :] # (2, 1, 100)
    injection = ops.convert_to_tensor(injection)
    
    hrss = gf_inj.calculate_hrss(injection)
    
    # Expected: sqrt(sum(x^2))
    # s1: sqrt(100 * 1^2) = 10.0
    # s2: sqrt(100 * 2^2) = 20.0
    
    assert ops.shape(hrss) == (2,)
    np.testing.assert_allclose(hrss, [10.0, 20.0], atol=1e-5)

def test_scale_to_hrss():
    t = np.ones((100,), dtype=np.float32)
    injection = ops.convert_to_tensor(t.reshape(1, 1, 100))
    
    target_hrss = 5.0
    scaled = gf_inj.scale_to_hrss(injection, target_hrss)
    
    new_hrss = gf_inj.calculate_hrss(scaled)
    np.testing.assert_allclose(new_hrss, [target_hrss], atol=1e-5)

def test_ensure_last_dim_even():
    # Odd length
    x = ops.ones((1, 5))
    y = gf_inj.ensure_last_dim_even(x)
    assert ops.shape(y) == (1, 4)
    
    # Even length
    x = ops.ones((1, 4))
    y = gf_inj.ensure_last_dim_even(x)
    assert ops.shape(y) == (1, 4)

def test_wnb_generator():
    # Test WNBGenerator class
    gen = gf_inj.WNBGenerator(
        duration_seconds=gf.Distribution(value=0.5, type_=gf.DistributionType.CONSTANT),
        min_frequency_hertz=gf.Distribution(value=10.0, type_=gf.DistributionType.CONSTANT),
        max_frequency_hertz=gf.Distribution(value=50.0, type_=gf.DistributionType.CONSTANT)
    )
    
    waveforms, params = gen.generate(
        num_waveforms=2,
        sample_rate_hertz=100.0,
        duration_seconds=1.0,
        seed=42
    )
    
    assert ops.shape(waveforms) == (2, 2, 100)
    assert gf_inj.WaveformParameters.DURATION_SECONDS in params

def test_ripple_generator():
    # Test RippleGenerator class
    # Needs valid approximant and parameters
    gen = gf_inj.RippleGenerator(
        mass_1_msun=gf.Distribution(value=30.0, type_=gf.DistributionType.CONSTANT),
        mass_2_msun=gf.Distribution(value=30.0, type_=gf.DistributionType.CONSTANT),
        distance_mpc=gf.Distribution(value=100.0, type_=gf.DistributionType.CONSTANT)
    )
    
    waveforms, params = gen.generate(
        num_waveforms=2,
        sample_rate_hertz=1024.0,
        duration_seconds=1.0,
        seed=0
    )
    
    # Ripple returns (Batch, 2, Time)
    # Time = 1.0 * 1024 = 1024
    assert ops.shape(waveforms) == (2, 2, 1024)
    assert gf_inj.WaveformParameters.MASS_1_MSUN in params
