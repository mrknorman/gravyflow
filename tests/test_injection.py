# Built-In imports:
import logging
from pathlib import Path
from itertools import islice
import os

# Library imports:
import numpy as np
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from tqdm import tqdm

# Local imports:
from gravyflow import gf

def test_iteration(
    num_tests : int = int(1.0E2)
    ):
    
     # Test Parameters:
    num_examples_per_generation_batch : int = 128
    num_examples_per_batch : int = num_tests
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 1.0
    crop_duration_seconds : float = 0.5
    
    # Define injection directory path:
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    injection_directory_path : Path = \
        Path(current_dir / "example_injection_parameters")
    
    phenom_d_generator : cuPhenomDGenerator = \
        gf.WaveformGenerator.load(
            injection_directory_path / "phenom_d_parameters.json", 
            sample_rate_hertz, 
            onsource_duration_seconds
        )
    
    generator : gf.InjectionGenerator = \
        gf.InjectionGenerator(
            phenom_d_generator,
            sample_rate_hertz,
            onsource_duration_seconds,
            crop_duration_seconds,
            num_examples_per_generation_batch,
            num_examples_per_batch,
            variables_to_return = \
                [gf.WaveformParameters.MASS_1_MSUN, gf.WaveformParameters.MASS_2_MSUN]
        )
    
    logging.info("Start iteration tests...")
    for index, _ in tqdm(enumerate(islice(generator.generate(), num_tests))):
        pass
    
    assert index == num_tests - 1, \
        "Warning! Injection generator does not iterate the required number of batches"
    
    logging.info("Compete!")
    
def test_phenom_d_injection(
    num_tests : int = 10,
    output_diretory_path : Path = Path("./gravyflow_data/tests/")
    ):
    
    # Test Parameters:
    num_examples_per_generation_batch : int = 128
    num_examples_per_batch : int = num_tests
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 1.0
    crop_duration_seconds : float = 0.5
    scale_factor : float = 1.0E21
        
    # Define injection directory path:
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    injection_directory_path : Path = \
        Path(current_dir / "example_injection_parameters")
    
    phenom_d_generator_high_mass : cuPhenomDGenerator = \
        gf.WaveformGenerator.load(
            injection_directory_path / "phenom_d_parameters_high_mass.json", 
            sample_rate_hertz, 
            onsource_duration_seconds
        )
    
    phenom_d_generator_low_mass : cuPhenomDGenerator = \
        gf.WaveformGenerator.load(
            injection_directory_path / "phenom_d_parameters_low_mass.json", 
            sample_rate_hertz, 
            onsource_duration_seconds
        )
    
    injection_generator : gf.InjectionGenerator = \
        gf.InjectionGenerator(
            [phenom_d_generator_high_mass, phenom_d_generator_low_mass],
            sample_rate_hertz,
            onsource_duration_seconds,
            crop_duration_seconds,
            num_examples_per_generation_batch,
            num_examples_per_batch,
            variables_to_return = \
                [gf.WaveformParameters.MASS_1_MSUN, gf.WaveformParameters.MASS_2_MSUN]
        )
        
    generator : Iterator = injection_generator.generate
    
    injections, _, parameters = next(generator())
        
    high_mass = [
        gf.generate_strain_plot(
            {"Plus": injection[0], "Cross": injection[1]},
            sample_rate_hertz,
            title=f"cuPhenomD injection example: mass_1 {m1} msun; mass_2 {m2} msun",
            scale_factor=scale_factor
        )
        for injection, m1, m2 in zip(
            injections[0], 
            parameters[gf.WaveformParameters.MASS_1_MSUN][0], 
            parameters[gf.WaveformParameters.MASS_2_MSUN][0]
        )
    ]

    low_mass = [
        gf.generate_strain_plot(
            {"Plus": injection[0], "Cross": injection[1]},
            sample_rate_hertz,
            title=f"cuPhenomD injection example: mass_1 {m1} msun; mass_2 {m2} msun",
            scale_factor=scale_factor
        )
        for injection, m1, m2 in zip(
            injections[1], 
            parameters[gf.WaveformParameters.MASS_1_MSUN][1], 
            parameters[gf.WaveformParameters.MASS_2_MSUN][1]
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
    
def test_wnb_injection(
    num_tests : int = 10,
    output_diretory_path : Path = Path("./gravyflow_data/tests/")
    ):
    
    # Test Parameters:
    num_examples_per_generation_batch : int = 128
    num_examples_per_batch : int = num_tests
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 1.0
    crop_duration_seconds : float = 0.5
    scale_factor : float = 1.0E21
    
    # Define injection directory path:
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    injection_directory_path : Path = \
        Path(current_dir / "example_injection_parameters")
    
    wnb_generator : gf.WNBGenerator = \
        gf.WaveformGenerator.load(
            injection_directory_path / "wnb_parameters.json", 
            sample_rate_hertz, 
            onsource_duration_seconds
        )
    
    wnb_generator.injection_chance = 1.0
    
    injection_generator : gf.InjectionGenerator = \
        gf.InjectionGenerator(
            [wnb_generator],
            sample_rate_hertz,
            onsource_duration_seconds,
            crop_duration_seconds,
            num_examples_per_generation_batch,
            num_examples_per_batch,
            variables_to_return = \
                [
                    gf.WaveformParameters.DURATION_SECONDS,
                    gf.WaveformParameters.MIN_FREQUENCY_HERTZ, 
                    gf.WaveformParameters.MAX_FREQUENCY_HERTZ
                ]
        )
        
    generator : Iterator = injection_generator.generate
    
    injections, _, parameters = next(generator())

    layout = [
        [gf.generate_strain_plot(
            {"Plus": injection[0], "Cross": injection[1]},
            sample_rate_hertz,
            title=f"WNB injection example: min frequency {min_frequency_hertz} "
            f"hertz; min frequency {max_frequency_hertz} hertz; duration "
            f"{duration} seconds.",
            scale_factor=scale_factor
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

if __name__ == "__main__":
    
    # ---- User parameters ---- #
    
    # GPU setup:
    min_gpu_memory_mb : int = 6000
    num_gpus_to_request : int = 1
    memory_to_allocate_tf : int = 4000
    
    # Setup CUDA
    gpus = gf.find_available_GPUs(min_gpu_memory_mb, num_gpus_to_request)
    strategy = gf.setup_cuda(
        gpus, 
        max_memory_limit = memory_to_allocate_tf, 
        logging_level=logging.WARNING
    )    
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    # Test injection generation:
    with strategy.scope():
        test_phenom_d_injection()
        test_wnb_injection()
        test_iteration()
