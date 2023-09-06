# Built-In imports:
import logging
from pathlib import Path

# Library imports:
import numpy as np
from bokeh.io import output_file, save
from bokeh.layouts import gridplot

# Local imports:
from ..maths import Distribution, DistributionType
from ..setup import find_available_GPUs, setup_cuda, ensure_directory_exists
from ..injection import (cuPhenomDGenerator, InjectionGenerator, 
                         WaveformParameters, WaveformGenerator)
from ..plotting import generate_strain_plot

def test_phenom_d_injection(
    num_tests : int = 10,
    output_diretory_path : Path = Path("./py_ml_data/tests/")
    ):
    
    # Test Parameters:
    num_examples_per_generation_batch : int = 2048
    num_examples_per_batch : int = num_tests
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 1.0
    crop_duration_seconds : float = 0.5
    scale_factor : float = 1.0E21
    
    phenom_d_generator_high_mass : cuPhenomDGenerator = \
        WaveformGenerator.load(
            Path("./py_ml_tools/tests/injection_parameters_high_mass.json"), 
            sample_rate_hertz, 
            onsource_duration_seconds
        )
    
    phenom_d_generator_low_mass : cuPhenomDGenerator = \
        WaveformGenerator.load(
            Path("./py_ml_tools/tests/injection_parameters_low_mass.json"), 
            sample_rate_hertz, 
            onsource_duration_seconds
        )
    
    injection_generator : InjectionGenerator = \
        InjectionGenerator(
            [phenom_d_generator_high_mass, phenom_d_generator_low_mass],
            sample_rate_hertz,
            onsource_duration_seconds,
            crop_duration_seconds,
            num_examples_per_generation_batch,
            num_examples_per_batch,
            scale_factor,
            variables_to_return = \
                [WaveformParameters.MASS_1_MSUN, WaveformParameters.MASS_2_MSUN]
        )
    
    total_onsource_duration_seconds : float = \
        onsource_duration_seconds + (crop_duration_seconds * 2.0)
        
    generator : Iterator = injection_generator.generate
    
    injections, mask, parameters = next(generator())
        
    high_mass = [
        generate_strain_plot(
            {"Injection Test": injection},
            sample_rate_hertz,
            total_onsource_duration_seconds,
            title=f"cuPhenomD injection example: mass_1 {m1} msun; mass_2 {m2} msun",
            scale_factor=scale_factor
        )
        for injection, m1, m2 in zip(
            injections[0], 
            parameters[WaveformParameters.MASS_1_MSUN][0], 
            parameters[WaveformParameters.MASS_2_MSUN][0]
        )
    ]

    low_mass = [
        generate_strain_plot(
            {"Injection Test": injection},
            sample_rate_hertz,
            total_onsource_duration_seconds,
            title=f"cuPhenomD injection example: mass_1 {m1} msun; mass_2 {m2} msun",
            scale_factor=scale_factor
        )
        for injection, m1, m2 in zip(
            injections[1], 
            parameters[WaveformParameters.MASS_1_MSUN][1], 
            parameters[WaveformParameters.MASS_2_MSUN][1]
        )
    ]
        
    layout = [list(item) for item in zip(low_mass, high_mass)]
    
    # Ensure output directory exists
    ensure_directory_exists(output_diretory_path)
    
    # Define an output path for the dashboard
    output_file(output_diretory_path / "injection_plots.html")

    # Arrange the plots in a grid. 
    grid = gridplot(layout)
        
    save(grid)

if __name__ == "__main__":
    
    # ---- User parameters ---- #
    
    # GPU setup:
    min_gpu_memory_mb : int = 4000
    num_gpus_to_request : int = 1
    memory_to_allocate_tf : int = 2000
    
    # Setup CUDA
    gpus = find_available_GPUs(min_gpu_memory_mb, num_gpus_to_request)
    strategy = setup_cuda(
        gpus, 
        max_memory_limit = memory_to_allocate_tf, 
        logging_level=logging.WARNING
    )    
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    # Test IFO noise generator:
    test_phenom_d_injection()