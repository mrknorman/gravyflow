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
from ..acquisition import (IFODataObtainer, SegmentOrder, ObservingRun, 
                          DataQuality, DataLabel, IFO)
from ..noise import NoiseObtainer, NoiseType
from ..plotting import generate_strain_plot, generate_spectrogram
from ..dataset import get_ifo_data, ReturnVariables

def test_generator(
    num_tests : int = 32,
    output_diretory_path : Path = Path("./py_ml_data/tests/")
    ):
    
    # Test Parameters:
    num_examples_per_generation_batch : int = 2048
    num_examples_per_batch : int = num_tests
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 1.0
    offsource_duration_seconds : float = 16.0
    crop_duration_seconds : float = 0.5
    scale_factor : float = 1.0E21
    
    # Load injection config:
    phenom_d_generator_high_mass : cuPhenomDGenerator = \
        WaveformGenerator.load(
            Path("./py_ml_tools/tests/injection_parameters.json"), 
            sample_rate_hertz, 
            onsource_duration_seconds,
            snr=Distribution(min_=8.0,max_=15.0,type_=DistributionType.UNIFORM)
        )
    
    # Setup ifo data acquisition object:
    ifo_data_obtainer : IFODataObtainer = \
        IFODataObtainer(
            ObservingRun.O3, 
            DataQuality.BEST, 
            [
                DataLabel.NOISE, 
                DataLabel.GLITCHES
            ],
            IFO.L1,
            SegmentOrder.RANDOM,
            force_acquisition = True,
            cache_segments = False
        )
    
    # Initilise noise generator wrapper:
    noise_obtainer: NoiseObtainer = \
        NoiseObtainer(
            ifo_data_obtainer = ifo_data_obtainer,
            noise_type = NoiseType.REAL
        )
    
    generator = get_ifo_data(    
        # Random Seed:
        seed= 1000,
        # Temporal components:
        sample_rate_hertz=sample_rate_hertz,   
        onsource_duration_seconds=onsource_duration_seconds,
        offsource_duration_seconds=offsource_duration_seconds,
        crop_duration_seconds=crop_duration_seconds,
        # Noise: 
        noise_obtainer=noise_obtainer,
        # Injections:
        injection_generators=phenom_d_generator_high_mass, 
        # Output configuration:
        num_examples_per_batch=num_examples_per_batch,
        input_variables = [
            ReturnVariables.WHITENED_ONSOURCE, 
            ReturnVariables.INJECTION_MASKS, 
            ReturnVariables.INJECTIONS,
            ReturnVariables.WHITENED_INJECTIONS,
            WaveformParameters.MASS_1_MSUN, 
            WaveformParameters.MASS_2_MSUN
        ],
    )
    
    input_dict, output_dict = next(generator)
        
    onsource = input_dict[ReturnVariables.WHITENED_ONSOURCE].numpy()
    injections = input_dict[ReturnVariables.INJECTIONS].numpy()
    whitened_injections = input_dict[ReturnVariables.WHITENED_INJECTIONS].numpy()
    masks = input_dict[ReturnVariables.INJECTION_MASKS].numpy()
    mass_1_msun = input_dict[WaveformParameters.MASS_1_MSUN].numpy()
    mass_2_msun = input_dict[WaveformParameters.MASS_2_MSUN].numpy()
    
    plots = [
        [generate_strain_plot(
            {
                "Whitened Onsouce + Injection": onsource_,
                "Whitened Injection" : whitened_injection,
                "Injection": injection
            },
            sample_rate_hertz,
            onsource_duration_seconds,
            title=f"cuPhenomD injection example: mass_1 {m1} msun; mass_2 {m2} msun",
            scale_factor=scale_factor
        ), 
        generate_spectrogram(
            onsource_, 
            sample_rate_hertz,
            onsource_duration_seconds
        )]
        for onsource_, whitened_injection, injection, m1, m2 in zip(
            onsource,
            whitened_injections[0],
            injections[0], 
            mass_1_msun[0], 
            mass_2_msun[0]
        )
    ]
    
    layout = [item for item in plots]
    
    # Ensure output directory exists
    ensure_directory_exists(output_diretory_path)
    
    # Define an output path for the dashboard
    output_file(output_diretory_path / "dataset_plots.html")

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
    stratergy = setup_cuda(
        gpus, 
        max_memory_limit = memory_to_allocate_tf, 
        logging_level=logging.WARNING
    )    
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    # Test IFO noise generator:
    test_generator()
    