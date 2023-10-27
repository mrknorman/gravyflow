# Built-In imports:
from pathlib import Path
import logging
from itertools import islice

# Library imports:
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from tqdm import tqdm

# Local imports:
import gravyflow as gw

def test_iteration(
    num_tests : int = int(1.0E2)
    ):
    
    # Test parameters:
    sample_rate_hertz : float = 1024.0
    onsource_duration_seconds : float = 1.0    
    crop_duration_seconds : float = 0.5
    offsource_duration_seconds : float = 4.0
    num_examples_per_batch : int = 32
    scale_factor : float = 1.0E20
    
    # Setup ifo data acquisition object:
    ifo_data_obtainer : gw.IFODataObtainer = \
        gw.IFODataObtainer(
            gw.ObservingRun.O3, 
            gw.DataQuality.BEST, 
            [
                gw.DataLabel.NOISE, 
                gw.DataLabel.GLITCHES
            ],
            gw.SegmentOrder.RANDOM,
            force_acquisition=True,
            cache_segments=False,
            logging_level=logging.INFO
        )
    
    # Initilise noise generator wrapper:
    noise : gw.NoiseObtainer = \
        gw.NoiseObtainer(
            ifo_data_obtainer = ifo_data_obtainer,
            noise_type = gw.NoiseType.REAL,
            ifos = gw.IFO.L1
        )
    
    # Create generator:
    generator : Iterator = \
        noise.init_generator(
            sample_rate_hertz,
            onsource_duration_seconds,
            crop_duration_seconds,
            offsource_duration_seconds,
            num_examples_per_batch,
            scale_factor
        )
    
    logging.info("Start iteration tests...")
    for index, _ in tqdm(enumerate(islice(generator, num_tests))):
        pass
    
    assert index == num_tests - 1, \
        "Warning! Noise generator does not iterate the required number of batches"
    
    logging.info("Complete")

def test_real_noise(
        num_tests : int = 8, 
        output_diretory_path : Path = Path("./py_ml_data/tests/")
    ):
    
    # Test parameters:
    sample_rate_hertz : float = 1024.0
    onsource_duration_seconds : float = 1.0    
    crop_duration_seconds : float = 0.5
    offsource_duration_seconds : float = 4.0
    num_examples_per_batch : int = num_tests
    scale_factor : float = 1.0E20
    
    # Setup ifo data acquisition object:
    ifo_data_obtainer : gw.IFODataObtainer = \
        gw.IFODataObtainer(
            gw.ObservingRun.O3, 
            gw.DataQuality.BEST, 
            [
                gw.DataLabel.NOISE, 
                gw.DataLabel.GLITCHES
            ],
            gw.SegmentOrder.RANDOM,
            force_acquisition = True,
            cache_segments = False
        )
    
    # Initilise noise generator wrapper:
    noise : gw.NoiseObtainer = \
        gw.NoiseObtainer(
            ifo_data_obtainer = ifo_data_obtainer,
            noise_type = gw.NoiseType.REAL,
            ifos = gw.IFO.L1
        )
    
    # Create generator:
    generator : Iterator = \
        noise.init_generator(
            sample_rate_hertz,
            onsource_duration_seconds,
            crop_duration_seconds,
            offsource_duration_seconds,
            num_examples_per_batch,
            scale_factor
        )
    
    # Calculate total onsource length including padding
    total_onsource_duration_seconds : float = \
        onsource_duration_seconds + (crop_duration_seconds * 2.0)
        
    # Iterate through num_tests batches to check correct operation:
    onsource, offsource, gps_times  = next(generator)
        
    layout = []
    for onsource_, offsource_, gps_time in zip(onsource, offsource, gps_times):
        
        onsource_strain_plot = \
            gw.generate_strain_plot(
                {"Onsource Noise" : onsource_},
                sample_rate_hertz,
                total_onsource_duration_seconds,
                title = f"Onsource Background noise at {gps_time}",
                scale_factor = scale_factor
            )
        
        offsource_strain_plot = \
            gw.generate_strain_plot(
                {"Offsource Noise" : offsource_},
                sample_rate_hertz,
                offsource_duration_seconds,
                title = f"Offsource Background noise at {gps_time}",
                scale_factor = scale_factor
            )
        
        layout.append([onsource_strain_plot, offsource_strain_plot])
    
    # Ensure output directory exists
    gw.ensure_directory_exists(output_diretory_path)
    
    # Define an output path for the dashboard
    output_file(output_diretory_path / "noise_plots.html")

    # Arrange the plots in a grid. 
    grid = gridplot(layout)
        
    save(grid)
    
def test_multi_noise(
        num_tests : int = 8, 
        output_diretory_path : Path = Path("./py_ml_data/tests/")
    ):
    
    # Test parameters:
    sample_rate_hertz : float = 1024.0
    onsource_duration_seconds : float = 1.0    
    crop_duration_seconds : float = 0.5
    offsource_duration_seconds : float = 4.0
    num_examples_per_batch : int = num_tests
    scale_factor : float = 1.0E20
    
    # Setup ifo data acquisition object:
    ifo_data_obtainer : gw.IFODataObtainer = \
        gw.IFODataObtainer(
            gw.ObservingRun.O3, 
            gw.DataQuality.BEST, 
            [
                gw.DataLabel.NOISE, 
                gw.DataLabel.GLITCHES
            ],
            gw.SegmentOrder.RANDOM,
            force_acquisition = True,
            cache_segments = False
        )
    
    # Initilise noise generator wrapper:
    noise : gw.NoiseObtainer = \
        gw.NoiseObtainer(
            ifo_data_obtainer = ifo_data_obtainer,
            noise_type = gw.NoiseType.REAL,
            ifos = [gw.IFO.L1, gw.IFO.H1]
        )
    
    # Create generator:
    generator : Iterator = \
        noise.init_generator(
            sample_rate_hertz,
            onsource_duration_seconds,
            crop_duration_seconds,
            offsource_duration_seconds,
            num_examples_per_batch,
            scale_factor
        )
    
    # Calculate total onsource length including padding
    total_onsource_duration_seconds : float = \
        onsource_duration_seconds + (crop_duration_seconds * 2.0)
        
    # Iterate through num_tests batches to check correct operation:
    onsource, offsource, gps_times  = next(generator)
        
    layout = []
    for onsource_, offsource_, gps_time in zip(onsource, offsource, gps_times):
        
        list_of_onsource = []
        for onsource_ifo in onsource_: 
            list_of_onsource.append(
                gw.generate_strain_plot(
                    {"Onsource Noise" : onsource_ifo},
                    sample_rate_hertz,
                    total_onsource_duration_seconds,
                    title = f"Onsource Background noise at {gps_time}",
                    scale_factor = scale_factor
                )
            )
        
        layout.append(list_of_onsource)
    
    # Ensure output directory exists
    gw.ensure_directory_exists(output_diretory_path)
    
    # Define an output path for the dashboard
    output_file(output_diretory_path / "multi_noise_plots.html")

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
    gpus = gw.find_available_GPUs(min_gpu_memory_mb, num_gpus_to_request)
    strategy = gw.setup_cuda(
        gpus, 
        max_memory_limit = memory_to_allocate_tf, 
        logging_level=logging.WARNING
    )    
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    # Test gw.IFO noise generator:
    with strategy.scope():
        test_multi_noise()
        test_real_noise()
        test_iteration()
    
    