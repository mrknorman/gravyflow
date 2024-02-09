# Built-In imports:
from pathlib import Path
import logging
from itertools import islice
from typing import Dict
import os

# Library imports:
import numpy as np
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from tqdm import tqdm

# Local imports:
import gravyflow as gf

def _test_real_noise_single(
        num_tests : int = 8, 
        output_diretory_path : Path = Path("./gravyflow_data/tests/"),
        plot_results : bool = False
    ):

    with gf.env():

        # Setup ifo data acquisition object:
        ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3, 
            gf.DataQuality.BEST, 
            [
                gf.DataLabel.NOISE, 
                gf.DataLabel.GLITCHES
            ],
            gf.SegmentOrder.RANDOM,
            force_acquisition = True,
            cache_segments = False
        )
        
        # Initilise noise generator wrapper:
        noise : gf.NoiseObtainer = gf.NoiseObtainer(
            ifo_data_obtainer = ifo_data_obtainer,
            noise_type = gf.NoiseType.REAL,
            ifos = gf.IFO.L1
        )
        
        # Iterate through num_tests batches to check correct operation:
        onsource, offsource, gps_times  = next(noise())

        parameters_dict = {
            "onsource" : onsource,
            "offsource" : offsource,
            "gps_times" : gps_times
        }

        parameters_file_path : Path = (
            gf.PATH / f"res/tests/real_noise_consistancy_single.hdf5"
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
            gf.ensure_directory_exists(output_diretory_path)
            
            # Define an output path for the dashboard
            output_file(output_diretory_path / "noise_plots.html")

            # Arrange the plots in a grid. 
            grid = gridplot(layout)
                
            save(grid)
    
def _test_real_noise_multi(
        num_tests : int = 8, 
        output_diretory_path : Path = Path("./gravyflow_data/tests/"),
        plot_results : bool = False 
    ):

    with gf.env():
        
        # Setup ifo data acquisition object:
        ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3, 
            gf.DataQuality.BEST, 
            [
                gf.DataLabel.NOISE, 
                gf.DataLabel.GLITCHES
            ],
            gf.SegmentOrder.RANDOM,
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
        onsource, offsource, gps_times  = next(noise())

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
        num_tests : int = int(1.0E3)
    ):
    
    with gf.env():
        
        # Setup ifo data acquisition object:
        ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3, 
            gf.DataQuality.BEST, 
            [
                gf.DataLabel.NOISE, 
                gf.DataLabel.GLITCHES
            ],
            gf.SegmentOrder.RANDOM,
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
            
        onsource, offsource, _ = next(iter(noise()))

        onsource_shape = onsource.shape
        offsource_shape = offsource.shape

        logging.info("Start shape tests...")
        for index, (onsource, offsource, _) in tqdm(
                enumerate(islice(noise(), num_tests))
            ):

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
):

    with gf.env():

        # Setup ifo data acquisition object:
        ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
            gf.ObservingRun.O3, 
            gf.DataQuality.BEST, 
            [
                gf.DataLabel.NOISE, 
                gf.DataLabel.GLITCHES
            ],
            gf.SegmentOrder.RANDOM,
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

def test_real_noise_single(pytestconfig : Dict):
    _test_real_noise_single(
        plot_results=pytestconfig.getoption("plot")
    )

def test_real_noise_multi(pytestconfig : Dict):
    _test_real_noise_multi(
        plot_results=pytestconfig.getoption("plot")
    )
    
def test_noise_shape(pytestconfig : Dict):
    _test_noise_shape(
        num_tests=gf.tests.num_tests_from_config(pytestconfig)
    )

def test_noise_iteration(pytestconfig : Dict):
    _test_noise_iteration(
        num_tests=gf.tests.num_tests_from_config(pytestconfig)
    )