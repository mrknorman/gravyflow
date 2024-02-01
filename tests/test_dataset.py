# Built-In imports:
import logging
from pathlib import Path
from itertools import islice
from copy import deepcopy
import os
from typing import Iterator, List, Optional

# Library imports:
import numpy as np
import tensorflow as tf
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from tqdm import tqdm
import h5py

# Local imports:
import gravyflow as gf

def _test_dataset(
        name : str,
        plot_examples : bool = True,
        num_tests : int = 32,
        output_diretory_path : Path = Path("./gravyflow_data/tests/"),
        ifos : Optional[List[gf.IFO]] = [gf.IFO.L1],
    ):

    logging.info(f"Running test: {name}")
    
    # Setup GPU environment
    with gf.env():
        
        # Define injection directory path:
        current_dir : Path = Path(os.path.dirname(os.path.abspath(__file__)))
        injection_directory_path : Path = current_dir / "example_injection_parameters"
        parameters_file_path : Path = current_dir.parent / f"res/tests/dataset_{name}.hdf5" 

        gf.ensure_directory_exists(parameters_file_path.parent)
        gf.ensure_directory_exists(injection_directory_path)

        # Intilise Scaling Method:
        scaling_method = gf.ScalingMethod(
            value=gf.Distribution(
                min_=8.0,
                max_=15.0,
                type_=gf.DistributionType.UNIFORM
            ),
            type_=gf.ScalingTypes.SNR
        )

        # Load injection config:
        phenom_d_generator : gf.cuPhenomDGenerator = gf.WaveformGenerator.load(
            path=injection_directory_path / "phenom_d_parameters.json", 
            scaling_method=scaling_method,
            network=ifos
        )

        # Ensure injection in all test cases:
        phenom_d_generator.injection_chance = 1.0

        # Setup ifo data acquisition object:
        ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
            observing_runs=gf.ObservingRun.O3, 
            data_quality=gf.DataQuality.BEST, 
            data_labels=[
                gf.DataLabel.NOISE
            ],
            force_acquisition = True,
            cache_segments = False
        )

        # Initilise noise generator wrapper:
        noise_obtainer: gf.NoiseObtainer = gf.NoiseObtainer(
            ifo_data_obtainer=ifo_data_obtainer,
            noise_type=gf.NoiseType.REAL,
            ifos=ifos
        )
        
        dataset : tf.data.Dataset = gf.Dataset(
            # Noise: 
            noise_obtainer=noise_obtainer,
            # Injections:
            waveform_generators=phenom_d_generator, 
            # Output configuration:
            num_examples_per_batch=num_tests,
            input_variables = [
                gf.ReturnVariables.WHITENED_ONSOURCE, 
                gf.ReturnVariables.INJECTION_MASKS, 
                gf.ReturnVariables.INJECTIONS,
                gf.ReturnVariables.WHITENED_INJECTIONS,
                gf.WaveformParameters.MASS_1_MSUN, 
                gf.WaveformParameters.MASS_2_MSUN
            ],
        )

        logging.info(f"Generating examples...")

        # Get one iteration from iterator, ignore outputs dict
        input_dict, _ = next(iter(dataset))
        
        logging.info(f"Extracting parameters...")

        # Extract parameters:
        onsource = input_dict[
            gf.ReturnVariables.WHITENED_ONSOURCE.name
        ].numpy()
        injections = input_dict[
            gf.ReturnVariables.INJECTIONS.name
        ].numpy()[0]
        whitened_injections = input_dict[
            gf.ReturnVariables.WHITENED_INJECTIONS.name
        ].numpy()[0]
        masks = input_dict[
            gf.ReturnVariables.INJECTION_MASKS.name
        ].numpy()[0]
        mass_1_msun = input_dict[
            gf.WaveformParameters.MASS_1_MSUN.name
        ].numpy()[0]
        mass_2_msun = input_dict[
            gf.WaveformParameters.MASS_2_MSUN.name
        ].numpy()[0]

        logging.info(f"Save parameters...")

        # Save the extracted parameters
        current_parameters = {
            'onsource': onsource,
            'injections': injections,
            'whitened_injections': whitened_injections,
            'masks': masks,
            'mass_1_msun': mass_1_msun,
            'mass_2_msun': mass_2_msun
        }

        logging.info(f"Check parameters...")

        if parameters_file_path.exists():
            with h5py.File(parameters_file_path, 'r') as hf:
                # Compare with previous parameters
                tolerance = 1e-6  # Set an appropriate tolerance level
                for key in current_parameters:
                    previous_data = hf[key][:]
                    if not np.allclose(previous_data, current_parameters[key], atol=tolerance):
                        raise ValueError(f"Parameter consistency check failed for {key}.")
                logging.info("Parameter consistency check passed.")
        else:
            # Save the current parameters for the first time
            with h5py.File(parameters_file_path, 'w') as hf:
                for key, data in current_parameters.items():
                    hf.create_dataset(key, data=data)
            logging.info("No previous parameters to compare. Saving current.")

        if plot_examples:

            logging.info(f"Plotting examples...")

            # Generate plots:
            layout = [
                [gf.generate_strain_plot(
                    {
                        "Whitened Onsouce + Injection": onsource_,
                        "Whitened Injection" : whitened_injection,
                        "Injection": injection
                    },
                    title=(f"cuPhenomD injection example: mass_1 {m1} msun; mass_2 {m2}"
                        " msun")
                ),
                gf.generate_spectrogram(
                    onsource_
                )]
                for onsource_, whitened_injection, injection, m1, m2 in zip(
                    onsource,
                    whitened_injections,
                    injections, 
                    mass_1_msun, 
                    mass_2_msun
                )
            ]

            # Ensure output directory exists
            gf.ensure_directory_exists(output_diretory_path)

            # Define an output path for the dashboard
            output_file(output_diretory_path / f"{name}.html")

            # Arrange the plots in a grid. 
            grid = gridplot(layout)

            save(grid)

def _test_dataset_incoherent(
        name : str,
        plot_examples : bool = True,
        num_tests : int = 32,
        output_diretory_path : Path = Path("./gravyflow_data/tests/"),
        ifos : Optional[List[gf.IFO]] = [gf.IFO.L1],
    ):

    logging.info(f"Running test: {name}")
    
    # Setup GPU environment
    with gf.env(gpus="3"):
        
        # Define injection directory path:
        current_dir : Path = Path(os.path.dirname(os.path.abspath(__file__)))
        injection_directory_path : Path = current_dir / "example_injection_parameters"
        parameters_file_path : Path = current_dir.parent / f"res/tests/dataset_{name}.hdf5" 

        gf.ensure_directory_exists(parameters_file_path.parent)
        gf.ensure_directory_exists(injection_directory_path)

        # Intilise Scaling Method:
        scaling_method = gf.ScalingMethod(
            value=gf.Distribution(
                min_=8.0,
                max_=15.0,
                type_=gf.DistributionType.UNIFORM
            ),
            type_=gf.ScalingTypes.SNR
        )

        # Load injection config:
        phenom_d_generator : gf.cuPhenomDGenerator = gf.WaveformGenerator.load(
            path=injection_directory_path / "phenom_d_parameters.json", 
            scaling_method=scaling_method,
            network=ifos
        )

        # Ensure injection in all test cases:
        phenom_d_generator.injection_chance = 1.0

        wnb_generator : WNBGenerator = gf.WaveformGenerator.load(
            path=injection_directory_path / "wnb_parameters.json", 
            scaling_method=scaling_method,
           network=ifos
        )

        # Ensure injection in all test cases:
        wnb_generator.injection_chance = 1.0

        incoherent_generator = gf.IncoherentGenerator(
            [phenom_d_generator, wnb_generator]
        )

         # Setup ifo data acquisition object:
        ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
            observing_runs=gf.ObservingRun.O3, 
            data_quality=gf.DataQuality.BEST, 
            data_labels=[
                gf.DataLabel.NOISE
            ],
            force_acquisition = True,
            cache_segments = False
        )

        # Initilise noise generator wrapper:
        noise_obtainer: gf.NoiseObtainer = gf.NoiseObtainer(
            ifo_data_obtainer=ifo_data_obtainer,
            noise_type=gf.NoiseType.REAL,
            ifos=ifos
        )
        
        dataset : tf.data.Dataset = gf.Dataset(
            # Noise: 
            noise_obtainer=noise_obtainer,
            # Injections:
            waveform_generators=incoherent_generator, 
            # Output configuration:
            num_examples_per_batch=num_tests,
            input_variables = [
                gf.ReturnVariables.WHITENED_ONSOURCE, 
                gf.ReturnVariables.INJECTION_MASKS, 
                gf.ReturnVariables.INJECTIONS,
                gf.ReturnVariables.WHITENED_INJECTIONS,
                gf.WaveformParameters.MASS_1_MSUN, 
                gf.WaveformParameters.MASS_2_MSUN
            ]
        )

        logging.info(f"Generating examples...")

        # Get one iteration from iterator, ignore outputs dict
        input_dict, _ = next(iter(dataset))
        
        logging.info(f"Extracting parameters...")

        # Extract parameters:
        onsource = input_dict[
            gf.ReturnVariables.WHITENED_ONSOURCE.name
        ].numpy()
        injections = input_dict[
            gf.ReturnVariables.INJECTIONS.name
        ].numpy()[0]
        whitened_injections = input_dict[
            gf.ReturnVariables.WHITENED_INJECTIONS.name
        ].numpy()[0]
        masks = input_dict[
            gf.ReturnVariables.INJECTION_MASKS.name
        ].numpy()[0]
        mass_1_msun = input_dict[
            gf.WaveformParameters.MASS_1_MSUN.name
        ].numpy()[0]
        mass_2_msun = input_dict[
            gf.WaveformParameters.MASS_2_MSUN.name
        ].numpy()[0]

        logging.info(f"Save parameters...")

        # Save the extracted parameters
        current_parameters = {
            'onsource': onsource,
            'injections': injections,
            'whitened_injections': whitened_injections,
            'masks': masks,
            'mass_1_msun': mass_1_msun,
            'mass_2_msun': mass_2_msun
        }

        logging.info(f"Check parameters...")

        if parameters_file_path.exists():
            with h5py.File(parameters_file_path, 'r') as hf:
                # Compare with previous parameters
                tolerance = 1e-6  # Set an appropriate tolerance level
                for key in current_parameters:
                    previous_data = hf[key][:]
                    if not np.allclose(previous_data, current_parameters[key], atol=tolerance):
                        raise ValueError(f"Parameter consistency check failed for {key}.")
                logging.info("Parameter consistency check passed.")
        else:
            # Save the current parameters for the first time
            with h5py.File(parameters_file_path, 'w') as hf:
                for key, data in current_parameters.items():
                    hf.create_dataset(key, data=data)
            logging.info("No previous parameters to compare. Saving current.")

        if plot_examples:

            logging.info(f"Plotting examples...")

            # Generate plots:
            layout = [
                [gf.generate_strain_plot(
                    {
                        "Whitened Onsouce + Injection": onsource_,
                        "Whitened Injection" : whitened_injection,
                        "Injection": injection
                    },
                    title=(f"cuPhenomD injection example: mass_1 {m1} msun; mass_2 {m2}"
                        " msun")
                ),
                gf.generate_spectrogram(
                    onsource_
                )]
                for onsource_, whitened_injection, injection, m1, m2 in zip(
                    onsource,
                    whitened_injections,
                    injections, 
                    mass_1_msun, 
                    mass_2_msun
                )
            ]

            # Ensure output directory exists
            gf.ensure_directory_exists(output_diretory_path)

            # Define an output path for the dashboard
            output_file(output_diretory_path / f"{name}.html")

            # Arrange the plots in a grid. 
            grid = gridplot(layout)

            save(grid)

def test_dataset_single():
    _test_dataset(
        name="dataset_single"
    )

def test_dataset_multi():
    _test_dataset(
        name="dataset_multi", 
        ifos=[gf.IFO.L1, gf.IFO.H1]
    )

def test_dataset_incoherent():
    _test_dataset_incoherent(
        name="dataset_incoherent",
        ifos=[gf.IFO.L1, gf.IFO.H1]
    )

def test_iteration(
    num_tests : int = int(1.0E2)
    ):
    
    with gf.env():        
        # Define injection directory path:
        current_dir : Path = Path(os.path.dirname(os.path.abspath(__file__)))
        injection_directory_path : Path = current_dir / "example_injection_parameters"

        # Intilise Scaling Method:
        scaling_method : gf.ScalingMethod = gf.ScalingMethod(
            value=gf.Distribution(
                min_=8.0,
                max_=15.0,
                type_=gf.DistributionType.UNIFORM
            ),
            type_=gf.ScalingTypes.SNR
        )

        # Load injection config:
        phenom_d_generator : gf.cuPhenomDGenerator = gf.WaveformGenerator.load(
            path=injection_directory_path / "phenom_d_parameters.json", 
            scaling_method=scaling_method
        )

        # Ensure injection in all test cases:
        phenom_d_generator.injection_chance = 1.0

        # Setup ifo data acquisition object:
        ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
            observing_runs=gf.ObservingRun.O3, 
            data_quality=gf.DataQuality.BEST, 
            data_labels=[
                gf.DataLabel.NOISE
            ],
            force_acquisition = True,
            cache_segments = False
        )

        # Initilise noise generator wrapper:
        noise_obtainer: gf.NoiseObtainer = gf.NoiseObtainer(
            ifo_data_obtainer=ifo_data_obtainer,
            noise_type=gf.NoiseType.REAL,
            ifos=ifos
        )

        # Extract parameters:
        input_variables = [
            gf.ReturnVariables.WHITENED_ONSOURCE, 
            gf.ReturnVariables.INJECTION_MASKS, 
            gf.ReturnVariables.INJECTIONS,
            gf.ReturnVariables.WHITENED_INJECTIONS,
            gf.WaveformParameters.MASS_1_MSUN, 
            gf.WaveformParameters.MASS_2_MSUN
        ]

        data_args : Dict = {
            # Random Seed:
            "seed" : 1000,
            # Temporal components:
            "sample_rate_hertz" : sample_rate_hertz,   
            "onsource_duration_seconds" : onsource_duration_seconds,
            "offsource_duration_seconds" : offsource_duration_seconds,
            "crop_duration_seconds" : crop_duration_seconds,
            # Noise: 
            "noise_obtainer" : noise_obtainer,
            "scale_factor" : scale_factor,
            # Injections:
            "waveform_generators" : phenom_d_generator, 
            # Output configuration:
            "num_examples_per_batch" : num_examples_per_batch,
            "input_variables" : input_variables
        }

        # Deepcopy before running tests to ensure args remain constant:
        dataset_args : Dict = deepcopy(data_args)

        data : Iterator = gf.data(
            **data_args
        )

        logging.info("Start iteration tests...")
        for index, _ in tqdm(enumerate(islice(data, num_tests))):
            pass
        logging.info("Complete.")

        assert index == num_tests - 1, "Warning! Data does not iterate the required number of batches"

        dataset : tf.data.Dataset = gf.Dataset(
            **dataset_args
        )

        for index, _ in tqdm(enumerate(islice(dataset, num_tests))):
            pass

        assert index == num_tests - 1, "Warning! Dataset does not iterate the required number of batches"
        
def test_feature_dataset(
    num_tests : int = 32,
    output_diretory_path : Path = Path("./gravyflow_data/tests/")
    ):
    
    with gf.env():
    
        # Test Parameters:
        num_examples_per_generation_batch : int = 2048
        num_examples_per_batch : int = num_tests
        sample_rate_hertz : float = 2048.0
        onsource_duration_seconds : float = 1.0
        offsource_duration_seconds : float = 16.0
        crop_duration_seconds : float = 0.5
        scale_factor : float = 1.0E21

        # Define injection directory path:
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        injection_directory_path : Path = \
            Path(current_dir / "example_injection_parameters")

        # Intilise Scaling Method:
        scaling_method = \
            gf.ScalingMethod(
                gf.Distribution(
                    min_=8.0,
                    max_=15.0,
                    type_=gf.DistributionType.UNIFORM
                ),
                gf.ScalingTypes.SNR
            )

        # Load injection config:
        phenom_d_generator : gf.cuPhenomDGenerator = \
            gf.WaveformGenerator.load(
                injection_directory_path / "phenom_d_parameters.json", 
                sample_rate_hertz, 
                onsource_duration_seconds,
                scaling_method=scaling_method
            )

        # Setup ifo data acquisition object:
        ifo_data_obtainer : gf.IFODataObtainer = \
            gf.IFODataObtainer(
                gf.ObservingRun.O3, 
                gf.DataQuality.BEST, 
                [
                    gf.DataLabel.EVENTS
                ],
                gf.SegmentOrder.RANDOM,
                force_acquisition = True,
                cache_segments = False
            )

        # Initilise noise generator wrapper:
        noise_obtainer: gf.NoiseObtainer = \
            gf.NoiseObtainer(
                ifo_data_obtainer=ifo_data_obtainer,
                noise_type=gf.NoiseType.REAL,
                ifos=gf.IFO.L1
            )

        dataset : tf.data.Dataset = gf.Dataset(
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
            waveform_generators=phenom_d_generator, 
            # Output configuration:
            num_examples_per_batch=num_examples_per_batch,
            input_variables = [
                gf.ReturnVariables.WHITENED_ONSOURCE, 
                gf.ReturnVariables.SPECTROGRAM_ONSOURCE
            ],
        )

        input_dict, _ = next(iter(dataset))

        onsource = input_dict[gf.ReturnVariables.WHITENED_ONSOURCE.name].numpy()
        injections = input_dict[gf.ReturnVariables.INJECTIONS.name].numpy()

        layout = [
            [gf.generate_strain_plot(
                {
                    "Whitened Onsouce + Injection": onsource_
                },
                sample_rate_hertz,
                scale_factor=scale_factor
            ), 
            gf.generate_spectrogram(
                onsource_, 
                sample_rate_hertz
            )]
            for onsource_ in zip(
                onsource,
            )
        ]

        # Ensure output directory exists
        gf.ensure_directory_exists(output_diretory_path)

        # Define an output path for the dashboard
        output_file(output_diretory_path / "event_plots.html")

        # Arrange the plots in a grid. 
        grid = gridplot(layout)

        save(grid)

if __name__ == "__main__":
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    # Test IFO noise dataset:

    test_dataset_incoherent()

    test_dataset_single()
    test_dataset_multi()

    #test_feature_dataset()
    test_iteration()

    