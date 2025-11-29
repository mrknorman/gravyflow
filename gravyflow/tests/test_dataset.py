# Built-In imports:
import logging
from pathlib import Path
from itertools import islice
from typing import List, Optional, Tuple, Dict, Union, Any

# Library imports:
import numpy as np
import tensorflow as tf
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from tqdm import tqdm
from _pytest.config import Config

# Local imports:
import gravyflow as gf

def validate_dataset_arguments(
        name: Union[str, None],
        waveform_type : Union[str, None],
        plot_examples: Union[bool, None],
        num_tests: Union[int, None],
        output_directory_path: Union[Path, None],
        ifos: Union[List[gf.IFO], None]
    ) -> None:

    """
    Validates the types of arguments.

    Args:
        name (Union[str, None]): Name of the test.
        waveform_type (Union[str, None]): Type of signal to be used in test.
        plot_examples (Union[bool, None]): Whether to plot examples.
        num_tests (Union[int, None]): Number of tests to run.
        output_directory_path (Union[Path, None]): Path to save output.
        ifos (Union[List[gf.IFO], None]): Interferometers to use.

    Raises:
        TypeError: If any argument does not match its expected type.
    """
    if not isinstance(name, str) and name is not None:
        raise TypeError(
            f"Expected 'name' to be of type 'str', got {type(name)}"
        )

    if not isinstance(waveform_type, str) and name is not None:
        raise TypeError(
            f"Expected 'signal_tyoe' to be of type 'str', got {type(name)}"
        )

    if not isinstance(plot_examples, bool) and plot_examples is not None:
        raise TypeError(
            f"Expected 'plot_examples' to be of type 'bool', got {type(plot_examples)}"
        )

    if not isinstance(num_tests, int) and num_tests is not None:
        raise TypeError(
            f"Expected 'num_tests' to be of type 'int', got {type(num_tests)}"
        )

    if not isinstance(output_directory_path, Path) and output_directory_path is not None:
        raise TypeError(
            f"Expected 'output_directory_path' to be of type 'Path', got {type(output_directory_path)}"
        )

    if ifos is not None:
        if not isinstance(ifos, list):
            raise TypeError(
                f"Expected 'ifos' to be of type 'list', got {type(ifos)}"
            )
        if not all(isinstance(ifo, gf.IFO) for ifo in ifos):
            raise TypeError(
                "All elements in 'ifos' should be of type 'gf.IFO'"
            )

def _test_dataset(
        name: str,
        waveform_type : str,
        plot_examples: Optional[bool] = True,
        num_tests: Optional[int] = 32,
        output_directory_path : Optional[Path] = None,
        ifos: Optional[List[gf.IFO]] = [gf.IFO.L1],
    ) -> None:

    """Test the dataset with specified parameters.

    Args:
        name (str): Name of the test.
        waveform_type (str) : Type of waveform to be injected into dataset.
        plot_examples (bool, optional): Whether to plot examples. Defaults to True.
        num_tests (int, optional): Number of tests to run. Defaults to 32.
        output_directory_path (Path, optional): Path to save output. Defaults to './gravyflow_data/tests/'.
        ifos (Optional[List[gf.IFO]], optional): IFOs to use. Defaults to gf.IFO.L1.
    """
    
    if output_directory_path is None:
        output_directory_path : Path = gf.PATH.parent.parent / "gravyflow_data/tests/"
    
    # Validate input arguments:
    validate_dataset_arguments(
        name,
        waveform_type,
        plot_examples,
        num_tests,
        output_directory_path,
        ifos
    )

    name : str = f"{name}_{waveform_type}"

    if waveform_type == "incoherent": 
        ifos = [gf.IFO.L1, gf.IFO.H1]

    logging.info(f"Running test: {name}")
    with gf.env():
        injection_directory_path, parameters_file_path = setup_paths(name)
        scaling_method = initialize_scaling_method()
        waveform_generator = get_waveform_generator(
            waveform_type,
            injection_directory_path, 
            scaling_method, 
            ifos
        )
        noise_obtainer = setup_noise_obtainer(ifos)
        dataset = create_dataset(
            waveform_type, noise_obtainer, waveform_generator, num_tests
        )
        input_dict, _ = next(iter(dataset))
        current_parameters = extract_parameters(waveform_type, input_dict)
        gf.tests.compare_and_save_parameters(
            current_parameters, parameters_file_path
        )
        if plot_examples:
            plot_dataset_examples(waveform_type, current_parameters, output_directory_path, name)

def setup_paths(name: str) -> Tuple[Path, Path, Path]:

    """Setup the directory paths for the test.

    Args:
        name (str): Name of the test.

    Returns:
        Tuple[Path, Path, Path]: Current directory, injection directory, and 
        parameters file path.
    """
    injection_directory_path = gf.tests.PATH / "example_injection_parameters"
    parameters_file_path = gf.PATH / f"res/tests/dataset_{name}.hdf5"
    gf.ensure_directory_exists(parameters_file_path.parent)
    gf.ensure_directory_exists(injection_directory_path)
    return injection_directory_path, parameters_file_path

def initialize_scaling_method() -> gf.ScalingMethod:

    """Initialize the scaling method.

    Returns:
        gf.ScalingMethod: The scaling method object.
    """
    return gf.ScalingMethod(
        value=gf.Distribution(
            min_=8.0,
            max_=15.0,
            type_=gf.DistributionType.UNIFORM
        ),
        type_=gf.ScalingTypes.SNR
    )

def get_waveform_generator(
        waveform_type : str,
        injection_directory_path: Path, 
        scaling_method: gf.ScalingMethod, 
        ifos: List[gf.IFO]
    ) -> gf.WaveformGenerator:

    match waveform_type:
        case "noise":
            return None
        case "phenomd":
            get_generator = get_phenom_generator
        case "wnb":
            get_generator = get_wnb_generator
        case "incoherent":
            get_generator = get_incoherent_generator
        case _:
            raise ValueError("Unknown test signal type {waveform_type} input!")
    
    return get_generator(
        injection_directory_path, 
        scaling_method, 
        ifos
    )

def get_phenom_generator(
        injection_directory_path: Path, 
        scaling_method: gf.ScalingMethod, 
        ifos: List[gf.IFO]
    ) -> gf.RippleGenerator:

    """Load the injection configuration.

    Args:
        injection_directory_path (Path): Path to the injection directory.
        scaling_method (gf.ScalingMethod): The scaling method to be used.
        ifos (List[gf.IFO]): The interferometers to use.

    Returns:
        gf.RippleGenerator: The waveform generator object.
    """
    phenom_d_generator = gf.WaveformGenerator.load(
        path=injection_directory_path / "phenom_d_parameters.json", 
        scaling_method=scaling_method,
        network=ifos
    )
    phenom_d_generator.injection_chance = 1.0
    return phenom_d_generator

def get_wnb_generator(
        injection_directory_path: Path, 
        scaling_method: gf.ScalingMethod, 
        ifos: List[gf.IFO]
    ) -> gf.WNBGenerator:

    """Load the injection configuration.

    Args:
        injection_directory_path (Path): Path to the injection directory.
        scaling_method (gf.ScalingMethod): The scaling method to be used.
        ifos (List[gf.IFO]): The interferometers to use.

    Returns:
        gf.WNBGenerator: The waveform generator object.
    """
    wnb_generator = gf.WaveformGenerator.load(
        path=injection_directory_path / "wnb_parameters.json", 
        scaling_method=scaling_method,
        network=ifos
    )
    wnb_generator.injection_chance = 1.0
    return wnb_generator

def get_incoherent_generator(
        injection_directory_path: Path, 
        scaling_method: gf.ScalingMethod,
        _ : Any
    ) -> Tuple[gf.RippleGenerator, gf.WNBGenerator]:

    """Load the waveform generators.

    Args:
        injection_directory_path (Path): Path to the injection directory.
        scaling_method (gf.ScalingMethod): The scaling method.
        ifos (List[gf.IFO]): Unused.

    Returns:
        gf.IncoherentGenerator: The incoherent generator object.
    """

    # Ifos must match num incoherent compents so hardcoded here:
    ifos: List[gf.IFO] = [gf.IFO.L1, gf.IFO.H1]

    phenom_d_generator = get_phenom_generator(
        injection_directory_path, 
        scaling_method, 
        ifos
    )
    wnb_generator = get_wnb_generator(
        injection_directory_path, 
        scaling_method, 
        ifos
    )

    return gf.IncoherentGenerator([phenom_d_generator, wnb_generator])

def setup_noise_obtainer(
        ifos: List[gf.IFO]
    ) -> gf.NoiseObtainer:

    """Setup the noise obtainer.

    Args:
        ifos (List[gf.IFO]): The interferometers to use.

    Returns:
        gf.NoiseObtainer: The noise obtainer object.
    """
    ifo_data_obtainer = gf.IFODataObtainer(
        observing_runs=gf.ObservingRun.O3, 
        data_quality=gf.DataQuality.BEST, 
        data_labels=[gf.DataLabel.NOISE],
        force_acquisition=True,
        cache_segments=False
    )
    return gf.NoiseObtainer(
        ifo_data_obtainer=ifo_data_obtainer,
        noise_type=gf.NoiseType.REAL,
        ifos=ifos
    )


def get_return_variables(waveform_type : str) -> List:
    match waveform_type:
        case "noise":
            input_variables=[
                gf.ReturnVariables.WHITENED_ONSOURCE
            ]
        case "phenomd":
            input_variables=[
                gf.ReturnVariables.WHITENED_ONSOURCE, 
                gf.ReturnVariables.INJECTION_MASKS,
                gf.ReturnVariables.INJECTIONS,
                gf.ReturnVariables.WHITENED_INJECTIONS,
                gf.WaveformParameters.MASS_1_MSUN,
                gf.WaveformParameters.MASS_2_MSUN,
            ]
        case "wnb":
            input_variables=[
                gf.ReturnVariables.WHITENED_ONSOURCE, 
                gf.ReturnVariables.INJECTION_MASKS,
                gf.ReturnVariables.INJECTIONS,
                gf.ReturnVariables.WHITENED_INJECTIONS,
                gf.WaveformParameters.DURATION_SECONDS
            ]
        case "incoherent":
            input_variables=[
                gf.ReturnVariables.WHITENED_ONSOURCE, 
                gf.ReturnVariables.INJECTION_MASKS,
                gf.ReturnVariables.INJECTIONS,
                gf.ReturnVariables.WHITENED_INJECTIONS,
                gf.WaveformParameters.MASS_1_MSUN,
                gf.WaveformParameters.MASS_2_MSUN,
            ]
        case _:
            raise ValueError("Unknown test signal type {waveform_type} input!")

    return input_variables

def create_dataset(
        waveform_type : str,
        noise_obtainer: gf.NoiseObtainer, 
        waveform_generator: gf.WaveformGenerator, 
        num_tests: int
    ) -> tf.data.Dataset:

    """Create the dataset.

    Args:
        waveform_type (str) : Type of the waveform being injected.
        noise_obtainer (gf.NoiseObtainer): The noise obtainer object.
        waveform_generator (gf.WaveformGenerator): The waveform generator.
        num_tests (int): Number of tests to run.

    Returns:
        tf.data.Dataset: The dataset object.
    """
    return gf.Dataset(
        noise_obtainer=noise_obtainer,
        waveform_generators=waveform_generator,
        num_examples_per_batch=num_tests,
        input_variables=get_return_variables(waveform_type)
    )

def extract_parameters(
        waveform_type : str,
        input_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:

    """Extract parameters from the input dictionary.

    Args:
        input_dict (Dict[str, np.ndarray]): The input dictionary.

    Returns:
        Dict[str, np.ndarray]: Dictionary of extracted parameters.
    """

    match waveform_type:
        case "noise":
            return_dict = {
                'onsource': input_dict[
                    gf.ReturnVariables.WHITENED_ONSOURCE.name
                ].numpy()
            }
        case "phenomd":
            return_dict = {
                'onsource': input_dict[
                    gf.ReturnVariables.WHITENED_ONSOURCE.name
                ].numpy(),
                'injections': input_dict[
                    gf.ReturnVariables.INJECTIONS.name
                ].numpy()[0],
                'whitened_injections': input_dict[
                    gf.ReturnVariables.WHITENED_INJECTIONS.name
                ].numpy()[0],
                'masks': input_dict[
                    gf.ReturnVariables.INJECTION_MASKS.name
                ].numpy()[0],
                'mass_1_msun': input_dict[
                    gf.WaveformParameters.MASS_1_MSUN.name
                ].numpy()[0],
                'mass_2_msun': input_dict[
                    gf.WaveformParameters.MASS_2_MSUN.name
                ].numpy()[0]
            }
        case "wnb":
            return_dict = {
                'onsource': input_dict[
                    gf.ReturnVariables.WHITENED_ONSOURCE.name
                ].numpy(),
                'injections': input_dict[
                    gf.ReturnVariables.INJECTIONS.name
                ].numpy()[0],
                'whitened_injections': input_dict[
                    gf.ReturnVariables.WHITENED_INJECTIONS.name
                ].numpy()[0],
                'masks': input_dict[
                    gf.ReturnVariables.INJECTION_MASKS.name
                ].numpy()[0],
                'duration_seconds': input_dict[
                    gf.WaveformParameters.DURATION_SECONDS.name
                ].numpy()[0]
            }
        case "incoherent":
            return_dict = {
                'onsource': input_dict[
                    gf.ReturnVariables.WHITENED_ONSOURCE.name
                ].numpy(),
                'injections': input_dict[
                    gf.ReturnVariables.INJECTIONS.name
                ].numpy()[0],
                'whitened_injections': input_dict[
                    gf.ReturnVariables.WHITENED_INJECTIONS.name
                ].numpy()[0],
                'masks': input_dict[
                    gf.ReturnVariables.INJECTION_MASKS.name
                ].numpy()[0],
                'mass_1_msun': input_dict[
                    gf.WaveformParameters.MASS_1_MSUN.name
                ].numpy()[0],
                'mass_2_msun': input_dict[
                    gf.WaveformParameters.MASS_2_MSUN.name
                ].numpy()[0]
            }
        case _:
            raise ValueError(f"waveform_type: {waveform_type} not recognised!")

    return return_dict

def plot_dataset_examples(
        waveform_type: str,
        current_parameters: Dict[str, np.ndarray],
        output_directory_path: Path,
        name: str
    ) -> None:

    """
    Plot dataset examples.

    Args:
        waveform_type (str): Type of waveform.
        current_parameters (Dict[str, np.ndarray]): The current parameters.
        output_directory_path (Path): Path to the output directory.
        name (str): Name of the test.
    """
    logging.info("Plotting examples...")

    onsource = current_parameters['onsource']
    layout = []

    match waveform_type:
        case "noise":
            plot_data = [(onsource_,) for onsource_ in onsource]
            plot_func = lambda data: [
                gf.generate_strain_plot({"Whitened Onsource": data[0]}, title="Noise example"),
                gf.generate_spectrogram(data[0])
            ]
        
        case "phenomd" | "incoherent":
            whitened_injections = current_parameters['whitened_injections']
            injections = current_parameters['injections']
            mass_1_msun = current_parameters['mass_1_msun']
            mass_2_msun = current_parameters['mass_2_msun']
            plot_data = zip(onsource, whitened_injections, injections, mass_1_msun, mass_2_msun)
            title = "PhenomD injection example" if waveform_type == "phenomd" else "Incoherent injection example"
            plot_func = lambda data: [
                gf.generate_strain_plot(
                    {"Whitened Onsource + Injection": data[0], "Whitened Injection": data[1], "Injection": data[2]},
                    title=f"{title}: mass_1 {data[3]} msun; mass_2 {data[4]} msun"
                ),
                gf.generate_spectrogram(data[0])
            ]

        case "wnb":
            whitened_injections = current_parameters['whitened_injections']
            injections = current_parameters['injections']
            duration = current_parameters['duration_seconds']
            plot_data = zip(onsource, whitened_injections, injections, duration)
            plot_func = lambda data: [
                gf.generate_strain_plot(
                    {"Whitened Onsource + Injection": data[0], "Whitened Injection": data[1], "Injection": data[2]},
                    title=f"WNB injection example: duration {data[3]} seconds."
                ),
                gf.generate_spectrogram(data[0])
            ]

        case _:
            raise ValueError(f"Waveform type {waveform_type} not recognised!")

    layout = [plot_func(data) for data in plot_data]

    # Save the plots
    gf.ensure_directory_exists(output_directory_path)
    output_file(output_directory_path / f"{name}.html")
    grid = gridplot(layout)
    save(grid)

    logging.info(f"Plotting completed. Output saved to {output_directory_path / f'{name}.html'}")

def _test_dataset_iteration(
        name: str,
        waveform_type : str,
        num_tests: Optional[int] = int(1.0E2),
        ifos: Optional[List[gf.IFO]] = [gf.IFO.L1],
    ) -> None:

    """Test dataset iteration.

    Args:
        name (str): Name of the test.
        num_tests (Optional[int], optional): Number of tests to run. Defaults to 1.0E2.
        ifos (Optional[List[gf.IFO]], optional): Interferometers to use. Defaults to [gf.IFO.L1].
    """

    # Validate input arguments:
    validate_dataset_arguments(
        name,
        waveform_type,
        None,
        num_tests,
        None,
        ifos
    )

    name : str = f"{name}_{waveform_type}"
    
    logging.info(f"Running test: {name}")
    with gf.env():
        injection_directory_path, parameters_file_path = setup_paths(name)
        scaling_method = initialize_scaling_method()
        waveform_generator = get_waveform_generator(
            waveform_type, injection_directory_path, scaling_method, ifos
        )
        noise_obtainer = setup_noise_obtainer(ifos)
        dataset_args = setup_dataset_arguments(
            waveform_type, noise_obtainer, waveform_generator
        )

        logging.info("Start data iteration tests...")
        iterate_data(gf.data, dataset_args, num_tests)
        logging.info("Complete.")

        logging.info("Start dataset iteration tests...")
        iterate_data(gf.Dataset, dataset_args, num_tests)
        logging.info("Complete.")

def setup_dataset_arguments(
        waveform_type : str,
        noise_obtainer: gf.NoiseObtainer, 
        phenom_d_generator: gf.RippleGenerator
    ) -> Dict:

    """Set up arguments for the dataset.

    Args:
        waveform_type (str): Type of waveform being returned by dataset.
        noise_obtainer (gf.NoiseObtainer): Noise obtainer object.
        phenom_d_generator (gf.RippleGenerator): Waveform generator object.

    Returns:
        Dict: Arguments for creating the dataset.
    """
    
    return {
        "noise_obtainer": noise_obtainer,
        "waveform_generators": phenom_d_generator,
        "input_variables": get_return_variables(waveform_type),
    }

def iterate_data(
        dataset_function, 
        dataset_args: Dict, 
        num_tests: int
    ) -> None:

    """
    Iterate through data or dataset.

    Args:
        dataset_function: A function to create the dataset.
        dataset_args (Dict): Arguments for creating the dataset.
        num_tests (int): Number of tests (iterations) to perform.
    """
    data = dataset_function(**dataset_args)
    for index, _ in tqdm(enumerate(islice(data, num_tests))):
        pass

    np.testing.assert_equal(
        index, 
        num_tests - 1, 
        err_msg="Warning! Data does not iterate the required number of batches."
    )

def test_dataset_consistency_single_ifo_noise(
        pytestconfig : Config
    ) -> None:

    _test_dataset(
        name="consistency_single",
        waveform_type="noise",
        plot_examples=pytestconfig.getoption("plot")
    )

def test_dataset_consistency_single_ifo_phenom(
        pytestconfig : Config
    ) -> None:

    _test_dataset(
        name="consistency_single",
        waveform_type="phenomd",
        plot_examples=pytestconfig.getoption("plot")
    )

def test_dataset_consistency_single_ifo_wnb(
        pytestconfig : Config
    ) -> None:
    _test_dataset(
        name="consistency_single",
        waveform_type="wnb",
        plot_examples=pytestconfig.getoption("plot")
    )

def test_dataset_consistency_multi_ifo_noise(
        pytestconfig : Config
    ) -> None:

    _test_dataset(
        name="consistency_multi", 
        waveform_type="noise",
        plot_examples=pytestconfig.getoption("plot"),
        ifos=[gf.IFO.L1, gf.IFO.H1]
    )

def test_dataset_consistency_multi_ifo_phenom(
     pytestconfig : Config
    ) -> None:
    _test_dataset(
        name="consistency_multi", 
        waveform_type="phenomd",
        ifos=[gf.IFO.L1, gf.IFO.H1]
    )

def test_dataset_consistency_multi_ifo_wnb(
        pytestconfig : Config
    ) -> None:

    _test_dataset(
        name="consistency_multi", 
        waveform_type="wnb",
        plot_examples=pytestconfig.getoption("plot"),
        ifos=[gf.IFO.L1, gf.IFO.H1]
    )

def test_dataset_consistency_multi_ifo_incoherent(
        pytestconfig : Config
    ) -> None:

    _test_dataset(
        name="consistency_multi", 
        waveform_type="incoherent",
        plot_examples=pytestconfig.getoption("plot"),
        ifos=[gf.IFO.L1, gf.IFO.H1]
    )

def test_dataset_iteration_single_ifo_noise(
        pytestconfig : Config
    ) -> None:

    _test_dataset_iteration(
        name="iteration_single_ifo",
        waveform_type="noise",
        num_tests=gf.tests.num_tests_from_config(pytestconfig),
        ifos=[gf.IFO.L1]
    )

def test_dataset_iteration_single_ifo_phenomd(
        pytestconfig : Config
    ) -> None:

    _test_dataset_iteration(
        name="iteration_single_ifo",
        waveform_type="phenomd",
        num_tests=gf.tests.num_tests_from_config(pytestconfig),
        ifos=[gf.IFO.L1]
    )

def test_dataset_iteration_single_ifo_wnb(
        pytestconfig : Config
    ) -> None:

    _test_dataset_iteration(
        name="iteration_single_ifo",
        waveform_type="wnb",
        num_tests=gf.tests.num_tests_from_config(pytestconfig),
        ifos=[gf.IFO.L1]
    )

def test_dataset_iteration_multi_ifo_noise(
        pytestconfig : Config
    ) -> None:

    _test_dataset_iteration(
        name="iteration_multi_ifo",
        waveform_type="noise",
        num_tests=gf.tests.num_tests_from_config(pytestconfig),
        ifos=[gf.IFO.L1, gf.IFO.H1]
    )

def test_dataset_iteration_multi_ifo_phenomd(
        pytestconfig : Config
    ) -> None:

    _test_dataset_iteration(
        name="iteration_multi_ifo",
        waveform_type="phenomd",
        num_tests=gf.tests.num_tests_from_config(pytestconfig),
        ifos=[gf.IFO.L1, gf.IFO.H1]
    )

def test_dataset_iteration_multi_ifo_wnb(
        pytestconfig : Config
    ) -> None:

    _test_dataset_iteration(
        name="iteration_multi_ifo",
        waveform_type="wnb",
        num_tests=gf.tests.num_tests_from_config(pytestconfig),
        ifos=[gf.IFO.L1, gf.IFO.H1]
    )

def test_dataset_iteration_multi_ifo_incoherent(
        pytestconfig : Config
    ) -> None:

    _test_dataset_iteration(
        name="iteration_multi_ifo",
        waveform_type="incoherent",
        num_tests=gf.tests.num_tests_from_config(pytestconfig),
        ifos=[gf.IFO.L1, gf.IFO.H1]
    )