import os
import time
from pathlib import Path

import numpy as np
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from tqdm import tqdm

from .cuphenom import imrphenomd


def test_generate_phenom():
    # Define reasonable input parameters for an average gravitational wave
    num_waveforms = 1000
    sample_rate_hertz = 2048.0
    duration_seconds = 2.0
    
    mass_1_msun = np.random.uniform(30, 30, size = num_waveforms)  # random float between 1 and 30
    mass_2_msun = np.random.uniform(10, 10, size = num_waveforms)
    inclination_radians = np.random.uniform(0, 0, size = num_waveforms)
    distance_mpc = np.random.uniform(10.0, 10.0, size = num_waveforms)
    reference_orbital_phase_in = np.random.uniform(0, 0, size = num_waveforms)
    ascending_node_longitude = np.random.uniform(0, 0, size = num_waveforms)
    eccentricity = np.random.uniform(0, 0.0, size = num_waveforms)
    mean_periastron_anomaly = np.random.uniform(0, 0, size = num_waveforms)
    spin_1_in = np.random.uniform(0, 0, size = num_waveforms*3)
    spin_2_in = np.random.uniform(0, 0, size = num_waveforms*3)
    
    # Call generatePhenom function
    result = imrphenomd(
        num_waveforms,
        sample_rate_hertz,
        duration_seconds,
        mass_1_msun,
        mass_2_msun,
        inclination_radians,
        distance_mpc,
        reference_orbital_phase_in,
        ascending_node_longitude,
        eccentricity,
        mean_periastron_anomaly,
        spin_1_in,
        spin_2_in
    )  
    result = np.array(result.tolist())
        
    # Assume that result corresponds to pairs of gravitational wave polarisations plus and cross
    times = np.arange(0, duration_seconds, 1/sample_rate_hertz)
    hplus = result[:, 0, :]
    hcross = result[:, 1, :]
    
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Load test waveforms:
    test_waveforms = np.load(current_dir / "res/test_waveforms.npy")
    
    # Assert current waveforms are equal:
    assert np.allclose(result, test_waveforms), \
        "Warning generated waveforms do not match fixed save"
    
def plot_test_waveforms(
    num_test_plots : int = 10
    ):
    
    # Define reasonable input parameters for an average gravitational wave
    num_waveforms = 10
    sample_rate_hertz = 2048.0
    duration_seconds = 2.0
    
    mass_1_msun = np.random.uniform(30, 30, size = num_waveforms)  # random float between 1 and 30
    mass_2_msun = np.random.uniform(10, 10, size = num_waveforms)
    inclination_radians = np.random.uniform(0, 0, size = num_waveforms)
    distance_mpc = np.random.uniform(10.0, 10.0, size = num_waveforms)
    reference_orbital_phase_in = np.random.uniform(0, 0, size = num_waveforms)
    ascending_node_longitude = np.random.uniform(0, 0, size = num_waveforms)
    eccentricity = np.random.uniform(0, 0.0, size = num_waveforms)
    mean_periastron_anomaly = np.random.uniform(0, 0, size = num_waveforms)
    spin_1_in = np.random.uniform(0, 0, size = num_waveforms*3)
    spin_2_in = np.random.uniform(0, 0, size = num_waveforms*3)
    
    # Call generatePhenom function
    result = imrphenomd(
        num_waveforms,
        sample_rate_hertz,
        duration_seconds,
        mass_1_msun,
        mass_2_msun,
        inclination_radians,
        distance_mpc,
        reference_orbital_phase_in,
        ascending_node_longitude,
        eccentricity,
        mean_periastron_anomaly,
        spin_1_in,
        spin_2_in
    )  
    result = np.array(result.tolist())
        
    # Assume that result corresponds to pairs of gravitational wave polarisations plus and cross
    times = np.arange(0, duration_seconds, 1/sample_rate_hertz)
    hplus = result[:, 0, :]
    hcross = result[:, 1, :]
    
    figures = []  # list to store all figures

    for i in range(num_waveforms):
        
        # Prepare data for bokeh
        data = ColumnDataSource(data=dict(
            time=times,
            plus=hplus[i],
            cross=hcross[i]
        ))
        
        # Create a new plot
        p = figure(
            title=f"Gravitational Wave Polarisation {i}", 
            x_axis_label='Time (s)', 
            y_axis_label='Strain'
        )

        # Add polarisation traces
        p.line(
            'time', 
            'plus', 
            source=data, 
            legend_label="Plus Polarisation",
            line_color="blue"
        )
        p.line(
            'time', 
            'cross', 
            source=data, 
            legend_label="Cross Polarisation", 
            line_color="red"
        )

        # Move the legend to the upper left corner
        p.legend.location = "top_left"

        # Append to the figures list
        figures.append(p)
    
    # Get directory outside git repo:
    grandparent_directory_path = Path(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
        )
    )
    
    output_directory_path = grandparent_directory_path / "cuphenom_outputs/"
    
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)
        
    output_file_name = output_directory_path / "example_plots.html"
    
    # Output to a static HTML file
    output_file(output_file_name)

    # Arrange plots in a grid, where each row has plots from the figures list
    grid = gridplot([figures])

    # Show the results
    show(grid)

def speed_test_generate_phenom(num_tests=100):

    # Prepare data storage for bokeh
    times = []
    runtimes = []
    
    # Define random input parameters
    num_waveforms = 1000
    sample_rate_hertz = 2048.0
    duration_seconds = 2.0
        
    pbar = tqdm(total=num_tests)
    
    mass_1_msun = np.random.uniform(5, 95, size = num_waveforms)  # random float between 1 and 30
    mass_2_msun = np.random.uniform(5, 95, size = num_waveforms)
    inclination_radians = np.random.uniform(0, np.pi, size = num_waveforms)
    distance_mpc = np.random.uniform(10.0, 1000.0, size = num_waveforms)
    reference_orbital_phase_in = np.random.uniform(0, np.pi*2, size = num_waveforms)
    ascending_node_longitude = np.random.uniform(0, np.pi*2, size = num_waveforms)
    eccentricity = np.random.uniform(0, 0.1, size = num_waveforms)
    mean_periastron_anomaly = np.random.uniform(0, np.pi*2, size = num_waveforms)
    spin_1_in = np.random.uniform(-0.5, 0.5, size = num_waveforms*3)
    spin_2_in = np.random.uniform(-0.5, 0.5, size = num_waveforms*3)

    for _ in range(num_tests // num_waveforms):
        
        # Start the timer
        start_time = time.time()
        
        # Call generatePhenom function
        result = imrphenomd(
            num_waveforms,
            sample_rate_hertz,
            duration_seconds,
            mass_1_msun,
            mass_2_msun,
            inclination_radians,
            distance_mpc,
            reference_orbital_phase_in,
            ascending_node_longitude,
            eccentricity,
            mean_periastron_anomaly,
            spin_1_in,
            spin_2_in
        )

        # Record the runtime
        runtime = time.time() - start_time
        times.append(start_time)
        runtimes.append(runtime)
        
        pbar.update(num_waveforms)
    
    pbar.close()
        
    logging.info("Runtimes", np.sum(runtimes))

    # Prepare data for bokeh
    data = ColumnDataSource(data=dict(
        time=times,
        runtime=runtimes,
    ))

    # Create a new plot
    p = figure(
        title="GeneratePhenom Run Time",
        x_axis_label='Start Time (s)',
        y_axis_label='Run Time (s)'
    )

    # Add runtime traces
    p.circle(
        'time', 
        'runtime', 
        source=data, 
        legend_label="Runtime", 
        line_color="blue"
    )

    # Move the legend to the upper left corner
    p.legend.location = "top_left"
    
    # Get directory outside git repo:
    grandparent_directory_path = Path(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
        )
    )
    
    output_directory_path = grandparent_directory_path / "cuphenom_outputs/"
    
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)
        
    output_file_name = output_directory_path / "runtimes.html"
    

    # Output to static HTML file
    output_file(output_file_name)

    # Show the results
    show(p)

if __name__ == "__main__":
    
    device_num = "2"
    
    # Later include auto GPU selector
    try:
        # Set the device number for CUDA to recognize.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    except Exception as e:
        logging.error(
            f"Failed to set CUDA_VISIBLE_DEVICES environment variable: {e}"
        )
        raise
    
    # Call the test function
    test_generate_phenom()
    speed_test_generate_phenom(num_tests=100000)
    plot_test_waveforms()