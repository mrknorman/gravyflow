from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
import h5py
import logging
from copy import deepcopy

import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow.data.experimental import AutoShardPolicy
from tensorflow.keras.callbacks import Callback
from bokeh.embed import components, file_html
from bokeh.io import export_png, output_file, save
from bokeh.layouts import column, gridplot
from bokeh.models import (ColumnDataSource, CustomJS, Dropdown, HoverTool, 
                          Legend, LogAxis, LogTicker, Range1d, Slider, Select,
                         Div)
from bokeh.plotting import figure, show
from bokeh.resources import INLINE, Resources
from bokeh.palettes import Bright

from .dataset import (get_ifo_dataset, extract_data_from_indicies, 
                      ReturnVariables)
from .plotting import generate_strain_plot, generate_spectrogram
from .injection import WaveformGenerator, WaveformParameters
from .maths import Distribution, DistributionType

def calculate_efficiency_scores(
        model : tf.keras.Model, 
        dataset_args : Dict[str, Union[float, List, int]],
        num_examples_per_batch : int = 32,
        max_snr : float = 20.0,
        num_snr_steps : int = 21,
        num_examples_per_snr_step : int = 2048,
    ) -> Dict:
    
    """
    Calculate the Efficiency scores for a given model.

    Parameters
    ----------
    model : tf.keras.Model
        The model used to predict FAR scores.
    dataset_args : dict
        Dictionary containing options for dataset generator.
    num_examples_per_batch : int, optional
        The number of examples per batch, default is 32.
    max_snr: float, optional
        The max SNR value to generate an efficiency score for, default is 20.0.
    num_snr_steps: int, optional 
        The number of snr values at which to generate and efficiency score,
        default is 21.
    num_examples_per_snr_step : float, optional
        The number of examples to be used for each efficiency score calculation, 
        default is 2048.
    
    Returns
    -------
    snr_array : np.ndarray
        Array of SNRs ar which the efficieny is calculated
    efficiency_scores : np.ndarray
        The calculated efficiency scores.
    """
    
    # Make copy of generator args so original is not affected:
    dataset_args = deepcopy(dataset_args)
        
    # Integer arguments are integers:
    num_examples_per_snr_step = int(num_examples_per_snr_step)
    num_examples_per_batch = int(num_examples_per_batch)
    num_snr_steps = int(num_snr_steps)
    
    # Calculate number of batches required given batch size:
    num_examples = num_examples_per_snr_step*num_snr_steps
    num_batches = num_examples // num_examples_per_batch
    
    # Generate array of snr values used in dataset generation:
    efficiency_snrs = np.linspace(0.0, max_snr, num_snr_steps)
    
    snr_values = \
        np.repeat(
            efficiency_snrs,
            num_examples_per_snr_step
        )
    
    #Ensure injection generators is list for subsequent logic:
    if not isinstance(dataset_args["injection_generators"], list):
        dataset_args["injection_generators"] = \
            [dataset_args["injection_generators"]]
    
    # Ensure dataset is full of injections:
    dataset_args["num_examples_per_batch"] = num_examples_per_batch
    dataset_args["output_variables"] = [ReturnVariables.INJECTION_MASKS]
    dataset_args["injection_generators"][0].injection_chance = 1.0
    dataset_args["injection_generators"][0].snr = snr_values
    
    # Initlize generator:
    dataset : tf.data.Dataset = get_ifo_dataset(
            **dataset_args
        ).take(num_batches)

    # Process all examples in one go:
    combined_scores = model.predict(dataset, steps=num_batches, verbose=2)
    
    # Split predictions back into separate arrays for each SNR level:
    scores = [ 
        combined_scores[
            index * num_examples_per_snr_step : 
            (index + 1) * num_examples_per_snr_step
        ] for index in range(num_snr_steps)
    ]
    
    return {"snrs" : efficiency_snrs, "scores": np.array(scores)}

def calculate_far_scores(
        model : tf.keras.Model, 
        dataset_args : dict, 
        num_examples_per_batch : int = 32,  
        num_examples : int = 1E5
    ) -> np.ndarray:
    
    """
    Calculate the False Alarm Rate (FAR) scores for a given model.

    Parameters
    ----------
    model : tf.keras.Model
        The model used to predict FAR scores.
    dataset_args : dict
        Dictionary containing options for dataset generator.
    num_examples_per_batch : int, optional
        The number of examples per batch, default is 32.
    num_examples : float, optional
        The total number of examples to be used, default is 1E5.
    
    Returns
    -------
    far_scores : np.ndarray
        The calculated FAR scores.
    """
    
    # Make copy of generator args so original is not affected:
    dataset_args = deepcopy(dataset_args)
    
    # Integer arguments are integers:
    num_examples = int(num_examples)
    num_examples_per_batch = int(num_examples_per_batch)
    
    # Calculate number of batches required given batch size:
    num_batches = num_examples // num_examples_per_batch

    # Setting options for data distribution:
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    
    # Ensure dataset has no injections:
    dataset_args["num_examples_per_batch"] = num_examples_per_batch
    dataset_args["injection_generators"] = []
    dataset_args["output_variables"] = []

    # Initlize generator:
    dataset : tf.data.Dataset = get_ifo_dataset(
            **dataset_args
        ).take(num_batches)
        
    # Predict the scores and get the second column ([:, 1]):
    far_scores = model.predict(dataset, steps = num_batches, verbose=2)
    
    return far_scores

def calculate_far_score_thresholds(
    far_scores: np.ndarray, 
    onsource_duration_seconds: float,
    fars: np.ndarray
) -> Dict[float, Tuple[float, float]]:

    """
    Calculate the score thresholds for False Alarm Rate (FAR).

    Parameters
    ----------
    far_scores : np.ndarray
        The FAR scores calculated previously.
    onsource_duration_seconds : float
        The duration of onsource in seconds.
    model_data : np.ndarray
        The data used to train the model.
    fars : np.ndarray
        Array of false alarm rates.

    Returns
    -------
    score_thresholds : Dict[float, Tuple[float, float]]
        Dictionary of false alarm rates and their corresponding score 
        thresholds.

    """
    # Sorting the FAR scores in descending order
    far_scores = np.sort(far_scores)[::-1]

    # Calculating the total number of seconds
    total_num_seconds = len(far_scores) * onsource_duration_seconds

    # Creating the far axis
    far_axis = (np.arange(total_num_seconds) + 1) / total_num_seconds
    
    # Find the indexes of the closest FAR values in the far_axis
    idxs = np.abs(np.subtract.outer(far_axis, fars)).argmin(axis=0)
    # Find the indexes of the closest scores in the far_scores
    idxs = np.abs(
        np.subtract.outer(far_scores, far_scores[idxs])
    ).argmin(axis=0)

    # Build the score thresholds dictionary
    score_thresholds = {
        far: (far, far_scores[idx]) for far, idx in zip(fars, idxs)
    }

    # If any score is 1, set the corresponding threshold to 1.1
    for far, (_, score) in score_thresholds.items():
        if score == 1:
            score_thresholds[far] = (far, 1.1)

    return score_thresholds

@tf.function
def roc_curve_and_auc(
        y_true, 
        y_scores, 
        chunk_size=500
    ):
    num_thresholds = 1000
    # Use logspace with a range between 0 and 6, which corresponds to values 
    # between 1 and 1e-6:
    log_thresholds = tf.exp(tf.linspace(0, -6, num_thresholds))
    # Generate thresholds focusing on values close to 1
    thresholds = 1 - log_thresholds
    
    thresholds = tf.cast(thresholds, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    num_samples = y_true.shape[0]
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    # Initialize accumulators for true positives, false positives, true 
    # negatives, and false negatives
    tp_acc = tf.zeros(num_thresholds, dtype=tf.float32)
    fp_acc = tf.zeros(num_thresholds, dtype=tf.float32)
    fn_acc = tf.zeros(num_thresholds, dtype=tf.float32)
    tn_acc = tf.zeros(num_thresholds, dtype=tf.float32)

    # Process data in chunks
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, num_samples)

        y_true_chunk = y_true[start_idx:end_idx]
        y_scores_chunk = y_scores[start_idx:end_idx]

        y_pred = tf.expand_dims(y_scores_chunk, 1) >= thresholds
        y_pred = tf.cast(y_pred, tf.float32)

        y_true_chunk = tf.expand_dims(y_true_chunk, axis=-1)
        tp = tf.reduce_sum(y_true_chunk * y_pred, axis=0)
        fp = tf.reduce_sum((1 - y_true_chunk) * y_pred, axis=0)
        fn = tf.reduce_sum(y_true_chunk * (1 - y_pred), axis=0)
        tn = tf.reduce_sum((1 - y_true_chunk) * (1 - y_pred), axis=0)

        # Update accumulators
        tp_acc += tp
        fp_acc += fp
        fn_acc += fn
        tn_acc += tn

    tpr = tp_acc / (tp_acc + fn_acc)
    fpr = fp_acc / (fp_acc + tn_acc)

    auc = tf.reduce_sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + tpr[1:])) / 2

    return fpr, tpr, auc

class CaptureWorstPredictions(Callback):
    def __init__(self, n_worst=10):
        super().__init__()
        self.n_worst = n_worst
        self.all_scores = []
        self.all_indices = []

    def on_predict_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_predictions = logs.get('outputs')
        
        if batch_predictions is not None:
            # Add scores and indices to the global list
            scores = batch_predictions.tolist()
            indices = list(range(batch * len(scores), (batch + 1) * len(scores)))

            self.all_scores.extend(scores)
            self.all_indices.extend(indices)
        else:
            print("Warning: 'outputs' not found in logs for batch", batch)

    def on_predict_end(self, logs=None):
        # Sort the global list based on scores to get the worst predictions
        sorted_indices = np.argsort(self.all_scores)[:self.n_worst]
        self.worst_global_indices = np.array(self.all_indices)[sorted_indices]
        self.worst_scores = np.array(self.all_scores)[sorted_indices]

def calculate_roc(    
    model: tf.keras.Model,
    dataset_args : dict,
    num_examples_per_batch: int = 32,
    num_examples: int = 1.0E5
    ) -> dict:
    
    """
    Calculate the ROC curve for a given model.

    Parameters
    ----------
    model : tf.keras.Model
        The model used to predict FAR scores.
    dataset_args : dict
        Dictionary containing options for dataset generator.        
    num_examples_per_batch : int, optional
        The number of examples per batch, default is 32.
    num_examples : float, optional
        The total number of examples to be used, default is 1E5.
    
    Returns
    -------
    roc_data : dict
        Dict containing {"fpr" : fpr, "tpr" : tpr, "auc" : auc}
    where:
    
    fpr : np.ndarray
        An array of false positive rates
    tpr : np.ndarray
        An array of true positive rates
    auc : float
        The area under the roc curve
    """
        
    # Make copy of generator args so original is not affected:
    dataset_args = deepcopy(dataset_args)
    
    # Integer arguments are integers:
    num_examples = int(num_examples)
    num_examples_per_batch = int(num_examples_per_batch)
    
    # Calculate number of batches required given batch size:
    num_batches = num_examples // num_examples_per_batch
    
    #Ensure injection generators is list for subsequent logic:
    if not isinstance(dataset_args["injection_generators"], list):
        dataset_args["injection_generators"] = \
            [dataset_args["injection_generators"]]
    
    # Ensure dataset has balanced injections:
    dataset_args["num_examples_per_batch"] = num_examples_per_batch
    dataset_args["output_variables"] = [ReturnVariables.INJECTION_MASKS]
    dataset_args["injection_generators"][0].injection_chance = 0.5
    
    # Initlize generator:
    dataset : tf.data.Dataset = get_ifo_dataset(
            **dataset_args
        ).take(num_batches)
    
    # Use .map() to extract the true labels and model inputs
    x_dataset = dataset.map(lambda x, y: x)
    y_true_dataset = dataset.map(
        lambda x, y: tf.cast(
            y[ReturnVariables.INJECTION_MASKS.name][0], tf.int32
        )
    )
        
    # Convert the true labels dataset to a tensor using reduce
    tensor_list = []
    for batch in y_true_dataset:
        tensor_list.append(batch)

    y_true = tf.concat(tensor_list, axis=0)

    # Get the model predictions
    y_scores = model.predict(
        x_dataset, 
        steps = num_batches, 
        verbose = 2
    )
    
    # Calculate the ROC curve and AUC
    fpr, tpr, roc_auc = roc_curve_and_auc(y_true, y_scores)
    
    return {'fpr': fpr.numpy(), 'tpr': tpr.numpy(), 'roc_auc': roc_auc.numpy()}

def calculate_multi_rocs(    
    model: tf.keras.Model,
    dataset_args : dict,
    num_examples_per_batch: int = 32,
    num_examples: int = 1.0E5,
    snr_ranges: list = [
        (8.0, 20.0),
        8.0,
        10.0,
        12.0
    ]
    ) -> dict:
    
    roc_results = {}
    for snr_range in snr_ranges:
        # Make copy of generator args so original is not affected:
        dataset_args = deepcopy(dataset_args)
        
        if isinstance(snr_range, tuple):
            snr = Distribution(
                min_=snr_range[0], 
                max_=snr_range[1],
                type_=DistributionType.UNIFORM
            )
        else:
            snr = Distribution(
                value=snr_range, 
                type_=DistributionType.CONSTANT
            )
        #Ensure injection generators is list for subsequent logic:
        if not isinstance(dataset_args["injection_generators"], list):
            dataset_args["injection_generators"] = \
                [dataset_args["injection_generators"]]
            
        # Set desired injection SNR
        dataset_args["injection_generators"][0].snr = snr
        
        roc_results[str(snr_range)] = \
            calculate_roc(    
                model,
                dataset_args,
                num_examples_per_batch,
                num_examples
            )
    
    return roc_results

def calculate_tar_scores(
        model : tf.keras.Model, 
        dataset_args : dict, 
        num_examples_per_batch : int = 32,  
        snr : int = 20.0,
        num_examples : int = 1E5
    ) -> np.ndarray:
    
    """
    Calculate the True Alarm Rate (FAR) scores for a given model.

    Parameters
    ----------
    model : tf.keras.Model
        The model used to predict FAR scores.
    dataset_args : dict
        Dictionary containing options for dataset generator.
    num_examples_per_batch : int, optional
        The number of examples per batch, default is 32.
    num_examples : float, optional
        The total number of examples to be used, default is 1E5.
    
    Returns
    -------
    far_scores : np.ndarray
        The calculated FAR scores.
    """
    
    # Make copy of generator args so original is not affected:
    dataset_args = deepcopy(dataset_args)
    
    # Integer arguments are integers:
    num_examples = int(num_examples)
    num_examples_per_batch = int(num_examples_per_batch)
    
    # Calculate number of batches required given batch size:
    num_batches = num_examples // num_examples_per_batch
    
    #Ensure injection generators is list for subsequent logic:
    if not isinstance(dataset_args["injection_generators"], list):
        dataset_args["injection_generators"] = \
            [dataset_args["injection_generators"]]
    
    # Ensure dataset is full of injections:
    dataset_args["num_examples_per_batch"] = num_examples_per_batch
    dataset_args["output_variables"] = []
    dataset_args["injection_generators"][0].injection_chance = 0.0
    dataset_args["injection_generators"][0].snr = \
        Distribution(value=snr, type_=DistributionType.CONSTANT)
    
    
    # Initlize generator:
    dataset : tf.data.Dataset = get_ifo_dataset(
            **dataset_args
        ).take(num_batches)
        
    callback = CaptureWorstPredictions(n_worst=10)
    
    model.summary()
    # Get the output layers
    output_layers = [layer.output for layer in model.layers]

    # Print the output shapes of each output layer
    for layer in output_layers:
        print(layer.name, layer.shape)
    
    print(num_examples_per_batch, num_examples)
    print(next(iter(dataset)))
    
    # Predict the scores and get the second column ([:, 1]):
    tar_scores = model.predict(
        dataset, 
        batch_size = num_examples_per_batch,
        callbacks = [callback],
        steps = num_batches, 
        verbose=2
    )
    
    print(num_batches)
    print(tar_scores)
    
    dataset_args["output_variables"] = \
        [
            ReturnVariables.WHITENED_INJECTIONS,
            ReturnVariables.INJECTIONS,
            WaveformParameters.MASS_1_MSUN,
            WaveformParameters.MASS_2_MSUN
        ]
    
    # Initlize generator:
    dataset : tf.data.Dataset = get_ifo_dataset(
            **dataset_args
        ).take(num_batches)
    
    worst_performers = \
        extract_data_from_indicies(
            dataset,
            callback.worst_global_indices,
            num_examples_per_batch
        )
    
    for index, element in enumerate(worst_performers):
        for key in element:
            element[key] = element[key].numpy()
        element["score"] = callback.worst_scores[index]
    
    return tar_scores, worst_performers

def check_equal_duration(
    validators : list
    ):
    
    if not validators:  # Check if list is empty
        return

    # Take the input_duration_seconds property of the first object as reference
    reference_duration = validators[0].input_duration_seconds

    for validator in validators[1:]:  # Start from the second object
        if validator.input_duration_seconds != reference_duration:
            raise ValueError(
                "All validators do not have the same input_duration_seconds "
                "property value."
            )
            
def snake_to_capitalized_spaces(snake_str: str) -> str:
    return ' '.join(word.capitalize() for word in snake_str.split('_'))

def generate_efficiency_curves(
        validators : list,
        fars : np.ndarray,
        colors : list[str] = Bright[7]
    ):
    
    # Check input durations are equal:
    check_equal_duration(validators)
    
    # Unpack values:
    input_duration_seconds = validators[0].input_duration_seconds
    
    plot_width = 800
    plot_height = 600

    p = figure(
        title = "Efficiency Curves",
        width=plot_width,
        height=plot_height,
        x_axis_label="SNR",
        y_axis_label="Accuracy",
        y_range=(0.0, 1.0)  # Set y-axis bounds here
    )

    # Set the initial plot title
    far_keys = list(fars)
    
    legend_items = []
    all_sources = {}
    acc_data = {}
    
    model_names = []
    
    for validator, color in zip(validators, colors):
        
        thresholds = calculate_far_score_thresholds(
            validator.far_scores, 
            input_duration_seconds, 
            fars
        )
        
        # Unpack arrays:
        scores = validator.efficiency_data["scores"]
        snrs = validator.efficiency_data["snrs"]
        name = validator.name
        title = snake_to_capitalized_spaces(name)
        
        model_names.append(title)
        
        acc_all_fars = []
        
        for far_index, far in enumerate(thresholds.keys()):
            threshold = thresholds[far][1]
            actual_far = thresholds[far][0]
            acc = []

            for score in scores:
                score = score[:, 1]
                if threshold != 0:
                    total = np.sum(score >= threshold)
                else:
                    total = np.sum(score > threshold)
                                        
                acc.append(total / len(score))
            
            acc_all_fars.append(acc)
                
        acc_data[name] = acc_all_fars
        source = ColumnDataSource(
            data=dict(x=snrs, y=acc_all_fars[0], name=[title] * len(snrs))
        )
        all_sources[name] = source
        line = p.line(
            x='x', 
            y='y', 
            source=source, 
            line_width=2, 
            line_color=color
        )
        legend_items.append((title, [line]))

    legend = Legend(items=legend_items, location="top_left")
    p.add_layout(legend)
    p.legend.click_policy = "hide"

    hover = HoverTool()
    hover.tooltips = [("Name", "@name"), ("SNR", "@x"), ("Accuracy", "@y")]
    p.add_tools(hover)

    slider = Slider(
        start=0, 
        end=len(fars) - 1, 
        value=0,
        step=1, 
        title=f"False Alarm Rate (FAR): {far_keys[0]}"
    )
    slider.background = 'white'

    callback = CustomJS(args=dict(
        slider=slider, 
        sources=all_sources, 
        plot_title=p.title, 
        acc_data=acc_data,
        thresholds=thresholds,
        model_names=model_names, 
        far_keys=far_keys
    ), code="""
        const far_index = slider.value;
        const far_value = far_keys[far_index];
        
        for (const key in sources) {
            if (sources.hasOwnProperty(key)) {
                const source = sources[key];
                source.data.y = acc_data[key][far_index];
                source.change.emit();
            }
        }
    """)             
    slider.js_on_change('value', callback)

    # Add a separate callback to update the slider's title
    slider_title_callback = \
        CustomJS(
            args=dict(
                slider=slider, 
                far_keys=far_keys
            ), 
            code = \
            """
                const far_index = slider.value;
                const far_value = far_keys[far_index];
                slider.title = 'False Alarm Rate (FAR): ' + far_value;
            """
        )
    slider.js_on_change('value', slider_title_callback)
    
    return p, slider
    
def downsample_data(x, y, num_points):
    """
    Downsample x, y data to a specific number of points using logarithmic 
    interpolation.
    
    Parameters:
        - x: Original x data.
        - y: Original y data.
        - num_points: Number of points in the downsampled data.

    Returns:
        - downsampled_x, downsampled_y: Downsampled data.
    """
    
    if len(x) <= num_points:
        return x, y

    interpolator = interp1d(x, y)
    downsampled_x = np.linspace(min(x), max(x), num_points)
    downsampled_y = interpolator(downsampled_x)
        
    return downsampled_x, downsampled_y
    
def generate_far_curves(
        validators : list,
        colors : list[str] = Bright[7]
    ):
    
    plot_width = 800
    plot_height = 600
    tooltips = [
        ("Name", "@name"),
        ("Score Threshold", "@x"),
        ("False Alarms per Second", "@y"),
    ]

    p = figure(
        title = "False Alarm Rate (FAR) curves",
        width=plot_width,
        height=plot_height,
        x_axis_label="Score Threshold",
        y_axis_label="False Alarms per Second (Hz)",
        tooltips=tooltips,
        x_axis_type="log",
        y_axis_type="log"
    )
        
    max_num_points = 2000

    for color, validator in zip(colors, validators):
        far_scores = validator.far_scores
        
        name = validator.name
        title = snake_to_capitalized_spaces(name)
                
        far_scores = np.sort(far_scores)[::-1]
        total_num_seconds = len(far_scores) * validator.input_duration_seconds
        far_axis = (
                np.arange(total_num_seconds, dtype=float) + 1
            ) / total_num_seconds

        downsampled_far_scores, downsampled_far_axis = \
            downsample_data(far_scores, far_axis, max_num_points)
        
        source = ColumnDataSource(
            data=dict(
                x=downsampled_far_scores, 
                y=downsampled_far_axis,
                name=[title]*len(downsampled_far_scores)
            )
        )
        
        line = p.line(
            "x", 
            "y", 
            source=source, 
            line_color=color,
            line_width = 2,
            legend_label=title
        )

    hover = HoverTool()
    hover.tooltips = tooltips
    p.add_tools(hover)
    
    p.legend.location = "bottom_left"
    p.legend.click_policy = "hide"
    p.legend.click_policy = "hide"
    
    return p

def generate_roc_curves(
    validators: list,
    colors : list[str] = Bright[7]
    ):
    
    p = figure(
        title="Receiver Operating Characteristic (ROC) Curves",
        x_axis_label='False Alarm Rate',
        y_axis_label='Accuracy',
        width=800, 
        height=600,
        x_axis_type='log', 
        x_range=[1e-6, 1], 
        y_range=[0.0, 1.0]
    )
    
    max_num_points = 500
    
    # Initial population key:
    initial_population_key = list(validators[0].roc_data.keys())[0]
    all_sources = {}
    
    for color, validator in zip(colors, validators):
        roc_data = validator.roc_data[initial_population_key]
        name = validator.name
        title = snake_to_capitalized_spaces(name)

        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]
        roc_auc = roc_data["roc_auc"]

        reduced_fpr, reduced_tpr = downsample_data(fpr, tpr, max_num_points)
        source = ColumnDataSource(
            data=dict(
                x=reduced_fpr, 
                y=reduced_tpr, 
                roc_auc=[roc_auc] * len(reduced_fpr))
            )
        all_sources[name] = source
        line = p.line(
            x='x', 
            y='y', 
            source=source,
            color=color, 
            width=2, 
            legend_label=f'{title} (area = {roc_auc:.5f})'
        )
        
        hover = HoverTool(
            tooltips=[
                ("Series", title),
                ("False Positive Rate", "$x{0.0000}"),
                ("True Positive Rate", "$y{0.0000}")
            ],
            renderers=[line]
        )
        p.add_tools(hover)

    p.legend.location = "bottom_right"
    p.legend.click_policy = "hide"
    p.legend.click_policy = "hide"
    
    # Dropdown to select the test population
    populations = list(validators[0].roc_data.keys())
    select = \
        Select(
            title="Test Population:", 
            value=initial_population_key, 
            options=populations
        )
    select.background = 'white'

    # JS code to update the curves when the test population changes
    update_code = """
        const selected_population = cb_obj.value;
        
        for (let name in all_sources) {
            const source = all_sources[name];
            const new_data = all_data[name][selected_population];
            source.data.x = new_data.fpr;
            source.data.y = new_data.tpr;
            source.data.roc_auc = new Array(new_data.fpr.length).fill(new_data.roc_auc);
            source.change.emit();
        }
    """

    # Organize all data in a structured way for JS to easily pick it
    all_data = {}
    for validator in validators:
        name = validator.name
        all_data[name] = {}
        for population, data in validator.roc_data.items():
            fpr, tpr = downsample_data(data["fpr"], data["tpr"], max_num_points)
            all_data[name][population] = {
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': data['roc_auc']
            }

    callback = CustomJS(
        args=
        {
            'all_sources': all_sources, 
            'all_data': all_data
        }, 
        code=update_code
    )
    select.js_on_change('value', callback)
    
    return p, select

def generate_waveform_plot(
    data : dict,
    onsource_duration_seconds : float,
    colors : list = Bright[7]
    ):
    
    p = figure(
        title=f"Worst Performing Input Score: {data['score']}, {data['mass_1_msun']}, {data['mass_2_msun']}",
        x_axis_label='Time Seconds',
        y_axis_label='Strain',
        width=800, 
        height=300
    )
    
    source = ColumnDataSource(
        data=dict(
            x=np.linspace(0,onsource_duration_seconds, len(data['onsource'])), 
            y=data['onsource']
        )
    )
    line = p.line(
        x='x', 
        y='y', 
        source=source,
        color=colors[0], 
        width=2, 
        legend_label=f'Whitened Strain + Injection'
    )
        
    source = ColumnDataSource(
        data=dict(
            x= np.linspace(0,onsource_duration_seconds, len(data['onsource'])),
            y= data['whitened_injections']
        )
    )
    line = p.line(
        x='x', 
        y='y', 
        source=source,
        color=colors[1], 
        width=2, 
        legend_label=f'Whitened Injection'
    )
    
    source = ColumnDataSource(
        data=dict(
            x=np.linspace(0,onsource_duration_seconds, len(data['onsource'])), 
            y=data['injections']*20.0
        )
    )
    line = p.line(
        x='x', 
        y='y', 
        source=source,
        color=colors[2], 
        width=2, 
        legend_label=f'Scaled Raw Injections'
    )
    
    p.legend.location = "bottom_right"
    p.legend.click_policy = "hide"
    p.legend.click_policy = "hide"
    
    return p

class Validator:
    
    @classmethod
    def validate(
        cls, 
        model : tf.keras.Model, 
        name : str,
        dataset_args : dict,
        num_examples_per_batch : int = 32,
        efficiency_config : dict = \
        {
            "max_snr" : 20.0, 
            "num_snr_steps" : 21, 
            "num_examples_per_snr_step" : 2048
        },
        far_config : dict = \
        {
            "num_exammples" : 1.0E5
        },
        roc_config : dict = \
        {
            "num_exammples" : 1.0E5,
            "snr_ranges" :  [
                (8.0, 20),
                8.0,
                10.0,
                12.0
            ]    
        }
    ):
        # Create a new instance without executing any logic
        validator = cls()
        
        validator.name = name
        validator.input_duration_seconds = \
            dataset_args["onsource_duration_seconds"]
        
        logging.info(f"Worst performing inputs for {validator.name}...")
        tar_scores, worst_performers = \
            calculate_tar_scores(
                model, 
                dataset_args, 
                num_examples_per_batch,
                snr=20.0,
                num_examples=1.0E3
            )
        validator.worst_performers = worst_performers
        logging.info(f"Done")
                
        logging.info(f"Calculating efficiency scores for {validator.name}...")
        validator.efficiency_data = \
            calculate_efficiency_scores(
                model, 
                dataset_args,
                num_examples_per_batch,
                **efficiency_config
            )
        logging.info(f"Done")

        logging.info(f"Calculating FAR scores for {validator.name}...")
        validator.far_scores = \
            calculate_far_scores(
                model, 
                dataset_args, 
                num_examples_per_batch,  
                **far_config
            )
        logging.info(f"Done")
        
        logging.info(f"Calculating ROC data for {validator.name}...")
        validator.roc_data = \
            calculate_multi_rocs(    
                model,
                dataset_args,
                num_examples_per_batch,
                **roc_config
            )
        logging.info(f"Done")
                
        return validator

    def save(
        self,
        file_path : Path
    ):
        
        logging.info(f"Saving validation data for {self.name}...")
        
        with h5py.File(file_path, 'w') as h5f:

            # Unpack:
            snrs = self.efficiency_data['snrs']
            efficiency_scores = self.efficiency_data['scores']

            # Save model title: 
            h5f.create_dataset('name', data=self.name.encode())
            h5f.create_dataset(
                'input_duration_seconds', 
                data=self.input_duration_seconds
            )

            # Save efficiency scores:
            eff_group = h5f.create_group('efficiency_data')
            eff_group.create_dataset(f'snrs', data=snrs)
            for i, score in enumerate(efficiency_scores):
                eff_group.create_dataset(f'score_{i}', data=score)

            # Save FAR scores
            far_group = h5f.create_group('far_scores')
            far_group.create_dataset('scores', data=self.far_scores)

            # Save ROC data:
            roc_group = h5f.create_group('roc_data')
            
            roc_data = self.roc_data

            keys_array = [str(item) for item in roc_data.keys()]

            string_dt = h5py.string_dtype(encoding='utf-8')
            keys_array = np.array(keys_array, dtype=string_dt)

            roc_group.create_dataset('keys', data=keys_array)

            for key, value in roc_data.items():
                roc_group.create_dataset(
                    f'{key}_fpr', 
                    data=value['fpr']
                )
                roc_group.create_dataset(
                    f'{key}_tpr', 
                    data=value['tpr']
                )
                roc_group.attrs[f'{key}_roc_auc'] = value['roc_auc']
                
            worst_performers = self.worst_performers
            
            for idx, entry in enumerate(worst_performers):
                group = h5f.create_group(f'worst_perfomers_{idx}')
                for key, value in entry.items():
                    group.create_dataset(key, data=value, dtype=np.float16)
        
            logging.info("Done.")
        
    @classmethod
    def load(
        cls, 
        file_path: Path
    ):
        # Create a new instance without executing any logic
        validator = cls()

        with h5py.File(file_path, 'r') as h5f:
            # Load title:
            validator.name = h5f['name'][()].decode()
            validator.input_duration_seconds = \
                float(h5f['input_duration_seconds'][()])
            logging.info(f"Loading validation data for {validator.name}...")
            
            # Load efficiency scores:
            eff_group = h5f['efficiency_data']
            efficiency_data = {
                'snrs': eff_group['snrs'][:],
                'scores': [
                    eff_group[f'score_{i}'][:] for i in range(len(eff_group) - 1)
                ]
            }

            # Load FAR scores:
            far_scores = h5f['far_scores']['scores'][:]

            # Load ROC data:
            roc_group = h5f['roc_data']
            keys_array = roc_group['keys'][:]
            roc_data = {
                key.decode('utf-8'): {
                    'fpr': roc_group[f'{key.decode("utf-8")}_fpr'][:],
                    'tpr': roc_group[f'{key.decode("utf-8")}_tpr'][:],
                    'roc_auc': roc_group.attrs[f'{key.decode("utf-8")}_roc_auc']
                } for key in keys_array
            }

            # Populate the Validator object's attributes with loaded data:
            validator.efficiency_data = efficiency_data
            validator.far_scores = far_scores
            validator.roc_data = roc_data
            
            worst_performers = []
            for group_name in h5f:
                group_data = {}
                if group_name.startswith(f"worst_performers_"): 
                    for key in h5f[group_name]:
                        group_data[key] = h5f[group_name][key][()]
                    print(group_data)
                    worst_performers.append(group_data)
                
            validator.worst_performers = worst_performers
                                
            logging.info("Done.")

        return validator

    def plot(
        self,
        file_path : Path,
        comparison_validators : list = [],
        fars : np.ndarray = np.logspace(-1, -7, 500)
    ):
        
        validators = comparison_validators + [self]
        
        efficiency_curves, slider = \
            generate_efficiency_curves(
                validators, 
                fars
            )

        far_curves = \
            generate_far_curves(
                validators
            )

        roc_curves, dropdown = \
            generate_roc_curves(
                validators
            )
        
        layout = [
            [dropdown, slider],
            [roc_curves, efficiency_curves], 
            [far_curves, None]
        ]
        
        for waveform in self.worst_performers:
            waveform_plot = generate_waveform_plot(
                waveform,
                self.input_duration_seconds
            )
            layout.append([waveform_plot, None])

        # Define an output path for the dashboard
        output_file(file_path)

        # Arrange the plots in a grid. 
        grid = gridplot(layout)
        
        # Define CSS to make background white
        div = Div(
            text="""
                <style>
                    body {
                        background-color: white !important;
                    }
                </style>
                """
        )

        # Save the combined dashboard
        save(column(div, grid))