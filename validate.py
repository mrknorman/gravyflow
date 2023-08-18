from .dataset import get_ifo_data, O3
from typing import Dict, Tuple, Optional
from pathlib import Path

import tensorflow as tf
import numpy as np

import h5py

import logging

from tensorflow.data.experimental import AutoShardPolicy
from .dataset import get_ifo_data_generator, O3

from itertools import cycle

from bokeh.embed import components, file_html
from bokeh.io import export_png, output_file, save
from bokeh.layouts import column, gridplot
from bokeh.models import (ColumnDataSource, CustomJS, Dropdown, HoverTool, 
                          Legend, LogAxis, LogTicker, Range1d, Slider)
from bokeh.plotting import figure, show
from bokeh.resources import INLINE, Resources

from scipy.interpolate import interp1d

def calculate_efficiency_scores(
        model : tf.keras.Model, 
        generator_args : dict,
        num_examples_per_batch : int = 32,
        max_snr : float = 20.0,
        num_snr_steps : int = 21,
        num_examples_per_snr_step : int = 2048,
    ) -> dict:
    
    """
    Calculate the Efficiency scores for a given model.

    Parameters
    ----------
    model : tf.keras.Model
        The model used to predict FAR scores.
    generator_args : dict
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
    generator_args = generator_args.copy()
    
    print(num_examples_per_snr_step)
    
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
    
    # Setting options for data distribution:
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    
    # Ensure generator args are correctly set up:
    generator_args["num_examples_per_batch"] = num_examples_per_batch
    injection_config = generator_args["injection_configs"][0].copy()
    injection_config.update({
        "injection_chance" : 1.0,
        "snr" : snr_values
    })
    generator_args["injection_configs"] = [injection_config]
    
    # Initlize generator:
    dataset = \
        get_ifo_data_generator(
            **generator_args
        ).with_options(options).take(num_batches)

    # Process all examples in one go:
    combined_scores = model.predict(dataset, steps = num_batches, verbose = 2)
    
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
        generator_args : dict, 
        num_examples_per_batch : int = 32,  
        num_examples : int = 1E5
    ) -> np.ndarray:
    
    """
    Calculate the False Alarm Rate (FAR) scores for a given model.

    Parameters
    ----------
    model : tf.keras.Model
        The model used to predict FAR scores.
    generator_args : dict
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
    generator_args = generator_args.copy()
    
    # Integer arguments are integers:
    num_examples = int(num_examples)
    num_examples_per_batch = int(num_examples_per_batch)
    
    # Calculate number of batches required given batch size:
    num_batches = num_examples // num_examples_per_batch

    # Setting options for data distribution:
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    
    # Ensure dataset has no injections:
    generator_args["num_examples_per_batch"] = num_examples_per_batch
    generator_args["injection_configs"] = []
    generator_args["output_keys"] = []

    # Initlize generator:
    dataset = \
        get_ifo_data_generator(
            **generator_args
        ).with_options(options).take(num_batches)
        
    # Predict the scores and get the second column ([:, 1]):
    far_scores = model.predict(dataset, steps = num_batches, verbose=2)[:, 1]
    
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
        Dictionary of false alarm rates and their corresponding score thresholds.

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
    idxs = np.abs(np.subtract.outer(far_scores, far_scores[idxs])).argmin(axis=0)

    # Build the score thresholds dictionary
    score_thresholds = {far: (far, far_scores[idx]) for far, idx in zip(fars, idxs)}

    # If any score is 1, set the corresponding threshold to 1.1
    for far, (_, score) in score_thresholds.items():
        if score == 1:
            score_thresholds[far] = (far, 1.1)

    return score_thresholds

def validate_far(
    model: tf.keras.Model,
    sample_rate_hertz: int,
    onsource_duration_seconds: float,
    ifo: str,
    time_interval: str = O3,
    num_examples_per_batch: int = 32,
    seed: int = 100,
    num_examples: float = 1E5,
    validation_fars: Optional[np.ndarray] = None
    ):
    """
    Validate the False Alarm Rate (FAR) of the model.

    Parameters
    ----------
    model : tf.keras.Model
        The model used to predict FAR scores.
    sample_rate_hertz : int
        The sample rate in hertz.
    onsource_duration_seconds : float
        The duration of onsource in seconds.
    ifo : str
        The name of the Interferometric Gravitational-Wave Observatory.
    time_interval : str, optional
        The time interval for the IFO data generator, default is 'O3'.
    num_examples_per_batch : int, optional
        The number of examples per batch, default is 32.
    seed : int, optional
        The seed used for random number generation, default is 100.
    num_examples : float, optional
        The total number of examples to be used, default is 1E5.
    validation_fars : np.ndarray, optional
        The array of false alarm rates for validation, default is logspace from 
        1/(num_examples*onsource_duration_seconds) to 1.

    """
    # If validation_fars not provided, generate it from num_examples and 
    # onsource_duration_seconds
    if validation_fars is None:
        start_power = np.log10(1/(num_examples*onsource_duration_seconds))
         # Calculate number of points to cover all integer exponents
        num_points = int(np.ceil(1 - start_power)) + 1 
        validation_fars = np.logspace(start_power, 0, num=num_points)
    
    far_scores = calculate_far_scores(
        model,
        sample_rate_hertz,
        onsource_duration_seconds,
        ifo,
        time_interval,
        num_examples_per_batch,
        seed,
        num_examples
    )
    
    score_thresholds = calculate_far_score_thresholds(
        far_scores, 
        onsource_duration_seconds,
        validation_fars
    )

    return score_thresholds

@tf.function
def roc_curve_and_auc(
        y_true, 
        y_scores, 
        chunk_size=500
    ):
    num_thresholds = 1000
     # Use logspace with a range between 0 and 6, which corresponds to values between 1 and 1e-6
    log_thresholds = tf.exp(tf.linspace(0, -6, num_thresholds))
    # Generate thresholds focusing on values close to 1
    thresholds = 1 - log_thresholds
    
    thresholds = tf.cast(thresholds, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    num_samples = y_true.shape[0]
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    # Initialize accumulators for true positives, false positives, true negatives, and false negatives
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

def calculate_roc(    
    model: tf.keras.Model,
    generator_args : dict,
    num_examples_per_batch: int = 32,
    num_examples: int = 1.0E5
    ) -> dict:
    
    """
    Calculate the ROC curve for a given model.

    Parameters
    ----------
    model : tf.keras.Model
        The model used to predict FAR scores.
    generator_args : dict
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
    generator_args = generator_args.copy()
    
    # Integer arguments are integers:
    num_examples = int(num_examples)
    num_examples_per_batch = int(num_examples_per_batch)
    
    # Calculate number of batches required given batch size:
    num_batches = num_examples // num_examples_per_batch
    
    # Setting options for data distribution
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    
    # Ensure dataset has no injections:
    generator_args["num_examples_per_batch"] = num_examples_per_batch
    injection_config = generator_args["injection_configs"][0].copy()
    injection_config.update({"injection_chance" : 0.5})
    generator_args["injection_configs"] = [injection_config]
    generator_args["output_keys"] = ["injection_masks"]
    
    # Initlize generator:
    dataset = \
        get_ifo_data_generator(
            **generator_args
        ).with_options(options).take(num_batches)
    
    # Use .map() to extract the true labels and model inputs
    x_dataset = dataset.map(lambda x, y: x)
    y_true_dataset = dataset.map(
        lambda x, y: tf.cast(y['injection_masks'][0], tf.int32)
    )
        
    # Convert the true labels dataset to a tensor using reduce
    tensor_list = []
    for batch in y_true_dataset:
        tensor_list.append(batch)

    y_true = tf.concat(tensor_list, axis=0)

    # Get the model predictions
    y_scores = model.predict(x_dataset, steps = num_batches, verbose = 2)[:, 1]

    # Calculate the ROC curve and AUC
    fpr, tpr, roc_auc = roc_curve_and_auc(y_true, y_scores)
    
    return {'fpr': fpr.numpy(), 'tpr': tpr.numpy(), 'roc_auc': roc_auc.numpy()}

def calculate_multi_rocs(    
    model: tf.keras.Model,
    generator_args : dict,
    num_examples_per_batch: int = 32,
    num_examples: int = 1.0E5,
    snr_ranges: list = [
        (8.0, 20),
        8.0,
        10.0,
        12.0
    ]
    ) -> dict:
    
    roc_results = {}
    for snr_range in snr_ranges:
        # Make copy of generator args so original is not affected:
        generator_args = generator_args.copy()
        
        if (type(snr_range) == tuple):
            snr = {
                "min_value"   : snr_range[0], 
                "max_value"   : snr_range[1],
                "distribiton" : "uniform"
            }
        else:
            snr = {
                "value"        : snr_range,
                "distribution" : "constant"
            }
        
        injection_config = generator_args["injection_configs"][0].copy()
        injection_config.update({
            "snr" : snr
        })
        generator_args["injection_configs"] = [injection_config]
        
        roc_results[str(snr_range)] = \
            calculate_roc(    
                model,
                generator_args,
                num_examples_per_batch,
                num_examples
            )
    
    return roc_results

def check_equal_duration(
    validators : list
    ):
    
    if not validators:  # Check if list is empty
        return

    # Take the `input_duration_seconds` property of the first object as reference
    reference_duration = validators[0].input_duration_seconds

    for validator in validators[1:]:  # Start from the second object
        if validator.input_duration_seconds != reference_duration:
            raise ValueError(
                "All validators do not have the same input_duration_seconds "
                "property value."
            )
            
def snake_to_capitalized_spaces(snake_str: str) -> str:
    return ' '.join(word.capitalize() for word in snake_str.split('_'))

def plot_efficiency_curves(
        validators : list,
        fars : np.ndarray,
        output_file_path : Path
    ):
    
    # Check input durations are equal:
    check_equal_duration(validators)
    
    # Unpack values:
    input_duration_seconds = validators[0].input_duration_seconds
    
    plot_width = 800
    plot_height = 600
    colors = cycle(['red', 'blue', 'green', 'orange'])

    p = figure(
        width=plot_width,
        height=plot_height,
        x_axis_label="SNR",
        y_axis_label="Accuracy"
    )

    # Set the initial plot title
    far_keys = list(fars)
    p.title.text = f'FAR: {far_keys[0]}'

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
        line = p.line(x='x', y='y', source=source, line_width=1, line_color=color)
        legend_items.append((title, [line]))

    legend = Legend(items=legend_items, location="bottom_right")
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
        title=f"FAR Index: {far_keys[0]}"
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
        plot_title.text = 'FAR: ' + far_value;

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
    slider_title_callback = CustomJS(args=dict(slider=slider, far_keys=far_keys), code="""
        const far_index = slider.value;
        const far_value = far_keys[far_index];
        slider.title = 'FAR Index: ' + far_value;
    """)
    slider.js_on_change('value', slider_title_callback)

    layout = column(slider, p)
    
    output_file(output_file_path)
    save(layout) 
    
def downsample_data(x, y, num_points):
    """
    Downsample x, y data to a specific number of points using logarithmic interpolation.
    
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
    
def plot_far_curves(
        validators : list,
        output_path : Path
    ):
    
    plot_width = 800
    plot_height = 600
    tooltips = [
        ("Name", "@name"),
        ("Score Threshold", "@x"),
        ("False Alarms per Second", "@y"),
    ]

    p = figure(
        width=plot_width,
        height=plot_height,
        x_axis_label="Score Threshold",
        y_axis_label="False Alarms per Second",
        tooltips=tooltips,
        x_axis_type="log",
        y_axis_type="log"
    )
    
    colors = cycle(['red', 'blue', 'green', 'orange'])
    
    max_num_points = 500

    for color, validator in zip(colors, validators):
        far_scores = validator.far_scores
        
        name = validator.name
        title = snake_to_capitalized_spaces(name)
                
        far_scores = np.sort(far_scores)[::-1]
        total_num_seconds = len(far_scores) * validator.input_duration_seconds
        far_axis = (np.arange(total_num_seconds, dtype=float) + 1) / total_num_seconds

        downsampled_far_scores, downsampled_far_axis = \
            downsample_data(far_scores, far_axis, max_num_points)
        
        source = ColumnDataSource(
            data=dict(x=downsampled_far_scores, y=downsampled_far_axis, name=[title]*len(downsampled_far_scores))
        )
        
        line = p.line("x", "y", source=source, line_color=color, legend_label=title)

    hover = HoverTool()
    hover.tooltips = tooltips
    p.add_tools(hover)
    
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    p.legend.click_policy = "hide"

    output_file(output_path, title="FAR Plot (Logarithmic)")
    save(p)
    
def plot_roc_curves(
    validators : list,
    output_file_path : Path
    ):
    
    p = figure(title="Receiver Operating Characteristic (ROC) Curves",
               x_axis_label='False Positive Rate',
               y_axis_label='True Positive Rate',
               width=800, height=600,
               x_axis_type='log', x_range=[1e-6, 1], y_range=[0, 1])

    p.line(
        [0, 1], 
        [0, 1], 
        color='navy', 
        line_width=2,
        line_dash='dashed', 
        legend_label="Random (area = 0.5)"
    )

    colors = cycle([
        'red', 
        'green', 
        'blue', 
        'purple',
        'orange', 
        'brown', 
        'pink', 
        'cyan',
        'magenta', 
        'yellow']
    )
    
    max_num_points = 500
    
    for color, validator in zip(colors, validators):
        print(validator.roc_data.keys)
        
        roc_data = validator.roc_data["12.0"]
        
        name = validator.name
        title = snake_to_capitalized_spaces(name)

        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]
        roc_auc = roc_data["roc_auc"]
        
        reduced_fpr, reduced_tpr = downsample_data(fpr, tpr, max_num_points)
        source = ColumnDataSource(data=dict(x=reduced_fpr, y=reduced_tpr))
        line = p.line(
            x='x',
            y='y', 
            source=source, 
            color=color,
            width=2, 
            legend_label=f'{title} (area = {roc_auc:.5f})'
        )
        
        hover = HoverTool(tooltips=[("Series", title),
                                     ("False Positive Rate", "$x{0.0000}"),
                                     ("True Positive Rate", "$y{0.0000}")],
                          renderers=[line])
        p.add_tools(hover)

    p.legend.location = "bottom_right"

    output_file(output_file_path)
    save(p)

def calculate_validation_data(
    model : tf.keras.Model, 
    generator_args : dict,
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
    ) -> dict:
    
    logging.info(f"Calculating efficiency scores...")
    efficiency_data = \
        calculate_efficiency_scores(
            model, 
            generator_args,
            num_examples_per_batch,
            **efficiency_config
        )
    logging.info(f"Done")
    
    logging.info(f"Calculating FAR scores...")
    far_scores = \
        calculate_far_scores(
            model, 
            generator_args, 
            num_examples_per_batch,  
            **far_config
        )
    logging.info(f"Done")
    
    logging.info(f"Calculating ROC data...")
    roc_data = \
        calculate_multi_rocs(    
            model,
            generator_args,
            num_examples_per_batch,
            **roc_config
        )
    logging.info(f"Done")
    report = {
        "efficiency_data" : efficiency_data, 
        "far_scores" : far_scores,
        "roc_data" : roc_data
    }
    
    return report

class Validator:
    
    @classmethod
    def validate(
        cls, 
        model : tf.keras.Model, 
        name : str,
        generator_args : dict,
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
            generator_args["onsource_duration_seconds"]
        
        logging.info(f"Calculating efficiency scores for {validator.name}...")
        validator.efficiency_data = \
            calculate_efficiency_scores(
                model, 
                generator_args,
                num_examples_per_batch,
                **efficiency_config
            )
        logging.info(f"Done")

        logging.info(f"Calculating FAR scores for {validator.name}...")
        validator.far_scores = \
            calculate_far_scores(
                model, 
                generator_args, 
                num_examples_per_batch,  
                **far_config
            )
        logging.info(f"Done")

        logging.info(f"Calculating ROC data for {validator.name}...")
        validator.roc_data = \
            calculate_multi_rocs(    
                model,
                generator_args,
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
            validator.input_duration_seconds = float(h5f['input_duration_seconds'][()])
            logging.info(f"Loading validation data for {validator.name}...")
            
            # Load efficiency scores:
            eff_group = h5f['efficiency_data']
            efficiency_data = {
                'snrs': eff_group['snrs'][:],
                'scores': [eff_group[f'score_{i}'][:] for i in range(len(eff_group) - 1)]
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
            
            logging.info("Done.")

        return validator

    @classmethod
    def plot(
        cls,
        validators : list
    ):
        pass

        
        
        



