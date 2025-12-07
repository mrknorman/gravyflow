from pathlib import Path
import sys
from typing import Dict, Tuple, Optional, List, Union
import h5py
import logging
from copy import deepcopy
from itertools import cycle
import math

import numpy as np
from scipy.interpolate import interp1d
import keras
from keras import ops
import jax.numpy as jnp
from jax import jit

import keras
from keras.callbacks import Callback
from bokeh.embed import components, file_html
from bokeh.io import export_png, output_file, save
from bokeh.layouts import column, gridplot
from bokeh.models import (ColumnDataSource, CustomJS, Dropdown, HoverTool, 
                          Legend, LogAxis, LogTicker, Range1d, Slider, Select,
                         Div)
from bokeh.plotting import figure, show
from bokeh.resources import INLINE, Resources
from bokeh.palettes import Bright

import gravyflow as gf

def pad_with_random_values(scores):
    # Determine the maximum length among all numpy arrays of 2-element arrays in scores
    max_length = max(len(score) for score in scores)
    
    def pad_array(arr, max_length):
        current_length = len(arr)
        if current_length < max_length:
            # Calculate the number of 2-element arrays needed
            num_arrays_needed = max_length - current_length
            # Randomly sample indices from the array
            sampled_indices = np.random.randint(0, current_length, size=num_arrays_needed)
            # Use the indices to select 2-element arrays to duplicate
            sampled_arrays = arr[sampled_indices]
            # Concatenate the original array with the sampled ones to pad it
            arr = np.concatenate([arr, sampled_arrays], axis=0)
        return arr
    
    # Apply padding to each numpy array in scores
    padded_scores = np.array([pad_array(score, max_length) for score in scores])
    
    return padded_scores

def calculate_efficiency_scores(
        model : keras.Model, 
        dataset_args : Dict[str, Union[float, List, int]],
        logger,
        file_path : Path = None,
        num_examples_per_batch : int = 32,
        max_scaling : float = 20.0,
        num_scaling_steps : int = 21,
        num_examples_per_scaling_step : int = 2048,
        heart : gf.Heart = None
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
    max_scaling: float, optional
        The max scaling value to generate an efficiency score for, default is 20.0.
    num_scaling_steps: int, optional 
        The number of scaling values at which to generate and efficiency score,
        default is 21.
    num_examples_per_scaling_step : float, optional
        The number of examples to be used for each efficiency score calculation, 
        default is 2048.
    
    Returns
    -------
    scaling_array : np.ndarray
        Array of scalings ar which the efficieny is calculated
    efficiency_scores : np.ndarray
        The calculated efficiency scores.
    """
    
    # Make copy of generator args so original is not affected:
    dataset_args = deepcopy(dataset_args)
        
    # Integer arguments are integers:
    num_examples_per_scaling_step = int(num_examples_per_scaling_step)
    num_examples_per_batch = int(num_examples_per_batch)
    num_scaling_steps = int(num_scaling_steps)
    
    # Calculate number of batches required given batch size:
    num_examples = num_examples_per_scaling_step*num_scaling_steps
    num_batches = math.ceil(num_examples / num_examples_per_batch)

    # Generate array of scaling values used in dataset generation:
    efficiency_scalings = np.linspace(0.0, max_scaling, num_scaling_steps)
    
    scaling_values = np.repeat(
        efficiency_scalings,
        num_examples_per_scaling_step
    )
    
    #Ensure injection generators is list for subsequent logic:
    if not isinstance(dataset_args["waveform_generators"], list):
        dataset_args["waveform_generators"] = \
            [dataset_args["waveform_generators"]]
    
    # Ensure dataset is full of injections:
    dataset_args["num_examples_per_batch"] = num_examples_per_batch
    dataset_args["output_variables"] = [gf.ReturnVariables.INJECTION_MASKS]
    dataset_args["waveform_generators"][0].injection_chance = 1.0
    dataset_args["waveform_generators"][0].scaling_method.value = scaling_values
    
    # Initialize generator:
    dataset = gf.Dataset(
        **dataset_args
    )
    callbacks = []
    if heart is not None:
        callbacks += [gf.HeartbeatCallback(heart)]

    combined_scores = None
    while combined_scores is None:
        try:
            verbose : int= 1
            if gf.is_redirected():
                verbose : int = 2

            combined_scores = model.predict(
                dataset, 
                steps=num_batches, 
                callbacks=callbacks,
                verbose=verbose
            )

            try:
                # Split predictions back into separate arrays for each scaling level:
                scores = [ 
                    combined_scores[
                        index * num_examples_per_scaling_step : 
                        (index + 1) * num_examples_per_scaling_step
                    ] for index in range(num_scaling_steps)
                ]

                scores = pad_with_random_values(scores)

            except Exception as e:
                raise Exception(f"Error splitting efficiency scores: {e}.")

        except Exception as e:
            logging.error(f"Error {e} calculating efficiency scores! Retrying.")
            combined_scores = None
            continue    

    if file_path is not None:
        with gf.open_hdf5_file(
            file_path, 
            logger, 
            mode = "a"
        ) as validation_file:
        
            # Unpack:
            scalings = efficiency_scalings

            # Save efficiency scores:
            if scores is not None:
                if 'efficiency_data' not in validation_file:
                    eff_group = validation_file.create_group('efficiency_data')
                else:
                    eff_group = validation_file['efficiency_data']

                if 'scalings' not in eff_group:
                    eff_group.create_dataset(f'scalings', data=scalings)
                else:
                    del eff_group['scalings']
                    eff_group.create_dataset(f'scalings', data=scalings)
                
                for i, score in enumerate(scores):

                    if f'score_{i}' not in eff_group:
                        eff_group.create_dataset(f'score_{i}', data=score)
                    else: 
                        del eff_group[f'score_{i}']
                        eff_group.create_dataset(f'score_{i}', data=score)
    
    return {"scalings" : efficiency_scalings, "scores": scores}

def calculate_far_scores(
        model : keras.Model, 
        dataset_args : dict, 
        logger,
        file_path : Path,
        num_examples_per_batch : int = 32,  
        num_seconds : float = 1E5,
        heart : gf.Heart = None
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
    num_examples = int(num_seconds/dataset_args["onsource_duration_seconds"])
    num_examples_per_batch = int(num_examples_per_batch)
    
    # Calculate number of batches required given batch size:
    num_batches = math.ceil(num_examples / num_examples_per_batch)

    # Ensure dataset has no injections:
    dataset_args["num_examples_per_batch"] = num_examples_per_batch
    dataset_args["waveform_generators"] = []
    dataset_args["output_variables"] = []

    # Initialize generator:
    dataset = gf.Dataset(
            **dataset_args
        )

    callbacks = []
    if heart is not None:
        callbacks += [gf.HeartbeatCallback(heart)]
        
    # Predict the scores and get the second column ([:, 1]):

    far_scores = None
    while far_scores is None:
        try:
            verbose : int= 1
            if gf.is_redirected():
                verbose : int = 2

            far_scores = model.predict(
                dataset, 
                steps=num_batches, 
                callbacks=callbacks,
                verbose=verbose
            )

            try:
                far_scores = far_scores[:,0]
            except:
                raise Exception(f"Error slicing FAR scores: {e}")

        except Exception as e:
            logging.error("Error calculating FAR scores because {e}. Retrying.")
            far_scores = None
            continue

    if file_path is not None:
        with gf.open_hdf5_file(
            file_path, 
            logger, 
            mode = "a"
        ) as validation_file:

            if "far_scores" not in validation_file:

                logger.info("Saving FAR Scores!")

                far_group = validation_file.create_group('far_scores')
                far_group.create_dataset('scores', data=far_scores)

    return far_scores

import numpy as np
from typing import Dict, Tuple

def calculate_far_score_thresholds(
    true_negative_scores: np.ndarray, 
    onsource_duration_seconds: float,
    fars: np.ndarray
) -> Dict[float, Tuple[float, float]]:
    """
    Calculate the score thresholds for False Alarm Rate (FAR) using interpolation,
    with true_negative_scores sorted in ascending order for compatibility with np.searchsorted.

    Parameters
    ----------
    true_negative_scores : np.ndarray
        The scores of the model when fed examples of noise only.
    onsource_duration_seconds : float
        The duration of onsource in seconds.
    fars : np.ndarray
        Array of false alarm rates.

    Returns
    -------
    score_thresholds : Dict[float, Tuple[float, float]]
        Dictionary of false alarm rates and their corresponding interpolated 
        score thresholds.
    """
    # Ensure true_negative_scores is in ascending order for correct interpolation
    sorted_scores = np.sort(true_negative_scores)[::-1]

    # Calculate the FAR for each threshold, directly reflecting the sorted order
    n_scores = len(sorted_scores)
    cumulative_far = np.arange(1, n_scores + 1) / (n_scores * onsource_duration_seconds)

    # Adjusting min_far and max_far based on the sorted order of scores and FAR calculation
    min_far = cumulative_far[0]  # Corresponds to the highest score due to ascending order
    max_far = cumulative_far[-1]  # Corresponds to the lowest score

    # Build the score thresholds dictionary with interpolation
    score_thresholds = {}
    for far in fars:
        if far > max_far:
            # Set to 1.1 if the desired FAR is higher than the maximum achievable FAR
            score_thresholds[far] = (far, cumulative_far[-1])
        elif far < min_far:
            # Also set to 1.1 if the desired FAR is lower than the minimum achievable FAR
            score_thresholds[far] = (far, 1.1)
        else:
            # Interpolating the score threshold for the given FAR
            interpolated_score = np.interp(far, cumulative_far, sorted_scores, right=1.1)
            score_thresholds[far] = (far, interpolated_score)

    return score_thresholds

@jit
def roc_curve_and_auc(
        y_true, 
        y_scores, 
        chunk_size=512
    ):
    
    num_thresholds = 512
    # Use logspace with a range between 0 and 6, which corresponds to values 
    # between 1 and 1e-6:
    log_thresholds = jnp.exp(jnp.linspace(0, -6, num_thresholds))
    # Generate thresholds focusing on values close to 1
    thresholds = 1 - log_thresholds
    
    thresholds = jnp.array(thresholds, dtype=jnp.float32)
    y_true = jnp.array(y_true, dtype=jnp.float32)

    num_samples = y_true.shape[0]
    num_chunks = num_samples // chunk_size 
    
    # Handle case where num_samples < chunk_size
    if num_chunks == 0:
        num_chunks = 1

    # Initialize accumulators for true positives, false positives, true 
    # negatives, and false negatives
    tp_acc = jnp.zeros(num_thresholds, dtype=jnp.float32)
    fp_acc = jnp.zeros(num_thresholds, dtype=jnp.float32)
    fn_acc = jnp.zeros(num_thresholds, dtype=jnp.float32)
    tn_acc = jnp.zeros(num_thresholds, dtype=jnp.float32)

    # Process data in chunks
    # Note: JAX loops are different, but for simplicity in this script we use python loop
    # or we can use lax.scan if performance is critical. 
    # Given this is validation, python loop over chunks might be acceptable if not too slow.
    # But strictly, JAX arrays are immutable. In-place update += won't work as expected if not careful.
    # We should use a loop that updates state.
    
    def body_fun(i, val):
        tp_acc, fp_acc, fn_acc, tn_acc = val
        start_idx = i * chunk_size
        end_idx = jnp.minimum((i + 1) * chunk_size, num_samples)

        y_true_chunk = y_true[start_idx:end_idx]
        y_scores_chunk = y_scores[start_idx:end_idx]

        y_pred = jnp.expand_dims(y_scores_chunk, 1) >= thresholds
        y_pred = jnp.array(y_pred, dtype=jnp.float32)

        y_true_chunk = jnp.expand_dims(y_true_chunk, axis=-1)
        tp = jnp.sum(y_true_chunk * y_pred, axis=0)
        fp = jnp.sum((1 - y_true_chunk) * y_pred, axis=0)
        fn = jnp.sum(y_true_chunk * (1 - y_pred), axis=0)
        tn = jnp.sum((1 - y_true_chunk) * (1 - y_pred), axis=0)
        
        return (tp_acc + tp, fp_acc + fp, fn_acc + fn, tn_acc + tn)

    # We need to handle the loop. Since num_chunks is dynamic (python int), we can use python range.
    # But for JIT, we might want lax.fori_loop.
    # However, num_chunks depends on input shape.
    
    # Let's just do vectorized op if memory allows, or python loop.
    # If chunk_size is small, python loop is slow.
    # If we assume memory is sufficient, we can do it all at once?
    # 1e5 samples * 512 thresholds * 4 bytes = 200MB. It fits in memory.
    # So we can remove chunking for JAX implementation to simplify and speed up.
    
    y_pred = jnp.expand_dims(y_scores, 1) >= thresholds
    y_pred = jnp.array(y_pred, dtype=jnp.float32)
    y_true_expanded = jnp.expand_dims(y_true, axis=-1)
    
    tp_acc = jnp.sum(y_true_expanded * y_pred, axis=0)
    fp_acc = jnp.sum((1 - y_true_expanded) * y_pred, axis=0)
    fn_acc = jnp.sum(y_true_expanded * (1 - y_pred), axis=0)
    tn_acc = jnp.sum((1 - y_true_expanded) * (1 - y_pred), axis=0)
    
    tpr = tp_acc / (tp_acc + fn_acc)
    fpr = fp_acc / (fp_acc + tn_acc)

    auc = jnp.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + tpr[1:])) / 2

    return fpr, tpr, auc



def calculate_roc(    
        model: keras.Model,
        dataset_args : dict,
        num_examples_per_batch: int = 32,
        num_examples: int = 1.0E5,
        heart : gf.Heart = None
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
    num_batches = math.ceil(num_examples / num_examples_per_batch)
    
    #Ensure injection generators is list for subsequent logic:
    if not isinstance(dataset_args["waveform_generators"], list):
        dataset_args["waveform_generators"] = \
            [dataset_args["waveform_generators"]]
    
    # Ensure dataset has balanced injections:
    dataset_args["num_examples_per_batch"] = num_examples_per_batch
    dataset_args["output_variables"] = [gf.ReturnVariables.INJECTION_MASKS]
    dataset_args["waveform_generators"][0].injection_chance = 0.5
    
    mask_history = []
    # Initialize generators
    dataset = gf.Dataset(
            **dataset_args,
            mask_history=mask_history
        )
    
    callbacks = []
    if heart is not None:
        callbacks += [gf.HeartbeatCallback(heart)]

    # Get the model predictions and true labels via manual iteration
    y_scores_list = []
    y_true_list = []
    
    try:
        verbose = 1
        if gf.is_redirected():
            verbose = 2
            
        # Manual iteration to capture y_true
        for _ in range(num_batches):
            x, y = dataset[0] # Get a batch
            scores = model.predict_on_batch(x)
            y_scores_list.append(scores)
            y_true_list.append(y[gf.ReturnVariables.INJECTION_MASKS.name])
            
            if heart is not None:
                heart.beat()
                
    except Exception as e:
        logging.error(f"Error calculating ROC scores because {e}!")
        return {'fpr': np.array([]), 'tpr': np.array([]), 'roc_auc': 0.0}

    y_scores = np.concatenate(y_scores_list)
    y_true = np.concatenate(y_true_list).flatten()
    
    if len(y_scores.shape) > 1:
        y_scores = y_scores[:, 0]

    # Calculate size difference (should be zero if logic is correct)
    size_a = y_true.size
    size_b = y_scores.size
    size_difference = size_a - size_b

    # Resize tensor_a
    if size_difference > 0:
        y_true = y_true[:size_a - size_difference]
    
    # Calculate the ROC curve and AUC
    fpr, tpr, roc_auc = roc_curve_and_auc(y_true, y_scores)
    
    return {'fpr': fpr.numpy(), 'tpr': tpr.numpy(), 'roc_auc': roc_auc.numpy()}

def calculate_multi_rocs(    
    model: keras.Model,
    dataset_args : dict,
    logger, 
    file_path : Path = None,
    num_examples_per_batch: int = 32,
    num_examples: int = 1.0E5,
    scaling_ranges: list = [
        (8.0, 20.0),
        8.0,
        10.0,
        12.0
    ],
    heart : gf.Heart = None
    ) -> dict:

    if file_path is not None:
        with gf.open_hdf5_file(
            file_path, 
            logger, 
            mode = "a"
        ) as validation_file:

            if "roc_data" not in validation_file:
                logger.info("Saving ROC data keys.")
                roc_group = validation_file.create_group('roc_data')
            
            if "keys" not in validation_file["roc_data"]:
                keys_array = [str(item) for item in scaling_ranges]
                string_dt = h5py.string_dtype(encoding='utf-8')
                keys_array = np.array(keys_array, dtype=string_dt)
                validation_file["roc_data"]["keys"]=keys_array
        
    roc_results = {}
    for scaling_range in scaling_ranges:

        # Make copy of generator args so original is not affected:
        dataset_args = deepcopy(dataset_args)

        range_name = str(scaling_range)

        if file_path is not None:
            with gf.open_hdf5_file(
                file_path, 
                logger, 
                mode = "r"
            ) as validation_file:

                roc_data = dict(validation_file["roc_data"])

                if f"{range_name}_fpr" in roc_data and \
                    f"{range_name}_tpr" in roc_data and \
                    f"{range_name}_roc_auc"  in roc_data:

                    logger.info(f"Range group: {range_name} already present in validation file. Skipping!")

                    roc_results[range_name] = {}
                    roc_results[range_name]["fpr"] = np.array(roc_data[f"{range_name}_fpr"])
                    roc_results[range_name]["tpr"] = np.array(roc_data[f"{range_name}_tpr"])
                    roc_results[range_name]["roc_auc"] = np.array(roc_data[f"{range_name}_roc_auc"])[0]
                    
                    continue
        
        if isinstance(scaling_range, tuple):
            scaling_disribution = gf.Distribution(
                min_=scaling_range[0], 
                max_=scaling_range[1],
                type_=gf.DistributionType.UNIFORM
            )
        else:
            scaling_disribution = gf.Distribution(
                value=scaling_range, 
                type_=gf.DistributionType.CONSTANT
            )
        #Ensure injection generators is list for subsequent logic:
        if not isinstance(dataset_args["waveform_generators"], list):
            dataset_args["waveform_generators"] = \
                [dataset_args["waveform_generators"]]
            
        # Set desired injection scalings:
        dataset_args["waveform_generators"][0].scaling_method.value = scaling_disribution
        
        roc_results[range_name] = \
            calculate_roc(    
                model,
                dataset_args,
                num_examples_per_batch,
                num_examples,
                heart
            )

        if file_path is not None:
            with gf.open_hdf5_file(
                file_path, 
                logger, 
                mode = "a"
            ) as validation_file:

                value = roc_results[range_name]

                logger.info(f"Save roc data {range_name}!")

                if f'roc_data/{range_name}_fpr' not in validation_file:
                    logger.info(f"Save roc data {range_name}_fpr!")

                    validation_file.create_dataset(
                        f'roc_data/{range_name}_fpr', 
                        data=value['fpr']
                    )

                if f'roc_data/{range_name}_tpr' not in validation_file:
                    logger.info(f"Save roc data {range_name}_tpr!")

                    validation_file.create_dataset(
                        f'roc_data/{range_name}_tpr', 
                        data=value['tpr']
                    )

                if f'roc_data/{range_name}_roc_auc' not in validation_file:
                    validation_file.create_dataset(f'roc_data/{range_name}_roc_auc', data=np.array([value['roc_auc']]))

        
    return roc_results

def calculate_tar_scores(
        model : keras.Model, 
        dataset_args : dict, 
        num_examples_per_batch : int = 32,  
        scaling : int = 20.0,
        num_examples : int = 1E5,
        heart : gf.Heart = None
    ) -> np.ndarray:
    
    """
    Calculate the True Alarm Rate (TAR) scores for a given model.

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
    num_batches = math.ceil(num_examples / num_examples_per_batch)
    
    #Ensure injection generators is list for subsequent logic:
    if not isinstance(dataset_args["waveform_generators"], list):
        dataset_args["waveform_generators"] = [dataset_args["waveform_generators"]]
    
    # Ensure dataset is full of injections:
    dataset_args["num_examples_per_batch"] = num_examples_per_batch
    dataset_args["output_variables"] = []
    dataset_args["waveform_generators"][0].injection_chance = 1.0
    dataset_args["waveform_generators"][0].scaling_method.value = \
        gf.Distribution(value=scaling, type_=gf.DistributionType.CONSTANT)
    
    # Initialize generator:
    dataset = gf.Dataset(
            **dataset_args
        )
        
    callbacks = []
    if heart is not None:
        callbacks += [gf.HeartbeatCallback(heart)]
    
    # Predict the scores and get the second column ([:, 1]):

    tar_scores_list = []
    data_batches = []
    
    try:
        verbose = 1
        if gf.is_redirected():
            verbose = 2
        
        for _ in range(num_batches):
            x, y = dataset[0]
            scores = model.predict_on_batch(x)
            tar_scores_list.append(scores)
            
            # Store data for worst performers extraction
            # We need to reconstruct the batch structure expected by extract_data_from_indicies?
            # Or we can just extract here.
            # But we don't know which are worst yet.
            # So we store x and y.
            #
            # Note: storing all data in memory might be expensive if num_examples is large (1E5).
            # 1E5 examples * 4096 floats * 4 bytes ~ 1.6 GB. It fits.
            data_batches.append((x, y))
            
            if heart is not None:
                heart.beat()

    except Exception as e:
        logging.error(f"Error calculating TAR score because {e}.")
        return np.array([]), []
            
    tar_scores = np.concatenate(tar_scores_list)
    if len(tar_scores.shape) > 1:
        tar_scores = tar_scores[:, 0]

    # Find worst performers
    # Sort indices by score (ascending)
    sorted_indices = np.argsort(tar_scores)
    worst_indices = sorted_indices[:10]
    worst_scores = tar_scores[worst_indices]
    
    worst_performers = []
    for i, idx in enumerate(worst_indices):
        batch_idx = idx // num_examples_per_batch
        sample_idx = idx % num_examples_per_batch
        
        if batch_idx < len(data_batches):
            x_batch, y_batch = data_batches[batch_idx]
            
            # We need to construct the element dict.
            # y_batch contains return variables like INJECTIONS, etc.
            # x_batch contains input variables.
            # We combine them.
            element = {}
            # Add inputs
            for k, v in x_batch.items():
                element[k] = v[sample_idx]
            # Add outputs
            for k, v in y_batch.items():
                element[k] = v[sample_idx]
                
            # Convert to numpy if needed (already numpy/jax array)
            for key in element:
                if hasattr(element[key], 'numpy'):
                    element[key] = element[key].numpy()
                elif hasattr(element[key], '__array__'):
                     element[key] = np.array(element[key])

            element["score"] = worst_scores[i]
            worst_performers.append(element)
        
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

def generate_efficiency_curves(
        validators : list,
        fars : np.ndarray,
        colors : List[str] = Bright[7],
        width : int = 800,
        height : int = 600
    ):

    colors = cycle(colors)
    
    # Check input durations are equal:
    check_equal_duration(validators)
    
    # Unpack values:
    input_duration_seconds = validators[0].input_duration_seconds

    p = figure(
        #title = "Efficiency Curves",
        width=width,
        height=height,
        x_axis_label="SNR",
        y_axis_label="Accuracy (Per Cent)",
        y_range=(0.0, 100.0)  # Set y-axis bounds here
    )

    # Set the initial plot title
    far_keys = list(fars)
    
    legend_items = []
    all_sources = {}
    acc_data = {}
    
    model_names = []
    
    for index, (validator, color) in enumerate(zip(validators, colors)):
        
        thresholds = calculate_far_score_thresholds(
            validator.far_scores, 
            input_duration_seconds, 
            fars
        )
        
        # Unpack arrays:
        scores = validator.efficiency_data["scores"]
        scalings = validator.efficiency_data["scalings"]
        name = validator.name

        if name is not None:
            title = gf.snake_to_capitalized_spaces(name)
        else:
            name = index
            title = f"default_{index}"
        
        model_names.append(title)
        
        acc_all_fars = []
        
        for far_index, far in enumerate(thresholds.keys()):
            threshold = thresholds[far][1]
            actual_far = thresholds[far][0]
            acc = []

            for score in scores:
                score = score[:, 0]
                if threshold != 0:
                    total = np.sum(score >= threshold)
                else:
                    total = np.sum(score > threshold)
                                        
                acc.append((total / len(score)) * 100)
            
            acc_all_fars.append(acc)
                
        acc_data[name] = acc_all_fars
        source = ColumnDataSource(
            data=dict(x=scalings, y=acc_all_fars[0], name=[title] * len(scalings))
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
    p.legend.label_text_font_size = "12pt"

    # Increase font sizes
    p.axis.axis_label_text_font_size = "14pt"  # Increase axis label font size
    p.axis.major_label_text_font_size = "12pt"  # Increase tick label font size

    # If you have titles
    p.title.text_font_size = '16pt'

    hover = HoverTool()
    hover.tooltips = [("Name", "@name"), ("SNR", "@x"), ("Accuracy (Per Cent)", "@y")]
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
        colors : List[str] = Bright[7],
        width : int = 800,
        height : int = 600
    ):

    colors = cycle(colors)
    
    tooltips = [
        ("Name", "@name"),
        ("Score Threshold", "@x"),
        ("False Alarm Rate (Hz)", "@y"),
    ]

    p = figure(
        #title = "False Alarm Rate (FAR) curves",
        width=width,
        height=height,
        x_axis_label="Score Threshold",
        y_axis_label="False Alarm Rate (Hz)",
        tooltips=tooltips,
        x_axis_type="log",
        y_axis_type="log"
    )
        
    max_num_points = 2000

    for index, (color, validator) in enumerate(zip(colors, validators)):
        far_scores = validator.far_scores
        
        name = validator.name

        if name is not None:
            title = gf.snake_to_capitalized_spaces(name)
        else:
            title = f"default_{index}"
            name = index
                
        far_scores = np.sort(far_scores)[::-1]
        total_num_seconds = len(far_scores) * validator.input_duration_seconds
        far_axis = np.arange(1, len(far_scores) + 1, dtype=float) / total_num_seconds
        
        downsampled_far_scores, downsampled_far_axis = far_scores, far_axis
        # downsample_data(
        #    far_scores, far_axis, max_num_points
        #)
        
        source = ColumnDataSource(
            data=dict(
                x=downsampled_far_scores, 
                y=downsampled_far_axis,
                name=[title]*len(downsampled_far_scores)
            )
        )
        
        p.line(
            "x", 
            "y", 
            source=source, 
            line_color=color,
            line_width=2,
            legend_label=title
        )

    hover = HoverTool()
    hover.tooltips = tooltips
    p.add_tools(hover)
    
    p.legend.location = "bottom_left"
    p.legend.click_policy = "hide"
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "12pt"

    # Increase font sizes
    p.axis.axis_label_text_font_size = "14pt"  # Increase axis label font size
    p.axis.major_label_text_font_size = "12pt"  # Increase tick label font size

    # If you have titles
    p.title.text_font_size = '16pt'

    return p

def generate_roc_curves(
    validators: list,
    colors : List[str] = Bright[7], 
    width : int = 800,
    height : int = 600
    ):

    colors = cycle(colors)
    
    p = figure(
        #title="Receiver Operating Characteristic (ROC) Curves",
        x_axis_label='False Alarm Rate (Hz)',
        y_axis_label='Accuracy (Per Cent)',
        width=width, 
        height=height,
        x_axis_type='log', 
        x_range=[1e-6, 1], 
        y_range=[0.0, 100.0]
    )
    
    max_num_points = 500

    initial_population_key = list(validators[0].roc_data.keys())[0]
    all_sources = {}
    
    for color, validator in zip(colors, validators):
        roc_data = validator.roc_data[initial_population_key]
        name = validator.name
                
        if name is not None:
            title = gf.snake_to_capitalized_spaces(name)
        else:
            title = f"default_{index}"
            name = index
        
        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]*100
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
    p.legend.label_text_font_size = "12pt"

    # Increase font sizes
    p.axis.axis_label_text_font_size = "14pt"  # Increase axis label font size
    p.axis.major_label_text_font_size = "12pt"  # Increase tick label font size

    # If you have titles
    p.title.text_font_size = '16pt'
    
    # Dropdown to select the test population
    populations = list(validators[0].roc_data.keys())
    select = Select(
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
            fpr, tpr = downsample_data(data["fpr"], data["tpr"]*100, max_num_points)
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

    colors = cycle(colors)
    
    p = figure(
        title=f"Worst Performing Input Score: {data['score']}, "
        f"{data[gf.WaveformParameters.MASS_1_MSUN.name]}, "
        f"{data[gf.WaveformParameters.MASS_2_MSUN.name]}.",
        x_axis_label='Time Seconds',
        y_axis_label='Strain',
        width=800, 
        height=300
    )
    
    source = ColumnDataSource(
        data=dict(
            x=np.linspace(
                0,
                onsource_duration_seconds, 
                len(data[gf.ReturnVariables.WHITENED_ONSOURCE.name])
            ), 
            y=data[gf.ReturnVariables.WHITENED_ONSOURCE.name]
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
            x= np.linspace(0,onsource_duration_seconds, len(data[gf.ReturnVariables.WHITENED_ONSOURCE.name])),
            y= data[gf.ReturnVariables.WHITENED_INJECTIONS.name]
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
            x=np.linspace(0,onsource_duration_seconds, len(data[gf.ReturnVariables.WHITENED_ONSOURCE.name])), 
            y=data[gf.ReturnVariables.WHITENED_INJECTIONS.name]*20.0
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
    p.legend.label_text_font_size = "12pt"

    # Increase font sizes
    p.axis.axis_label_text_font_size = "14pt"  # Increase axis label font size
    p.axis.major_label_text_font_size = "12pt"  # Increase tick label font size

    # If you have titles
    p.title.text_font_size = '16pt'
    
    return p

class Validator:
    
    @classmethod
    def validate(
        cls, 
        model : keras.Model, 
        name : str,
        dataset_args : dict,
        num_examples_per_batch : int = None,
        efficiency_config : dict = \
        {
            "max_scaling" : 20.0, 
            "num_scaling_steps" : 21, 
            "num_examples_per_scaling_step" : 2048
        },
        far_config : dict = \
        {
            "num_seconds" : 1.0E5
        },
        roc_config : dict = \
        {
            "num_examples" : 1.0E5,
            "scaling_ranges" :  [
                (8.0, 20),
                8.0,
                10.0,
                12.0
            ]    
        },
        checkpoint_file_path : Path = None,
        logging_level : int = logging.INFO,
        heart : gf.Heart = None
    ):
        if num_examples_per_batch is None:
            num_examples_per_batch = gf.Defaults.num_examples_per_batch

        # Save model title: 
        if isinstance(name, Path):
            name = name.name

        if not isinstance(name, str):
            raise ValueError("Requires model name!")

        validator = cls()

        # Initiate logging:
        validator.logger = logging.getLogger("validator")
        stream_handler = logging.StreamHandler(sys.stdout)
        validator.logger.addHandler(stream_handler)
        validator.logger.setLevel(logging_level)

        validator.efficiency_data = None
        validator.far_scores = None
        validator.roc_data = None

        validator.name = name

        if "onsource_duration_seconds" in dataset_args:
            if dataset_args["onsource_duration_seconds"] is None:
                validator.input_duration_seconds = gf.Defaults.onsource_duration_seconds
            else:
                validator.input_duration_seconds = dataset_args["onsource_duration_seconds"]
        else: 
            validator.input_duration_seconds = gf.Defaults.onsource_duration_seconds

        if checkpoint_file_path is not None:

            if checkpoint_file_path.exists():

                validator = validator.load(
                    checkpoint_file_path,
                    logging_level=logging_level
                )

                if validator.name is None:
                    if name is not None:
                        validator.name = name
                    else:
                        validator.name = "Default"

                    with gf.open_hdf5_file(
                        checkpoint_file_path, 
                        validator.logger, 
                        mode = "w"
                    ) as validation_file:

                        if "name" not in validation_file:
                            if validator.name is not None:
                                validation_file.create_dataset('name', data=validator.name.encode())
                            else:
                                validation_file.create_dataset('name', data="default".encode())
                       
            else:

                gf.ensure_directory_exists(checkpoint_file_path.parent)

                with gf.open_hdf5_file(
                        checkpoint_file_path, 
                        validator.logger, 
                        mode = "w"
                    ) as validation_file:

                    if "name" not in validation_file:
                        if validator.name is not None:
                            validation_file.create_dataset('name', data=validator.name.encode())
                        else:
                            validation_file.create_dataset('name', data="default".encode())

                    if "input_duration_seconds" not in validation_file:             
                        validation_file.create_dataset(
                            'input_duration_seconds', 
                            data=validator.input_duration_seconds
                        )

        validator.heart = heart
        
        validator.worst_performers = None
        """
        validator.logger.info(f"Worst performing inputs for {validator.name}...")
        tar_scores, worst_performers = \
            calculate_tar_scores(
                model, 
                dataset_args, 
                num_examples_per_batch,
                scaling=20.0,
                num_examples=1.0E3,
                heart=validator.heart
            )
        validator.worst_performers = worst_performers
        validator.logger.info(f"Done")
        """
        validator.worst_performmers = None

        if validator.efficiency_data is None:   
            validator.logger.info(f"Calculating efficiency scores for {validator.name}...")
            validator.efficiency_data = \
                calculate_efficiency_scores(
                    model, 
                    dataset_args,
                    validator.logger,
                    file_path=checkpoint_file_path,
                    num_examples_per_batch=num_examples_per_batch,
                    heart=validator.heart,
                    **efficiency_config
                )

            validator.logger.info(f"Done")
        else:
            validator.logger.info("Validation file already contains efficiency scores! Loading...")

        if (validator.far_scores is None):
            validator.logger.info(f"Calculating FAR scores for {validator.name}...")
            validator.far_scores = calculate_far_scores(
                    model, 
                    dataset_args, 
                    validator.logger,
                    file_path=checkpoint_file_path,
                    num_examples_per_batch=num_examples_per_batch,  
                    heart=validator.heart,
                    **far_config
                )
            validator.logger.info(f"Done")
        else:
            validator.logger.info("Validation file already contains FAR scores! Loading...")
        
        validator.logger.info(f"Collecting ROC data for {validator.name}...")
        validator.roc_data = calculate_multi_rocs(    
                model,
                dataset_args,
                validator.logger,
                file_path=checkpoint_file_path,
                num_examples_per_batch=num_examples_per_batch,
                heart=validator.heart,
                **roc_config
            )
        validator.logger.info(f"Done")
                
        return validator

    def save(
        self,
        file_path : Path
        ):

        self.logger.info(f"Saving validation data for {self.name}...")

        gf.ensure_directory_exists(file_path.parent)

        with gf.open_hdf5_file(
                file_path, 
                self.logger, 
                mode = "a"
            ) as validation_file:
            
            # Unpack:
            scalings = self.efficiency_data['scalings']
            efficiency_scores = self.efficiency_data['scores']

            if self.name is not None:
                validation_file.create_dataset('name', data=self.name.encode())
            else:
                validation_file.create_dataset('name', data="default".encode())
            
            validation_file.create_dataset(
                'input_duration_seconds', 
                data=self.input_duration_seconds
            )

            # Save efficiency scores:
            if efficiency_scores is not None:
                eff_group = validation_file.create_group('efficiency_data')
                eff_group.create_dataset(f'scalings', data=scalings)
                for i, score in enumerate(efficiency_scores):
                    eff_group.create_dataset(f'score_{i}', data=score)

            # Save FAR scores
            if self.far_scores is not None:
                far_group = validation_file.create_group('far_scores')
                far_group.create_dataset('scores', data=self.far_scores)

            # Save ROC data:
            if roc_data is not None:
                roc_group = validation_file.create_group('roc_data')
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
                    roc_group.create_dataset(
                        f'{key}_roc_auc', data=np.array([value['roc_auc']])
                    )

            if self.worst_perfomers is not None:         
                worst_performers = self.worst_performers
                
                for idx, entry in enumerate(worst_performers):
                    group = validation_file.create_group(f'worst_perfomers_{idx}')
                    for key, value in entry.items():
                        group.create_dataset(key, data=value, dtype=np.float16)
        
            self.logger.info("Done.")
        
    @classmethod
    def load(
        cls, 
        file_path: Path,
        logging_level = logging.INFO
    ):
        # Create a new instance without executing any logic
        validator = cls()

        with h5py.File(file_path, 'r') as h5f:
            # Check and load title:
            validator.name = h5f['name'][()].decode() if 'name' in h5f else None

            # Check and load input_duration_seconds
            if 'input_duration_seconds' in h5f:
                validator.input_duration_seconds = float(h5f['input_duration_seconds'][()])
            else:
                validator.input_duration_seconds = gf.Defaults.onsource_duration_seconds
            
            validator.logger = logging.getLogger("validator")
            validator.logger.setLevel(logging_level)
            validator.logger.info(f"Loading validation data for {validator.name}...")

            # Check and load efficiency scores:
            if 'efficiency_data' in h5f:
                eff_group = h5f['efficiency_data']
                efficiency_data = {
                    'scalings': eff_group['scalings'][:] if 'scalings' in eff_group else None,
                    'scores': [
                        eff_group[f'score_{i}'][:] for i in range(len(eff_group) - 1)
                    ] if all(f'score_{i}' in eff_group for i in range(len(eff_group) - 1)) else None
                }
            else:
                efficiency_data = None

            # Check and load FAR scores:
            far_scores = h5f['far_scores']['scores'][:] if 'far_scores' in h5f and 'scores' in h5f['far_scores'] else None
            
            roc_data = {}
            # Check and load ROC data:
            if 'roc_data' in h5f:
                roc_group = dict(h5f['roc_data'])
                keys_array = roc_group['keys'][:] if 'keys' in roc_group else []

                # Check if keys_array is not empty
                for key in keys_array:
                    key_str = key.decode('utf-8')

                    # Check if all required metrics are present for the current key
                    metrics_present = True
                    for metric in ['fpr', 'tpr', 'roc_auc']:
                        if f'{key_str}_{metric}' not in roc_group:
                            metrics_present = False
                            break

                    if metrics_present:
                        # All metrics are present, add them to the dictionary
                        roc_data[key_str] = {
                            'fpr': roc_group[f'{key_str}_fpr'][:],
                            'tpr': roc_group[f'{key_str}_tpr'][:],
                            'roc_auc': roc_group[f'{key_str}_roc_auc'][0],
                        }


            # Populate the Validator object's attributes with loaded data:
            validator.efficiency_data = efficiency_data
            validator.far_scores = far_scores
            validator.roc_data = roc_data
            
            # Check and load worst performers:
            worst_performers = []
            for group_name in h5f:
                group_data = {}
                if group_name.startswith(f"worst_performers_"): 
                    for key in h5f[group_name]:
                        group_data[key] = h5f[group_name][key][()] if key in h5f[group_name] else None
                    worst_performers.append(group_data)
                
            validator.worst_performers = worst_performers if worst_performers else None
                                    
            validator.logger.info("Done.")

        return validator

    def plot(
        self,
        file_path : Path,
        comparison_validators : list = [],
        fars : np.ndarray = np.logspace(-1, -7, 500),
        colors = Bright[7], 
        width : int = 800,
        height : int = 600
    ):
        gf.ensure_directory_exists(file_path.parent)

        validators = comparison_validators + [self]
        
        efficiency_curves, slider = generate_efficiency_curves(
            validators, 
            fars,
            colors=colors,
            width=width,
            height=height
        )
        
        far_curves = generate_far_curves(
            validators,
            colors=colors,
            width=width,
            height=height
        )

        roc_curves, dropdown = generate_roc_curves(
            validators,
            colors=colors,
            width=width,
            height=height
        )
        
        layout = [
            [dropdown, slider],
            [roc_curves, efficiency_curves], 
            [far_curves, None]
        ]
        
        if self.worst_performers is not None:
            for waveform in self.worst_performers:

                pass
                """
                waveform_plot = generate_waveform_plot(
                    waveform,
                    self.input_duration_seconds
                )
                layout.append([waveform_plot, None])
                """
            
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