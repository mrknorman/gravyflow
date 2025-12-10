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

# Panel for dashboard layout and HTML export
import panel as pn
pn.extension('bokeh')

# Bokeh for plotting
from bokeh.embed import components, file_html
from bokeh.io import output_file, save
from bokeh.layouts import column, gridplot
from bokeh.models import (ColumnDataSource, CustomJS, HoverTool,
                          Legend, Slider, Select, Div)
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.palettes import Bright, Category10

import gravyflow as gf
from dataclasses import dataclass, field
from scipy.optimize import curve_fit
import heapq


@dataclass
class ValidationConfig:
    """Configuration for unified validation pipeline."""
    snr_range: Tuple[float, float] = (0.0, 20.0)
    num_examples: int = 100_000
    batch_size: int = 512
    snr_bin_width: float = 5.0
    num_worst_per_bin: int = 10
    far_thresholds: List[float] = field(default_factory=lambda: [1e-1, 1e-2, 1e-3, 1e-4])
    

class UnifiedValidationBank:
    """
    Unified data bank for validation metrics.
    
    Generates a single dataset with uniform SNR distribution and computes
    all validation metrics (efficiency, TAR, worst performers) from the same data.
    """
    
    def __init__(
        self,
        model: keras.Model,
        dataset_args: dict,
        config: ValidationConfig = None,
        heart: gf.Heart = None,
        logger: logging.Logger = None
    ):
        self.model = model
        self.dataset_args = deepcopy(dataset_args)
        self.config = config or ValidationConfig()
        self.heart = heart
        self.logger = logger or logging.getLogger("validation_bank")
        
        # Results storage
        self.snrs = None
        self.scores = None  
        self.injection_masks = None
        self.sample_data = {}  # For worst performers
        
        # Computed metrics
        self._efficiency_data = None
        self._tar_data = None
        self._worst_per_bin = None
        
    def generate(self) -> None:
        """Generate all validation data in a single pass."""
        config = self.config
        dataset_args = deepcopy(self.dataset_args)
        
        # Configure for uniform SNR sampling
        min_snr, max_snr = config.snr_range
        
        # Ensure injection generators use uniform SNR
        if "waveform_generators" not in dataset_args:
            raise ValueError("dataset_args must contain waveform_generators")
            
        if not isinstance(dataset_args["waveform_generators"], list):
            dataset_args["waveform_generators"] = [dataset_args["waveform_generators"]]
        
        # Set uniform SNR distribution for all generators
        for gen in dataset_args["waveform_generators"]:
            gen.scaling_method.value = gf.Distribution(
                min_=min_snr,
                max_=max_snr,
                type_=gf.DistributionType.UNIFORM
            )
            gen.injection_chance = 1.0  # Always inject for efficiency/TAR
        
        # Configure output variables to get SNR values
        dataset_args["num_examples_per_batch"] = config.batch_size
        dataset_args["output_variables"] = [
            gf.ReturnVariables.WHITENED_ONSOURCE,
            gf.ReturnVariables.WHITENED_INJECTIONS,
            gf.ReturnVariables.INJECTION_MASKS,
            gf.ReturnVariables.GPS_TIME,
            gf.ScalingTypes.SNR,
            gf.WaveformParameters.MASS_1_MSUN,
            gf.WaveformParameters.MASS_2_MSUN
        ]
        
        num_batches = math.ceil(config.num_examples / config.batch_size)
        dataset_args["steps_per_epoch"] = num_batches
        dataset_args["group"] = "test"  # Use test data split for validation
        
        dataset = gf.Dataset(**dataset_args)
        
        # Preallocate arrays
        all_snrs = []
        all_scores = []
        all_masks = []
        
        # Per-bin worst performers using min-heaps (store negative score for max-heap behavior)
        num_bins = int((max_snr - min_snr) / config.snr_bin_width)
        worst_heaps = [[] for _ in range(num_bins)]
        
        self.logger.info(f"Generating {config.num_examples} examples across SNR range {config.snr_range}")
        
        for batch_idx in range(num_batches):
            if self.heart:
                self.heart.beat()
                
            x_batch, y_batch = dataset[batch_idx]
            
            # Get model predictions
            predictions = self.model.predict(x_batch, verbose=0)
            
            # Handle different output formats
            if len(predictions.shape) == 2:
                if predictions.shape[1] == 2:
                    batch_scores = predictions[:, 1]  # Binary: signal probability
                else:
                    batch_scores = predictions[:, 0]
            else:
                batch_scores = predictions.flatten()
            
            # Extract SNRs and masks
            batch_snrs = np.array(y_batch.get("SNR", np.zeros(len(batch_scores))))
            batch_masks = np.array(y_batch.get(gf.ReturnVariables.INJECTION_MASKS.name, 
                                                np.ones(len(batch_scores))))
            
            # Handle mask shape (may be (gen, batch, time))
            if len(batch_masks.shape) == 3:
                batch_masks = np.max(batch_masks[0], axis=-1)
            elif len(batch_masks.shape) == 2:
                batch_masks = np.max(batch_masks, axis=-1)
            
            all_snrs.extend(batch_snrs)
            all_scores.extend(batch_scores)
            all_masks.extend(batch_masks)
            
            # Track worst performers per bin
            for i in range(len(batch_scores)):
                snr_val = float(batch_snrs[i])
                score_val = float(batch_scores[i])
                
                bin_idx = min(int((snr_val - min_snr) / config.snr_bin_width), num_bins - 1)
                bin_idx = max(0, bin_idx)  # Clamp to valid range
                
                # Use min-heap with negative score for max-heap behavior (keep lowest scores)
                if len(worst_heaps[bin_idx]) < config.num_worst_per_bin:
                    # Extract sample data for worst performer
                    sample_data = _extract_sample_data(x_batch, y_batch, i, score_val)
                    heapq.heappush(worst_heaps[bin_idx], (-score_val, batch_idx * config.batch_size + i, sample_data))
                elif score_val < -worst_heaps[bin_idx][0][0]:
                    # This score is worse (lower) than the best (highest) in heap
                    sample_data = _extract_sample_data(x_batch, y_batch, i, score_val)
                    heapq.heapreplace(worst_heaps[bin_idx], (-score_val, batch_idx * config.batch_size + i, sample_data))
            
            if (batch_idx + 1) % 50 == 0:
                self.logger.info(f"Progress: {batch_idx + 1}/{num_batches} batches")
        
        # Store results
        self.snrs = np.array(all_snrs)
        self.scores = np.array(all_scores)
        self.injection_masks = np.array(all_masks)
        
        # Extract worst performers from heaps
        self._worst_per_bin = {}
        for bin_idx, heap in enumerate(worst_heaps):
            bin_start = min_snr + bin_idx * config.snr_bin_width
            bin_end = bin_start + config.snr_bin_width
            bin_key = f"{bin_start:.0f}-{bin_end:.0f}"
            
            # Sort by score (ascending - worst first)
            sorted_worst = sorted(heap, key=lambda x: -x[0])  # Undo negative
            self._worst_per_bin[bin_key] = [item[2] for item in sorted_worst]
        
        self.logger.info(f"Generated {len(self.snrs)} samples. SNR range: [{self.snrs.min():.1f}, {self.snrs.max():.1f}]")
    
    def get_efficiency_data(self) -> Dict:
        """
        Get efficiency scatter data and fitted curve.
        
        Returns dict with:
        - snrs: array of SNR values
        - scores: array of model scores
        - fit_snrs: x values for fitted curve
        - fit_efficiency: y values (efficiency) for fitted curve
        - fit_lower: lower confidence bound
        - fit_upper: upper confidence bound
        """
        if self.snrs is None:
            raise ValueError("Must call generate() first")
        
        # Filter to injection samples only
        mask = self.injection_masks > 0.5
        snrs = self.snrs[mask]
        scores = self.scores[mask]
        
        # Bin statistics for confidence bands
        config = self.config
        min_snr, max_snr = config.snr_range
        num_bins = int((max_snr - min_snr) / config.snr_bin_width)
        
        bin_centers = []
        bin_means = []
        bin_stds = []
        
        for i in range(num_bins):
            bin_start = min_snr + i * config.snr_bin_width
            bin_end = bin_start + config.snr_bin_width
            bin_mask = (snrs >= bin_start) & (snrs < bin_end)
            
            if np.sum(bin_mask) > 10:
                bin_centers.append((bin_start + bin_end) / 2)
                bin_means.append(np.mean(scores[bin_mask]))
                bin_stds.append(np.std(scores[bin_mask]))
        
        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        
        # Fit sigmoid curve: efficiency = 1 / (1 + exp(-k*(x - x0)))
        def sigmoid(x, x0, k, L, b):
            return L / (1 + np.exp(-k * (x - x0))) + b
        
        try:
            popt, _ = curve_fit(
                sigmoid, bin_centers, bin_means,
                p0=[10.0, 0.5, 0.8, 0.1],  # Initial guess
                bounds=([0, 0.01, 0, 0], [30, 5, 1, 0.5]),
                maxfev=5000
            )
            
            fit_snrs = np.linspace(min_snr, max_snr, 100)
            fit_efficiency = sigmoid(fit_snrs, *popt)
        except:
            # Fallback to interpolation if fit fails
            fit_snrs = bin_centers
            fit_efficiency = bin_means
        
        return {
            "snrs": snrs,
            "scores": scores,
            "fit_snrs": fit_snrs,
            "fit_efficiency": fit_efficiency,
            "bin_centers": bin_centers,
            "bin_means": bin_means,
            "bin_stds": bin_stds
        }
    
    def get_tar_at_far(self, far_thresholds: List[float] = None) -> Dict:
        """
        Compute True Acceptance Rate at given False Alarm Rates.
        
        Uses noise-only samples (injection_mask = 0) to estimate FAR thresholds,
        then computes TAR on injection samples.
        """
        if self.snrs is None:
            raise ValueError("Must call generate() first")
        
        far_thresholds = far_thresholds or self.config.far_thresholds
        
        # For this unified bank, we need noise-only samples for FAR
        # If all samples have injections, we can't compute FAR properly
        # This is a limitation - suggest adding noise-only samples
        
        # For now, use a threshold approach based on score distribution
        injection_mask = self.injection_masks > 0.5
        
        if not np.any(~injection_mask):
            self.logger.warning("No noise-only samples in bank. TAR@FAR requires noise-only samples.")
            return {}
        
        noise_scores = self.scores[~injection_mask]
        signal_scores = self.scores[injection_mask]
        signal_snrs = self.snrs[injection_mask]
        
        results = {}
        for far in far_thresholds:
            # Find threshold that gives this FAR on noise
            threshold_percentile = (1 - far) * 100
            threshold = np.percentile(noise_scores, threshold_percentile)
            
            # Compute TAR per SNR bin
            config = self.config
            min_snr, max_snr = config.snr_range
            num_bins = int((max_snr - min_snr) / config.snr_bin_width)
            
            tar_per_bin = []
            bin_centers = []
            
            for i in range(num_bins):
                bin_start = min_snr + i * config.snr_bin_width
                bin_end = bin_start + config.snr_bin_width
                bin_mask = (signal_snrs >= bin_start) & (signal_snrs < bin_end)
                
                if np.sum(bin_mask) > 0:
                    tar = np.mean(signal_scores[bin_mask] > threshold)
                    tar_per_bin.append(tar)
                    bin_centers.append((bin_start + bin_end) / 2)
            
            results[far] = {
                "threshold": threshold,
                "bin_centers": np.array(bin_centers),
                "tar_per_bin": np.array(tar_per_bin)
            }
        
        return results
    
    def get_worst_performers(self) -> Dict[str, List[dict]]:
        """Get worst performing samples per SNR bin."""
        if self._worst_per_bin is None:
            raise ValueError("Must call generate() first")
        return self._worst_per_bin



def pad_with_random_values(scores):
    """Pad score arrays to uniform length with random values."""
    # Determine the maximum length among all numpy arrays of 2-element arrays in scores
    max_length = max(len(score) for score in scores)
    
    # If all arrays are empty, return empty array
    if max_length == 0:
        return np.array(scores)
    
    def pad_array(arr, max_length):
        current_length = len(arr)
        if current_length == 0:
            # Cannot pad an empty array by sampling from it.
            # Return an array of NaNs with the target shape to indicate missing data.
            if hasattr(arr, 'shape') and len(arr.shape) > 1:
                return np.full((max_length,) + arr.shape[1:], np.nan)
            else:
                return np.full((max_length,), np.nan)
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
    
    # Set steps_per_epoch to match num_batches to prevent data exhaustion
    dataset_args["steps_per_epoch"] = num_batches
    dataset_args["group"] = "test"  # Use test data split for validation
    
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
            import traceback
            traceback.print_exc()
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
    # Warn if efficiency is suspiciously low at high SNR
    if scores is not None and len(scores) > 0:
        # Check efficiency at highest SNR
        high_snr_scores = scores[-1]  # Last scaling step (highest SNR)
        if high_snr_scores is not None and len(high_snr_scores) > 0:
            # Get mean prediction at highest SNR
            mean_score = np.nanmean(high_snr_scores[:, 1] if len(high_snr_scores.shape) > 1 else high_snr_scores)
            if mean_score < 0.5:
                logging.warning(
                    f"LOW EFFICIENCY WARNING: Mean detection probability at SNR={efficiency_scalings[-1]:.1f} "
                    f"is only {mean_score:.2f}. Expected > 0.5 for a working model. "
                    "Check: 1) Model trained correctly? 2) Input preprocessing matches training? "
                    "3) Noise type compatible with training data?"
                )
    
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
    dataset_args["steps_per_epoch"] = num_batches
    dataset_args["group"] = "test"  # Use test data split for validation

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
                far_scores = far_scores[:,1]  # Use signal+noise probability (column 1)
            except:
                raise Exception(f"Error slicing FAR scores: shape={far_scores.shape}")

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
    dataset_args["steps_per_epoch"] = num_batches
    dataset_args["group"] = "test"  # Use test data split for validation
    
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
        y_scores = y_scores[:, 1]  # Use signal+noise probability (column 1)

    # Calculate size difference (should be zero if logic is correct)
    size_a = y_true.size
    size_b = y_scores.size
    size_difference = size_a - size_b

    # Resize tensor_a
    if size_difference > 0:
        y_true = y_true[:size_a - size_difference]
    
    # Calculate the ROC curve and AUC
    fpr, tpr, roc_auc = roc_curve_and_auc(y_true, y_scores)
    
    # Warn if ROC AUC is suspiciously low
    roc_auc_val = float(roc_auc)
    if roc_auc_val < 0.6:
        logging.warning(
            f"LOW ROC AUC WARNING: AUC = {roc_auc_val:.3f} (expected > 0.6 for a working model). "
            "An AUC near 0.5 indicates random-chance performance. "
            "Check: 1) Model trained correctly? 2) Input preprocessing matches training? "
            "3) Noise type compatible with training data?"
        )
    
    return {'fpr': np.asarray(fpr), 'tpr': np.asarray(tpr), 'roc_auc': np.asarray(roc_auc)}

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
        scaling_range : tuple = (8.0, 15.0),
        num_examples : int = 1E5,
        num_worst : int = 10,
        heart : gf.Heart = None
    ) -> np.ndarray:
    
    """
    Calculate the True Alarm Rate (TAR) scores for a given model.
    Uses streaming approach to avoid storing all data in memory.

    Parameters
    ----------
    model : keras.Model
        The model used to predict scores.
    dataset_args : dict
        Dictionary containing options for dataset generator.
    scaling_range : tuple
        (min_snr, max_snr) for injection SNR distribution.
    num_examples : int
        The total number of examples to evaluate.
    num_worst : int
        Number of worst performers to return.
    
    Returns
    -------
    tar_scores : np.ndarray
        The calculated TAR scores.
    worst_performers : list
        List of dicts containing data for worst performing samples.
    """
    import heapq
    
    # Make copy of generator args so original is not affected:
    dataset_args = deepcopy(dataset_args)
    
    # Integer arguments:
    num_examples = int(num_examples)
    num_examples_per_batch = int(num_examples_per_batch)
    num_batches = math.ceil(num_examples / num_examples_per_batch)
    
    # Ensure injection generators is list:
    if not isinstance(dataset_args["waveform_generators"], list):
        dataset_args["waveform_generators"] = [dataset_args["waveform_generators"]]
    
    # Configure dataset for worst performers extraction:
    dataset_args["num_examples_per_batch"] = num_examples_per_batch
    dataset_args["output_variables"] = [
        gf.ReturnVariables.WHITENED_ONSOURCE,
        gf.ReturnVariables.WHITENED_INJECTIONS,
        gf.ReturnVariables.GPS_TIME,
        gf.ScalingTypes.SNR,
        gf.WaveformParameters.MASS_1_MSUN,
        gf.WaveformParameters.MASS_2_MSUN
    ]
    dataset_args["waveform_generators"][0].injection_chance = 1.0
    
    # Use a range of SNRs - lower SNR samples are more challenging
    min_snr, max_snr = scaling_range
    dataset_args["waveform_generators"][0].scaling_method.value = \
        gf.Distribution(
            min_=min_snr,
            max_=max_snr,
            type_=gf.DistributionType.UNIFORM
        )
    dataset_args["steps_per_epoch"] = num_batches
    dataset_args["group"] = "test"  # Use test data split for validation
    
    # Initialize dataset:
    dataset = gf.Dataset(**dataset_args)
    
    # Use a max-heap to keep track of worst performers (lowest scores)
    # We use negative scores because heapq is a min-heap
    # Heap entries: (-score, counter, data_dict) - counter prevents dict comparison
    worst_heap = []
    all_scores = []
    counter = 0  # Unique counter to break ties in heap comparison
    
    try:
        for batch_idx in range(num_batches):
            x, y = dataset[batch_idx]
            scores = model.predict_on_batch(x)
            
            # Extract signal probability
            if len(scores.shape) > 1:
                batch_scores = scores[:, 1]
            else:
                batch_scores = scores
            
            all_scores.append(batch_scores)
            
            # Process each sample in batch
            for sample_idx in range(len(batch_scores)):
                score = float(batch_scores[sample_idx])
                
                # Check if this is a candidate for worst performers
                if len(worst_heap) < num_worst:
                    # Still building up the heap, add this sample
                    element = _extract_sample_data(x, y, sample_idx, score)
                    heapq.heappush(worst_heap, (-score, counter, element))
                    counter += 1
                elif score < -worst_heap[0][0]:  # score < current max in heap
                    # This is worse than the best of our current worst
                    element = _extract_sample_data(x, y, sample_idx, score)
                    heapq.heapreplace(worst_heap, (-score, counter, element))
                    counter += 1
            
            if heart is not None:
                heart.beat()

    except Exception as e:
        logging.error(f"Error calculating TAR score: {e}")
        return np.array([]), []
            
    tar_scores = np.concatenate(all_scores)
    
    # Extract worst performers from heap (sorted by score ascending)
    worst_performers = []
    sorted_worst = sorted(worst_heap, key=lambda x: x[0], reverse=True)  # Most negative first = lowest score
    for neg_score, counter, element in sorted_worst:
        worst_performers.append(element)
        
    return tar_scores, worst_performers


def _extract_sample_data(x_batch, y_batch, sample_idx, score):
    """Extract data for a single sample from batch for worst performers."""
    element = {}
    
    # Add inputs - shape is typically (batch, detectors, samples)
    for k, v in x_batch.items():
        if hasattr(v, 'shape') and len(v.shape) >= 1:
            element[k] = np.array(v[sample_idx])
        else:
            element[k] = v
    
    # Add outputs - handle different shapes
    for k, v in y_batch.items():
        if hasattr(v, 'shape'):
            if len(v.shape) == 4:  # (generators, batch, det, samples)
                element[k] = np.array(v[0, sample_idx])
            elif len(v.shape) >= 2:
                element[k] = np.array(v[sample_idx])
            elif len(v.shape) == 1:
                element[k] = np.array(v[sample_idx])
            else:
                element[k] = v
        else:
            element[k] = v

    element["score"] = score
    return element

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

def generate_efficiency_scatter_plot(
    bank: UnifiedValidationBank,
    colors: List[str] = Bright[7],
    width: int = 800,
    height: int = 500,
    downsample_points: int = 5000
):
    """
    Generate efficiency scatter plot with fitted curve and confidence bands.
    
    Shows all (SNR, score) points as scatter, with sigmoid fit and ±1σ bands.
    
    Args:
        bank: UnifiedValidationBank with generated data
        colors: Color palette
        downsample_points: Max scatter points to render (for performance)
    """
    efficiency_data = bank.get_efficiency_data()
    
    snrs = efficiency_data["snrs"]
    scores = efficiency_data["scores"]
    fit_snrs = efficiency_data["fit_snrs"]
    fit_efficiency = efficiency_data["fit_efficiency"]
    bin_centers = efficiency_data["bin_centers"]
    bin_means = efficiency_data["bin_means"]
    bin_stds = efficiency_data["bin_stds"]
    
    p = figure(
        title="Detection Efficiency vs SNR",
        x_axis_label="Signal-to-Noise Ratio (SNR)",
        y_axis_label="Detection Score",
        width=width,
        height=height,
        tools="pan,box_zoom,wheel_zoom,reset,hover"
    )
    
    # Downsample scatter points for performance
    if len(snrs) > downsample_points:
        idx = np.random.choice(len(snrs), downsample_points, replace=False)
        plot_snrs = snrs[idx]
        plot_scores = scores[idx]
    else:
        plot_snrs = snrs
        plot_scores = scores
    
    # Scatter plot of all points (semi-transparent)
    scatter_source = ColumnDataSource(data=dict(x=plot_snrs, y=plot_scores))
    p.circle(
        x="x", y="y", source=scatter_source,
        size=3, alpha=0.1, color=colors[0],
        legend_label="Samples"
    )
    
    # Confidence band (±1σ)
    if len(bin_centers) > 1:
        upper = bin_means + bin_stds
        lower = np.maximum(bin_means - bin_stds, 0)
        
        band_x = np.concatenate([bin_centers, bin_centers[::-1]])
        band_y = np.concatenate([upper, lower[::-1]])
        
        p.patch(
            band_x, band_y,
            fill_alpha=0.2, fill_color=colors[1],
            line_alpha=0, legend_label="±1σ Band"
        )
    
    # Bin means as points
    if len(bin_centers) > 0:
        means_source = ColumnDataSource(data=dict(x=bin_centers, y=bin_means))
        p.circle(
            x="x", y="y", source=means_source,
            size=10, color=colors[1], alpha=0.8,
            legend_label="Bin Means"
        )
    
    # Fitted curve
    if len(fit_snrs) > 0:
        fit_source = ColumnDataSource(data=dict(x=fit_snrs, y=fit_efficiency))
        p.line(
            x="x", y="y", source=fit_source,
            line_width=3, color=colors[2],
            legend_label="Sigmoid Fit"
        )
    
    p.legend.location = "bottom_right"
    p.legend.click_policy = "hide"
    
    # Add hover tool info
    hover = p.select_one(HoverTool)
    hover.tooltips = [("SNR", "@x{0.1f}"), ("Score", "@y{0.3f}")]
    
    return p


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
                score = score[:, 1]  # Use signal+noise probability (column 1)
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
    Downsample x, y data to a specific number of points using linear 
    interpolation.
    
    Parameters:
        - x: Original x data.
        - y: Original y data.
        - num_points: Number of points in the downsampled data.

    Returns:
        - downsampled_x, downsampled_y: Downsampled data.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Handle empty arrays
    if len(x) == 0 or len(y) == 0:
        return x, y
    
    # No downsampling needed if already small enough
    if len(x) <= num_points:
        return x, y
    
    # Remove NaN and Inf values
    valid_mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid_mask):
        return x, y  # All values invalid, return as-is
    
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    # Handle case where all x values are the same (would cause divide by zero)
    if np.all(x_valid == x_valid[0]):
        # Return single point repeated
        return np.full(num_points, x_valid[0]), np.full(num_points, np.mean(y_valid))
    
    # Remove duplicate x values (keep first occurrence) to avoid interpolation issues
    _, unique_indices = np.unique(x_valid, return_index=True)
    unique_indices = np.sort(unique_indices)  # Preserve original order
    x_unique = x_valid[unique_indices]
    y_unique = y_valid[unique_indices]
    
    # Need at least 2 points for interpolation
    if len(x_unique) < 2:
        return x, y
    
    interpolator = interp1d(x_unique, y_unique, bounds_error=False, fill_value='extrapolate')
    downsampled_x = np.linspace(np.min(x_unique), np.max(x_unique), num_points)
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

    from datetime import datetime, timedelta
    from bokeh.layouts import column, row
    
    # Don't use cycle - we index directly into colors list
    if not isinstance(colors, (list, tuple)):
        colors = list(colors)
    
    # Extract and flatten data for plotting
    # Data may have shape (detectors, samples) or just (samples,)
    onsource_data = np.array(data[gf.ReturnVariables.WHITENED_ONSOURCE.name])
    injection_data = np.array(data[gf.ReturnVariables.WHITENED_INJECTIONS.name])
    
    # Flatten multi-detector data - take first detector if multi-dimensional
    if onsource_data.ndim > 1:
        onsource_data = onsource_data.flatten() if onsource_data.shape[0] == 1 else onsource_data[0]
    if injection_data.ndim > 1:
        injection_data = injection_data.flatten() if injection_data.shape[0] == 1 else injection_data[0]
    
    # Helper to extract scalar value from data
    def get_scalar(key, default=None):
        val = data.get(key, default)
        if val is None:
            return default
        try:
            # Convert to numpy array first for consistent handling
            arr = np.asarray(val)
            if arr.ndim == 0:
                return float(arr)
            elif arr.size > 0:
                return float(arr.flatten()[0])
            else:
                return default
        except (TypeError, ValueError, IndexError):
            # Fallback for non-numeric types
            try:
                return float(val)
            except:
                return default
    
    # Extract parameters
    mass1 = get_scalar(gf.WaveformParameters.MASS_1_MSUN.name, 0)
    mass2 = get_scalar(gf.WaveformParameters.MASS_2_MSUN.name, 0)
    score = get_scalar('score', 0)
    snr = get_scalar(gf.ScalingTypes.SNR.name)
    gps_time = get_scalar(gf.ReturnVariables.GPS_TIME.name)
    
    # Convert GPS to human readable (approximate - ignores leap seconds)
    human_time = "N/A"
    if gps_time is not None and gps_time > 0:
        # GPS epoch: Jan 6, 1980. GPS is ~18s ahead of UTC due to leap seconds
        gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
        try:
            dt = gps_epoch + timedelta(seconds=float(gps_time) - 18)  # Approximate UTC
            human_time = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except:
            human_time = "Error"
    
    # Build info panel HTML
    info_items = []
    info_items.append(f"<b>Score:</b> {score:.3f}")
    if snr is not None:
        info_items.append(f"<b>SNR:</b> {snr:.1f}")
    if gps_time is not None:
        info_items.append(f"<b>GPS:</b> {gps_time:.1f}")
        info_items.append(f"<b>Time:</b> {human_time}")
    if mass1 > 0:
        info_items.append(f"<b>M₁:</b> {mass1:.1f} M☉")
    if mass2 > 0:
        info_items.append(f"<b>M₂:</b> {mass2:.1f} M☉")
    if mass1 > 0 and mass2 > 0:
        chirp_mass = ((mass1 * mass2) ** 0.6) / ((mass1 + mass2) ** 0.2)
        info_items.append(f"<b>Mchirp:</b> {chirp_mass:.1f} M☉")
    
    # Add any other WaveformParameters found in data
    extra_params = [
        (gf.WaveformParameters.INCLINATION_RADIANS.name, "Inclination", "rad"),
        (gf.WaveformParameters.DISTANCE_MPC.name, "Distance", "Mpc"),
    ]
    for param_name, label, unit in extra_params:
        val = get_scalar(param_name)
        if val is not None:
            info_items.append(f"<b>{label}:</b> {val:.2f} {unit}")
    
    info_html = "<br>".join(info_items)
    info_panel = Div(
        text=f"""
        <div style="
            background: #f8f9fa; 
            padding: 12px; 
            border-radius: 6px; 
            border: 1px solid #dee2e6;
            font-size: 11px;
            line-height: 1.6;
            min-width: 150px;
            max-width: 200px;
        ">
            <div style="font-weight: bold; margin-bottom: 8px; border-bottom: 1px solid #dee2e6; padding-bottom: 4px;">
                Injection Parameters
            </div>
            {info_html}
        </div>
        """,
        width=200
    )
    
    p = figure(
        title="Worst Performing Input",
        x_axis_label='Time (seconds)',
        y_axis_label='Whitened Strain',
        width=600, 
        height=300
    )
    
    # Create time axis
    num_samples = len(onsource_data)
    time = np.linspace(0, onsource_duration_seconds, num_samples)
    
    source = ColumnDataSource(
        data=dict(
            x=time, 
            y=onsource_data
        )
    )
    line = p.line(
        x='x', 
        y='y', 
        source=source,
        color=colors[0], 
        width=2, 
        legend_label='Whitened Strain + Injection'
    )
        
    source = ColumnDataSource(
        data=dict(
            x=time,
            y=injection_data
        )
    )
    line = p.line(
        x='x', 
        y='y', 
        source=source,
        color=colors[1], 
        width=2, 
        legend_label='Whitened Injection'
    )
    
    # Add scaled injection with interactive slider
    scaled_source = ColumnDataSource(
        data=dict(
            x=time, 
            y=injection_data,  # Store unscaled, scale via JS
            y_original=injection_data.copy()  # Keep original for scaling
        )
    )
    scaled_line = p.line(
        x='x', 
        y='y', 
        source=scaled_source,
        color=colors[2], 
        width=2, 
        legend_label='Scaled Injection'
    )
    
    # Create slider for injection scale
    scale_slider = Slider(
        start=1, 
        end=50, 
        value=20, 
        step=1, 
        title="Injection Scale"
    )
    
    # JavaScript callback to update scaled injection
    scale_callback = CustomJS(
        args=dict(source=scaled_source, slider=scale_slider),
        code="""
            const data = source.data;
            const scale = slider.value;
            const y_orig = data['y_original'];
            const y = data['y'];
            for (let i = 0; i < y_orig.length; i++) {
                y[i] = y_orig[i] * scale;
            }
            source.change.emit();
        """
    )
    scale_slider.js_on_change('value', scale_callback)
    
    p.legend.location = "bottom_right"
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "10pt"

    # Increase font sizes
    p.axis.axis_label_text_font_size = "12pt"
    p.axis.major_label_text_font_size = "10pt"
    p.title.text_font_size = '14pt'
    
    # Return plot with slider and info panel in a row layout
    plot_column = column(scale_slider, p)
    return row(plot_column, info_panel)

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
        tar_config : dict = \
        {
            "scaling_range" : (4.5, 15.0),  # SNR range for finding challenging samples
            "num_examples" : 1.0E5,
            "num_worst" : 10
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
        
        # Calculate worst performing inputs (useful for debugging)
        validator.worst_performers = None
        if tar_config is not None:
            validator.logger.info(f"Finding worst performing inputs for {validator.name}...")
            tar_scores, worst_performers = \
                calculate_tar_scores(
                    model, 
                    dataset_args, 
                    num_examples_per_batch,
                    heart=validator.heart,
                    **tar_config
                )
            validator.worst_performers = worst_performers
            validator.logger.info(f"Found {len(worst_performers)} worst performers")
            
            # Save worst performers to checkpoint file
            if checkpoint_file_path is not None and len(worst_performers) > 0:
                with gf.open_hdf5_file(
                    checkpoint_file_path, 
                    validator.logger, 
                    mode = "a"
                ) as validation_file:
                    for idx, entry in enumerate(worst_performers):
                        group_name = f'worst_performers_{idx}'
                        if group_name not in validation_file:
                            group = validation_file.create_group(group_name)
                            for key, value in entry.items():
                                if hasattr(value, '__array__'):
                                    group.create_dataset(key, data=np.array(value))
                                else:
                                    group.create_dataset(key, data=value)

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

    @classmethod
    def validate_unified(
        cls,
        model: keras.Model,
        name: str,
        dataset_args: dict,
        config: ValidationConfig = None,
        checkpoint_file_path: Path = None,
        logging_level: int = logging.INFO,
        heart: gf.Heart = None
    ):
        """
        Unified validation using single data bank.
        
        More efficient than validate() as it generates all data in one pass
        and computes efficiency, TAR, and worst performers from the same data.
        
        Args:
            model: Trained Keras model
            name: Model name for display
            dataset_args: Dataset configuration
            config: ValidationConfig instance (uses defaults if None)
            checkpoint_file_path: Path to save/load validation data
            logging_level: Logging verbosity
            heart: Heartbeat callback for progress monitoring
        """
        validator = cls()
        
        # Setup logging
        validator.logger = logging.getLogger("validator")
        stream_handler = logging.StreamHandler(sys.stdout)
        validator.logger.addHandler(stream_handler)
        validator.logger.setLevel(logging_level)
        
        validator.name = name if isinstance(name, str) else str(name)
        validator.config = config or ValidationConfig()
        
        # Extract duration from dataset_args
        validator.input_duration_seconds = dataset_args.get(
            "onsource_duration_seconds", 
            gf.Defaults.onsource_duration_seconds
        )
        
        # Create and generate unified bank
        validator.logger.info(f"=== Unified Validation for {validator.name} ===")
        
        bank = UnifiedValidationBank(
            model=model,
            dataset_args=dataset_args,
            config=validator.config,
            heart=heart,
            logger=validator.logger
        )
        bank.generate()
        
        # Store bank and extracted data
        validator.bank = bank
        validator.efficiency_data = bank.get_efficiency_data()
        validator.worst_per_bin = bank.get_worst_performers()
        
        # Note: TAR@FAR requires noise-only samples which we don't have
        # in the unified bank (injection_chance=1.0). For proper TAR,
        # use separate calculate_far_scores + calculate_tar_scores.
        validator.tar_data = None
        
        validator.logger.info("Unified validation complete!")
        
        return validator
    
    def plot_unified(
        self,
        output_path: Path = None,
        include_roc: bool = True, 
        include_worst: bool = True
    ):
        """
        Generate plots from unified validation results.
        
        Creates:
        - Efficiency scatter plot with fitted curve
        - Worst performers per SNR bin (in tabs)
        - ROC curve (if available)
        """
        if not hasattr(self, 'bank'):
            raise ValueError("Must run validate_unified() first")
        
        tabs = []
        
        # Efficiency scatter plot
        efficiency_plot = generate_efficiency_scatter_plot(self.bank)
        tabs.append(("Efficiency", pn.pane.Bokeh(efficiency_plot)))
        
        # Worst performers per SNR bin
        if include_worst and self.worst_per_bin:
            worst_tabs = []
            for bin_key, samples in self.worst_per_bin.items():
                if samples:
                    bin_plots = []
                    for sample in samples[:5]:  # Limit to 5 per bin for display
                        try:
                            plot = generate_waveform_plot(
                                sample,
                                self.input_duration_seconds
                            )
                            bin_plots.append(pn.pane.Bokeh(plot))
                        except Exception as e:
                            self.logger.warning(f"Failed to plot sample: {e}")
                    
                    if bin_plots:
                        worst_tabs.append((
                            f"SNR {bin_key}",
                            pn.Column(*bin_plots)
                        ))
            
            if worst_tabs:
                worst_panel = pn.Tabs(*worst_tabs)
                tabs.append(("Worst Performers", worst_panel))
        
        # Build final dashboard
        dashboard = pn.Tabs(*tabs)
        
        if output_path:
            output_path = Path(output_path)
            gf.ensure_directory_exists(output_path.parent)
            dashboard.save(output_path)
            self.logger.info(f"Saved unified validation report to {output_path}")
        
        return dashboard

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

            if self.worst_performers is not None:         
                worst_performers = self.worst_performers
                
                for idx, entry in enumerate(worst_performers):
                    group = validation_file.create_group(f'worst_performers_{idx}')
                    for key, value in entry.items():
                        group.create_dataset(key, data=value)
        
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
        """Generate validation dashboard using Panel Tabs."""
        gf.ensure_directory_exists(file_path.parent)

        validators = comparison_validators + [self]
        
        # Generate individual plots (Bokeh figures)
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
        
        # Create Panel tabs for each section
        # Wrap Bokeh models with pn.pane.Bokeh for proper embedding
        tabs = pn.Tabs(
            ('ROC Curves', pn.Column(
                pn.pane.Bokeh(dropdown),
                pn.pane.Bokeh(roc_curves)
            )),
            ('Efficiency', pn.Column(
                pn.pane.Bokeh(slider),
                pn.pane.Bokeh(efficiency_curves)
            )),
            ('FAR Curves', pn.pane.Bokeh(far_curves)),
        )
        
        # Add worst performers tab if available
        if self.worst_performers is not None and len(self.worst_performers) > 0:
            waveform_plots = []
            for waveform in self.worst_performers:
                if gf.ReturnVariables.WHITENED_ONSOURCE.name in waveform and \
                   gf.ReturnVariables.WHITENED_INJECTIONS.name in waveform:
                    waveform_plot = generate_waveform_plot(
                        waveform,
                        self.input_duration_seconds
                    )
                    waveform_plots.append(waveform_plot)
            
            if waveform_plots:
                tabs.append(('Worst Performers', pn.Column(*waveform_plots)))
        
        # Save tabs directly as static HTML (Templates don't support embed)
        tabs.save(str(file_path), embed=True, resources='inline')