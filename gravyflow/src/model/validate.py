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
import holoviews as hv
from holoviews.operation.datashader import datashade
import pandas as pd


@dataclass
class ValidationConfig:
    """Configuration for unified validation pipeline."""
    snr_range: Tuple[float, float] = (0.0, 20.0)
    num_examples: int = 100_000
    batch_size: int = 512
    snr_bin_width: float = 5.0
    num_worst_per_bin: int = 5
    far_thresholds: List[float] = field(
        default_factory=lambda: np.logspace(-1, -4.5, 50).tolist()
    )
    

class ValidationBank:
    """
    Validation Bank for managing noise and injection data generation.
    
    Supports two-pass validation:
    1. generate_noise(): Calculates scores on noise-only data for FAR.
    2. generate(): Calculates scores on injection data for Efficiency/TAR.
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
        self.far_scores = None        # From generate_noise()
        self.snrs = None              # From generate()
        self.scores = None            # From generate()
        self.injection_masks = None   # From generate()
        self._worst_per_bin = None    # From generate()
        
    def generate_noise(self) -> None:
        """
        Generate noise-only scores for False Alarm Rate (FAR) calculation.
        Sets injection_chance = 0.0.
        """
        config = self.config
        dataset_args = deepcopy(self.dataset_args)
        
        # Configure for noise-only (no injections)
        dataset_args["waveform_generators"] = []
        dataset_args["output_variables"] = []  # No injection masks/params needed
        
        # Calculate number of batches for desired duration
        num_examples = int(config.num_examples) # Use num_examples or calculate from duration?
        # Legacy used num_seconds = 1E5. Let's use config.num_examples or derived.
        # If config doesn't have num_seconds, we stick to num_examples.
        # But FAR usually needs stats over TIME. 
        # config.num_examples is 100,000. 
        
        num_batches = math.ceil(config.num_examples / config.batch_size)
        dataset_args["num_examples_per_batch"] = config.batch_size
        dataset_args["steps_per_epoch"] = num_batches
        dataset_args["group"] = "test"
        
        dataset = gf.Dataset(**dataset_args)
        
        self.logger.info(f"Generating noise data: {num_batches} batches ({config.num_examples} examples)")
        
        all_scores = []
        
        for batch_idx in range(num_batches):
            if self.heart:
                self.heart.beat()
            
            x_batch, _ = dataset[batch_idx]
            
            # Predict
            predictions = self.model.predict_on_batch(x_batch)
            
            # Handle output shape
            if len(predictions.shape) == 2:
                if predictions.shape[1] == 2:
                    batch_scores = predictions[:, 1]
                else:
                    batch_scores = predictions[:, 0]
            else:
                batch_scores = predictions.flatten()
            
            all_scores.append(batch_scores)
            
            if (batch_idx + 1) % 50 == 0:
                self.logger.info(f"Noise Progress: {batch_idx + 1}/{num_batches} batches")
        
        self.far_scores = np.concatenate([np.array(s) for s in all_scores])
        self.logger.info(f"Generated {len(self.far_scores)} noise scores")

    def generate(self) -> None:
        """
        Generate injection scores for Efficiency calculation.
        Sets injection_chance = 1.0 and samples Uniform SNR.
        """
        config = self.config
        dataset_args = deepcopy(self.dataset_args)
        
        min_snr, max_snr = config.snr_range
        
        if "waveform_generators" not in dataset_args:
            raise ValueError("dataset_args must contain waveform_generators")
        if not isinstance(dataset_args["waveform_generators"], list):
            dataset_args["waveform_generators"] = [dataset_args["waveform_generators"]]
            
        for gen in dataset_args["waveform_generators"]:
            gen.scaling_method.value = gf.Distribution(
                min_=min_snr,
                max_=max_snr,
                type_=gf.DistributionType.UNIFORM
            )
            gen.injection_chance = 1.0
            
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
        dataset_args["group"] = "test"
        
        dataset = gf.Dataset(**dataset_args)
        
        all_snrs = []
        all_scores = []
        all_masks = []
        
        # Worst performer tracking
        num_bins = int((max_snr - min_snr) / config.snr_bin_width)
        worst_heaps = [[] for _ in range(num_bins)]
        
        self.logger.info(f"Generating injection data: {num_batches} batches")
        
        for batch_idx in range(num_batches):
            if self.heart:
                self.heart.beat()
                
            x_batch, y_batch = dataset[batch_idx]
            
            predictions = self.model.predict_on_batch(x_batch)
            
            if len(predictions.shape) == 2:
                if predictions.shape[1] == 2:
                    batch_scores = predictions[:, 1]
                else:
                    batch_scores = predictions[:, 0]
            else:
                batch_scores = predictions.flatten()
                
            batch_snrs = np.array(y_batch.get("SNR", np.zeros(len(batch_scores))))
            # Injections are 100%, so marks are arguably all 1, but let's read if available
            batch_masks = np.ones(len(batch_scores)) 
            
            all_snrs.extend(batch_snrs)
            all_scores.extend(batch_scores)
            all_masks.extend(batch_masks)
            
            # Worst performers
            for i in range(len(batch_scores)):
                snr_val = float(batch_snrs[i])
                score_val = float(batch_scores[i])
                
                bin_idx = min(int((snr_val - min_snr) / config.snr_bin_width), num_bins - 1)
                bin_idx = max(0, bin_idx)
                
                if len(worst_heaps[bin_idx]) < config.num_worst_per_bin:
                    sample_data = _extract_sample_data(x_batch, y_batch, i, score_val)
                    heapq.heappush(worst_heaps[bin_idx], (-score_val, batch_idx * config.batch_size + i, sample_data))
                elif score_val < -worst_heaps[bin_idx][0][0]:
                    sample_data = _extract_sample_data(x_batch, y_batch, i, score_val)
                    heapq.heapreplace(worst_heaps[bin_idx], (-score_val, batch_idx * config.batch_size + i, sample_data))
            
            if (batch_idx + 1) % 50 == 0:
                self.logger.info(f"Injection Progress: {batch_idx + 1}/{num_batches} batches")
                
        self.snrs = np.array(all_snrs)
        self.scores = np.array(all_scores)
        self.injection_masks = np.array(all_masks)
        
        self._worst_per_bin = {}
        for bin_idx, heap in enumerate(worst_heaps):
            bin_start = min_snr + bin_idx * config.snr_bin_width
            bin_end = bin_start + config.snr_bin_width
            bin_key = f"{bin_start:.0f}-{bin_end:.0f}"
            sorted_worst = sorted(heap, key=lambda x: -x[0])
            self._worst_per_bin[bin_key] = [item[2] for item in sorted_worst]
            
    def get_efficiency_data(self) -> Dict:
        """
        Get efficiency scatter data and fitted curve.
        
        Returns dict with:
        - snrs: array of SNR values
        - scores: array of model scores
        - fit_snrs: x values for fitted curve
        - fit_efficiency: y values for fitted curve
        - bin_centers, bin_means, bin_stds: binned statistics
        """
        if self.snrs is None:
            raise ValueError("Must call generate() first")
        
        # All samples are injections (injection_chance=1.0)
        snrs = self.snrs
        scores = self.scores
        
        # Bin statistics for confidence bands
        config = self.config
        min_snr, max_snr = config.snr_range
        num_bins = max(1, int((max_snr - min_snr) / config.snr_bin_width))
        
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
        
        # Fit sigmoid curve
        def sigmoid(x, x0, k, L, b):
            return L / (1 + np.exp(-k * (x - x0))) + b
        
        try:
            popt, _ = curve_fit(
                sigmoid, bin_centers, bin_means,
                p0=[10.0, 0.5, 0.8, 0.1],
                bounds=([0, 0.01, 0, 0], [30, 5, 1, 0.5]),
                maxfev=5000
            )
            
            fit_snrs = np.linspace(min_snr, max_snr, 100)
            fit_efficiency = sigmoid(fit_snrs, *popt)
        except:
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
    
    def get_worst_performers(self) -> Dict[str, List[dict]]:
        """Get worst performing samples per SNR bin."""
        if self._worst_per_bin is None:
            raise ValueError("Must call generate() first")
        return self._worst_per_bin

    def get_roc_curves(self, scaling_ranges: List[Union[Tuple[float, float], float]] = None) -> Dict:
        """
        Calculate ROC curves for different SNR ranges/values using generated data.
        Reuses far_scores (negatives) and filters injection scores (positives) by SNR.
        """
        if self.far_scores is None or self.scores is None:
            raise ValueError("Must call generate_noise() and generate() first")
            
        if scaling_ranges is None:
            # Default ranges if none provided
            scaling_ranges = [
                (8.0, 20.0),
                8.0,
                10.0,
                12.0
            ]
            
        results = {}
        noise_scores = self.far_scores
        
        # Ensure we have noise scores
        if len(noise_scores) == 0:
            self.logger.warning("No noise scores found. ROC calculation requires noise data.")
            return {}
        
        for scaling in scaling_ranges:
            # Determine SNR mask
            if isinstance(scaling, (tuple, list)):
                min_s, max_s = scaling
                mask = (self.snrs >= min_s) & (self.snrs < max_s)
                # Filter injection mask too (ensure they are injections)
                mask &= (self.injection_masks > 0.5)
                key = str(scaling)
            else:
                # Point estimate: use small bin
                center = float(scaling)
                width = 0.5 # +/- 0.5 width
                mask = (self.snrs >= center - width) & (self.snrs < center + width)
                mask &= (self.injection_masks > 0.5)
                key = str(center)
            
            signal_scores = self.scores[mask]
            
            if len(signal_scores) == 0:
                self.logger.warning(f"No signal samples found for SNR range {key}")
                continue
            
            # Combine for ROC
            y_scores = np.concatenate([noise_scores, signal_scores])
            y_true = np.concatenate([np.zeros(len(noise_scores)), np.ones(len(signal_scores))])
            
            # Use JAX helper function defined globally
            fpr, tpr, auc = roc_curve_and_auc(y_true, y_scores)
            
            results[key] = {
                "fpr": np.array(fpr),
                "tpr": np.array(tpr),
                "roc_auc": float(auc)
            }
            
        return results



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
    import pandas as pd
    
    # Don't use cycle - we index directly into colors list
    if not isinstance(colors, (list, tuple)):
        colors = list(colors)
    
    # Extract and flatten data for plotting
    onsource_data = np.array(data[gf.ReturnVariables.WHITENED_ONSOURCE.name])
    injection_data = np.array(data[gf.ReturnVariables.WHITENED_INJECTIONS.name])
    
    # Flatten multi-detector data - take first detector if multi-dimensional
    if onsource_data.ndim > 1:
        onsource_data = onsource_data.flatten() if onsource_data.shape[0] == 1 else onsource_data[0]
    if injection_data.ndim > 1:
        injection_data = injection_data.flatten() if injection_data.shape[0] == 1 else injection_data[0]
    
    # Cast onsource to float32 for Datashader (avoids float16 error)
    onsource_data = onsource_data.astype(np.float32)
    # Cast injection to float64 for Bokeh/JS compatibility (avoids serialization/browser errors)
    injection_data = injection_data.astype(np.float64)
    
    # Helper to extract scalar value from data
    def get_scalar(key, default=None):
        val = data.get(key, default)
        if val is None:
            return default
        try:
            arr = np.asarray(val)
            if arr.ndim == 0:
                return float(arr)
            elif arr.size > 0:
                return float(arr.flatten()[0])
            else:
                return default
        except:
             return default
    
    # Extract parameters
    mass1 = get_scalar(gf.WaveformParameters.MASS_1_MSUN.name, 0)
    mass2 = get_scalar(gf.WaveformParameters.MASS_2_MSUN.name, 0)
    score = get_scalar('score', 0)
    snr = get_scalar(gf.ScalingTypes.SNR.name)
    GPS_TIME_KEY = gf.ReturnVariables.GPS_TIME.name
    gps_time = get_scalar(GPS_TIME_KEY)
    
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
    
    # Create time axis
    num_samples = len(onsource_data)
    time = np.linspace(0, onsource_duration_seconds, num_samples)
    
    p = figure(
        title="Worst Performing Input",
        x_axis_label='Time (seconds)',
        y_axis_label='Whitened Strain',
        width=600, 
        height=300
    )

    # Plot Onsource Noise (Full Resolution, Standard Bokeh)
    source = ColumnDataSource(
        data=dict(
            x=time, 
            y=onsource_data
        )
    )
    p.line(
        x='x', 
        y='y', 
        source=source,
        color=colors[0], 
        width=1,  # Thinner line for dense noise
        legend_label='Whitened Strain + Injection'
    )

    # Now add the interactive lines (Injection) using standard Bokeh
    # These remain as vectors so they are sharp and interact fast with JS
    
    # Downsample injection data for client-side plotting (reduce HTML size)
    MAX_DISPLAY_POINTS = 1500
    if len(injection_data) > MAX_DISPLAY_POINTS:
        # Simple decimation is sufficient for visual line inspection
        step = int(np.ceil(len(injection_data) / MAX_DISPLAY_POINTS))
        injection_plot = injection_data[::step]
        time_plot = time[::step]
    else:
        injection_plot = injection_data
        time_plot = time

    # Injection Line (Static)
    source_inj = ColumnDataSource(data=dict(x=time_plot, y=injection_plot))
    p.line(x='x', y='y', source=source_inj, color=colors[1], width=2, legend_label='Whitened Injection')
    
    # Scaled Injection Line (Interactive)
    scaled_source = ColumnDataSource(
        data=dict(
            x=time_plot, 
            y=injection_plot, 
            y_original=injection_plot.copy()
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
    
    # Add hover tool for scaled injection
    hover = HoverTool(renderers=[scaled_line], tooltips=[("Time", "@x{0.000}"), ("Strain", "@y")])
    p.add_tools(hover)
    
    # Create slider for injection scale
    scale_slider = Slider(start=1, end=2, value=1, step=1, title="Injection Scale")
    
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
    
    # Info Panel
    info_items = []
    info_items.append(f"<b>Score:</b> {score:.3f}")
    if snr: info_items.append(f"<b>SNR:</b> {snr:.1f}")
    if gps_time is not None:
        info_items.append(f"<b>GPS:</b> {gps_time:.1f}")
        info_items.append(f"<b>Time:</b> {human_time}")
    if mass1: info_items.append(f"<b>M₁:</b> {mass1:.1f} M☉")
    if mass2: info_items.append(f"<b>M₂:</b> {mass2:.1f} M☉")
    
    info_html = "<br>".join(info_items)
    info_panel = Div(
        text=f"""<div style="background:#f8f9fa;padding:12px;border:1px solid #dee2e6;font-size:11px;">
            <b>Parameters</b><br>{info_html}</div>""",
        width=200
    )
    
    return row(column(scale_slider, p), info_panel)

def generate_efficiency_plot(
    validators: list,
    fars: List[float] = [1e-1, 1e-2, 1e-3, 1e-4],
    colors: List[str] = Bright[7],
    width: int = 800,
    height: int = 600
):
    """
    Generate interactive efficiency plot with slider for FAR.
    Calculates efficiency curves (Recall vs SNR) for each FAR.
    """
    colors = cycle(colors)
    
    p = figure(
        #title="Efficiency vs SNR",
        x_axis_label="SNR",
        y_axis_label="Efficiency (Recall) / Score",
        y_range=(0, 1.05),
        width=width,
        height=height,
        tools="pan,box_zoom,wheel_zoom,reset,hover"
    )
    
    # Add Datashaded Scatter of Scores (Background)
    all_snrs = []
    all_scores = []
    # Collect data from all validators
    for validator in validators:
        if validator.efficiency_data:
             all_snrs.append(validator.efficiency_data["snrs"])
             all_scores.append(validator.efficiency_data["scores"])
    
    if all_snrs:
        combined_snrs = np.concatenate(all_snrs).astype(np.float32)
        combined_scores = np.concatenate(all_scores).astype(np.float32)
        
        # Use HoloViews for datashading
        points = hv.Points((combined_snrs, combined_scores))
        shaded = datashade(points, cmap=["grey"]) # Grey background for scores
        
        # Overlay datashaded image
        # Note: hv.render creates a new figure, but we want to put it into 'p'.
        # Actually simplest is to render the shaded object to a figure properties, 
        # OR use hv.render(shaded, backend='bokeh') and extract the renderer.
        # BUT getting the renderer out is tricky.
        # EASIER: Use hv.save scheme? No.
        # PROPER WAY: Use hv.render to get a figure, then add our lines to THAT figure.
        
        # Re-create figure from HoloViews
        p = hv.render(shaded.opts(width=width, height=height, xlim=(0, 20), ylim=(0, 1.05)), backend='bokeh')
        p.xaxis.axis_label = "SNR"
        p.yaxis.axis_label = "Efficiency (Recall) / Score"
        p.tools += [HoverTool()] # Re-add hover since render might reset tools

    hover = p.select_one(HoverTool)
    if hover:
        hover.tooltips = [("SNR", "@x{0.1f}"), ("Efficiency", "@y{0.3f}")]
    else:
         # Fallback if select_one failed
         hover = HoverTool(tooltips=[("SNR", "@x{0.1f}"), ("Efficiency", "@y{0.3f}")])
         p.add_tools(hover)
    
    all_data = {}
    sources = {}
    all_thresholds = {}
    thresh_sources = {}
    initial_far = fars[0]
    
    for i, (validator, color) in enumerate(zip(validators, colors)):
        name = validator.name or f"Model {i}"
        
        if validator.efficiency_data is None or validator.far_scores is None:
            continue
            
        snrs = validator.efficiency_data["snrs"]
        scores = validator.efficiency_data["scores"]
        far_scores = validator.far_scores
        
        # Calculate thresholds for requested FARs
        thresholds = calculate_far_score_thresholds(
            far_scores,
            validator.input_duration_seconds,
            np.array(fars)
        )
        
        # Binning setup
        min_snr = np.floor(snrs.min())
        max_snr = np.ceil(snrs.max())
        bin_width = 1.0
        bins = np.arange(min_snr, max_snr + bin_width, bin_width)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        model_curves = {}
        
        for far in fars:
            if far not in thresholds:
                continue
                
            thresh = thresholds[far][1]
            recalls = []
            valid_centers = []
            
            for b_idx in range(len(bins)-1):
                b_min, b_max = bins[b_idx], bins[b_idx+1]
                mask = (snrs >= b_min) & (snrs < b_max)
                if np.sum(mask) > 10:
                    recall = np.mean(scores[mask] > thresh)
                    recalls.append(recall)
                    valid_centers.append(bin_centers[b_idx])
            
            try:
                def sigmoid(x, x0, k): return 1 / (1 + np.exp(-k * (x - x0)))
                popt, _ = curve_fit(sigmoid, valid_centers, recalls, p0=[10, 1], maxfev=5000)
                x_fit = np.linspace(min_snr, max_snr, 100)
                y_fit = sigmoid(x_fit, *popt)
                model_curves[f"{far:.1e}"] = {"x": x_fit, "y": y_fit}
            except:
                 model_curves[f"{far:.1e}"] = {"x": valid_centers, "y": recalls}

        all_data[name] = model_curves
        
        # Store thresholds for this validator
        thresh_map = {}
        for far in fars:
            if far in thresholds:
                thresh_map[f"{far:.1e}"] = thresholds[far][1]
            else:
                thresh_map[f"{far:.1e}"] = 0.0 # Should not happen if filtered
        
        all_thresholds[name] = thresh_map
        
        init_curve = model_curves.get(f"{initial_far:.1e}", {"x": [], "y": []})
        source = ColumnDataSource(data=dict(x=init_curve["x"], y=init_curve["y"]))
        sources[name] = source
        
        p.line(
            x='x', y='y', source=source,
            line_width=3, color=color, legend_label=name
        )
        
        # Add dynamic threshold line
        init_thresh = thresh_map.get(f"{initial_far:.1e}", 0.0)
        thresh_source = ColumnDataSource(data=dict(x=[0, 20], y=[init_thresh, init_thresh]))
        thresh_sources[name] = thresh_source
        
        p.line(
            x='x', y='y', source=thresh_source,
            line_width=2, color=color, line_dash='dashed', alpha=0.7,
            legend_label=f"{name} Threshold"
        )
    
    p.legend.location = "bottom_right"
    p.legend.click_policy = "hide"
    
    far_options = [f"{f:.1e}" for f in fars]
    slider = Slider(
        start=0, 
        end=len(far_options)-1, 
        value=0, 
        step=1, 
        title=f"False Alarm Rate (FAR): {far_options[0]}"
    )
    
    code = """
        const far_index = cb_obj.value;
        const far_val = far_options[far_index];
        cb_obj.title = "False Alarm Rate (FAR): " + far_val;
        
        for (const name in sources) {
            const source = sources[name];
            const data = all_data[name][far_val];
            if (data) {
                source.data.x = data.x;
                source.data.y = data.y;
                source.change.emit();
            }
            
            // Update threshold line
            const thresh_source = thresh_sources[name];
            const thresh = all_thresholds[name][far_val];
            if (thresh_source && thresh !== undefined) {
                thresh_source.data.y = [thresh, thresh];
                thresh_source.change.emit();
            }
        }
    """
    slider.js_on_change('value', CustomJS(args=dict(sources=sources, all_data=all_data, far_options=far_options, thresh_sources=thresh_sources, all_thresholds=all_thresholds), code=code))
    
    return p, slider

class Validator:
    
    @classmethod
    def validate(
        cls, 
        model : keras.Model, 
        name : str,
        dataset_args : dict,
        config: ValidationConfig = None,
        checkpoint_file_path : Path = None,
        logging_level : int = logging.INFO,
        heart : gf.Heart = None,
        **kwargs # Absorb legacy kwargs
    ):
        validator = cls()
        
        validator.logger = logging.getLogger("validator")
        if not validator.logger.handlers:
            stream_handler = logging.StreamHandler(sys.stdout)
            validator.logger.addHandler(stream_handler)
        validator.logger.setLevel(logging_level)
        
        validator.name = str(name) if name else "model"
        validator.config = config or ValidationConfig()
        
        validator.input_duration_seconds = dataset_args.get(
            "onsource_duration_seconds", 
            gf.Defaults.onsource_duration_seconds
        )
        
        validator.efficiency_data = None
        validator.far_scores = None
        validator.roc_data = None
        validator.worst_performers = None
        
        if checkpoint_file_path and checkpoint_file_path.exists():
            try:
                validator = validator.load(checkpoint_file_path, logging_level)
                if validator.efficiency_data is not None and validator.far_scores is not None:
                    validator.config = config or ValidationConfig()
                    validator.logger.info("Loaded complete validation data from checkpoint.")
                    return validator
            except Exception as e:
                validator.logger.warning(f"Failed to load checkpoint: {e}. Regenerating.")
        
        bank = ValidationBank(
            model=model,
            dataset_args=dataset_args,
            config=validator.config,
            heart=heart,
            logger=validator.logger
        )
        
        if validator.far_scores is None:
            bank.generate_noise()
            validator.far_scores = bank.far_scores
            
        if validator.efficiency_data is None:
            bank.generate()
            validator.efficiency_data = bank.get_efficiency_data()
            validator.worst_performers = bank.get_worst_performers()
            
        validator.roc_data = bank.get_roc_curves()
        
        if checkpoint_file_path:
            validator.save(checkpoint_file_path)
            
        validator.bank = bank
        return validator

    def save(self, file_path: Path):
        self.logger.info(f"Saving validation data to {file_path}")
        gf.ensure_directory_exists(file_path.parent)
        
        with gf.open_hdf5_file(file_path, self.logger, mode="w") as f:
            f.create_dataset('name', data=self.name.encode())
            f.create_dataset('input_duration_seconds', data=self.input_duration_seconds)
            
            if self.efficiency_data:
                g = f.create_group('efficiency_data')
                g.create_dataset('snrs', data=self.efficiency_data['snrs'])
                g.create_dataset('scores', data=self.efficiency_data['scores'])
                
            if self.far_scores is not None:
                g = f.create_group('far_scores')
                g.create_dataset('scores', data=self.far_scores)
                
            if self.roc_data:
                g = f.create_group('roc_data')
                keys = list(self.roc_data.keys())
                g.create_dataset('keys', data=np.array([k.encode() for k in keys], dtype=h5py.string_dtype()))
                for k, v in self.roc_data.items():
                    sub = g.create_group(k)
                    sub.create_dataset('fpr', data=v['fpr'])
                    sub.create_dataset('tpr', data=v['tpr'])
                    sub.create_dataset('roc_auc', data=v['roc_auc'])
            
            if self.worst_performers:
                g = f.create_group('worst_performers')
                for bin_key, samples in self.worst_performers.items():
                    bg = g.create_group(bin_key)
                    for i, sample in enumerate(samples):
                        sg = bg.create_group(f"sample_{i}")
                        for k, v in sample.items():
                            if isinstance(v, np.ndarray):
                                sg.create_dataset(k, data=v)
                            else:
                                sg.create_dataset(k, data=v)

    @classmethod
    def load(cls, file_path: Path, logging_level=logging.INFO):
        validator = cls()
        validator.logger = logging.getLogger("validator")
        validator.logger.setLevel(logging_level)
        
        with h5py.File(file_path, 'r') as f:
            validator.name = f['name'][()].decode() if 'name' in f else "Unknown"
            validator.input_duration_seconds = float(f['input_duration_seconds'][()]) if 'input_duration_seconds' in f else 1.0
            
            if 'efficiency_data' in f:
                g = f['efficiency_data']
                validator.efficiency_data = {
                    "snrs": g['snrs'][:],
                    "scores": g['scores'][:]
                }
            
            if 'far_scores' in f:
                validator.far_scores = f['far_scores']['scores'][:]
                
            if 'roc_data' in f:
                g = f['roc_data']
                validator.roc_data = {}
                keys = [k.decode() for k in g['keys'][:]]
                for k in keys:
                    if k in g:
                        sub = g[k]
                        validator.roc_data[k] = {
                            "fpr": sub['fpr'][:],
                            "tpr": sub['tpr'][:],
                            "roc_auc": float(sub['roc_auc'][()])
                        }
            
            if 'worst_performers' in f:
                g = f['worst_performers']
                validator.worst_performers = {}
                for bin_key in g:
                    validator.worst_performers[bin_key] = []
                    bg = g[bin_key]
                    for sample_key in bg:
                        sg = bg[sample_key]
                        sample = {}
                        for k in sg:
                            sample[k] = sg[k][()]
                        validator.worst_performers[bin_key].append(sample)

        return validator

    def plot(self, output_path: Path = None, comparison_validators: list = []):
        """Generate validation dashboard."""
        
        all_validators = comparison_validators + [self]
        
        eff_plot, eff_slider = generate_efficiency_plot(
            all_validators, 
            fars=self.config.far_thresholds
        )
        far_plot = generate_far_curves(all_validators)
        roc_plot, roc_select = generate_roc_curves(all_validators)
        
        tabs = pn.Tabs(
            ("Efficiency", pn.Column(eff_slider, eff_plot)),
            ("ROC Curves", pn.Column(roc_select, roc_plot)),
            ("FAR Curves", far_plot)
        )
        
        if self.worst_performers:
            worst_tabs = []
            sorted_bins = sorted(self.worst_performers.keys(), key=lambda x: float(str(x).split('-')[0]) if '-' in str(x) else 0)
            
            for bin_key in sorted_bins:
                samples = self.worst_performers[bin_key]
                if not samples: continue
                
                plots = []
                for sample in samples[:10]:
                     try:
                         plots.append(generate_waveform_plot(sample, self.input_duration_seconds))
                     except Exception as e:
                         self.logger.warning(f"Error plotting sample: {e}")
                
                if plots:
                    worst_tabs.append((f"SNR {bin_key}", pn.Column(*plots)))
            
            if worst_tabs:
                tabs.append(("Worst Performers", pn.Tabs(*worst_tabs)))
                
        if output_path:
            gf.ensure_directory_exists(Path(output_path).parent)
            tabs.save(str(output_path), embed=True, resources='inline')
            self.logger.info(f"Saved report to {output_path}")
            
        return tabs