import numpy as np
import jax.numpy as jnp
from jax import jit
from scipy.interpolate import interp1d
from typing import Dict, Tuple, List, Optional, Union

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
    # Avoid division by zero if duration is 0 (should shouldn't happen)
    duration = max(onsource_duration_seconds, 1e-9)
    cumulative_far = np.arange(1, n_scores + 1) / (n_scores * duration)

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
    
    # Vectorized implementation for JIT efficiency
    y_pred = jnp.expand_dims(y_scores, 1) >= thresholds
    y_pred = jnp.array(y_pred, dtype=jnp.float32)
    y_true_expanded = jnp.expand_dims(y_true, axis=-1)
    
    tp_acc = jnp.sum(y_true_expanded * y_pred, axis=0)
    fp_acc = jnp.sum((1 - y_true_expanded) * y_pred, axis=0)
    fn_acc = jnp.sum(y_true_expanded * (1 - y_pred), axis=0)
    tn_acc = jnp.sum((1 - y_true_expanded) * (1 - y_pred), axis=0)
    
    # Avoid division by zero
    tpr = tp_acc / (tp_acc + fn_acc + 1e-10)
    fpr = fp_acc / (fp_acc + tn_acc + 1e-10)

    auc = jnp.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + tpr[1:])) / 2

    return fpr, tpr, auc

def _extract_sample_data(x_batch, y_batch, sample_idx, score):
    """Extract data for a single sample from batch for worst performers."""
    element = {}
    
    # Add inputs - shape is typically (batch, detectors, samples)
    if x_batch:
        for k, v in x_batch.items():
            if hasattr(v, 'shape') and len(v.shape) >= 1:
                element[k] = np.array(v[sample_idx])
            else:
                element[k] = v
    
    # Add outputs - handle different shapes
    if y_batch:
        for k, v in y_batch.items():
            if hasattr(v, 'shape'):
                if len(v.shape) == 4:  # (generators, batch, det, samples)
                    # Check first dim bound
                    if v.shape[0] > 0:
                        element[k] = np.array(v[0, sample_idx])
                    else:
                        element[k] = np.array(v[sample_idx]) # Fallback
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
