import numpy as np
from scipy.interpolate import interp1d


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
