from .dataset import get_ifo_data, O3
from typing import Dict, Tuple, Optional

import tensorflow as tf
import numpy as np

from tensorflow.data.experimental import AutoShardPolicy
from .dataset import get_ifo_data_generator, O3

def calculate_efficiency_scores(
        model : tf.keras.Model, 
        generator_args : dict,
        num_examples_per_batch : int,
        max_snr : float,
        num_snr_steps : int,
        examples_per_snr_step : int,
    ) -> np.ndarray:
    
    snr_values = \
        np.repeat(
            np.linspace(0.0, max_snr, num_snr_steps),
            examples_per_snr_step
        )
        
    injection_config = generator_args["injection_configs"][0].copy()
    injection_config.update({
        "injection_chance" : 1.0,
        "snr" : snr_values
    })
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    
    generator_args["injection_configs"] = [injection_config]

    efficiency_dataset = \
        get_ifo_data_generator(
            **generator_args
        ).with_options(options).take(
            (examples_per_snr_step*num_snr_steps) // num_examples_per_batch
        )

    # Process all examples in one go
    combined_scores = model.predict(efficiency_dataset, verbose=1)
    
    # Split predictions back into separate arrays for each SNR level
    scores = [
        combined_scores[
            i * examples_per_snr_step: (i + 1) * examples_per_snr_step
        ] for i in range(num_snr_steps)
    ]
    
    return np.array(scores)

@tf.function 
def nan_to_zero(tensor):
    return tf.where(tf.math.is_nan(tensor), 0.0, tensor)

def calculate_far_scores(
    model: tf.keras.Model,
    sample_rate_hertz: int,
    onsource_duration_seconds: float,
    ifo: str,
    time_interval = O3,
    num_examples_per_batch: int = 32,
    seed: int = 100,
    num_examples: int = int(1E5)
) -> np.ndarray:
    """
    Calculate the False Alarm Rate (FAR) scores for a given model.

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

    Returns
    -------
    far_scores : np.ndarray
        The calculated FAR scores.

    """
    
    num_examples = int(num_examples)
    
    # Setting options for data distribution
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    
    # Creating the noise dataset
    noise_ds = get_ifo_data_generator(
        time_interval = time_interval,
        data_labels = ["noise", "glitches"],
        ifo = ifo,
        sample_rate_hertz = sample_rate_hertz,
        onsource_duration_seconds = onsource_duration_seconds,
        max_segment_size = 3600,
        num_examples_per_batch = num_examples_per_batch,
        order = "random",
        seed = seed,
        apply_whitening = True,
        return_keys = ["onsource"], 
    ).map(getInput, num_parallel_calls=tf.data.AUTOTUNE)\
    .prefetch(tf.data.experimental.AUTOTUNE).with_options(options)
    
    # Calculating the number of steps for model prediction
    num_batches = int(num_examples // num_examples_per_batch)
    noise_ds = noise_ds.take(num_batches)
    
    # Predicting the scores and getting the second column ([:, 1])
    far_scores = model.predict(noise_ds, steps = num_batches, verbose=2)
    far_scores = nan_to_zero(far_scores)
    
    return far_scores[:, 1]

def calculate_far_score_thresholds(
    far_scores: np.ndarray, 
    onsource_duration_seconds: float,
    fars: np.ndarray  # Changed this to np.ndarray to leverage vectorized operations
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
def roc_curve_and_auc(y_true, y_scores, chunk_size=500):
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

def validate_roc(    
    model: tf.keras.Model,
    sample_rate_hertz: int,
    onsource_duration_seconds: float,
    ifo: str,
    time_interval: str = O3,
    num_examples: int = 1.0E5,
    num_examples_per_batch: int = 32,
    seed: int = 100
    ):
    
    injection_configs = [
        {
            "type" : "cbc",
            "snr"  : \
                {"min_value" : 10, "max_value": 50, "mean_value": 20, "std": 10,  "distribution_type": "normal"},
            "injection_chance" : 0.5,
            "padding_seconds" : {"front" : 0.2, "back" : 0.1},
            "args" : {
                "approximant_enum" : \
                    {"value" : 1, "distribution_type": "constant", "dtype" : int}, 
                "mass_1_msun" : \
                    {"min_value" : 5, "max_value": 95, "distribution_type": "uniform"},
                "mass_2_msun" : \
                    {"min_value" : 5, "max_value": 95, "distribution_type": "uniform"},
                "sample_rate_hertz" : \
                    {"value" : sample_rate_hertz, "distribution_type": "constant"},
                "duration_seconds" : \
                    {"value" : onsource_duration_seconds, "distribution_type": "constant"},
                "inclination_radians" : \
                    {"min_value" : 0, "max_value": np.pi, "distribution_type": "uniform"},
                "distance_mpc" : \
                    {"min_value" : 10, "max_value": 1000, "distribution_type": "uniform"},
                "reference_orbital_phase_in" : \
                    {"min_value" : 0, "max_value": 2*np.pi, "distribution_type": "uniform"},
                "ascending_node_longitude" : \
                    {"min_value" : 0, "max_value": 2*np.pi, "distribution_type": "uniform"},
                "eccentricity" : \
                    {"min_value" : 0, "max_value": 0.1, "distribution_type": "uniform"},
                "mean_periastron_anomaly" : \
                    {"min_value" : 0, "max_value": 2*np.pi, "distribution_type": "uniform"},
                "spin_1_in" : \
                    {"min_value" : -0.5, "max_value": 0.5, "distribution_type": "uniform", "num_values" : 3},
                "spin_2_in" : \
                    {"min_value" : -0.5, "max_value": 0.5, "distribution_type": "uniform", "num_values" : 3}
            }
        }
    ]
    
    # Setting options for data distribution
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    
     # Creating the noise dataset
    cbc_ds = get_ifo_data_generator(
        time_interval = time_interval,
        data_labels = ["noise", "glitches"],
        ifo = ifo,
        injection_configs = injection_configs,
        sample_rate_hertz = sample_rate_hertz,
        onsource_duration_seconds = onsource_duration_seconds,
        max_segment_size = 3600,
        num_examples_per_batch = num_examples_per_batch,
        order = "random",
        seed = seed,
        apply_whitening = True,
        return_keys = ["onsource"], 
    ).map(getInput, num_parallel_calls=tf.data.AUTOTUNE) \
    .prefetch(tf.data.experimental.AUTOTUNE).with_options(options)
    
    # Use .map() to extract the true labels and model inputs
    x_dataset = dataset.map(lambda x, y: x)
    y_true_dataset = dataset.map(lambda x, y: y)

    # Convert the true labels dataset to a tensor using reduce
    y_true = y_true_dataset.reduce(tf.constant([], dtype=tf.float32), concat_labels)

    # Get the model predictions
    y_scores = model.predict(x_dataset, verbose = 2)[:, 1]

    # Calculate the ROC curve and AUC
    fpr, tpr, roc_auc = roc_curve_and_auc(y_true, y_scores)

    return fpr.numpy(), tpr.numpy(), roc_auc.numpy()