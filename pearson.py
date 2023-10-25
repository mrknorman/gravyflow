import tensorflow as tf

# Compute Pearson correlation
@tf.function
def pearson(x, y):

    mean_x, mean_y = tf.reduce_mean(x, -1, keepdims=True), tf.reduce_mean(y, -1, keepdims=True)
    numerator = tf.reduce_sum((x - mean_x) * (y - mean_y), axis=-1)
    denominator = tf.sqrt(tf.reduce_sum(tf.square(x - mean_x), axis=-1) * tf.reduce_sum(tf.square(y - mean_y), axis=-1))
    
    return numerator / (denominator + 1e-5) # Adding epsilon for stability

@tf.function
def rolling_pearson(
        tensor: tf.Tensor, 
        max_arival_time_difference_seconds: float,
        sample_rate_hertz: float
    ) -> tf.Tensor:
    
    # Calculate max arrival time difference in samples
    max_arival_time_difference_samples = int(max_arival_time_difference_seconds * sample_rate_hertz)

    # Create pairs of indices for the arrays (for non-duplicate pairs)
    NUM_BATCHES, NUM_ARRAYS, ARRAY_SIZE = tensor.shape
    i, j = tf.meshgrid(tf.range(NUM_ARRAYS), tf.range(NUM_ARRAYS), indexing='ij')
    
    # Only consider unique pairs (i < j)
    mask = i < j
    i, j = tf.boolean_mask(i, mask), tf.boolean_mask(j, mask)
    NUM_PAIRS = tf.shape(i)[0]

    # Create a tensor array to store correlations for all pairs
    all_correlations = tf.TensorArray(dtype=tf.float32, size=NUM_PAIRS, dynamic_size=True)
    
    for pair_idx in tf.range(NUM_PAIRS):
        x = tensor[:, i[pair_idx], :]
        y = tensor[:, j[pair_idx], :]

        # Expand dimensions for broadcasting
        x = tf.expand_dims(x, axis=-2)  # shape: [NUM_BATCHES, ARRAY_SIZE, 1]
        
        y_collect = tf.TensorArray(dtype=tf.float32, size=max_arival_time_difference_samples, dynamic_size=True)
        for offset in tf.range(max_arival_time_difference_samples):
            y_collect = y_collect.write(offset, tf.roll(y, shift=-offset, axis=-1))

        # Shift y for all possible offsets
        y_shifted = tf.transpose(y_collect.stack(), [1, 0, 2])
        
        # Compute correlations using broadcasting
        corr = pearson(x, y_shifted)
                 
        corr = tf.expand_dims(corr, axis=1)  # Add an additional axis for pair
        all_correlations = all_correlations.write(pair_idx, corr)
        
    # Concatenate the results
    correlations = all_correlations.stack()
    correlations = tf.reshape(correlations, [NUM_BATCHES, NUM_PAIRS, max_arival_time_difference_samples])

    return correlations