# =============================================================================
# Glitch Classification Training Example
# =============================================================================
# Train a Keras classifier to classify glitch types from whitened strain data.
# Uses TransientObtainer with new augmentation options.

import gravyflow as gf
import numpy as np
from keras import ops, layers, models, callbacks
from bokeh.io import output_file, save
from bokeh.layouts import column

# =============================================================================
# Configuration
# =============================================================================

SAMPLE_RATE = 2048.0
ONSOURCE_DURATION = 1.0
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = len(gf.GlitchType)

print(f"Glitch types: {NUM_CLASSES}")
for i, gt in enumerate(gf.GlitchType):
    print(f"  {i}: {gt.name}")

# =============================================================================
# Create Data Obtainer with Augmentation
# =============================================================================

ifo_data_obtainer = gf.IFODataObtainer(
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.GLITCHES],
    observing_runs=[gf.ObservingRun.O3],
    saturation=1.0,
    # Augmentations for training variety
    random_sign_reversal=True,
    random_time_reversal=True,
    random_shift=True,       # Move glitch off-center
    shift_fraction=0.2,      # Up to 20% shift
    add_noise=True,          # Add noise perturbations
    noise_amplitude=0.05,    # Small noise
    augmentation_probability=0.5,
)

# Create TransientObtainer for glitches
glitch_obtainer = gf.TransientObtainer(
    ifo_data_obtainer=ifo_data_obtainer,
    ifos=[gf.IFO.L1],
)

# =============================================================================
# Simple Classification Model
# =============================================================================

def create_classifier(input_shape, num_classes):
    """Create a 1D CNN classifier for glitch classification."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(32, 7, activation='relu'),
        layers.MaxPooling1D(4),
        layers.Conv1D(64, 5, activation='relu'),
        layers.MaxPooling1D(4),
        layers.Conv1D(128, 3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# =============================================================================
# Load Training Data
# =============================================================================

print("\nAcquiring glitch data...")

# Create generator with cropping and whitening
glitch_generator = glitch_obtainer(
    sample_rate_hertz=SAMPLE_RATE,
    onsource_duration_seconds=ONSOURCE_DURATION,
    offsource_duration_seconds=16.0,
    num_examples_per_batch=BATCH_SIZE * 10,  # Get multiple batches worth
    crop=True,
    whiten=True,
    seed=42
)

# Get training data (single large batch for demo)
try:
    onsource, offsource, gps_times = next(glitch_generator)
    print(f"Acquired {onsource.shape[0]} samples")
    print(f"Shape: {onsource.shape}")
    
    # For demo: create synthetic labels (in real use, track from source)
    # This assigns labels round-robin - in production, would track actual type
    num_samples = onsource.shape[0]
    labels = np.array([i % NUM_CLASSES for i in range(num_samples)])
    labels_onehot = np.eye(NUM_CLASSES)[labels]
    
    # Reshape for 1D CNN: (batch, samples) -> (batch, samples, 1)
    X = ops.convert_to_numpy(onsource[:, 0, :])  # Take first IFO
    X = X.reshape(-1, X.shape[-1], 1)  # Add channel dim
    y = labels_onehot
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
except Exception as e:
    print(f"Error: {e}")
    X, y = None, None

# =============================================================================
# Train Model
# =============================================================================

if X is not None:
    # Split train/val
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"\nTraining: {len(X_train)} samples, Validation: {len(X_val)} samples")
    
    # Create and compile model
    input_shape = (X.shape[1], 1)
    model = create_classifier(input_shape, NUM_CLASSES)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )
    
    print(f"\nFinal validation accuracy: {history.history['val_accuracy'][-1]:.3f}")
