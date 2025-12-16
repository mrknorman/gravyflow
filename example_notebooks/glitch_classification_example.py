# =============================================================================
# Glitch Classification Training Example (Production Ready)
# =============================================================================
# Train a Keras classifier to classify glitch types using GravyflowDataset.
# This follows the style of 07_training_a_model.ipynb.

# Built-in imports
from typing import List, Dict
from pathlib import Path
import os

# Import the GravyFlow module
import gravyflow as gf

# Dependency imports
import numpy as np
import keras
from keras import ops
from keras.layers import Input, Permute, Conv1D, MaxPooling1D, Dense, Flatten, Dropout, ELU
from keras.models import Model

# =============================================================================
# Configuration
# =============================================================================

SAMPLE_RATE = 2048.0
ONSOURCE_DURATION = 1.0
OFFSOURCE_DURATION = 16.0
BATCH_SIZE = 32
STEPS_PER_EPOCH = 5000  # ~160K samples per epoch
VALIDATION_STEPS = 1000  # ~32K validation samples
EPOCHS = 100
NUM_CLASSES = len(gf.GlitchType)

# Output directory for checkpoints and logs
OUTPUT_DIR = Path("training_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Glitch types: {NUM_CLASSES}")
for i, gt in enumerate(gf.GlitchType):
    print(f"  {i}: {gt.name}")

# =============================================================================
# Define the Classification Model
# =============================================================================

def create_glitch_classifier(
        input_shape_onsource: tuple,
        input_shape_offsource: tuple,
        num_classes: int
    ) -> keras.Model:
    """
    Create a CNN classifier for glitch classification.
    Uses whitening layer and outputs softmax probabilities.
    """
    # Define inputs
    onsource_input = Input(shape=input_shape_onsource, name="ONSOURCE")
    offsource_input = Input(shape=input_shape_offsource, name="OFFSOURCE")
    
    # Whiten the data
    x = gf.Whiten()([onsource_input, offsource_input])
    
    # Permute for Conv1D: (IFO, Samples) -> (Samples, IFO)
    x = Permute((2, 1))(x)
    
    # Convolutional layers
    x = Conv1D(8, 64, padding='valid', name="Conv_1")(x)
    x = ELU(name="ELU_1")(x)
    x = MaxPooling1D(pool_size=4, strides=4, name="Pool_1")(x)
    
    x = Conv1D(16, 32, padding='valid', name="Conv_2")(x)
    x = ELU(name="ELU_2")(x)
    x = MaxPooling1D(pool_size=4, strides=4, name="Pool_2")(x)
    
    x = Conv1D(32, 16, padding='valid', name="Conv_3")(x)
    x = ELU(name="ELU_3")(x)
    x = MaxPooling1D(pool_size=4, strides=4, name="Pool_3")(x)
    
    # Flatten and dense layers
    x = Flatten(name="Flatten")(x)
    x = Dense(128, name="Dense_1")(x)
    x = ELU(name="ELU_4")(x)
    x = Dropout(0.5, name="Dropout_1")(x)
    
    x = Dense(64, name="Dense_2")(x)
    x = ELU(name="ELU_5")(x)
    x = Dropout(0.5, name="Dropout_2")(x)
    
    # Output: softmax for multi-class classification
    outputs = Dense(num_classes, activation='softmax', name=gf.ReturnVariables.GLITCH_TYPE.name)(x)
    
    model = Model(
        inputs=[onsource_input, offsource_input],
        outputs={gf.ReturnVariables.GLITCH_TYPE.name: outputs},
        name="glitch_classifier"
    )
    
    return model

# =============================================================================
# Adapter to Process Labels for Keras
# =============================================================================

class GlitchAdapterDataset(keras.utils.PyDataset):
    """Adapter to convert integer labels to one-hot encoding for Keras."""
    
    def __init__(self, dataset, num_classes):
        super().__init__(workers=dataset.workers, use_multiprocessing=dataset.use_multiprocessing)
        self.dataset = dataset
        self.num_classes = num_classes
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        features, labels = self.dataset[index]
        
        # Convert integer labels to one-hot
        if gf.ReturnVariables.GLITCH_TYPE.name in labels:
            int_labels = labels[gf.ReturnVariables.GLITCH_TYPE.name]
            # Handle negative labels (unknown)
            int_labels = np.clip(int_labels, 0, self.num_classes - 1)
            # One-hot encode
            one_hot = np.eye(self.num_classes, dtype='float32')[int_labels]
            labels[gf.ReturnVariables.GLITCH_TYPE.name] = one_hot
        
        return features, labels

# =============================================================================
# Create Data Obtainers (Separate for Train and Validation)
# =============================================================================

print("\nSetting up data obtainers...")

# TRAINING: IFODataObtainer with augmentation and class balancing
train_ifo_obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.GLITCHES],
    segment_order=gf.SegmentOrder.RANDOM,
    # Augmentations
    random_sign_reversal=True,
    random_time_reversal=True,
    random_shift=True,
    shift_fraction=0.2,
    add_noise=True,
    noise_amplitude=0.05,
    # Class balancing
    balanced_glitch_types=True,
)

train_noise_obtainer = gf.TransientObtainer(
    ifo_data_obtainer=train_ifo_obtainer,
    ifos=[gf.IFO.L1]
)

# VALIDATION: Separate obtainer with NO augmentation and different seed
val_ifo_obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.GLITCHES],
    segment_order=gf.SegmentOrder.RANDOM,
    # NO augmentation for validation
    random_sign_reversal=False,
    random_time_reversal=False,
    random_shift=False,
    add_noise=False,
    # Class balancing still on for fair evaluation
    balanced_glitch_types=True,
)

val_noise_obtainer = gf.TransientObtainer(
    ifo_data_obtainer=val_ifo_obtainer,
    ifos=[gf.IFO.L1]
)

# =============================================================================
# Create Datasets
# =============================================================================

print("\nCreating datasets...")

training_dataset = gf.Dataset(
    noise_obtainer=train_noise_obtainer,
    input_variables=[
        gf.ReturnVariables.ONSOURCE,
        gf.ReturnVariables.OFFSOURCE,
    ],
    output_variables=[
        gf.ReturnVariables.GLITCH_TYPE
    ],
    steps_per_epoch=STEPS_PER_EPOCH,
)

validation_dataset = gf.Dataset(
    noise_obtainer=val_noise_obtainer,
    seed=1001,  # Different seed for validation
    group="validate",
    input_variables=[
        gf.ReturnVariables.ONSOURCE,
        gf.ReturnVariables.OFFSOURCE,
    ],
    output_variables=[
        gf.ReturnVariables.GLITCH_TYPE
    ],
    steps_per_epoch=VALIDATION_STEPS,
)

# =============================================================================
# Create Model
# =============================================================================

print("\nCreating model...")

# Get input shapes from a sample batch
for input_example, _ in [training_dataset[0]]:
    input_shape_onsource = input_example["ONSOURCE"].shape[1:]
    input_shape_offsource = input_example["OFFSOURCE"].shape[1:]
    print(f"Onsource shape: {input_shape_onsource}")
    print(f"Offsource shape: {input_shape_offsource}")

model = create_glitch_classifier(input_shape_onsource, input_shape_offsource, NUM_CLASSES)
model.summary()

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Wrap datasets with adapter for one-hot encoding
training_dataset = GlitchAdapterDataset(training_dataset, NUM_CLASSES)
validation_dataset = GlitchAdapterDataset(validation_dataset, NUM_CLASSES)

# =============================================================================
# Setup Callbacks
# =============================================================================

callbacks = [
    # Save best model
    keras.callbacks.ModelCheckpoint(
        filepath=OUTPUT_DIR / "best_model.keras",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    # Save every epoch checkpoint (for resuming)
    keras.callbacks.ModelCheckpoint(
        filepath=OUTPUT_DIR / "checkpoint_epoch_{epoch:03d}.keras",
        save_freq='epoch',
        verbose=0
    ),
    # TensorBoard logging
    keras.callbacks.TensorBoard(
        log_dir=OUTPUT_DIR / "logs",
        histogram_freq=1,
        write_graph=True
    ),
    # Early stopping
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    # Learning rate reduction
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        patience=5,
        factor=0.5,
        min_lr=1e-6,
        verbose=1
    ),
    # CSV logging for analysis
    keras.callbacks.CSVLogger(
        OUTPUT_DIR / "training_log.csv"
    ),
]

# =============================================================================
# Train Model
# =============================================================================

print("\nTraining model...")
print(f"Checkpoints will be saved to: {OUTPUT_DIR}")

history = model.fit(
    training_dataset,
    verbose=2,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =============================================================================
# Results
# =============================================================================

print(f"\nTraining complete!")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.3f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.3f}")
print(f"\nBest model saved to: {OUTPUT_DIR / 'best_model.keras'}")
print(f"Training logs saved to: {OUTPUT_DIR / 'training_log.csv'}")
