# =============================================================================
# Small Scale Glitch Classification Test
# =============================================================================
# A minimal version of glitch_classification_example.py configured for a 
# short 2-hour window to verify the pipeline quickly.

import gravyflow as gf
import keras
from keras import ops
from keras.layers import Input, Permute, Conv1D, MaxPooling1D, Dense, Flatten, Dropout, ELU
from keras.models import Model
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

SAMPLE_RATE = 2048.0
ONSOURCE_DURATION = 1.0  # Training requirement
# We expect the cache to standardize to 4.0s/32.0s internally

BATCH_SIZE = 16
STEPS_PER_EPOCH = 50
VALIDATION_STEPS = 20
EPOCHS = 2
NUM_CLASSES = len(gf.GlitchType)

# Start of O3a: April 1, 2019
START_GPS = 1238166018
END_GPS = START_GPS + 720000  # 200 hours of data to try and catch more glitches (if cache allows)
SEED = 42 # Force deterministic split to ensure validation set gets glitches

print(f"Running small verification test: {START_GPS} - {END_GPS} (200 hours)")

# =============================================================================
# Model Definition (Simplified)
# =============================================================================

def create_glitch_classifier(input_shape_onsource, input_shape_offsource, num_classes):
    onsource_input = Input(shape=input_shape_onsource, name="ONSOURCE")
    offsource_input = Input(shape=input_shape_offsource, name="OFFSOURCE")
    
    x = gf.Whiten()([onsource_input, offsource_input])
    x = Permute((2, 1))(x)
    
    x = Conv1D(8, 16, padding='valid')(x)
    x = ELU()(x)
    x = MaxPooling1D(4)(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    
    outputs = Dense(num_classes, activation='softmax', name=gf.ReturnVariables.GLITCH_TYPE.name)(x)
    
    model = Model(inputs=[onsource_input, offsource_input], outputs={gf.ReturnVariables.GLITCH_TYPE.name: outputs})
    return model

class GlitchAdapterDataset(keras.utils.PyDataset):
    def __init__(self, dataset, num_classes):
        super().__init__(workers=dataset.workers, use_multiprocessing=dataset.use_multiprocessing)
        self.dataset = dataset
        self.num_classes = num_classes
    
    def __len__(self): return len(self.dataset)
    
    def __getitem__(self, index):
        features, labels = self.dataset[index]
        if gf.ReturnVariables.GLITCH_TYPE.name in labels:
            int_labels = np.clip(labels[gf.ReturnVariables.GLITCH_TYPE.name], 0, self.num_classes - 1)
            labels[gf.ReturnVariables.GLITCH_TYPE.name] = np.eye(self.num_classes, dtype='float32')[int_labels]
        return features, labels

# =============================================================================
# Setup Data
# =============================================================================

print("\nSetting up data obtainers...")

# For small test with limited cache, use 80/20 split instead of default 98/1/1
groups = {"train": 0.8, "validate": 0.2}

# Explicitly set start/end time to restrict download size
ifo_data_obtainer = gf.IFODataObtainer(
    observing_runs=gf.ObservingRun.O3,  # Provides frame types/channels
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.GLITCHES],
    segment_order=gf.SegmentOrder.RANDOM,
    balanced_glitch_types=True,
)

# Manually override time range to restrict to 2 hours
ifo_data_obtainer.start_gps_times = [START_GPS]
ifo_data_obtainer.end_gps_times = [END_GPS]

noise_obtainer = gf.TransientObtainer(
    ifo_data_obtainer=ifo_data_obtainer,
    ifos=gf.IFO.L1,
    groups=groups
)

SEED = 42

print("\nCreating datasets (this will trigger precaching if needed)...")

training_dataset = gf.Dataset(
    sample_rate_hertz=SAMPLE_RATE,
    onsource_duration_seconds=ONSOURCE_DURATION,
    noise_obtainer=noise_obtainer,
    group="train",
    seed=SEED,
    input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],  # For reduced memory/complexity if needed
    output_variables=[gf.ReturnVariables.GLITCH_TYPE],
    steps_per_epoch=STEPS_PER_EPOCH,
    num_examples_per_batch=BATCH_SIZE
)

validation_dataset = gf.Dataset(
    sample_rate_hertz=SAMPLE_RATE,
    onsource_duration_seconds=ONSOURCE_DURATION,
    # crop_duration_seconds=0.5,
    noise_obtainer=noise_obtainer,
    group="validate",
    seed=SEED,
    input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
    output_variables=[gf.ReturnVariables.GLITCH_TYPE],
    steps_per_epoch=VALIDATION_STEPS,
    num_examples_per_batch=BATCH_SIZE
)

# =============================================================================
# Diagnostics
# =============================================================================
print("\nRUNNING DIAGNOSTICS...")

def get_stats(dataset, name="Dataset"):
    print(f"\nAnalyzing {name}...")
    # Create a temporary diagnostic dataset with GPS time
    diag_ds = gf.Dataset(
        noise_obtainer=noise_obtainer,
        group=dataset.group,
        seed=dataset.seed,
        input_variables=[gf.ReturnVariables.GPS_TIME],
        output_variables=[gf.ReturnVariables.GLITCH_TYPE],
        steps_per_epoch=len(dataset), # One epoch
        num_examples_per_batch=BATCH_SIZE
    )
    
    all_gps = []
    all_labels = []
    
    # Iterate through one epoch
    for i in range(len(diag_ds)):
        feats, labs = diag_ds[i]
        if gf.ReturnVariables.GPS_TIME in feats: # Input var
            all_gps.extend(feats[gf.ReturnVariables.GPS_TIME].flatten())
        elif gf.ReturnVariables.GPS_TIME.name in feats: # Key name
            all_gps.extend(feats[gf.ReturnVariables.GPS_TIME.name].flatten())
            
        if gf.ReturnVariables.GLITCH_TYPE.name in labs:
            all_labels.extend(labs[gf.ReturnVariables.GLITCH_TYPE.name].flatten())
            
    # Convert to standard types for hashing
    all_gps = [float(x) for x in all_gps]
    
    print(f"  Total samples: {len(all_gps)}")
    unique_gps = len(set(all_gps))
    print(f"  Unique GPS times: {unique_gps}")
    
    # Label distribution
    from collections import Counter
    counts = Counter(all_labels)
    print(f"  Class distribution:")
    for label, count in sorted(counts.items()):
        print(f"    Class {label}: {count}")
        
    return set(all_gps)

train_gps = get_stats(training_dataset, "Training Set")
val_gps = get_stats(validation_dataset, "Validation Set")

overlap = train_gps.intersection(val_gps)
print(f"\nOverlap between Train and Validate: {len(overlap)} samples")
if len(overlap) > 0:
    print("  WARNING: LEAKAGE DETECTED!")
else:
    print("  VERIFIED: Splits are disjoint.")
    
print("-" * 60)

# =============================================================================
# Train
# =============================================================================

# Get shapes
for x, _ in [training_dataset[0]]:
    shape_on = x["ONSOURCE"].shape[1:]
    shape_off = x["OFFSOURCE"].shape[1:]
    break

model = create_glitch_classifier(shape_on, shape_off, NUM_CLASSES)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

training_dataset = GlitchAdapterDataset(training_dataset, NUM_CLASSES)
validation_dataset = GlitchAdapterDataset(validation_dataset, NUM_CLASSES)

print("\nStarting training...")
model.fit(
    training_dataset, 
    validation_data=validation_dataset,
    epochs=EPOCHS
)
print("\nSuccess! Small run completed.")
