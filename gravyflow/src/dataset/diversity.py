"""
Dataset diversity metrics and callbacks for monitoring class balance during training.
"""

import numpy as np
import keras
from collections import Counter
from typing import Dict, List


def compute_diversity_score(labels: np.ndarray) -> float:
    """
    Compute a diversity score for label distribution.
    
    Uses normalized entropy:
    - 1.0 = perfectly balanced (equal split among all present classes)
    - 0.0 = all samples are same class
    
    Args:
        labels: Array of integer class labels
        
    Returns:
        Diversity score between 0.0 and 1.0
    """
    if len(labels) == 0:
        return 0.0
    
    # Count occurrences of each class
    counts = Counter(labels.flatten())
    
    if len(counts) == 1:
        return 0.0  # Only one class = no diversity
    
    # Compute probabilities
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    
    # Compute entropy (log base 2)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    # Normalize by max possible entropy (log2 of number of classes present)
    max_entropy = np.log2(len(counts))
    
    return entropy / max_entropy if max_entropy > 0 else 0.0


class DiversityCallback(keras.callbacks.Callback):
    """
    Keras callback to monitor dataset diversity during training.
    
    Tracks the distribution of class labels and reports a diversity score:
    - 1.0 = perfectly balanced (equal representation of all classes)
    - 0.0 = all samples are the same class
    
    Usage:
        callback = DiversityCallback(dataset, label_key='SUB_TYPE')
        model.fit(..., callbacks=[callback])
    """
    
    def __init__(
        self, 
        dataset,
        label_key: str = 'SUB_TYPE',
        log_every_n_batches: int = 100,
        num_classes: int = 20
    ):
        """
        Args:
            dataset: The GravyflowDataset or PyDataset to monitor
            label_key: Key for labels in the dataset output
            log_every_n_batches: How often to compute and log diversity
            num_classes: Expected number of classes (for ideal diversity)
        """
        super().__init__()
        self.dataset = dataset
        self.label_key = label_key
        self.log_every_n_batches = log_every_n_batches
        self.num_classes = num_classes
        
        self._batch_labels = []
        self._batch_count = 0
        self._epoch_labels = []
        
    def on_batch_end(self, batch, logs=None):
        self._batch_count += 1
        
        if self._batch_count % self.log_every_n_batches == 0:
            if len(self._batch_labels) > 0:
                all_labels = np.concatenate(self._batch_labels)
                diversity = compute_diversity_score(all_labels)
                
                # Count distribution
                counts = Counter(all_labels.flatten())
                unique_classes = len(counts)
                
                print(f"\n[Diversity] Batch {self._batch_count}: "
                      f"score={diversity:.3f}, "
                      f"classes_seen={unique_classes}/{self.num_classes}, "
                      f"samples={len(all_labels)}")
                
                # Add to logs for TensorBoard etc
                if logs is not None:
                    logs['diversity_score'] = diversity
                    logs['unique_classes'] = unique_classes
                
                # Reset for next window
                self._batch_labels = []
    
    def on_train_batch_end(self, batch, logs=None):
        # Try to capture labels from the current batch
        # This requires the dataset to expose labels
        try:
            # If we can access the last batch labels from the dataset...
            if hasattr(self.dataset, '_last_labels'):
                labels = self.dataset._last_labels
                if labels is not None:
                    self._batch_labels.append(np.array(labels))
                    self._epoch_labels.append(np.array(labels))
        except Exception:
            pass
        
        self.on_batch_end(batch, logs)
    
    def on_epoch_end(self, epoch, logs=None):
        if len(self._epoch_labels) > 0:
            all_labels = np.concatenate(self._epoch_labels)
            diversity = compute_diversity_score(all_labels)
            
            counts = Counter(all_labels.flatten())
            unique_classes = len(counts)
            
            print(f"\n[Diversity] Epoch {epoch + 1} Summary: "
                  f"score={diversity:.3f}, "
                  f"classes_seen={unique_classes}/{self.num_classes}, "
                  f"total_samples={len(all_labels)}")
            
            # Class distribution
            print(f"  Distribution: {dict(sorted(counts.items()))}")
            
            if logs is not None:
                logs['epoch_diversity_score'] = diversity
                logs['epoch_unique_classes'] = unique_classes
        
        # Reset for next epoch
        self._epoch_labels = []
        self._batch_count = 0


class LabelTrackingDataset(keras.utils.PyDataset):
    """
    Wrapper dataset that tracks labels for diversity monitoring.
    
    Wraps an existing dataset and exposes _last_labels for the DiversityCallback.
    """
    
    def __init__(self, dataset, label_key: str = 'SUB_TYPE'):
        super().__init__(workers=dataset.workers, use_multiprocessing=dataset.use_multiprocessing)
        self.dataset = dataset
        self.label_key = label_key
        self._last_labels = None
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        features, labels = self.dataset[index]
        
        # Track labels for diversity monitoring
        if self.label_key in labels:
            label_data = labels[self.label_key]
            # Convert one-hot to class indices if needed
            if len(label_data.shape) > 1 and label_data.shape[-1] > 1:
                self._last_labels = np.argmax(label_data, axis=-1)
            else:
                self._last_labels = np.array(label_data).flatten()
        
        return features, labels
