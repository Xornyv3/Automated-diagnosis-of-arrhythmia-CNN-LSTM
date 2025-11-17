"""ECG Arrhythmia Classification package (MIT-BIH).

Modules:
- data_download: Fetch MIT-BIH via wfdb to local cache
- preprocess: Load records, read annotations, segment beats, normalize
- dataset: Build arrays for training with selected classes
- model: Keras model (CNN+LSTM)
- train_eval: KFold training and evaluation
- metrics: Metric utilities (sensitivity, specificity, PPV)
- inference: Utilities for CLI inference
"""

__all__ = [
    "config",
    "data_download",
    "preprocess",
    "dataset",
    "model",
    "train_eval",
    "metrics",
    "inference",
]
