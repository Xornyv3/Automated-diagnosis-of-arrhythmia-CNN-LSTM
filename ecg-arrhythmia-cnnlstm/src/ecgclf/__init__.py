"""ecgclf: ECG arrhythmia classification (CNN+LSTM) package.

Modules:
- data: dataset download and loading
- preprocess: segmentation around R-peaks
- model: Keras model factory
- train: training utilities/CLI
- evaluate: evaluation utilities/CLI
- predict: inference CLI
"""

__all__ = [
    "data",
    "preprocess",
    "model",
    "train",
    "evaluate",
    "predict",
]
