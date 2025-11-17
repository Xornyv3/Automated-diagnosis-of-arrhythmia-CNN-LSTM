from __future__ import annotations
from typing import Any, Tuple

NUM_CLASSES = 5
INPUT_LEN = 1000


def _build_keras_model(input_len: int, n_classes: int):
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=(input_len, 1), name="ecg")

    # Three Conv1D blocks with ReLU, use_bias=False, kernel sizes ~20/10/5
    x = layers.Conv1D(32, 21, padding="same", use_bias=False)(inp)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

    x = layers.Conv1D(64, 11, padding="same", use_bias=False)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

    x = layers.Conv1D(128, 5, padding="same", use_bias=False)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

    # LSTM
    x = layers.LSTM(20, recurrent_dropout=0.2, return_sequences=False)(x)

    # Dense head: 20 -> Dropout -> 10 -> Dropout -> Softmax(n_classes)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(10, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="softmax", name="probs")(x)

    model = keras.Model(inp, out, name="ecg_cnn_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_torch_model(input_len: int, n_classes: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TorchECGModel(nn.Module):
        def __init__(self, input_len: int, n_classes: int):
            super().__init__()
            # Expect input shape (B, T, C=1)
            self.conv1 = nn.Conv1d(1, 32, kernel_size=21, padding=10, bias=False)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=11, padding=5, bias=False)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.lstm = nn.LSTM(input_size=128, hidden_size=20, batch_first=True, dropout=0.2)
            self.fc1 = nn.Linear(20, 20)
            self.fc2 = nn.Linear(20, 10)
            self.fc3 = nn.Linear(10, n_classes)
            self.dropout = nn.Dropout(0.2)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            # x: (B, T, C)
            x = x.transpose(1, 2)  # (B, C, T)
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.pool(x)  # (B, 128, T')
            x = x.transpose(1, 2)  # (B, T', 128)
            out, (h, c) = self.lstm(x)
            x = h[-1]  # (B, 20)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            x = self.softmax(x)
            return x

    return TorchECGModel(input_len=input_len, n_classes=n_classes)


def build_model(
    input_len: int = INPUT_LEN,
    n_classes: int = NUM_CLASSES,
    backend: str = "keras",
    **kwargs: Any,
):
    """Factory for the ECG CNN+LSTM classifier.

    Args:
        input_len: sequence length (default 1000)
        n_classes: number of classes (default 5)
        backend: "keras" (default) or "torch"
        **kwargs: compatibility parameters (input_length, num_classes)

    Returns:
        - Keras backend: a compiled keras.Model
        - Torch backend: a torch.nn.Module
    """
    # Backward-compat params mapping
    if "input_length" in kwargs and kwargs["input_length"] is not None:
        input_len = int(kwargs["input_length"])
    if "num_classes" in kwargs and kwargs["num_classes"] is not None:
        n_classes = int(kwargs["num_classes"])

    backend = (backend or "keras").lower()
    if backend == "keras":
        return _build_keras_model(input_len, n_classes)
    if backend == "torch":
        return _build_torch_model(input_len, n_classes)
    raise ValueError(f"Unsupported backend: {backend}")


def ensure_3d(X):
    """Ensure input has shape (N, T, C) by adding a trailing channel dim if needed."""
    import numpy as np

    if X.ndim == 2:
        return np.expand_dims(X, -1)
    return X
