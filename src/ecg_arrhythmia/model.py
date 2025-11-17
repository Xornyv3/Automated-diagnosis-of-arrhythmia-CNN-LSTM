from __future__ import annotations
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import SEGMENT_SAMPLES, NUM_CLASSES, LR


def build_model(input_length: int = SEGMENT_SAMPLES, num_classes: int = NUM_CLASSES) -> keras.Model:
    inputs = keras.Input(shape=(input_length, 1), name="ecg")

    x = layers.Conv1D(32, kernel_size=7, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.LSTM(20, recurrent_dropout=0.2, return_sequences=False)(x)

    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(10, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    model = keras.Model(inputs, outputs, name="ecg_cnn_lstm")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def expand_channels(X):
    """Ensure input has shape (N, L, 1)."""
    import numpy as np

    if X.ndim == 2:
        return np.expand_dims(X, -1)
    return X
