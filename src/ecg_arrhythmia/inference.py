from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np
from tensorflow import keras

from .config import SEGMENT_SAMPLES, CLASS_MAP
from .preprocess import read_record, read_annotations, make_beat_segments
from .model import expand_channels


CLASS_IDX_TO_NAME = {v: k for k, v in CLASS_MAP.items()}


def predict_record(record_path: Path | str, model_path: Path | str) -> Dict[str, Any]:
    """
    Load a trained model and predict class probabilities for each beat in a WFDB record.

    Returns:
      dict with keys: record, r_indices, symbols, y_pred, y_true (if symbols in target map), probs
    """
    model = keras.models.load_model(str(model_path))

    sig, _ = read_record(record_path)
    r_locs, symbols = read_annotations(record_path)
    beats = make_beat_segments(Path(record_path).stem, sig, r_locs, symbols)

    X = np.stack([b.segment for b in beats]).astype(np.float32)
    X = expand_channels(X)

    probs = model.predict(X, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    # True labels when available
    y_true = np.array([CLASS_MAP[s] if s in CLASS_MAP else -1 for s in symbols], dtype=int)

    return {
        "record": str(record_path),
        "r_indices": np.array([b.r_index for b in beats], dtype=int),
        "symbols": symbols,
        "y_pred": y_pred,
        "y_true": y_true,
        "probs": probs,
    }
