from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple
import json
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from .config import (
    DATA_DIR,
    MODELS_DIR,
    TENSORBOARD_DIR,
    KFOLDS,
    PATIENCE,
    EPOCHS,
    BATCH_SIZE,
    LR,
    NUM_CLASSES,
    SEED,
)
from .dataset import build_dataset
from .model import build_model, expand_channels
from .metrics import per_class_metrics, summarize_metrics


def run_kfold_training(
    data_dir: Path | str | None = None,
    kfolds: int = KFOLDS,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int = PATIENCE,
    seed: int = SEED,
) -> Dict[str, Any]:
    X, y, rec_ids = build_dataset(data_dir=data_dir)
    X = expand_channels(X)

    skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=seed)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        x_tr, x_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Class weights to mitigate imbalance
        classes = np.unique(y_tr)
        cw_vals = compute_class_weight("balanced", classes=classes, y=y_tr)
        class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}

        model = build_model(input_length=X.shape[1], num_classes=NUM_CLASSES)

        log_dir = TENSORBOARD_DIR / f"fold_{fold_idx}_{int(time.time())}"
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(
                filepath=str(MODELS_DIR / f"fold_{fold_idx}.keras"),
                monitor="val_loss",
                save_best_only=True,
            ),
            keras.callbacks.TensorBoard(log_dir=str(log_dir)),
        ]

        history = model.fit(
            x_tr,
            y_tr,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            verbose=2,
            callbacks=callbacks,
        )

        # Evaluate best model on val
        val_probs = model.predict(x_val, batch_size=batch_size, verbose=0)
        y_pred = np.argmax(val_probs, axis=1)
        mets = per_class_metrics(y_val, y_pred, NUM_CLASSES)
        summary = summarize_metrics(mets)

        fold_results.append({
            "fold": fold_idx,
            "summary": summary,
            "class_weight": class_weight,
        })

    # Aggregate summary
    avg_acc = float(np.mean([fr["summary"]["accuracy"] for fr in fold_results]))
    avg_sens = float(np.mean([fr["summary"]["sensitivity_mean"] for fr in fold_results]))
    avg_spec = float(np.mean([fr["summary"]["specificity_mean"] for fr in fold_results]))
    avg_ppv = float(np.mean([fr["summary"]["ppv_mean"] for fr in fold_results]))

    result = {
        "kfolds": kfolds,
        "avg_accuracy": avg_acc,
        "avg_sensitivity": avg_sens,
        "avg_specificity": avg_spec,
        "avg_ppv": avg_ppv,
        "folds": fold_results,
    }

    (Path.cwd() / "logs" / "metrics.json").parent.mkdir(parents=True, exist_ok=True)
    with open(Path.cwd() / "logs" / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
