from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .preprocess import (
    build_arrays,
    compute_class_weights,
    stratified_kfold_indices,
)
from .model import build_model, ensure_3d
from .viz import plot_examples_by_class, plot_training_curves


def main() -> None:
    """Train the CNN+LSTM with stratified K-fold CV and log per-fold artifacts."""
    parser = argparse.ArgumentParser(description="Train ECG CNN+LSTM with stratified K-fold CV")
    parser.add_argument("--data_dir", type=str, default="data/mitdb", help="Path to mitdb directory (ignored when --synthetic)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fold_k", type=int, default=10)
    parser.add_argument("--early_stopping", action="store_true", help="Enable EarlyStopping (patience=15)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", type=str, default="runs", help="TensorBoard log directory")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--augment", action="store_true", help="Apply simple augmentation (amplitude jitter + time warp) to training data")
    parser.add_argument("--bandpass", action="store_true", help="Apply 0.5-40Hz bandpass to raw signals before segmentation")
    parser.add_argument("--synthetic", action="store_true", help="Use a synthetic offline dataset (no download)")
    parser.add_argument("--synthetic_n", type=int, default=200, help="Synthetic samples per class (when --synthetic)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path("data/mitdb")
    logdir = Path(args.logdir)
    models_dir = Path(args.models_dir)
    logdir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    if args.synthetic:
        from .preprocess import build_synthetic_arrays

        X, y, _ = build_synthetic_arrays(n_per_class=args.synthetic_n, seed=args.seed)
    else:
        if not data_dir.exists():
            raise SystemExit(f"--data_dir not found: {data_dir}. Provide a valid dataset path or use --synthetic.")
        # z-scored per segment inside build_arrays; optionally bandpass raw signals first
        X, y, _ = build_arrays(data_dir, bandpass=args.bandpass)
    X = ensure_3d(X)

    # Save a figure of one example per class
    try:
        plot_examples_by_class(X, y, class_names=["Normal", "LBBB", "RBBB", "APB", "PVC"], out_path=Path("artifacts/figures/examples.png"))
    except Exception:
        pass

    # Prepare folds
    folds = list(stratified_kfold_indices(y, n_splits=args.fold_k, random_state=args.seed))

    # Train per fold
    fold_metrics = []
    for i, (tr_idx, va_idx) in enumerate(folds, start=1):
        x_tr, x_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Optionally augment training data (simple, low-risk augmentations)
        def amplitude_jitter(x: np.ndarray, sigma: float = 0.05, rng=None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            noise = rng.normal(loc=0.0, scale=sigma, size=(x.shape[0], 1, 1)).astype(np.float32)
            return x * (1.0 + noise)

        def time_warp(x: np.ndarray, max_factor: float = 0.05, rng=None) -> np.ndarray:
            # Simple per-sample small stretch/shrink via 1D interpolation
            if rng is None:
                rng = np.random.default_rng()
            out = np.zeros_like(x)
            L = x.shape[1]
            orig = np.arange(L)
            for ii in range(x.shape[0]):
                factor = 1.0 + rng.uniform(-max_factor, max_factor)
                # New positions (may compress/stretch). Clip to valid range for interp.
                src_pos = np.clip(orig * factor, 0, L - 1)
                seq = x[ii, :, 0]
                warped = np.interp(orig, src_pos, seq).astype(np.float32)
                out[ii, :, 0] = warped
            return out

        if args.augment:
            rng = np.random.default_rng(args.seed + i)
            x_aug = amplitude_jitter(x_tr, sigma=0.05, rng=rng)
            x_aug = time_warp(x_aug, max_factor=0.03, rng=rng)
            # Concatenate augmented copies to training set
            x_tr = np.concatenate([x_tr, x_aug], axis=0)
            y_tr = np.concatenate([y_tr, y_tr], axis=0)

        # Class weights (recomputed after augmentation if used)
        class_weight = compute_class_weights(y_tr)

        # Build model (Keras backend) and recompile with desired LR
        model = build_model(input_len=X.shape[1], n_classes=int(np.max(y) + 1), backend="keras")
        try:
            # override optimizer LR
            from tensorflow import keras as _keras  # local import

            model.compile(
                optimizer=_keras.optimizers.Adam(learning_rate=args.lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
        except Exception:
            pass  # fall back to default compile inside builder

        # Callbacks
        callbacks = []
        if args.early_stopping:
            callbacks.append(
                _keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True
                )
            )
        ckpt_path = models_dir / f"fold_{i}.keras"
        callbacks.append(
            _keras.callbacks.ModelCheckpoint(
                filepath=str(ckpt_path), monitor="val_loss", save_best_only=True
            )
        )
        run_dir = logdir / f"fold_{i}_{int(time.time())}"
        callbacks.append(_keras.callbacks.TensorBoard(log_dir=str(run_dir)))

        history = model.fit(
            x_tr,
            y_tr,
            validation_data=(x_va, y_va),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weight,
            verbose=2,
            callbacks=callbacks,
        )

        # Save training curves to artifacts/figures/fold_{i}
        try:
            fold_fig_dir = Path("artifacts/figures") / f"fold_{i}"
            plot_training_curves(history.history, out_dir=fold_fig_dir)
        except Exception:
            pass

        # Evaluate on validation set
        val_probs = model.predict(x_va, batch_size=args.batch_size, verbose=0)
        y_pred = np.argmax(val_probs, axis=1)
        acc = float(np.mean(y_pred == y_va))
        fold_metrics.append({"fold": i, "val_accuracy": acc})

        # Save loss/acc curves (history)
        hist_out = run_dir / "history.json"
        hist_out.parent.mkdir(parents=True, exist_ok=True)
        with open(hist_out, "w", encoding="utf-8") as f:
            json.dump({k: list(map(float, v)) for k, v in history.history.items()}, f, indent=2)

        print(f"Fold {i}: val_accuracy={acc:.4f} | model -> {ckpt_path}")

    # Summary
    avg_acc = float(np.mean([m["val_accuracy"] for m in fold_metrics]))
    print("\n==== K-fold summary ====")
    for m in fold_metrics:
        print(f"Fold {m['fold']}: val_accuracy={m['val_accuracy']:.4f}")
    print(f"Average val_accuracy over {len(fold_metrics)} folds: {avg_acc:.4f}")


if __name__ == "__main__":
    main()
