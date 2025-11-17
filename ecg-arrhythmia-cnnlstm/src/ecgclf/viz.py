from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def plot_examples_by_class(
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    out_path: Path | str = "artifacts/figures/examples.png",
    per_class: int = 1,
) -> Path:
    """Plot example segments for each class and save to a PNG.

    Args:
        X: (N, L[, C]) segments (if 3D, C should be 1)
        y: (N,) integer labels
        class_names: list of class names
        out_path: path to save the figure
        per_class: number of examples per class to plot (default 1)
    """
    import matplotlib.pyplot as plt

    X2 = X
    if X.ndim == 3:
        X2 = X[:, :, 0]
    L = X2.shape[1]

    out_path = Path(out_path)
    _ensure_dir(out_path)

    n_classes = len(class_names)
    n_rows = n_classes
    n_cols = per_class
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 1.8 * n_rows), squeeze=False)

    rng = np.random.default_rng(12345)
    for ci in range(n_classes):
        idxs = np.where(y == ci)[0]
        if idxs.size == 0:
            # no sample of this class, leave blank
            for c in range(n_cols):
                axes[ci, c].axis("off")
            continue
        chosen = rng.choice(idxs, size=min(n_cols, idxs.size), replace=False)
        for c, idx in enumerate(chosen):
            axes[ci, c].plot(np.arange(L), X2[idx], lw=1.0)
            axes[ci, c].set_title(class_names[ci])
            axes[ci, c].set_xlim(0, L - 1)
            axes[ci, c].set_xticks([])
            axes[ci, c].set_yticks([])
        # hide any remaining empty cells
        for c in range(len(chosen), n_cols):
            axes[ci, c].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_training_curves(history: Dict[str, Iterable[float]], out_dir: Path | str = "artifacts/figures") -> Path:
    """Plot train/val loss and accuracy curves from a Keras History-like dict.

    Saves a single PNG named training_curves.png in out_dir.
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_path = out_dir / "training_curves.png"
    _ensure_dir(out_path)

    loss = list(map(float, history.get("loss", [])))
    val_loss = list(map(float, history.get("val_loss", [])))
    acc = list(map(float, history.get("accuracy", [])))
    val_acc = list(map(float, history.get("val_accuracy", [])))

    epochs = range(1, max(len(loss), len(val_loss), len(acc), len(val_acc)) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Loss
    axes[0].plot(epochs, loss, label="train")
    if val_loss:
        axes[0].plot(epochs[: len(val_loss)], val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy
    if acc:
        axes[1].plot(epochs[: len(acc)], acc, label="train")
    if val_acc:
        axes[1].plot(epochs[: len(val_acc)], val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_confusion_matrix_norm(
    cm: np.ndarray,
    class_names: List[str],
    out_path: Path | str = "artifacts/figures/confusion_matrix_norm.png",
) -> Path:
    """Plot normalized confusion matrix (row-normalized) and save.

    Args:
        cm: confusion matrix of shape (C, C)
        class_names: list of class names (length C)
        out_path: file path to save PNG
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    out_path = Path(out_path)
    _ensure_dir(out_path)

    plt.figure(figsize=(6, 5))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, vmin=0.0, vmax=1.0)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
