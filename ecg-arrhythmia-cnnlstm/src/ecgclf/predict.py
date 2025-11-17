from __future__ import annotations
import argparse
from pathlib import Path
import csv
from typing import Optional, Tuple, List

import numpy as np

from .preprocess import segment_beats, zscore, CLASSES


def _load_ecg_signal(path: Path) -> np.ndarray:
    """Load mono-lead ECG from CSV or .mat and return 1D float32 array.

    CSV: first numeric column is used (header allowed).
    MAT: tries common keys and then the largest 1D array.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(path)
        # pick first numeric column
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                arr = df[col].to_numpy(dtype=np.float32)
                return arr.reshape(-1)
        raise ValueError("CSV contains no numeric columns")
    elif suffix == ".mat":
        from scipy.io import loadmat

        mat = loadmat(path)
        # common keys first
        for key in ["ecg", "signal", "val", "data", "x"]:
            if key in mat:
                arr = np.asarray(mat[key]).squeeze()
                if arr.ndim == 1:
                    return arr.astype(np.float32)
        # otherwise choose the largest 1D array
        best = None
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            a = np.asarray(v).squeeze()
            if a.ndim == 1:
                if best is None or a.size > best.size:
                    best = a
        if best is not None:
            return best.astype(np.float32)
        raise ValueError("MAT file does not contain a 1D ECG vector")
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")


def _load_r_indices(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in (".csv", ".txt"):
        vals: List[int] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for tok in line.replace(",", " ").split():
                    if tok.isdigit() or (tok.startswith("-") and tok[1:].isdigit()):
                        vals.append(int(tok))
        if not vals:
            raise ValueError("No indices found in R index file")
        return np.array(vals, dtype=int)
    elif path.suffix.lower() == ".npy":
        return np.load(path).astype(int)
    else:
        raise ValueError("Unsupported R index file format (use .csv, .txt, or .npy)")


def _detect_r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    """Very simple R-peak detector using find_peaks on a preprocessed signal.

    This is a lightweight fallback; for robust detection, prefer annotation files or
    specialized detectors.
    """
    from scipy.signal import butter, filtfilt, find_peaks

    # Bandpass 5-15 Hz to emphasize QRS
    low = 5 / (fs / 2)
    high = 15 / (fs / 2)
    b, a = butter(2, [low, high], btype="bandpass")
    x = filtfilt(b, a, signal.astype(np.float32))
    x = np.abs(np.gradient(x)).astype(np.float32)
    x = zscore(x)
    distance = int(0.25 * fs)  # at least 250ms between beats
    peaks, _ = find_peaks(x, distance=max(1, distance), height=np.percentile(x, 75))
    return peaks.astype(int)


def main() -> None:
    """Run inference on a mono-lead ECG file and write predictions and a preview figure."""
    ap = argparse.ArgumentParser(description="Predict classes for ECG beats from CSV/MAT using a trained model")
    ap.add_argument("--input", type=str, required=True, help="Path to ECG file (.csv or .mat)")
    ap.add_argument("--model", type=str, required=True, help="Path to Keras model (.keras/.h5)")
    ap.add_argument("--fs", type=int, default=360, help="Sampling frequency (Hz)")
    ap.add_argument("--r_indices", type=str, default=None, help="Optional CSV/TXT/NPY of R indices (samples)")
    ap.add_argument("--detect_r", action="store_true", help="Run simple R-peak detection if no indices provided")
    ap.add_argument("--pre", type=int, default=500, help="Samples before R")
    ap.add_argument("--post", type=int, default=500, help="Samples after R")
    ap.add_argument("--out_csv", type=str, default="runs/predict/predictions.csv")
    ap.add_argument("--out_fig", type=str, default="runs/predict/predictions.png")
    args = ap.parse_args()

    inp = Path(args.input)
    out_csv = Path(args.out_csv)
    out_fig = Path(args.out_fig)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    # Load signal
    signal = _load_ecg_signal(inp)

    # Load or detect R indices
    r_idx = _load_r_indices(Path(args.r_indices)) if args.r_indices else None
    if r_idx is None:
        if args.detect_r:
            r_idx = _detect_r_peaks(signal, fs=args.fs)
            if r_idx.size == 0:
                raise RuntimeError("R-peak detector failed to find peaks; provide --r_indices.")
        else:
            raise ValueError("No R indices provided. Use --r_indices or --detect_r.")

    # Dummy labels (not used for inference)
    labels = np.zeros(len(r_idx), dtype=int)
    X, _ = segment_beats(signal, r_idx, labels, pre=args.pre, post=args.post, apply_z=True)
    X = np.expand_dims(X, -1)

    # Load model and predict
    from tensorflow import keras

    model = keras.models.load_model(args.model)
    probs = model.predict(X, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    # Save CSV: r_index, pred_idx, pred_label, probs...
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["r_index", "pred_idx", "pred_label"] + [f"p_{cls}" for cls in CLASSES]
        w.writerow(header)
        for r, yp, p in zip(r_idx, y_pred, probs):
            row = [int(r), int(yp), CLASSES[int(yp)]] + [float(pi) for pi in p]
            w.writerow(row)

    # Summary
    unique, counts = np.unique(y_pred, return_counts=True)
    summary = {CLASSES[int(k)]: int(v) for k, v in zip(unique, counts)}
    total = int(len(y_pred))
    print(f"Wrote predictions for {total} beats to {out_csv}")
    print("Predicted distribution:", summary)

    # Figure: illustrate first few segments and their predicted class
    import matplotlib.pyplot as plt

    n_show = min(6, X.shape[0])
    cols = 3
    rows = int(np.ceil(n_show / cols))
    plt.figure(figsize=(12, 3.5 * rows))
    for i in range(n_show):
        ax = plt.subplot(rows, cols, i + 1)
        ax.plot(X[i].squeeze(), lw=1.0)
        ax.set_title(f"r={int(r_idx[i])}  pred={CLASSES[int(y_pred[i])]}  p={probs[i, y_pred[i]]:.2f}")
        ax.set_xlim([0, X.shape[1]])
    plt.tight_layout()
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"Saved figure to {out_fig}")


if __name__ == "__main__":
    main()
