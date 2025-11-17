from __future__ import annotations
"""Preprocessing utilities for ECG arrhythmia classification.

Includes:
- z-score normalization per segment
- beat-centric segmentation around R-peaks
- dataset assembly from MIT-BIH (via WFDB)
- balanced class weights and stratified splitting helpers
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import numpy as np

FS = 360
SEGMENT_SAMPLES = 1000
PRE_SAMPLES = SEGMENT_SAMPLES // 2
POST_SAMPLES = SEGMENT_SAMPLES - PRE_SAMPLES

CLASS_MAP = {"N": 0, "L": 1, "R": 2, "A": 3, "V": 4}
CLASSES = ["Normal", "LBBB", "RBBB", "APB", "PVC"]


@dataclass
class Beat:
    record: str
    r_idx: int
    segment: np.ndarray
    symbol: str


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-segment z-score normalization."""
    m = float(np.mean(x))
    s = float(np.std(x))
    return (x - m) / (s + eps)


def read_signal(record_path: Path | str) -> Tuple[np.ndarray, List[str]]:
    import wfdb  # local import to avoid hard dependency during testing

    rec = wfdb.rdrecord(str(record_path))
    sig = rec.p_signal
    names = list(rec.sig_name)
    ch = 0
    for pref in ("MLII", "II"):
        if pref in names:
            ch = names.index(pref)
            break
    return sig[:, ch].astype(np.float32), names


def read_ann(record_path: Path | str) -> Tuple[np.ndarray, List[str]]:
    import wfdb  # local import

    ann = wfdb.rdann(str(record_path), "atr")
    return np.asarray(ann.sample, dtype=np.int64), [str(s) for s in ann.symbol]


def segment(signal: np.ndarray, r_idx: int, pre: int = PRE_SAMPLES, post: int = POST_SAMPLES,
            apply_z: bool = True) -> np.ndarray:
    """Segment a single beat around r_idx with zero padding/truncation.

    Returns a vector of length pre+post (default 1000). Optionally applies z-score.
    """
    total = pre + post
    out = np.zeros((total,), dtype=np.float32)
    start = int(r_idx) - pre
    end = int(r_idx) + post
    s0 = max(start, 0)
    s1 = min(end, signal.shape[0])
    d0 = s0 - start
    d1 = d0 + (s1 - s0)
    if s1 > s0:
        out[d0:d1] = signal[s0:s1]
    if apply_z:
        out = zscore(out)
    return out


def segment_beats(signal: np.ndarray, r_locs: Iterable[int], labels: Iterable[int],
                  pre: int = 500, post: int = 500, apply_z: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Extract fixed-length segments centered on R-peaks.

    - Pads with zeros if the window extends beyond the signal
    - Truncates if necessary
    - Per-segment z-score normalization

    Args:
        signal: 1D array ECG signal
        r_locs: iterable of R indices (samples)
        labels: iterable of class indices for each beat (same length as r_locs)
        pre: samples before R
        post: samples after R
        apply_z: whether to apply z-score per segment

    Returns:
        X: (N, pre+post) float32
        y: (N,) int64
    """
    r_arr = np.asarray(list(r_locs), dtype=int)
    y_arr = np.asarray(list(labels), dtype=int)
    if r_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("r_locs and labels must have the same length")
    X = np.stack([segment(signal, int(r), pre=pre, post=post, apply_z=apply_z) for r in r_arr]).astype(np.float32)
    return X, y_arr.astype(np.int64)


def build_synthetic_arrays(n_per_class: int = 200, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate a small synthetic dataset for offline quick runs.

    Creates 5 classes of length-1000 beats with simple morphological differences.
    Deterministic given seed. Per-beat z-score is applied via segment().

    Returns:
        X: (N, 1000) float32
        y: (N,) int64 in 0..4
        rid: list[str] with pseudo record ids
    """
    rng = np.random.default_rng(seed)
    N = int(n_per_class)
    L = SEGMENT_SAMPLES
    t = np.linspace(0, 1, L, dtype=np.float32)

    def qrs(center: float, width: float, amp: float) -> np.ndarray:
        # Simple Gaussian bump
        return amp * np.exp(-0.5 * ((t - center) / width) ** 2)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    rid: List[str] = []

    # Class 0: Normal — narrow positive bump
    base0 = qrs(0.5, 0.015, 1.0)
    # Class 1: LBBB — wider bump
    base1 = qrs(0.5, 0.03, 0.9)
    # Class 2: RBBB — double bump
    base2 = qrs(0.47, 0.015, 0.7) + qrs(0.53, 0.015, 0.7)
    # Class 3: APB — slight timing shift
    base3 = qrs(0.46, 0.015, 0.9)
    # Class 4: PVC — inverted bump
    base4 = -qrs(0.5, 0.02, 1.0)

    bases = [base0, base1, base2, base3, base4]

    for cls_idx, base in enumerate(bases):
        for i in range(N):
            noise = rng.normal(0.0, 0.05, size=L).astype(np.float32)
            drift = (rng.normal(0.0, 0.005) * np.linspace(-1, 1, L, dtype=np.float32))
            sig = (base + noise + drift).astype(np.float32)
            # Use a synthetic R at the center and segment with z-score
            seg = segment(sig, int(L // 2), pre=PRE_SAMPLES, post=POST_SAMPLES, apply_z=True)
            X_list.append(seg)
            y_list.append(cls_idx)
            rid.append(f"syn_{cls_idx}")

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, rid


def build_arrays(data_dir: Path | str, bandpass: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build feature/label arrays from a directory of MIT-BIH records.

    Args:
        data_dir: Path to directory containing WFDB files (.hea/.dat/.atr)
        bandpass: if True, apply a 0.5-40 Hz bandpass filter to each raw signal
                  before segmentation and z-scoring.

    Returns:
        X: (N, SEGMENT_SAMPLES) float32 segments (z-scored)
        y: (N,) int64 class indices in 0..4
        rec_ids: list of record ids for each beat

    Raises:
        RuntimeError: if no valid beats are found in the directory.
    """
    base = Path(data_dir)
    records = sorted([p.stem for p in base.glob("*.hea")])
    X, y, rid = [], [], []
    for rec in records:
        rp = base / rec
        try:
            sig, _ = read_signal(rp)
            rlocs, syms = read_ann(rp)
        except Exception:
            continue

        # Optionally apply bandpass to the raw signal before segmentation
        if bandpass:
            try:
                from scipy.signal import butter, filtfilt

                low = 0.5 / (FS / 2.0)
                high = 40.0 / (FS / 2.0)
                b, a = butter(3, [low, high], btype="bandpass")
                sig = filtfilt(b, a, sig.astype(np.float32))
            except Exception:
                # If scipy not available or filtering fails, continue with raw signal
                pass

        # Keep only classes in CLASS_MAP
        y_raw = [CLASS_MAP[s] for s in syms if s in CLASS_MAP]
        r_keep = [int(r) for r, s in zip(rlocs, syms) if s in CLASS_MAP]
        if not r_keep:
            continue
        Xi, yi = segment_beats(sig, r_keep, y_raw, pre=PRE_SAMPLES, post=POST_SAMPLES, apply_z=True)
        X.append(Xi)
        y.append(yi)
        rid.extend([rec] * len(yi))
    if not X:
        raise RuntimeError(f"No beats found in {base}; ensure dataset is available.")
    X_all = np.concatenate(X, axis=0).astype(np.float32)
    y_all = np.concatenate(y, axis=0).astype(np.int64)
    return X_all, y_all, rid


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute class weights inversely proportional to class frequency (balanced)."""
    y = np.asarray(y, dtype=int)
    classes, counts = np.unique(y, return_counts=True)
    n_samples = y.shape[0]
    n_classes = classes.shape[0]
    weights = n_samples / (n_classes * counts.astype(np.float64))
    return {int(c): float(w) for c, w in zip(classes, weights)}


def stratified_kfold_indices(y: np.ndarray, n_splits: int = 10, random_state: int = 42):
    """Yield stratified train/val indices for K-fold cross-validation."""
    from sklearn.model_selection import StratifiedKFold  # local import

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_idx, va_idx in skf.split(np.zeros_like(y), y):
        yield tr_idx, va_idx


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    """Stratified split into train/val/test.

    Note: val_size is relative to the remaining after taking the test set.
    """
    from sklearn.model_selection import train_test_split  # local import

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_rel = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_rel, random_state=random_state, stratify=y_train
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
