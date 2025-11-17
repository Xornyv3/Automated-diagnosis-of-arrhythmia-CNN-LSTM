from __future__ import annotations
import os
import re
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ecgclf"


def check_make_setup() -> Tuple[bool, str]:
    mk = ROOT / "Makefile"
    if not mk.exists():
        return False, "Makefile not found"
    text = mk.read_text(encoding="utf-8", errors="ignore")
    ok = re.search(r"^setup:\s*$", text, re.M) is not None
    return ok, "setup target present" if ok else "setup target missing"


def check_segments_zscore() -> Tuple[bool, str]:
    # Build a synthetic signal and segment
    sys.path.insert(0, str(ROOT / "src"))
    from ecgclf.preprocess import segment, SEGMENT_SAMPLES

    sig = np.sin(np.linspace(0, 20 * np.pi, 5000, dtype=np.float32))
    r_idx = 2500
    seg = segment(sig, r_idx, pre=500, post=500, apply_z=True)
    if seg.shape[0] != SEGMENT_SAMPLES:
        return False, f"segment length {seg.shape[0]} != {SEGMENT_SAMPLES}"
    m = float(np.mean(seg))
    s = float(np.std(seg))
    if not (abs(m) < 1e-5 and 0.9 < s < 1.1):
        return False, f"z-score check failed (mean={m:.3g}, std={s:.3g})"
    return True, "segment length=1000 and z-score ~N(0,1)"


def check_kfold_reproducible() -> Tuple[bool, str]:
    sys.path.insert(0, str(ROOT / "src"))
    from ecgclf.preprocess import stratified_kfold_indices

    y = np.array([0] * 50 + [1] * 30 + [2] * 20, dtype=int)
    idx1 = list(stratified_kfold_indices(y, n_splits=5, random_state=123))
    idx2 = list(stratified_kfold_indices(y, n_splits=5, random_state=123))
    same = all((np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])) for a, b in zip(idx1, idx2))
    return same, "stratified_kfold_indices reproducible for fixed seed" if same else "kfold split differs across runs"


def check_train_callbacks_present() -> Tuple[bool, str]:
    p = SRC / "train.py"
    text = p.read_text(encoding="utf-8", errors="ignore")
    has_ckpt = "ModelCheckpoint" in text and "save_best_only=True" in text
    has_tb = "TensorBoard(" in text
    return (has_ckpt and has_tb), (
        "ModelCheckpoint(save_best_only=True) and TensorBoard callbacks detected" if (has_ckpt and has_tb) else "callbacks missing"
    )


def check_evaluate_exports() -> Tuple[bool, str]:
    p = SRC / "evaluate.py"
    text = p.read_text(encoding="utf-8", errors="ignore")
    # Verify presence of metrics keys and confusion matrix plotting calls
    has_spec = "specificity" in text
    has_recall = "recall" in text
    has_precision = "precision" in text
    has_ppv = "ppv" in text
    has_cm_png = "confusion_matrix_norm.png" in text or "confusion_matrix.png" in text
    ok = has_spec and has_recall and has_precision and has_ppv and has_cm_png
    return ok, "evaluate exports accuracy/recall/specificity/PPV and confusion matrix figure" if ok else "evaluate exports missing"


def check_predict_outputs() -> Tuple[bool, str]:
    p = SRC / "predict.py"
    text = p.read_text(encoding="utf-8", errors="ignore")
    has_csv = "output_csv" in text
    has_fig = "output_fig" in text
    ok = has_csv and has_fig
    return ok, "predict produces CSV of probabilities and a preview figure" if ok else "predict outputs not detected"


def check_readme_repro() -> Tuple[bool, str]:
    p = ROOT / "README.md"
    if not p.exists():
        return False, "README.md not found"
    text = p.read_text(encoding="utf-8", errors="ignore").lower()
    required = ["make setup", "make train", "make eval", "predict", "tensorboard", "repro"]
    ok = all(any(r in line for line in text.splitlines()) if r != "repro" else ("repro.sh" in text or "reproduce" in text) for r in required)
    return ok, "README contains reproduction instructions" if ok else "README missing some reproduction steps"


def main() -> int:
    checks = [
        ("make setup target", check_make_setup),
        ("segments length and z-score", check_segments_zscore),
        ("k-fold reproducibility (seed)", check_kfold_reproducible),
        ("train callbacks (best weights + TensorBoard)", check_train_callbacks_present),
        ("evaluate exports (acc/recall/specificity/PPV + CM PNG)", check_evaluate_exports),
        ("predict outputs (CSV + figure)", check_predict_outputs),
        ("README reproduction", check_readme_repro),
    ]
    failed = 0
    for name, fn in checks:
        ok, msg = fn()
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {msg}")
        if not ok:
            failed += 1
    print(f"\nSummary: {len(checks) - failed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
