from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .preprocess import CLASSES, build_arrays, stratified_kfold_indices
from .model import ensure_3d
from .viz import plot_confusion_matrix_norm


def _load_keras_model(path: Path):
    from tensorflow import keras

    return keras.models.load_model(str(path))


def _metrics_and_cm(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict:
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float(accuracy_score(y_true, y_pred))

    # Precision, Recall (Sensitivity), F1 per class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # Specificity per class = TN / (TN + FP)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    with np.errstate(divide="ignore", invalid="ignore"):
        specificity = np.where(tn + fp > 0, tn / (tn + fp), 0.0)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "support": support,
        "confusion_matrix": cm,
    }


def _report_dataframe(metrics: Dict) -> "pandas.DataFrame":
    import pandas as pd

    df = pd.DataFrame(
        {
            "class": CLASSES,
            "precision": metrics["precision"],
            "ppv": metrics["precision"],
            "recall": metrics["recall"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "support": metrics["support"],
        }
    )
    return df


def _save_confusion_matrix(cm: np.ndarray, out_png: Path):
    # Wrapper retained for compatibility; now plots normalized confusion via viz utility.
    plot_confusion_matrix_norm(cm, class_names=CLASSES, out_path=out_png)


def _save_reports(df, out_prefix: Path, overall_acc: float):
    import pandas as pd

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    # CSV
    df.to_csv(out_prefix.with_suffix(".csv"), index=False)
    # Markdown (avoid dependency on tabulate)
    md_lines = ["| class | precision | PPV | recall | specificity | f1 | support |", "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        md_lines.append(
            f"| {r['class']} | {r['precision']:.4f} | {r['ppv']:.4f} | {r['recall']:.4f} | {r['specificity']:.4f} | {r['f1']:.4f} | {int(r['support'])} |"
        )
    md_lines.append("")
    md_lines.append(f"Overall accuracy: {overall_acc:.4f}")
    (out_prefix.with_suffix(".md")).write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    """Evaluate saved fold models, save per-fold metrics, and aggregate results."""
    ap = argparse.ArgumentParser(description="Evaluate trained fold models on validation splits and report metrics")
    ap.add_argument("--data_dir", type=str, default="data/mitdb", help="Path to mitdb directory (ignored when --synthetic)")
    ap.add_argument("--models_dir", type=str, default="models")
    ap.add_argument("--logdir", type=str, default="runs/eval")
    ap.add_argument("--fold_k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--fold_avg", action="store_true", help="Also compute mean/std aggregation across folds")
    ap.add_argument("--synthetic", action="store_true", help="Use synthetic offline dataset")
    ap.add_argument("--synthetic_n", type=int, default=200, help="Synthetic samples per class (when --synthetic)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path("data/mitdb")
    models_dir = Path(args.models_dir)
    out_dir = Path(args.logdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data and folds
    if args.synthetic:
        from .preprocess import build_synthetic_arrays

        X, y, _ = build_synthetic_arrays(n_per_class=args.synthetic_n, seed=args.seed)
    else:
        if not data_dir.exists():
            raise SystemExit(f"--data_dir not found: {data_dir}. Provide a valid dataset path or use --synthetic.")
        X, y, _ = build_arrays(data_dir)
    X = ensure_3d(X)
    folds = list(stratified_kfold_indices(y, n_splits=args.fold_k, random_state=args.seed))

    all_metrics: List[Dict] = []
    all_cms: List[np.ndarray] = []

    for i, (_, va_idx) in enumerate(folds, start=1):
        model_path = models_dir / f"fold_{i}.keras"
        if not model_path.exists():
            print(f"[WARN] Missing model for fold {i}: {model_path}")
            continue
        model = _load_keras_model(model_path)
        x_va, y_va = X[va_idx], y[va_idx]
        probs = model.predict(x_va, batch_size=args.batch_size, verbose=0)
        y_pred = np.argmax(probs, axis=1)

        mets = _metrics_and_cm(y_va, y_pred, num_classes=len(CLASSES))
        all_metrics.append(mets)
        all_cms.append(mets["confusion_matrix"])

        # Save confusion matrix and report per fold
        fold_dir = out_dir / f"fold_{i}"
        # Save normalized confusion matrix figure to artifacts/figures path, as requested
        art_fig_path = Path("artifacts/figures") / f"fold_{i}" / "confusion_matrix_norm.png"
        _save_confusion_matrix(mets["confusion_matrix"], art_fig_path)
        df = _report_dataframe(mets)
        _save_reports(df, fold_dir / "classification_report", overall_acc=mets["accuracy"])

        mean_spec = float(np.mean(mets["specificity"]))
        print(f"Fold {i}: acc={mets['accuracy']:.4f} | mean_specificity={mean_spec:.4f}")

    if not all_metrics:
        print("No folds evaluated (missing models?).")
        return

    if args.fold_avg:
        # Aggregate metrics: mean and std across folds
        accs = np.array([m["accuracy"] for m in all_metrics], dtype=float)
        prec = np.stack([m["precision"] for m in all_metrics], axis=0)
        rec = np.stack([m["recall"] for m in all_metrics], axis=0)
        spec = np.stack([m["specificity"] for m in all_metrics], axis=0)
        f1 = np.stack([m["f1"] for m in all_metrics], axis=0)
        cm_sum = np.sum(np.stack(all_cms, axis=0), axis=0)

        agg = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "precision_mean": np.mean(prec, axis=0).tolist(),
            "precision_std": np.std(prec, axis=0).tolist(),
            "recall_mean": np.mean(rec, axis=0).tolist(),
            "recall_std": np.std(rec, axis=0).tolist(),
            "specificity_mean": np.mean(spec, axis=0).tolist(),
            "specificity_std": np.std(spec, axis=0).tolist(),
            "f1_mean": np.mean(f1, axis=0).tolist(),
            "f1_std": np.std(f1, axis=0).tolist(),
        }

        # Save aggregate JSON
        (out_dir / "aggregate_metrics.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

        # Save aggregate normalized confusion matrix to artifacts/figures
        _save_confusion_matrix(cm_sum, Path("artifacts/figures") / "aggregate_confusion_matrix_norm.png")

        # Also save an aggregate report (means only)
        import pandas as pd

        # Create a mean-report DataFrame with the same column names as per-fold reports
        # Use mean values and approximate support as the mean support across folds
        supports = np.stack([m["support"] for m in all_metrics], axis=0)
        support_mean = np.mean(supports, axis=0).astype(int).tolist()

        df_mean = pd.DataFrame(
            {
                "class": CLASSES,
                "precision": agg["precision_mean"],
                "recall": agg["recall_mean"],
                "ppv": agg["precision_mean"],
                "specificity": agg["specificity_mean"],
                "f1": agg["f1_mean"],
                "support": support_mean,
            }
        )
        _save_reports(df_mean, out_dir / "aggregate_classification_report", overall_acc=agg["accuracy_mean"])
        print(f"Aggregate accuracy mean: {agg['accuracy_mean']:.4f} | aggregate mean_specificity: {float(np.mean(agg['specificity_mean'])):.4f}")


if __name__ == "__main__":
    main()
