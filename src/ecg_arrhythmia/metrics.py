from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, np.ndarray]:
    """
    Compute sensitivity (recall), specificity, PPV (precision) per class.
    Specificity per class i is TN / (TN + FP) for one-vs-rest.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivity = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        specificity = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
        ppv = np.where(tp + fp > 0, tp / (tp + fp), 0.0)

    return {
        "confusion_matrix": cm,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "accuracy": accuracy_score(y_true, y_pred),
    }


def summarize_metrics(mets: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {
        "accuracy": float(mets["accuracy"]),
        "sensitivity_mean": float(np.mean(mets["sensitivity"])),
        "specificity_mean": float(np.mean(mets["specificity"])),
        "ppv_mean": float(np.mean(mets["ppv"])),
    }
