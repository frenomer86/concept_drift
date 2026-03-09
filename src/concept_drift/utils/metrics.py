from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_prob.ndim == 2 and y_prob.shape[1] > 1:
        try:
            metrics["auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        except ValueError:
            metrics["auc_ovr"] = float("nan")
    return metrics
