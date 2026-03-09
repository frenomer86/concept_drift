from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score


def classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auc_roc"] = float("nan")
    out["ece"] = expected_calibration_error(y_true, y_prob)
    return out


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if np.any(m):
            conf = y_prob[m].mean()
            acc = y_true[m].mean()
            ece += (m.mean()) * abs(acc - conf)
    return float(ece)
