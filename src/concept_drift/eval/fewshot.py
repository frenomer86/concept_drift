from __future__ import annotations

import numpy as np
from sklearn.base import clone


def fewshot_scores(model, X_train, y_train, X_test, y_test, shots: list[int], repeats: int = 20):
    rng = np.random.default_rng(42)
    malicious_idx = np.where(y_train == 1)[0]
    benign_idx = np.where(y_train == 0)[0]

    out: list[dict] = []
    for n in shots:
        scores = []
        for _ in range(repeats):
            if len(malicious_idx) < n or len(benign_idx) < n:
                continue
            sel_m = rng.choice(malicious_idx, size=n, replace=False)
            sel_b = rng.choice(benign_idx, size=n, replace=False)
            idx = np.concatenate([sel_m, sel_b])
            Xs, ys = X_train[idx], y_train[idx]

            m = clone(model)
            m.fit(Xs, ys)
            if hasattr(m, "predict_proba"):
                p = m.predict_proba(X_test)[:, 1]
            else:
                pred = m.predict(X_test)
                p = pred.astype(float)
            from concept_drift.eval.metrics import classification_metrics
            scores.append(classification_metrics(y_test, p)["f1"])

        if scores:
            out.append({"shots": n, "f1_mean": float(np.mean(scores)), "f1_std": float(np.std(scores))})
    return out
