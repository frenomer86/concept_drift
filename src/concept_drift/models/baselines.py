from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


@dataclass
class BaselineModel:
    name: str
    model: Any

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(X)
            if p.ndim == 2 and p.shape[1] > 1:
                return p[:, 1]
            return p.reshape(-1)
        scores = self.model.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores


class IFDriftBaseline:
    """IsolationForest drift trigger + LogisticRegression classifier."""

    def __init__(self, contamination: float = 0.05):
        self.detector = IsolationForest(contamination=contamination, random_state=42)
        self.classifier = LogisticRegression(max_iter=1000, n_jobs=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.detector.fit(X)
        self.classifier.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(X)[:, 1]

    def drift_rate(self, X: np.ndarray) -> float:
        pred = self.detector.predict(X)
        return float((pred == -1).mean())


def build_baselines(seed: int = 42) -> list[BaselineModel]:
    return [
        BaselineModel("logreg", LogisticRegression(max_iter=1000, n_jobs=1, random_state=seed)),
        BaselineModel("rf", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)),
        BaselineModel("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50, random_state=seed)),
    ]
