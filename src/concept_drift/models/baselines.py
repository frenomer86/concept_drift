from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def build_baselines(cfg: dict):
    return {
        "random_forest": RandomForestClassifier(**cfg["random_forest"], random_state=42, n_jobs=-1),
        "mlp": MLPClassifier(**cfg["mlp"], random_state=42),
        "logistic_regression": LogisticRegression(**cfg["logistic_regression"], random_state=42, n_jobs=-1),
    }
