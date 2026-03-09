from __future__ import annotations

import numpy as np


def apply_idp(X: np.ndarray, p: float) -> np.ndarray:
    noise = np.random.normal(0, 0.05, size=X.shape).astype(np.float32)
    mask = (np.random.rand(*X.shape) < p).astype(np.float32)
    return X + noise * mask


def apply_ibp(X: np.ndarray, p: float) -> np.ndarray:
    X2 = X.copy()
    idx = np.random.choice(len(X2), size=len(X2), replace=True)
    benign_like = X2[idx]
    mask = (np.random.rand(*X.shape) < p).astype(np.float32)
    return X2 * (1 - mask) + benign_like * mask


def apply_apr(X: np.ndarray, pct: float) -> np.ndarray:
    scale = 1.0 + np.random.uniform(-pct, pct, size=X.shape).astype(np.float32)
    return X * scale


def apply_inp(X: np.ndarray, p: float) -> np.ndarray:
    jitter = np.random.laplace(0, 0.03, size=X.shape).astype(np.float32)
    mask = (np.random.rand(*X.shape) < p).astype(np.float32)
    return X + jitter * mask
