from __future__ import annotations

import numpy as np


def temporal_split_indices(n: int, train_ratio: float, val_ratio: float, test_ratio: float):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    idx_train = np.arange(0, n_train)
    idx_val = np.arange(n_train, n_train + n_val)
    idx_test = np.arange(n_train + n_val, n)
    return idx_train, idx_val, idx_test
