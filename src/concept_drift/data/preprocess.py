from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


def _coerce_timestamp(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        ts = pd.to_datetime(df[col], errors="coerce")
        if ts.notna().sum() > 0:
            return ts.astype("int64") // 10**9
    return pd.Series(np.arange(len(df)), index=df.index)


def build_features(
    df: pd.DataFrame,
    target_col: str,
    timestamp_col: str,
    selected_features: list[str] | None,
    scale: str = "standard",
    dropna: bool = True,
):
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe")

    df["_time"] = _coerce_timestamp(df, timestamp_col)
    df = df.sort_values("_time").reset_index(drop=True)

    y_raw = df[target_col].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    feature_df = df.drop(columns=[c for c in [target_col] if c in df.columns])
    for c in feature_df.columns:
        if feature_df[c].dtype == "object":
            feature_df[c] = pd.factorize(feature_df[c].astype(str))[0]

    if selected_features:
        available = [c for c in selected_features if c in feature_df.columns]
        if available:
            feature_df = feature_df[available]

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    if dropna:
        mask = feature_df.notna().all(axis=1)
        feature_df = feature_df.loc[mask]
        y = y[mask.values]

    X = feature_df.astype(float).values
    scaler = StandardScaler() if scale == "standard" else MinMaxScaler()
    X = scaler.fit_transform(X)

    times = df.loc[feature_df.index, "_time"].to_numpy()
    return X, y, times, scaler, le, feature_df.columns.tolist()


def temporal_split(X, y, times, train_ratio: float, val_ratio: float, test_ratio: float):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6
    n = len(X)
    idx = np.argsort(times)
    X, y = X[idx], y[idx]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_end = n_train
    val_end = n_train + n_val
    return {
        "train": (X[:train_end], y[:train_end]),
        "val": (X[train_end:val_end], y[train_end:val_end]),
        "test": (X[val_end:], y[val_end:]),
    }
