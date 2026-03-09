from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


NSL_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files",
    "num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","label","difficulty"
]


def load_nsl_kdd(path: Path) -> pd.DataFrame:
    train = pd.read_csv(path / "KDDTrain%2B.txt", names=NSL_COLUMNS)
    test = pd.read_csv(path / "KDDTest%2B.txt", names=NSL_COLUMNS)
    df = pd.concat([train, test], ignore_index=True)
    df["timestamp"] = np.arange(len(df), dtype=np.int64)
    df["target"] = (df["label"] != "normal").astype(int)
    cat_cols = ["protocol_type", "service", "flag"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    drop_cols = ["label", "difficulty"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=c)
    return df


def load_generic_csv_dir(path: Path) -> pd.DataFrame:
    csvs = sorted(path.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {path}")
    frames = []
    for csv in csvs:
        frames.append(pd.read_csv(csv, low_memory=False))
    df = pd.concat(frames, ignore_index=True)

    # Attempt common schema harmonization for CIC datasets.
    cols = {c.lower().strip(): c for c in df.columns}
    label_col = cols.get("label") or cols.get(" label")
    if not label_col:
        raise ValueError("Expected label column not found in provided CSV files.")

    time_col = None
    for candidate in ["timestamp", " flow start time", "flow start time"]:
        if candidate in cols:
            time_col = cols[candidate]
            break
    if time_col is None:
        df["timestamp"] = np.arange(len(df), dtype=np.int64)
        time_col = "timestamp"

    y = df[label_col].astype(str).str.lower().str.strip()
    df["target"] = (~y.isin(["benign", "normal"])).astype(int)

    keep = [c for c in df.columns if c not in [label_col]]
    df = df[keep]
    df = df.replace([np.inf, -np.inf], np.nan)

    # Force temporal column.
    if time_col != "timestamp":
        df = df.rename(columns={time_col: "timestamp"})
    return df


def preprocess_dataframe(df: pd.DataFrame, max_rows: int | None = None):
    if max_rows is not None:
        df = df.iloc[:max_rows].copy()

    df = df.sort_values("timestamp").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ["target", "timestamp"]]

    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    y = df["target"].astype(int).to_numpy()
    t = df["timestamp"].to_numpy()

    return X_scaled.astype(np.float32), y.astype(np.int64), t, feature_cols, imputer, scaler
