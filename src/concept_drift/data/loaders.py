from __future__ import annotations

import logging
from pathlib import Path
import zipfile
import pandas as pd

LOGGER = logging.getLogger(__name__)


NSL_COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]


def load_cicids2017(raw_files: list[Path]) -> pd.DataFrame:
    data_frames = []
    for f in raw_files:
        if f.suffix.lower() == ".zip":
            with zipfile.ZipFile(f) as zf:
                for member in zf.namelist():
                    if member.lower().endswith(".csv"):
                        with zf.open(member) as h:
                            data_frames.append(pd.read_csv(h, low_memory=False))
        elif f.suffix.lower() == ".csv":
            data_frames.append(pd.read_csv(f, low_memory=False))
    if not data_frames:
        raise FileNotFoundError("No CICIDS2017 CSV files found in provided raw files")
    df = pd.concat(data_frames, ignore_index=True)
    return df


def load_nsl_kdd(raw_files: list[Path]) -> pd.DataFrame:
    txt_files = [p for p in raw_files if p.suffix == ".txt"]
    if not txt_files:
        raise FileNotFoundError("NSL-KDD raw .txt files not found")
    dfs = [pd.read_csv(fp, names=NSL_COLS) for fp in txt_files]
    df = pd.concat(dfs, ignore_index=True)
    df["_pseudo_time"] = range(len(df))
    return df


def load_cicddos2019(raw_files: list[Path]) -> pd.DataFrame:
    csv_files = [p for p in raw_files if p.suffix.lower() == ".csv"]
    if not csv_files:
        raise FileNotFoundError(
            "CICDDoS2019 CSV not found. The default source only downloads official info page. "
            "Provide direct dataset CSV URLs in config before training."
        )
    return pd.concat([pd.read_csv(f, low_memory=False) for f in csv_files], ignore_index=True)


def load_dataset(dataset_name: str, raw_files: list[Path]) -> pd.DataFrame:
    if dataset_name == "cicids2017":
        return load_cicids2017(raw_files)
    if dataset_name == "nsl_kdd":
        return load_nsl_kdd(raw_files)
    if dataset_name == "cicddos2019":
        return load_cicddos2019(raw_files)
    raise ValueError(f"Unsupported dataset: {dataset_name}")
