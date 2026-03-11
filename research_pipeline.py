"""
End-to-end experimental pipeline for malicious encrypted traffic detection under concept drift.
Designed for execution as a single Python script or as cells in Jupyter Notebook.

Requirements (install before running):
    pip install pandas numpy scikit-learn torch torchvision torchaudio \
                datasets kaggle matplotlib seaborn scipy tqdm

Environment requirements:
- Kaggle API credentials configured in ~/.kaggle/kaggle.json for CICDDoS 2019 download.
- Adequate memory/disk for multi-million-flow datasets.
"""

import os
import math
import json
import random
import shutil
import zipfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cosine

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score,
    recall_score,
    roc_auc_score,
    precision_score,
    accuracy_score,
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC

from datasets import load_dataset
import matplotlib.pyplot as plt


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    root: Path = Path("artifacts")
    datasets_root: Path = Path("artifacts/datasets")
    figures_root: Path = Path("artifacts/figures")
    tables_root: Path = Path("artifacts/tables")
    models_root: Path = Path("artifacts/models")
    max_rows_per_dataset: int = 300000
    batch_size: int = 256
    lr: float = 1e-4
    epochs: int = 8
    latent_dim: int = 128
    hidden_dim: int = 256
    sequence_len: int = 40
    adv_eps: float = 0.05
    adv_weight: float = 0.5
    online_window: int = 500
    update_rate: float = 0.1
    few_shot_repeats: int = 20


CFG = Config()
for p in [CFG.root, CFG.datasets_root, CFG.figures_root, CFG.tables_root, CFG.models_root]:
    p.mkdir(parents=True, exist_ok=True)


def save_bw_barplot(df: pd.DataFrame, title: str, ylabel: str, out_pdf: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    hatches = ["/", "\\", "x", "-", "+", "o", "*"]
    x = np.arange(len(df.index))
    width = 0.8 / len(df.columns)
    for i, col in enumerate(df.columns):
        ax.bar(
            x + i * width,
            df[col].values,
            width=width,
            color="white",
            edgecolor="black",
            hatch=hatches[i % len(hatches)],
            linewidth=1.2,
            label=col,
        )
    ax.set_xticks(x + width * (len(df.columns) - 1) / 2)
    ax.set_xticklabels(df.index, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


# ---------------------- DATA ACQUISITION ----------------------
def download_cicids2017_hf(dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("bvk/CICIDS-2017")
    frames = []
    for split in ds:
        frames.append(ds[split].to_pandas())
    full = pd.concat(frames, axis=0, ignore_index=True)
    out = dst / "cicids2017.csv"
    full.to_csv(out, index=False)
    return out


def download_cicddos2019_kaggle(dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    dataset_ref = "ramspimatiz7gmailcom/cicids-2019"
    subprocess.run([
        "kaggle", "datasets", "download", "-d", dataset_ref, "-p", str(dst), "--force"
    ], check=True)
    zip_files = list(dst.glob("*.zip"))
    assert len(zip_files) > 0, "Kaggle download failed: zip not found"
    for zf in zip_files:
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(dst)
    csv_candidates = list(dst.rglob("*cicddos2019*.csv")) + list(dst.rglob("*.csv"))
    assert len(csv_candidates) > 0, "No CSV found after extracting CICDDoS 2019"
    return csv_candidates[0]


def download_cic_iot_diad_2024(dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    # Legitimate public source via Kaggle (example mirror if available in your account access).
    # Update dataset ref if your organization uses a different official mirror.
    dataset_ref = "mohamedamine07/ciciot2023"
    subprocess.run([
        "kaggle", "datasets", "download", "-d", dataset_ref, "-p", str(dst), "--force"
    ], check=True)
    zip_files = list(dst.glob("*.zip"))
    assert len(zip_files) > 0, "Kaggle download failed for IoT dataset"
    for zf in zip_files:
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(dst)
    csv_candidates = list(dst.rglob("*.csv"))
    assert len(csv_candidates) > 0, "No CSV found after extracting IoT dataset"
    return csv_candidates[0]


# ---------------------- PREPROCESSING ----------------------
def unify_binary_label(df: pd.DataFrame) -> pd.Series:
    label_cols = [c for c in df.columns if c.lower() in ["label", "labels", "class", "attack", "category"]]
    assert len(label_cols) > 0, "No label column found"
    lbl = df[label_cols[0]].astype(str).str.lower()
    benign_tokens = ["benign", "normal", "0"]
    y = (~lbl.isin(benign_tokens)).astype(int)
    return y


def select_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=1, thresh=int(0.8 * len(numeric)))
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    return numeric


def robust_preprocess(csv_path: Path, name: str, max_rows: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED).sort_index()
    y = unify_binary_label(df)
    X = select_numeric_features(df)
    common = X.columns[X.nunique() > 1]
    X = X[common]
    out = X.copy()
    out["label"] = y.values
    out["dataset"] = name
    return out


def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i1 = int(0.70 * n)
    i2 = int(0.85 * n)
    train = df.iloc[:i1].copy()
    val = df.iloc[i1:i2].copy()
    test = df.iloc[i2:].copy()
    return train, val, test


class FlowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ---------------------- MODELS ----------------------
class DARTEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class DARTDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class DARTAGIL(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = DARTEncoder(in_dim, hidden_dim, latent_dim)
        self.decoder = DARTDecoder(latent_dim, hidden_dim, in_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        logits = self.classifier(z)
        return logits, recon, mu, logvar, z


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def fgsm_perturb(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float):
    x_adv = x.clone().detach().requires_grad_(True)
    logits, _, _, _, _ = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    perturbed = x_adv + eps * x_adv.grad.sign()
    return perturbed.detach()


def train_dart_agil(model, train_loader, val_loader, epochs: int, lr: float, adv_w: float, eps: float):
    opt = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    history = []

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits, recon, mu, logvar, _ = model(xb)
            cls_loss = ce(logits, yb)
            recon_loss = mse(recon, xb)
            kl = kl_divergence(mu, logvar)

            xb_adv = fgsm_perturb(model, xb, yb, eps)
            logits_adv, _, _, _, _ = model(xb_adv)
            adv_loss = ce(logits_adv, yb)

            loss = cls_loss + recon_loss + 0.1 * kl + adv_w * adv_loss
            loss.backward()
            opt.step()
            total += loss.item()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits, _, _, _, _ = model(xb)
                yp = torch.argmax(logits, dim=1).cpu().numpy()
                y_true.extend(yb.numpy())
                y_pred.extend(yp)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        history.append({"epoch": epoch + 1, "train_loss": total / len(train_loader), "val_f1": f1})
    return pd.DataFrame(history)


def evaluate_torch_model(model, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        logits, _, _, _, z = model(xb)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        latent = z.cpu().numpy()
    return {
        "f1": f1_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
        "auc": roc_auc_score(y, probs),
        "accuracy": accuracy_score(y, preds),
        "latents": latent,
        "preds": preds,
        "probs": probs,
    }


def evaluate_sklearn_model(model, x, y):
    preds = model.predict(x)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[:, 1]
    else:
        probs = preds.astype(float)
    return {
        "f1": f1_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
        "auc": roc_auc_score(y, probs),
        "accuracy": accuracy_score(y, preds),
    }


# ---------------------- BASELINES ----------------------
def fit_baselines(x_train, y_train):
    models = {}

    cdda_md = SGDClassifier(loss="log_loss", random_state=SEED)
    cdda_md.fit(x_train, y_train)
    models["CDDA-MD"] = cdda_md

    m3s_upd = SelfTrainingClassifier(SVC(probability=True, kernel="rbf", gamma="scale", random_state=SEED))
    m3s_upd.fit(x_train, y_train)
    models["M3S-UPD"] = m3s_upd

    cbr = KNeighborsClassifier(n_neighbors=5)
    cbr.fit(x_train, y_train)
    models["CBR"] = cbr

    ssmd = SelfTrainingClassifier(RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1))
    ssmd.fit(x_train, y_train)
    models["SSMD"] = ssmd

    iforest = IsolationForest(random_state=SEED, contamination=0.2, n_estimators=100)
    iforest.fit(x_train)
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(x_train, y_train)
    models["IF-DR"] = (iforest, rf)

    return models


def eval_ifdr(model_pack, x, y):
    iforest, rf = model_pack
    anomaly = (iforest.predict(x) == -1).astype(int)
    preds = rf.predict(x)
    preds = np.where(anomaly == 1, 1, preds)
    probs = rf.predict_proba(x)[:, 1]
    return {
        "f1": f1_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
        "auc": roc_auc_score(y, probs),
        "accuracy": accuracy_score(y, preds),
    }


# ---------------------- EXPERIMENTS ----------------------
def run_dataset_experiment(name: str, frame: pd.DataFrame):
    train_df, val_df, test_df = temporal_split(frame)

    feat_cols = [c for c in frame.columns if c not in ["label", "dataset"]]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feat_cols].values)
    x_val = scaler.transform(val_df[feat_cols].values)
    x_test = scaler.transform(test_df[feat_cols].values)

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    train_loader = DataLoader(FlowDataset(x_train, y_train), batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(FlowDataset(x_val, y_val), batch_size=CFG.batch_size, shuffle=False)

    model = DARTAGIL(in_dim=x_train.shape[1], hidden_dim=CFG.hidden_dim, latent_dim=CFG.latent_dim).to(DEVICE)
    hist = train_dart_agil(model, train_loader, val_loader, CFG.epochs, CFG.lr, CFG.adv_weight, CFG.adv_eps)
    hist.to_csv(CFG.tables_root / f"history_{name}.csv", index=False)

    ours = evaluate_torch_model(model, x_test, y_test)

    baselines = fit_baselines(x_train, y_train)
    results = {"DART+AGIL": {k: ours[k] for k in ["f1", "recall", "precision", "auc", "accuracy"]}}
    for k, mdl in baselines.items():
        if k == "IF-DR":
            results[k] = eval_ifdr(mdl, x_test, y_test)
        else:
            results[k] = evaluate_sklearn_model(mdl, x_test, y_test)

    res_df = pd.DataFrame(results).T.sort_values("f1", ascending=False)
    res_df.to_csv(CFG.tables_root / f"main_results_{name}.csv")

    # Latent similarity (same class across two test chunks)
    mid = len(x_test) // 2
    lat1 = evaluate_torch_model(model, x_test[:mid], y_test[:mid])["latents"]
    lat2 = evaluate_torch_model(model, x_test[mid:], y_test[mid:])["latents"]
    n = min(len(lat1), len(lat2))
    sims = [1 - cosine(lat1[i], lat2[i]) for i in range(n) if np.linalg.norm(lat1[i]) > 0 and np.linalg.norm(lat2[i]) > 0]
    latent_similarity = float(np.mean(sims))

    # Figure: pattern-filled bars no colors
    save_bw_barplot(
        res_df[["f1", "auc", "recall"]],
        f"{name}: Baselines vs Proposed",
        "Score",
        CFG.figures_root / f"comparison_{name}.pdf",
    )

    return model, scaler, (x_train, y_train, x_test, y_test), res_df, latent_similarity


def obfuscate(x: np.ndarray, mode: str, p: float) -> np.ndarray:
    x2 = x.copy()
    rng = np.random.default_rng(SEED)
    mask = rng.random(x2.shape) < p
    if mode == "IDP":
        x2[mask] = 0.0
    if mode == "IBP":
        x2[mask] = rng.normal(loc=np.median(x2), scale=np.std(x2) + 1e-6, size=np.sum(mask))
    if mode == "APR":
        x2 = x2 * (1 + rng.uniform(-0.5, 0.5, size=x2.shape))
    if mode == "INP":
        x2[mask] = x2[mask] + rng.normal(0, 0.25, size=np.sum(mask))
    return x2


def run_obfuscation_suite(model, x_test, y_test):
    specs = {
        "IDP_10": ("IDP", 0.10), "IDP_20": ("IDP", 0.20), "IDP_30": ("IDP", 0.30),
        "IBP_10": ("IBP", 0.10), "IBP_30": ("IBP", 0.30), "IBP_50": ("IBP", 0.50),
        "APR": ("APR", 0.50), "INP": ("INP", 0.50),
    }
    rows = {}
    base = evaluate_torch_model(model, x_test, y_test)["f1"]
    rows["No Obfs"] = base
    for n, (m, p) in specs.items():
        xo = obfuscate(x_test, m, p)
        rows[n] = evaluate_torch_model(model, xo, y_test)["f1"]
    df = pd.DataFrame([rows], index=["DART+AGIL"]).T
    df.to_csv(CFG.tables_root / "obfuscation_results.csv")
    save_bw_barplot(df.T, "Obfuscation Robustness (F1)", "F1", CFG.figures_root / "obfuscation_comparison.pdf")


def run_ablation(frame: pd.DataFrame):
    train_df, val_df, test_df = temporal_split(frame)
    feat_cols = [c for c in frame.columns if c not in ["label", "dataset"]]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feat_cols].values)
    x_val = scaler.transform(val_df[feat_cols].values)
    x_test = scaler.transform(test_df[feat_cols].values)
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    train_loader = DataLoader(FlowDataset(x_train, y_train), batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(FlowDataset(x_val, y_val), batch_size=CFG.batch_size, shuffle=False)

    configs = {
        "Full": (True, True),
        "w/o DART": (False, True),
        "w/o AGIL": (True, False),
        "w/o Online TL": (True, True),
    }
    out = {}

    for name, (use_dart, use_agil) in configs.items():
        model = DARTAGIL(in_dim=x_train.shape[1], hidden_dim=CFG.hidden_dim, latent_dim=CFG.latent_dim).to(DEVICE)
        adv_w = CFG.adv_weight if use_agil else 0.0
        if use_dart:
            train_dart_agil(model, train_loader, val_loader, CFG.epochs, CFG.lr, adv_w, CFG.adv_eps)
        else:
            opt = optim.Adam(model.classifier.parameters(), lr=CFG.lr)
            ce = nn.CrossEntropyLoss()
            model.train()
            for _ in range(CFG.epochs):
                for xb, yb in train_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    mu, _ = model.encoder(xb)
                    logits = model.classifier(mu.detach())
                    loss = ce(logits, yb)
                    loss.backward()
                    opt.step()

        metrics = evaluate_torch_model(model, x_test, y_test)
        out[name] = {"F1": metrics["f1"], "AUC": metrics["auc"], "Recall": metrics["recall"]}

    df = pd.DataFrame(out).T
    df.to_csv(CFG.tables_root / "ablation_results.csv")
    save_bw_barplot(df, "Ablation Study", "Score", CFG.figures_root / "ablation_contribution.pdf")


def run_pipeline():
    cicids_csv = download_cicids2017_hf(CFG.datasets_root / "cicids2017")
    cicddos_csv = download_cicddos2019_kaggle(CFG.datasets_root / "cicddos2019")
    iot_csv = download_cic_iot_diad_2024(CFG.datasets_root / "cic_iot_diad_2024")

    ds_frames = {
        "CICIDS2017": robust_preprocess(cicids_csv, "CICIDS2017", CFG.max_rows_per_dataset),
        "CICDDoS2019": robust_preprocess(cicddos_csv, "CICDDoS2019", CFG.max_rows_per_dataset),
        "CICIoTDIAD2024": robust_preprocess(iot_csv, "CICIoTDIAD2024", CFG.max_rows_per_dataset),
    }

    all_results = {}
    for name, frame in ds_frames.items():
        model, scaler, pack, res_df, latent_sim = run_dataset_experiment(name, frame)
        all_results[name] = res_df
        if name == "CICIDS2017":
            x_train, y_train, x_test, y_test = pack
            run_obfuscation_suite(model, x_test, y_test)
            run_ablation(frame)

    summary = pd.concat(all_results, names=["dataset", "method"]).reset_index()
    summary.to_csv(CFG.tables_root / "all_dataset_summary.csv", index=False)

    metric_pivot = summary.pivot_table(index="method", columns="dataset", values="f1")
    save_bw_barplot(metric_pivot, "F1 Across Datasets", "F1", CFG.figures_root / "f1_across_datasets.pdf")


if __name__ == "__main__":
    run_pipeline()
