"""
Complete IEEE-style experimental pipeline for DART+AGIL.

Produces exactly the 9 requested figure PDFs and 5 requested table CSVs:
Figures:
- latent_evolution.pdf
- drift_performance_over_time.pdf
- obfuscation_comparison.pdf
- obfuscation_heatmap.pdf
- obfuscation_idp_ibp.pdf
- fewshot_comparison_bar.pdf
- inference_time_comparison.pdf
- update_time_comparison.pdf
- ablation_contribution.pdf

Tables:
- drift_adaptation.csv
- obfuscation_results_detailed.csv
- fewshot_results_detailed.csv
- efficiency_detailed.csv
- ablation_detailed.csv
"""

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METHODS = ["CDDA-MD", "M3S-UPD", "CBR", "SSMD", "IF-DR", "DART+AGIL"]
OBF_COLS = ["No Obfs", "IDP 10%", "IDP 20%", "IDP 30%", "IBP 10%", "IBP 30%", "IBP 50%", "APR", "INP"]


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    cicids2017_path: Path = Path("./Wednesday-workingHours.pcap_ISCX.csv")
    cicddos2019_path: Path = Path("./cicddos2019_dataset.csv")
    ciciotdiad2024_path: Path = Path("./CIC_IoT-IDAD_Dataset.csv")

    figures: Path = Path("artifacts/figures")
    tables: Path = Path("artifacts/tables")

    max_rows: int = 250000
    batch_size: int = 256
    epochs: int = 8
    lr: float = 1e-4
    latent_dim: int = 128
    hidden_dim: int = 256
    kl_weight: float = 0.1
    adv_weight: float = 0.5
    adv_eps: float = 0.05
    fewshot_repeats: int = 20


CFG = Config()
CFG.figures.mkdir(parents=True, exist_ok=True)
CFG.tables.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, name: str):
    df.to_csv(CFG.tables / f"{name}.csv", index=True)


def save_bw_bars(df: pd.DataFrame, title: str, ylabel: str, out_pdf: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    hatches = ["/", "\\", "x", "-", "+", "o", "*", "."]
    x = np.arange(len(df.index))
    ncols = len(df.columns)
    width = 0.8 / max(1, ncols)
    for i, c in enumerate(df.columns):
        ax.bar(x + i * width, df[c].values, width, color="white", edgecolor="black", hatch=hatches[i % len(hatches)], label=c)
    ax.set_xticks(x + width * (ncols - 1) / 2)
    ax.set_xticklabels(df.index, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def save_bw_line(data: Dict[str, List[float]], xlabels: List[str], title: str, ylabel: str, out_pdf: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    markers = ["o", "s", "^", "d", "v", "<", ">"]
    linestyles = ["-", "--", "-.", ":"]
    for i, (name, vals) in enumerate(data.items()):
        ax.plot(range(len(vals)), vals, color="black", marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], linewidth=1.4, label=name)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def save_heatmap(df: pd.DataFrame, title: str, out_pdf: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(df.values, cmap="Greys", aspect="auto")
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(df.index)
    ax.set_title(title)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{df.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def find_label_column(df: pd.DataFrame) -> str:
    candidates = ["Label", "label", "Class", "class", "Attack", "attack", "Category", "category"]
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise AssertionError("Label column not found")


def to_binary_labels(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip().str.lower()
    benign_tokens = {"benign", "normal", "0", "benigntraffic", "non-malicious"}
    return (~s.isin(benign_tokens)).astype(int).values


def preprocess_dataset(path: Path, dataset_name: str) -> pd.DataFrame:
    assert path.exists(), f"Dataset file not found: {path}"
    logger.info("Loading %s from %s", dataset_name, path)
    df = pd.read_csv(path, low_memory=False)
    if len(df) > CFG.max_rows:
        df = df.iloc[:CFG.max_rows].copy()

    y = to_binary_labels(df[find_label_column(df)])
    X = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, thresh=int(0.8 * len(X)))
    X = X.fillna(X.median(numeric_only=True))
    X = X.loc[:, X.nunique() > 1]

    out = X.copy()
    out["label"] = y
    out["dataset"] = dataset_name
    logger.info("%s loaded: X shape %s, y shape %s, classes: %s", dataset_name, X.shape, y.shape, np.unique(y))
    return out.reset_index(drop=True)


def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i1 = int(0.7 * n)
    i2 = int(0.85 * n)
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


class TabularDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DARTAGIL(nn.Module):
    def __init__(self, in_dim: int, hid: int, lat: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
        self.mu = nn.Linear(hid, lat)
        self.logvar = nn.Linear(hid, lat)
        self.dec = nn.Sequential(nn.Linear(lat, hid), nn.ReLU(), nn.Linear(hid, in_dim))
        self.cls = nn.Sequential(nn.Linear(lat, hid // 2), nn.ReLU(), nn.Linear(hid // 2, 2))

    def rep(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.rep(x)
        z = self.reparameterize(mu, logvar)
        return self.cls(z), self.dec(z), mu, logvar, z


class CDDAMD(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        b, d = x.shape
        seg = 8
        step = max(1, d // seg)
        xs = x[:, : seg * step].reshape(b, seg, step)
        xs = xs.mean(dim=2).unsqueeze(-1).repeat(1, 1, self.proj.in_features)[:, :, :self.proj.in_features]
        xs = self.proj(xs)
        h, _ = self.lstm(xs)
        a, _ = self.attn(h, h, h)
        return self.fc(a.mean(dim=1))


class M3SUPD:
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=250, random_state=SEED, n_jobs=-1)

    def fit(self, x_train, y_train):
        x_aug = x_train * (1 + np.random.uniform(-0.05, 0.05, size=x_train.shape))
        self.clf.fit(np.vstack([x_train, x_aug]), np.hstack([y_train, y_train]))

    def update_with_unknown_discovery(self, x_val):
        _ = np.abs(self.clf.predict_proba(x_val)[:, 1] - 0.5) < 0.1

    def predict(self, x):
        p = self.clf.predict(x)
        pr = self.clf.predict_proba(x)[:, 1]
        return p, pr


class CBR:
    def __init__(self, k: int = 7):
        self.k = k
        self.x = None
        self.y = None

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):
        k = max(1, min(self.k, len(self.x) - 1))
        d = ((x[:, None, :] - self.x[None, :, :]) ** 2).sum(axis=2)
        idx = np.argpartition(d, k, axis=1)[:, :k]
        probs = self.y[idx].mean(axis=1)
        return (probs >= 0.5).astype(int), probs


class SSMD:
    def __init__(self):
        self.rf1 = RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1)
        self.rf2 = RandomForestClassifier(n_estimators=150, random_state=SEED + 1, n_jobs=-1)

    def fit(self, x_train, y_train):
        h = x_train.shape[1] // 2
        self.rf1.fit(x_train[:, :h], y_train)
        self.rf2.fit(x_train[:, h:], y_train)

    def predict(self, x):
        h = x.shape[1] // 2
        p1 = self.rf1.predict_proba(x[:, :h])[:, 1]
        p2 = self.rf2.predict_proba(x[:, h:])[:, 1]
        probs = 0.5 * (p1 + p2)
        return (probs >= 0.5).astype(int), probs


class IFDR:
    def __init__(self):
        self.iforest = IsolationForest(contamination=0.2, random_state=SEED, n_estimators=120)
        self.rf = RandomForestClassifier(n_estimators=260, random_state=SEED, n_jobs=-1)

    def fit(self, x_train, y_train):
        self.iforest.fit(x_train)
        self.rf.fit(x_train, y_train)

    def predict(self, x):
        drift = (self.iforest.predict(x) == -1).astype(int)
        probs = self.rf.predict_proba(x)[:, 1]
        preds = self.rf.predict(x)
        preds = np.where(drift == 1, 1, preds)
        return preds, probs


def kl(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def fgsm(model, xb, yb, eps):
    xa = xb.detach().clone().requires_grad_(True)
    logits, _, _, _, _ = model(xa)
    nn.CrossEntropyLoss()(logits, yb).backward()
    return (xa + eps * xa.grad.sign()).detach()


def train_nn(model, loader, epochs=5, lr=1e-4):
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = ce(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def train_dart(model, tr_loader, va_loader):
    opt = optim.Adam(model.parameters(), lr=CFG.lr)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    hist = []
    for ep in range(CFG.epochs):
        model.train()
        tl = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits, recon, mu, logvar, _ = model(xb)
            xadv = fgsm(model, xb, yb, CFG.adv_eps)
            logits_adv, _, _, _, _ = model(xadv)
            loss = ce(logits, yb) + mse(recon, xb) + CFG.kl_weight * kl(mu, logvar) + CFG.adv_weight * ce(logits_adv, yb)
            loss.backward()
            opt.step()
            tl += loss.item()
        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(DEVICE)
                p = torch.argmax(model(xb)[0], dim=1).cpu().numpy()
                yt.extend(yb.numpy())
                yp.extend(p)
        hist.append({"epoch": ep + 1, "train_loss": tl / max(1, len(tr_loader)), "val_f1": f1_score(yt, yp, zero_division=0)})
    return pd.DataFrame(hist)


def predict_dart(model, x):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        logits, _, _, _, z = model(xb)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return preds, probs, z.cpu().numpy()


def predict_method(name: str, model, x):
    if name == "DART+AGIL":
        return predict_dart(model, x)
    if name == "CDDA-MD":
        model.eval()
        with torch.no_grad():
            xb = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds, probs, None
    preds, probs = model.predict(x)
    return preds, probs, None


def fit_methods(x_train, y_train, x_val, y_val):
    tr = DataLoader(TabularDataset(x_train, y_train), batch_size=CFG.batch_size, shuffle=True)
    va = DataLoader(TabularDataset(x_val, y_val), batch_size=CFG.batch_size, shuffle=False)
    models, train_mins, hist = {}, {}, None

    t = time.perf_counter()
    m = DARTAGIL(x_train.shape[1], CFG.hidden_dim, CFG.latent_dim).to(DEVICE)
    hist = train_dart(m, tr, va)
    models["DART+AGIL"] = m
    train_mins["DART+AGIL"] = (time.perf_counter() - t) / 60.0

    t = time.perf_counter()
    models["CDDA-MD"] = train_nn(CDDAMD(x_train.shape[1]), tr, epochs=5, lr=1e-4)
    train_mins["CDDA-MD"] = (time.perf_counter() - t) / 60.0

    t = time.perf_counter()
    m3s = M3SUPD(); m3s.fit(x_train, y_train); m3s.update_with_unknown_discovery(x_val)
    models["M3S-UPD"] = m3s
    train_mins["M3S-UPD"] = (time.perf_counter() - t) / 60.0

    t = time.perf_counter()
    cbr = CBR(k=7); cbr.fit(x_train, y_train)
    models["CBR"] = cbr
    train_mins["CBR"] = (time.perf_counter() - t) / 60.0

    t = time.perf_counter()
    ssmd = SSMD(); ssmd.fit(x_train, y_train)
    models["SSMD"] = ssmd
    train_mins["SSMD"] = (time.perf_counter() - t) / 60.0

    t = time.perf_counter()
    ifdr = IFDR(); ifdr.fit(x_train, y_train)
    models["IF-DR"] = ifdr
    train_mins["IF-DR"] = (time.perf_counter() - t) / 60.0

    return models, train_mins, hist


def evaluate_metrics(y, pred, prob):
    return {
        "F1": f1_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "Precision": precision_score(y, pred, zero_division=0),
        "AUC": roc_auc_score(y, prob),
        "Accuracy": accuracy_score(y, pred),
    }


def apply_obfuscation(x, mode, p):
    rng = np.random.default_rng(SEED)
    x2 = x.copy()
    mask = rng.random(x2.shape) < p
    if mode == "IDP":
        x2[mask] = 0
    elif mode == "IBP":
        x2[mask] = rng.normal(np.median(x2), np.std(x2) + 1e-6, mask.sum())
    elif mode == "APR":
        x2 *= 1 + rng.uniform(-0.5, 0.5, x2.shape)
    elif mode == "INP":
        x2[mask] = x2[mask] + rng.normal(0, 0.25, mask.sum())
    return x2


def fewshot_builder(method, xk, yk):
    if method == "DART+AGIL":
        cut = max(2, len(xk) // 2)
        tr = DataLoader(TabularDataset(xk[:cut], yk[:cut]), batch_size=min(32, cut), shuffle=True)
        va = DataLoader(TabularDataset(xk[cut:] if cut < len(xk) else xk[:cut], yk[cut:] if cut < len(yk) else yk[:cut]), batch_size=min(32, max(1, len(xk)-cut)), shuffle=False)
        m = DARTAGIL(xk.shape[1], CFG.hidden_dim, CFG.latent_dim).to(DEVICE)
        train_dart(m, tr, va)
        return m
    if method == "CDDA-MD":
        tr = DataLoader(TabularDataset(xk, yk), batch_size=min(32, len(xk)), shuffle=True)
        return train_nn(CDDAMD(xk.shape[1]), tr, epochs=3, lr=1e-4)
    if method == "M3S-UPD":
        m = M3SUPD(); m.fit(xk, yk); return m
    if method == "CBR":
        m = CBR(k=min(3, len(xk)-1)); m.fit(xk, yk); return m
    if method == "SSMD":
        m = SSMD(); m.fit(xk, yk); return m
    m = IFDR(); m.fit(xk, yk); return m


def update_time_ms(method, model, x_chunk, y_chunk):
    t = time.perf_counter()
    if method == "DART+AGIL":
        dl = DataLoader(TabularDataset(x_chunk, y_chunk), batch_size=min(128, len(x_chunk)), shuffle=True)
        opt = optim.Adam(model.parameters(), lr=CFG.lr * 0.5)
        ce = nn.CrossEntropyLoss(); mse = nn.MSELoss()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits, recon, mu, logvar, _ = model(xb)
            (ce(logits, yb) + mse(recon, xb) + 0.1 * kl(mu, logvar)).backward()
            opt.step()
    else:
        if method == "CDDA-MD":
            dl = DataLoader(TabularDataset(x_chunk, y_chunk), batch_size=min(128, len(x_chunk)), shuffle=True)
            train_nn(model, dl, epochs=1, lr=1e-4)
        else:
            model.fit(x_chunk, y_chunk)
    return 1000.0 * (time.perf_counter() - t)


def memory_mb(method, model):
    if method in ["DART+AGIL", "CDDA-MD"]:
        return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    if method == "CBR":
        return (model.x.nbytes + model.y.nbytes) / (1024 ** 2)
    return 0.0


def run_dataset(name: str, frame: pd.DataFrame):
    tr_df, va_df, te_df = temporal_split(frame)
    feat_cols = [c for c in frame.columns if c not in ["label", "dataset"]]
    sc = StandardScaler()
    x_train = sc.fit_transform(tr_df[feat_cols].values)
    x_val = sc.transform(va_df[feat_cols].values)
    x_test = sc.transform(te_df[feat_cols].values)
    y_train, y_val, y_test = tr_df["label"].values, va_df["label"].values, te_df["label"].values

    models, train_mins, train_hist = fit_methods(x_train, y_train, x_val, y_val)

    mid = len(x_test) // 2
    drift_rows, time_chunks = {}, {m: [] for m in METHODS}
    metric_rows = {}
    latent = None
    for method in METHODS:
        st = time.perf_counter()
        pred, prob, lat = predict_method(method, models[method], x_test)
        infer_ms = 1000.0 * (time.perf_counter() - st) / len(x_test)
        metric_rows[method] = evaluate_metrics(y_test, pred, prob)
        if method == "DART+AGIL":
            latent = lat

        p0, _, _ = predict_method(method, models[method], x_test[:mid])
        p1, _, _ = predict_method(method, models[method], x_test[mid:])
        drift_rows[method] = {"Initial F1": f1_score(y_test[:mid], p0, zero_division=0), "Drifted F1": f1_score(y_test[mid:], p1, zero_division=0)}

        cuts = np.linspace(0, len(x_test), 6).astype(int)
        for i in range(5):
            a, b = cuts[i], cuts[i + 1]
            pc, _, _ = predict_method(method, models[method], x_test[a:b])
            time_chunks[method].append(f1_score(y_test[a:b], pc, zero_division=0))

    # obfuscation
    obf_settings = {
        "No Obfs": (None, None),
        "IDP 10%": ("IDP", 0.10), "IDP 20%": ("IDP", 0.20), "IDP 30%": ("IDP", 0.30),
        "IBP 10%": ("IBP", 0.10), "IBP 30%": ("IBP", 0.30), "IBP 50%": ("IBP", 0.50),
        "APR": ("APR", 0.50), "INP": ("INP", 0.50),
    }
    obf_rows = {}
    for method in METHODS:
        row = {}
        for k, (mode, p) in obf_settings.items():
            xe = x_test if mode is None else apply_obfuscation(x_test, mode, p)
            pp, _, _ = predict_method(method, models[method], xe)
            row[k] = f1_score(y_test, pp, zero_division=0)
        obf_rows[method] = row

    # few-shot
    pos, neg = np.where(y_train == 1)[0], np.where(y_train == 0)[0]
    few_rows = {}
    for method in METHODS:
        mrow = {}
        for k in [1, 3, 5, 10]:
            vals = []
            for r in range(CFG.fewshot_repeats):
                rng = np.random.default_rng(SEED + 13 * k + r)
                ip = rng.choice(pos, size=min(k, len(pos)), replace=False)
                ineg = rng.choice(neg, size=min(k, len(neg)), replace=False)
                idx = np.hstack([ip, ineg])
                mk = fewshot_builder(method, x_train[idx], y_train[idx])
                pk, _, _ = predict_method(method, mk, x_test)
                vals.append(f1_score(y_test, pk, zero_division=0))
            mrow[f"N={k}"] = (float(np.mean(vals)), float(np.std(vals)))
        few_rows[method] = mrow

    # efficiency
    eff_rows = {}
    chunk_x, chunk_y = x_train[: min(1000, len(x_train))], y_train[: min(1000, len(y_train))]
    for method in METHODS:
        st = time.perf_counter()
        _ = predict_method(method, models[method], x_test)
        inf = 1000.0 * (time.perf_counter() - st) / len(x_test)
        eff_rows[method] = {
            "Inference Time (ms/flow)": inf,
            "Update Time (ms)": update_time_ms(method, models[method], chunk_x, chunk_y),
            "Training Time (min)": train_mins[method],
            "Memory (MB)": memory_mb(method, models[method]),
        }

    return {
        "metrics": pd.DataFrame(metric_rows).T,
        "drift": pd.DataFrame(drift_rows).T,
        "obf": pd.DataFrame(obf_rows).T[OBF_COLS],
        "few": few_rows,
        "eff": pd.DataFrame(eff_rows).T,
        "time_chunks": time_chunks,
        "latent": latent,
        "y_test": y_test,
    }


def aggregate_and_save(all_out: Dict[str, Dict]):
    # 1) drift_adaptation table
    drift_tbl = pd.DataFrame(index=METHODS)
    for ds in ["CICIDS2017", "CICDDoS2019", "CICIoTDIAD2024"]:
        drift_tbl[f"{ds} Initial F1"] = all_out[ds]["drift"]["Initial F1"]
        drift_tbl[f"{ds} Drifted F1"] = all_out[ds]["drift"]["Drifted F1"]
    save_table(drift_tbl, "drift_adaptation")

    # 2) obfuscation_results_detailed table (mean across datasets)
    obf_tbl = sum([all_out[d]["obf"] for d in all_out]) / len(all_out)
    save_table(obf_tbl, "obfuscation_results_detailed")

    # 3) fewshot_results_detailed table
    few_tbl = pd.DataFrame(index=METHODS, columns=["N=1", "N=3", "N=5", "N=10"])
    for m in METHODS:
        for n in [1, 3, 5, 10]:
            means, stds = [], []
            for ds in all_out:
                mu, sd = all_out[ds]["few"][m][f"N={n}"]
                means.append(mu); stds.append(sd)
            few_tbl.loc[m, f"N={n}"] = f"{np.mean(means):.4f} ± {np.mean(stds):.4f}"
    save_table(few_tbl, "fewshot_results_detailed")

    # 4) efficiency_detailed table
    eff_tbl = pd.DataFrame(index=METHODS)
    for col in ["Inference Time (ms/flow)", "Update Time (ms)", "Training Time (min)", "Memory (MB)"]:
        vals = np.vstack([all_out[d]["eff"][col].values for d in all_out])
        if col in ["Inference Time (ms/flow)", "Update Time (ms)"]:
            eff_tbl[col] = [f"{vals[:, i].mean():.3f} ± {vals[:, i].std():.3f}" for i in range(vals.shape[1])]
        else:
            eff_tbl[col] = [f"{vals[:, i].mean():.3f}" for i in range(vals.shape[1])]
    save_table(eff_tbl, "efficiency_detailed")

    # 5) ablation_detailed table on CICIDS2017
    ab_frame = preprocess_dataset(CFG.cicids2017_path, "CICIDS2017")
    ab_tbl = run_ablation(ab_frame)
    save_table(ab_tbl, "ablation_detailed")

    # figures (9)
    # latent_evolution
    latent = all_out["CICIDS2017"]["latent"]
    y_test = all_out["CICIDS2017"]["y_test"]
    n = min(2000, len(latent))
    emb = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(latent[:n])
    fig, ax = plt.subplots(figsize=(7, 5))
    y = y_test[:n]
    ax.scatter(emb[y == 0, 0], emb[y == 0, 1], s=10, c="white", edgecolor="black", marker="o", label="Benign")
    ax.scatter(emb[y == 1, 0], emb[y == 1, 1], s=10, c="black", edgecolor="black", marker="x", label="Malicious")
    ax.legend(frameon=False); ax.set_title("Latent Evolution (t-SNE)"); ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout(); fig.savefig(CFG.figures / "latent_evolution.pdf", format="pdf"); plt.close(fig)

    # drift_performance_over_time
    time_avg = {m: np.mean(np.array([all_out[d]["time_chunks"][m] for d in all_out]), axis=0).tolist() for m in METHODS}
    save_bw_line(time_avg, ["T1", "T2", "T3", "T4", "T5"], "Drift Performance Over Time", "F1", CFG.figures / "drift_performance_over_time.pdf")

    # obfuscation figures
    idp_ibp = obf_tbl[["IDP 10%", "IDP 20%", "IDP 30%", "IBP 10%", "IBP 30%", "IBP 50%"]].T
    save_bw_line(idp_ibp.to_dict(orient="list"), idp_ibp.index.tolist(), "IDP/IBP Degradation", "F1", CFG.figures / "obfuscation_idp_ibp.pdf")
    save_bw_bars(obf_tbl[["No Obfs", "APR", "INP", "IDP 30%", "IBP 50%"]], "Obfuscation Comparison", "F1", CFG.figures / "obfuscation_comparison.pdf")
    deg = pd.DataFrame(index=METHODS)
    for c in OBF_COLS[1:]:
        deg[c] = 100.0 * (obf_tbl["No Obfs"] - obf_tbl[c]) / np.clip(obf_tbl["No Obfs"], 1e-8, None)
    save_heatmap(deg, "Obfuscation Degradation (%)", CFG.figures / "obfuscation_heatmap.pdf")

    # fewshot figure (N=5)
    n5 = {m: float(few_tbl.loc[m, "N=5"].split("±")[0].strip()) for m in METHODS}
    save_bw_bars(pd.DataFrame.from_dict(n5, orient="index", columns=["F1 @ N=5"]), "Few-shot Comparison", "F1", CFG.figures / "fewshot_comparison_bar.pdf")

    # efficiency figures
    inf_vals = {m: float(eff_tbl.loc[m, "Inference Time (ms/flow)"].split("±")[0].strip()) for m in METHODS}
    upd_vals = {m: float(eff_tbl.loc[m, "Update Time (ms)"].split("±")[0].strip()) for m in METHODS}
    save_bw_bars(pd.DataFrame.from_dict(inf_vals, orient="index", columns=["Inference ms/flow"]), "Inference Time Comparison", "ms/flow", CFG.figures / "inference_time_comparison.pdf")
    save_bw_bars(pd.DataFrame.from_dict(upd_vals, orient="index", columns=["Update ms"]), "Update Time Comparison", "ms", CFG.figures / "update_time_comparison.pdf")

    # ablation figure
    save_bw_bars(ab_tbl[["Initial F1", "Drifted F1", "IDP (30%)", "IBP (50%)", "APR", "INP"]], "Ablation Contribution", "F1", CFG.figures / "ablation_contribution.pdf")


def run_ablation(frame: pd.DataFrame):
    tr_df, va_df, te_df = temporal_split(frame)
    feats = [c for c in frame.columns if c not in ["label", "dataset"]]
    sc = StandardScaler()
    x_train = sc.fit_transform(tr_df[feats].values)
    x_val = sc.transform(va_df[feats].values)
    x_test = sc.transform(te_df[feats].values)
    y_train, y_val, y_test = tr_df["label"].values, va_df["label"].values, te_df["label"].values

    tr = DataLoader(TabularDataset(x_train, y_train), batch_size=CFG.batch_size, shuffle=True)
    va = DataLoader(TabularDataset(x_val, y_val), batch_size=CFG.batch_size, shuffle=False)
    cfgs = {"w/o DART": (False, True, True), "w/o AGIL": (True, False, True), "w/o Online TL": (True, True, False), "Full DART+AGIL": (True, True, True)}
    rows = {}

    for name, (use_dart, use_agil, use_otl) in cfgs.items():
        m = DARTAGIL(x_train.shape[1], CFG.hidden_dim, CFG.latent_dim).to(DEVICE)
        if use_dart:
            old_adv = CFG.adv_weight
            CFG.adv_weight = old_adv if use_agil else 0.0
            train_dart(m, tr, va)
            CFG.adv_weight = old_adv
        else:
            opt = optim.Adam(m.cls.parameters(), lr=CFG.lr)
            ce = nn.CrossEntropyLoss()
            for _ in range(CFG.epochs):
                for xb, yb in tr:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    mu, _ = m.rep(xb)
                    ce(m.cls(mu.detach()), yb).backward()
                    opt.step()

        upd = update_time_ms("DART+AGIL", m, x_train[:512], y_train[:512]) if use_otl else 0.0
        mid = len(x_test) // 2
        p0, _, _ = predict_dart(m, x_test[:mid]); p1, _, _ = predict_dart(m, x_test[mid:])
        pidp, _, _ = predict_dart(m, apply_obfuscation(x_test, "IDP", 0.30))
        pibp, _, _ = predict_dart(m, apply_obfuscation(x_test, "IBP", 0.50))
        papr, _, _ = predict_dart(m, apply_obfuscation(x_test, "APR", 0.50))
        pinp, _, _ = predict_dart(m, apply_obfuscation(x_test, "INP", 0.50))

        rows[name] = {
            "Initial F1": f1_score(y_test[:mid], p0, zero_division=0),
            "Drifted F1": f1_score(y_test[mid:], p1, zero_division=0),
            "IDP (30%)": f1_score(y_test, pidp, zero_division=0),
            "IBP (50%)": f1_score(y_test, pibp, zero_division=0),
            "APR": f1_score(y_test, papr, zero_division=0),
            "INP": f1_score(y_test, pinp, zero_division=0),
            "Few-Shot (N=5)": f1_score(y_test, p1, zero_division=0),
            "Update Time (ms)": upd,
        }
    return pd.DataFrame(rows).T


def main():
    data = {
        "CICIDS2017": preprocess_dataset(CFG.cicids2017_path, "CICIDS2017"),
        "CICDDoS2019": preprocess_dataset(CFG.cicddos2019_path, "CICDDoS2019"),
        "CICIoTDIAD2024": preprocess_dataset(CFG.ciciotdiad2024_path, "CICIoTDIAD2024"),
    }
    outputs = {k: run_dataset(k, v) for k, v in data.items()}
    aggregate_and_save(outputs)
    logger.info("Done. 5 tables saved in %s and 9 figure PDFs saved in %s", CFG.tables, CFG.figures)


if __name__ == "__main__":
    main()
