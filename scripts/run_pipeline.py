#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running the script directly without installing the package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import torch

from concept_drift.utils.config import load_config
from concept_drift.utils.reproducibility import set_seed
from concept_drift.utils.io import ensure_dir, write_json
from concept_drift.data.download import ensure_dataset
from concept_drift.data.preprocess import load_nsl_kdd, load_generic_csv_dir, preprocess_dataframe
from concept_drift.data.splits import temporal_split_indices
from concept_drift.data.dataloaders import make_loader
from concept_drift.models.dart_agil import DARTAGIL
from concept_drift.models.baselines import build_baselines, IFDriftBaseline
from concept_drift.training.trainer import fit
from concept_drift.eval.metrics import classification_metrics
from concept_drift.eval.fewshot import fewshot_scores
from concept_drift.eval.obfuscation import apply_idp, apply_ibp, apply_apr, apply_inp
from concept_drift.eval.reporting import export_all_reports


def infer_probs_torch(model, X, device):
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X).to(device)
        logits, _, _, _ = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def evaluate_obfuscations(y_true, model_name, predict_fn, X, cfg):
    rows = []
    for p in cfg["evaluation"]["obfuscation"]["idp"]:
        rows.append({"model": model_name, "scenario": f"idp_{p}", **classification_metrics(y_true, predict_fn(apply_idp(X, p)))})
    for p in cfg["evaluation"]["obfuscation"]["ibp"]:
        rows.append({"model": model_name, "scenario": f"ibp_{p}", **classification_metrics(y_true, predict_fn(apply_ibp(X, p)))})
    for p in cfg["evaluation"]["obfuscation"]["apr"]:
        rows.append({"model": model_name, "scenario": f"apr_{p}", **classification_metrics(y_true, predict_fn(apply_apr(X, p)))})
    for p in cfg["evaluation"]["obfuscation"]["inp"]:
        rows.append({"model": model_name, "scenario": f"inp_{p}", **classification_metrics(y_true, predict_fn(apply_inp(X, p)))})
    return rows


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    out = Path(cfg["output_dir"])
    for d in ["logs", "figures", "tables", "checkpoints", "reports"]:
        ensure_dir(out / d)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_rows: list[dict] = []
    fewshot_rows: list[dict] = []
    drift_rows: list[dict] = []

    for ds_name in cfg["data"]["datasets"]:
        ds_path = ensure_dataset(ds_name, cfg["data"]["root"])
        if ds_name == "nsl_kdd":
            df = load_nsl_kdd(ds_path)
        else:
            try:
                df = load_generic_csv_dir(ds_path)
            except Exception as exc:
                print(f"[WARN] Skipping {ds_name}: {exc}")
                continue

        X, y, _t, _features, _imputer, _scaler = preprocess_dataframe(df, cfg["data"]["max_rows_per_dataset"])
        tr, va, te = temporal_split_indices(len(y), **cfg["data"]["temporal_split"])

        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]
        X_te, y_te = X[te], y[te]

        # Baselines
        for baseline in build_baselines(seed=cfg["seed"]):
            t0 = time.perf_counter()
            baseline.fit(X_tr, y_tr)
            train_sec = time.perf_counter() - t0
            y_prob = baseline.predict_proba(X_te)
            base_metrics = classification_metrics(y_te, y_prob)
            metrics_rows.append({"dataset": ds_name, "model": baseline.name, "scenario": "clean", "train_sec": train_sec, **base_metrics})

            if cfg["evaluation"]["run_obfuscation"]:
                metrics_rows.extend(
                    [{"dataset": ds_name, **r} for r in evaluate_obfuscations(y_te, baseline.name, baseline.predict_proba, X_te, cfg)]
                )

            fs = fewshot_scores(baseline.model, X_tr, y_tr, X_te, y_te, cfg["evaluation"]["few_shot_shots"], repeats=20)
            for item in fs:
                fewshot_rows.append({"dataset": ds_name, "model": baseline.name, **item})

        # IF-DR baseline
        ifdr = IFDriftBaseline()
        ifdr.fit(X_tr, y_tr)
        y_prob = ifdr.predict_proba(X_te)
        m = classification_metrics(y_te, y_prob)
        drift_rows.append({"dataset": ds_name, "model": "ifdr", "drift_rate": ifdr.drift_rate(X_te), **m})

        # DART+AGIL and ablations
        train_loader = make_loader(X_tr, y_tr, cfg["training"]["batch_size"], shuffle=True)
        val_loader = make_loader(X_va, y_va, cfg["training"]["batch_size"], shuffle=False)

        variants = cfg["ablation"]["variants"] if cfg["ablation"]["run"] else ["full"]
        for variant in variants:
            run_cfg = json.loads(json.dumps(cfg))
            run_cfg["variant"] = variant
            if variant == "no_dart":
                run_cfg["training"]["kl_weight"] = 0.0
            if variant == "no_agil":
                run_cfg["training"]["agil_weight"] = 0.0

            model = DARTAGIL(
                input_dim=X.shape[1],
                hidden_dim=cfg["model"]["hidden_dim"],
                latent_dim=cfg["model"]["latent_dim"],
                dropout=cfg["model"]["dropout"],
            ).to(device)

            opt = torch.optim.Adam(model.parameters(), lr=run_cfg["training"]["lr"], weight_decay=run_cfg["training"]["weight_decay"])
            ckpt = out / "checkpoints" / f"{ds_name}_{variant}.pt"

            t0 = time.perf_counter()
            fit(model, train_loader, val_loader, opt, device, run_cfg, ckpt)
            train_sec = time.perf_counter() - t0

            y_prob = infer_probs_torch(model, X_te, device)
            base_metrics = classification_metrics(y_te, y_prob)
            metrics_rows.append({"dataset": ds_name, "model": f"dart_agil_{variant}", "scenario": "clean", "train_sec": train_sec, **base_metrics})

            if cfg["evaluation"]["run_obfuscation"]:
                predict_fn = lambda Xin: infer_probs_torch(model, Xin, device)
                metrics_rows.extend(
                    [{"dataset": ds_name, **r} for r in evaluate_obfuscations(y_te, f"dart_agil_{variant}", predict_fn, X_te, cfg)]
                )

    if not metrics_rows:
        raise RuntimeError("No dataset could be loaded. Provide dataset files in data/ directories.")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out / "tables" / "metrics_all.csv", index=False)
    if fewshot_rows:
        pd.DataFrame(fewshot_rows).to_csv(out / "tables" / "fewshot.csv", index=False)
    if drift_rows:
        pd.DataFrame(drift_rows).to_csv(out / "tables" / "drift.csv", index=False)

    export_all_reports(
        metrics_df=metrics_df,
        fewshot_df=(pd.DataFrame(fewshot_rows) if fewshot_rows else None),
        drift_df=(pd.DataFrame(drift_rows) if drift_rows else None),
        out_dir=out,
    )

    write_json(out / "logs" / "run_summary.json", {
        "datasets_processed": sorted(metrics_df["dataset"].unique().tolist()),
        "rows_metrics": int(len(metrics_df)),
        "rows_fewshot": int(len(fewshot_rows)),
        "rows_drift": int(len(drift_rows)),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    main(args.config)
