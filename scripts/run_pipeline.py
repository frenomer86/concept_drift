from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import joblib
import pandas as pd
import torch

from concept_drift.utils.io import load_yaml, ensure_dir
from concept_drift.utils.logging_utils import setup_logging
from concept_drift.utils.seed import set_global_seed
from concept_drift.data.downloader import download_dataset_sources
from concept_drift.data.loaders import load_dataset
from concept_drift.data.preprocess import build_features, temporal_split
from concept_drift.models.baselines import build_baselines
from concept_drift.training.trainer import DARTAGILTrainer
from concept_drift.utils.metrics import compute_classification_metrics
from concept_drift.utils.plotting import save_pattern_barplot


def main(config_path: str):
    cfg = load_yaml(config_path)
    setup_logging()
    logger = logging.getLogger("run_pipeline")
    set_global_seed(cfg["seed"])

    out_root = ensure_dir(cfg["output_dir"])
    datasets_root = ensure_dir(cfg["datasets"]["root_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_metrics = []

    for dataset_name, ds_cfg in cfg["datasets"].items():
        if dataset_name == "root_dir" or not isinstance(ds_cfg, dict) or not ds_cfg.get("enabled", False):
            continue

        logger.info("Processing dataset: %s", dataset_name)
        raw_files = download_dataset_sources(dataset_name, ds_cfg, datasets_root)
        if not raw_files:
            logger.warning("No files downloaded for %s; skipping.", dataset_name)
            continue

        try:
            df = load_dataset(dataset_name, raw_files)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dataset load failed for %s: %s", dataset_name, exc)
            continue

        if cfg["preprocessing"]["max_rows_per_dataset"]:
            df = df.head(int(cfg["preprocessing"]["max_rows_per_dataset"]))

        X, y, times, scaler, label_encoder, feature_names = build_features(
            df=df,
            target_col=ds_cfg["target_col"],
            timestamp_col=ds_cfg["timestamp_col"],
            selected_features=cfg["features"]["selected"],
            scale=cfg["preprocessing"]["scale"],
            dropna=cfg["preprocessing"]["dropna"],
        )
        split = temporal_split(
            X, y, times,
            cfg["preprocessing"]["temporal_split"]["train"],
            cfg["preprocessing"]["temporal_split"]["val"],
            cfg["preprocessing"]["temporal_split"]["test"],
        )

        X_train, y_train = split["train"]
        X_val, y_val = split["val"]
        X_test, y_test = split["test"]

        ds_out = ensure_dir(out_root / dataset_name)
        joblib.dump({"scaler": scaler, "label_encoder": label_encoder, "feature_names": feature_names}, ds_out / "preprocess.joblib")

        # Baselines
        baselines = build_baselines(cfg["baselines"])
        for name, model in baselines.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_test)
            else:
                prob = pd.get_dummies(pd.Series(pred)).reindex(columns=range(len(set(y))), fill_value=0).values
            m = compute_classification_metrics(y_test, prob, pred)
            m.update({"dataset": dataset_name, "model": name})
            all_metrics.append(m)

        # Proposed
        trainer = DARTAGILTrainer(input_dim=X_train.shape[1], n_classes=len(set(y)), cfg=cfg["training"], device=device)
        history = trainer.fit(X_train, y_train, X_val, y_val)
        pred, prob, latent = trainer.predict(X_test)
        m = compute_classification_metrics(y_test, prob, pred)
        m.update({"dataset": dataset_name, "model": "dart_agil"})
        all_metrics.append(m)

        pd.DataFrame(history).to_csv(ds_out / "train_history.csv", index=False)
        pd.DataFrame(latent).to_csv(ds_out / "latent_test.csv", index=False)

    if not all_metrics:
        raise RuntimeError("No datasets successfully processed. Check dataset sources and schema in config.")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_root / "metrics_summary.csv", index=False)

    for metric_col in ["f1_macro", "recall_macro", "accuracy"]:
        fig_df = metrics_df[["dataset", "model", metric_col]].copy()
        save_pattern_barplot(
            fig_df,
            x_col="dataset",
            y_col=metric_col,
            hue_col="model",
            out_path=out_root / f"{metric_col}_comparison.pdf",
            hatches=cfg["plots"]["hatch_patterns"],
        )

    with open(out_root / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/concept_drift/configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
