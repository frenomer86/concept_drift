from __future__ import annotations

from pathlib import Path
import pandas as pd

from concept_drift.eval.plots import pattern_barplot, grouped_pattern_barplot


METRICS = ["accuracy", "precision", "recall", "f1", "auc_roc", "ece"]


def export_all_reports(metrics_df: pd.DataFrame, fewshot_df: pd.DataFrame | None, drift_df: pd.DataFrame | None, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    tables = out_dir / "tables"
    figs = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    clean = metrics_df[metrics_df["scenario"] == "clean"].copy()
    clean.to_csv(tables / "clean_metrics.csv", index=False)

    # Table: dataset x model summary with all metrics.
    clean_summary = clean.groupby(["dataset", "model"], as_index=False)[METRICS].mean()
    clean_summary.to_csv(tables / "table_clean_summary.csv", index=False)

    # Table: obfuscation degradation.
    obf = metrics_df[metrics_df["scenario"] != "clean"].copy()
    if not obf.empty:
        merged = obf.merge(
            clean[["dataset", "model", "f1"]].rename(columns={"f1": "clean_f1"}),
            on=["dataset", "model"],
            how="left",
        )
        merged["f1_drop"] = merged["clean_f1"] - merged["f1"]
        merged.to_csv(tables / "table_obfuscation_degradation.csv", index=False)

    # Table: ablations only.
    ablation = clean[clean["model"].str.startswith("dart_agil_")].copy()
    if not ablation.empty:
        ablation.to_csv(tables / "table_ablation.csv", index=False)

    # Plots for each dataset and metric on clean setup.
    for ds in sorted(clean["dataset"].unique()):
        ds_df = clean[clean["dataset"] == ds]
        for metric in METRICS:
            vals = {r.model: float(getattr(r, metric)) for r in ds_df.itertuples()}
            pattern_barplot(vals, f"{metric.upper()} on {ds}", metric.upper(), figs / f"clean_{ds}_{metric}.pdf")

    # Grouped plot: model x metrics averaged over datasets.
    avg = clean.groupby("model", as_index=False)[METRICS].mean()
    series = {r.model: {m: float(getattr(r, m)) for m in METRICS} for r in avg.itertuples()}
    grouped_pattern_barplot(series, "Average clean metrics by model", "Score", figs / "clean_all_models_all_metrics.pdf")

    if fewshot_df is not None and not fewshot_df.empty:
        fewshot_df.to_csv(tables / "table_fewshot.csv", index=False)
        for ds in sorted(fewshot_df["dataset"].unique()):
            ds_df = fewshot_df[fewshot_df["dataset"] == ds]
            model_to_shot = {}
            for model in sorted(ds_df["model"].unique()):
                mdf = ds_df[ds_df["model"] == model]
                model_to_shot[model] = {f"shot_{int(r.shots)}": float(r.f1_mean) for r in mdf.itertuples()}
            grouped_pattern_barplot(
                model_to_shot,
                f"Few-shot F1 means on {ds}",
                "F1",
                figs / f"fewshot_{ds}.pdf",
            )

    if drift_df is not None and not drift_df.empty:
        drift_df.to_csv(tables / "table_drift.csv", index=False)
        for ds in sorted(drift_df["dataset"].unique()):
            ds_df = drift_df[drift_df["dataset"] == ds]
            vals = {r.model: float(r.drift_rate) for r in ds_df.itertuples()}
            pattern_barplot(vals, f"Estimated drift rate on {ds}", "Drift Rate", figs / f"drift_rate_{ds}.pdf")
