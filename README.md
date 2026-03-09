# Concept Drift in Encrypted Traffic: DART+AGIL (Complete Experiment Pipeline)

This repository provides a reproducible implementation for encrypted-traffic malicious flow detection under temporal drift.

## Implemented scope
- Proposed method: DART+AGIL (`full`) plus ablations (`no_dart`, `no_agil`, `no_online_tl` config variant)
- Baselines: Logistic Regression, Random Forest, MLP, IF-DR
- Evaluations: clean performance, obfuscation robustness (IDP/IBP/APR/INP), few-shot, drift-rate summary
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, ECE
- Outputs: automatic tables and PDF figures (pattern-filled, no color)

## Dataset handling
- NSL-KDD auto-download is implemented.
- CICIDS2017/CICDDoS2019 may require manual placement due source access restrictions.

## Binary-safe policy
- `.gitignore` excludes `__pycache__`, `*.pyc`, outputs, and data.
- No binary artifacts should be committed.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_pipeline.py --config configs/default.json
```

## Generated artifacts
- `outputs/tables/metrics_all.csv`
- `outputs/tables/clean_metrics.csv`
- `outputs/tables/table_clean_summary.csv`
- `outputs/tables/table_obfuscation_degradation.csv`
- `outputs/tables/table_ablation.csv`
- `outputs/tables/table_fewshot.csv` (if few-shot rows exist)
- `outputs/tables/table_drift.csv` (if drift rows exist)
- `outputs/figures/clean_*.pdf`
- `outputs/figures/fewshot_*.pdf`
- `outputs/figures/drift_rate_*.pdf`
- `outputs/logs/run_summary.json`


## Jupyter notebook (single complete file)
- `notebooks/complete_dart_agil_pipeline.ipynb` contains full end-to-end code in one notebook: downloads, preprocessing, proposed model, baselines, metrics, ablations, tables, and PDF plots.
