# Concept Drift in Encrypted Traffic: DART+AGIL (Regenerated, Binary-Safe)

This repository provides a reproducible implementation for encrypted-traffic malicious flow detection under temporal drift.

## What is implemented
- Real dataset handling (NSL-KDD auto-download, CIC-style CSV ingestion)
- Temporal train/val/test splits
- DART+AGIL model and ablations (`full`, `no_dart`, `no_agil`, `no_online_tl` config hook)
- Baseline models (LogReg, RandomForest, MLP, IF-DR)
- Obfuscation evaluation (IDP/IBP/APR/INP)
- Few-shot evaluation
- Metric export tables and PDF pattern-filled bar plots (no color)

## Binary-safe policy
- `.gitignore` excludes `__pycache__`, `*.pyc`, outputs, and data.
- The repository intentionally avoids tracked binary artifacts.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_pipeline.py --config configs/default.json
```

## Outputs
- `outputs/tables/metrics_all.csv`
- `outputs/tables/fewshot.csv` (if generated)
- `outputs/tables/drift.csv` (if generated)
- `outputs/figures/*.pdf`
- `outputs/logs/run_summary.json`
