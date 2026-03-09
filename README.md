# Concept Drift Encrypted Traffic Research Pipeline

This repository provides an implementation-aligned pipeline for encrypted traffic classification under temporal drift, with:

- DART+AGIL PyTorch model
- baseline models (RF / MLP / LR)
- automated data download hooks for CICIDS2017, CICDDoS2019, NSL-KDD
- temporal split protocol (70/15/15)
- evaluation, CSV summaries, and pattern-only PDF bar plots
- ablation runner

## Important scientific constraints

- This code **does not fabricate** metrics.
- If a dataset cannot be downloaded from the configured source, that dataset is skipped and reported.
- CICDDoS2019 official site may require manual URL updates for direct CSV archives; update `src/concept_drift/configs/default.yaml` accordingly.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
```

## Run full pipeline

```bash
python scripts/run_pipeline.py --config src/concept_drift/configs/default.yaml
```

## Run ablations

```bash
python scripts/run_ablation.py --config src/concept_drift/configs/default.yaml
```

## Outputs

- `outputs/metrics_summary.csv`
- `outputs/metrics_summary.json`
- `outputs/*_comparison.pdf`
- `outputs/<dataset>/train_history.csv`
- `outputs/<dataset>/latent_test.csv`

All plots are generated as PDF with black/white pattern-filled bars.
