# Execution Instructions

## 1. Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. Run full pipeline
```bash
python scripts/run_pipeline.py --config configs/default.json
```

## 3. Smoke run (minimal)
```bash
python scripts/run_pipeline.py --config configs/smoke.json
```

## 4. Dataset placement notes
- NSL-KDD is auto-downloaded to `data/nsl_kdd/`.
- CICIDS2017 and CICDDoS2019 may require manual placement in `data/cicids2017/` and `data/cicddos2019/` if source access is gated.
- Each directory should contain one or more CSV files with a `Label` (or `label`) column.
