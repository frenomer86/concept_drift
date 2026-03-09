from __future__ import annotations

import argparse
import copy
import subprocess
import pandas as pd

from concept_drift.utils.io import load_yaml


def write_cfg(cfg, out_path):
    import yaml
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main(config_path: str):
    base = load_yaml(config_path)
    variants = {
        "full": {},
        "w_o_agil": {"training.lambda_agil": 0.0},
        "w_o_adv": {"training.adv_steps": 0},
        "low_latent": {"training.latent_dim": 32},
    }

    for name, overrides in variants.items():
        cfg = copy.deepcopy(base)
        for k, v in overrides.items():
            keys = k.split(".")
            p = cfg
            for kk in keys[:-1]:
                p = p[kk]
            p[keys[-1]] = v
        cfg["output_dir"] = f"outputs/ablation_{name}"
        tmp_cfg = f"outputs/tmp_{name}.yaml"
        write_cfg(cfg, tmp_cfg)
        subprocess.run(["python", "scripts/run_pipeline.py", "--config", tmp_cfg], check=True)

    rows = []
    for name in variants:
        fp = f"outputs/ablation_{name}/metrics_summary.csv"
        df = pd.read_csv(fp)
        df["variant"] = name
        rows.append(df)
    pd.concat(rows, ignore_index=True).to_csv("outputs/ablation_summary.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="src/concept_drift/configs/default.yaml")
    args = p.parse_args()
    main(args.config)
