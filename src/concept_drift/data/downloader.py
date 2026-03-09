from __future__ import annotations

import logging
from pathlib import Path
import requests

LOGGER = logging.getLogger(__name__)


def download_file(url: str, out_path: Path, timeout: int = 60) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        LOGGER.info("Downloaded %s -> %s", url, out_path)
        return True
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to download %s: %s", url, exc)
        return False


def download_dataset_sources(dataset_name: str, dataset_cfg: dict, root_dir: Path) -> list[Path]:
    ds_dir = root_dir / dataset_name / "raw"
    ds_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for source in dataset_cfg.get("sources", []):
        out_path = ds_dir / source["filename"]
        if out_path.exists() and out_path.stat().st_size > 0:
            downloaded.append(out_path)
            continue
        ok = download_file(source["url"], out_path)
        if ok:
            downloaded.append(out_path)
    return downloaded
