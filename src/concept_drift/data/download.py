from __future__ import annotations

from pathlib import Path
import zipfile
import io
import requests


DATASET_URLS = {
    "nsl_kdd": [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
    ],
    # Primary public links; may change. Errors are surfaced honestly.
    "cicids2017": [
        "https://www.unb.ca/cic/datasets/ids-2017.html",
    ],
    "cicddos2019": [
        "https://www.unb.ca/cic/datasets/ddos-2019.html",
    ],
}


def _download_file(url: str, out_path: Path) -> None:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    out_path.write_bytes(r.content)


def download_nsl_kdd(root: Path) -> Path:
    target = root / "nsl_kdd"
    target.mkdir(parents=True, exist_ok=True)
    for url in DATASET_URLS["nsl_kdd"]:
        fname = url.split("/")[-1]
        _download_file(url, target / fname)
    return target


def ensure_dataset(name: str, root: str | Path) -> Path:
    root = Path(root)
    if name == "nsl_kdd":
        return download_nsl_kdd(root)

    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "MANUAL_DOWNLOAD_REQUIRED.txt"
    if not marker.exists():
        marker.write_text(
            "This dataset may require manual download or provider acceptance. "
            f"Attempted source: {DATASET_URLS.get(name, [])}\n"
            "Place extracted CSV files into this directory and rerun.\n",
            encoding="utf-8",
        )
    return target
