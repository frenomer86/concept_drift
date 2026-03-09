from __future__ import annotations

from pathlib import Path
import json


def load_config(path: str | Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        return json.loads(text)

    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except Exception as e:
        raise RuntimeError(
            "YAML parsing requires PyYAML. Provide a .json config or install pyyaml."
        ) from e
