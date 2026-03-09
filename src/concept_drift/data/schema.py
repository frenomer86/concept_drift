from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetFrame:
    name: str
    feature_columns: list[str]
    label_column: str
    time_column: str
