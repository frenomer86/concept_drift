from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_pattern_barplot(df, x_col: str, y_col: str, hue_col: str, out_path: Path, hatches: list[str]) -> None:
    labels = list(df[x_col].unique())
    groups = list(df[hue_col].unique())
    x = np.arange(len(labels))
    width = 0.8 / max(len(groups), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, g in enumerate(groups):
        sub = df[df[hue_col] == g]
        ys = [float(sub[sub[x_col] == lab][y_col].iloc[0]) if (sub[x_col] == lab).any() else 0 for lab in labels]
        bars = ax.bar(
            x + i * width - ((len(groups) - 1) * width / 2),
            ys,
            width=width,
            edgecolor="black",
            color="white",
            linewidth=1.0,
            label=str(g),
        )
        hatch = hatches[i % len(hatches)]
        for b in bars:
            b.set_hatch(hatch)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(y_col)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
