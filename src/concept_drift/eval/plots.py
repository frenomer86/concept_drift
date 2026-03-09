from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


HATCHES = ["/", "\\", "x", "-", "+", "o", ".", "*"]


def _save(fig, out_pdf: str | Path):
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def pattern_barplot(values: dict[str, float], title: str, ylabel: str, out_pdf: str | Path):
    labels = list(values.keys())
    vals = [values[k] for k in labels]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, vals, color="white", edgecolor="black", linewidth=1.2)
    for i, bar in enumerate(bars):
        bar.set_hatch(HATCHES[i % len(HATCHES)])

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    _save(fig, out_pdf)


def grouped_pattern_barplot(
    series: dict[str, dict[str, float]],
    title: str,
    ylabel: str,
    out_pdf: str | Path,
):
    groups = list(series.keys())
    if not groups:
        return
    cats = list(next(iter(series.values())).keys())

    x = np.arange(len(groups))
    width = 0.8 / max(1, len(cats))

    fig, ax = plt.subplots(figsize=(12, 5))
    for j, cat in enumerate(cats):
        vals = [series[g].get(cat, np.nan) for g in groups]
        offset = (j - (len(cats) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            label=cat,
            color="white",
            edgecolor="black",
            linewidth=1.0,
        )
        for b in bars:
            b.set_hatch(HATCHES[j % len(HATCHES)])

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.legend(frameon=False, ncol=2)
    _save(fig, out_pdf)
