from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


HATCHES = ["/", "\\", "x", "-", "+", "o", ".", "*"]


def pattern_barplot(values: dict[str, float], title: str, ylabel: str, out_pdf: str | Path):
    labels = list(values.keys())
    vals = [values[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, vals, color="white", edgecolor="black", linewidth=1.2)
    for i, bar in enumerate(bars):
        bar.set_hatch(HATCHES[i % len(HATCHES)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
