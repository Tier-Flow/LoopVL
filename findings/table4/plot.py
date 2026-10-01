"""Render the manuscript's Table 4 from versioned score data (no model required)."""
from pathlib import Path
import sys
_FINDINGS_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "paths.py").is_file())
sys.path.insert(0, str(_FINDINGS_ROOT))
from common.paths import MODEL_ROOT, DATA_ROOT, OUTPUT_ROOT, repo_path, tsv_directory
import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=repo_path, default=HERE / "data/paper_scores.json")
    parser.add_argument("--output", type=repo_path, default=OUTPUT_ROOT / 'table4')
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    matrix = np.asarray([row["scores"] for row in data["rows"]])
    assert matrix.shape == (12, 5) and np.isfinite(matrix).all()
    assert ((matrix >= 0) & (matrix <= 100)).all()
    best = matrix.max(axis=0)
    second = np.array([sorted(set(matrix[:, i]))[-2] for i in range(5)])
    cells = []
    for row in data["rows"]:
        h, l = map(int, re.fullmatch(r"H(\d+)L(\d+)", row["configuration"]).groups())
        assert row["unrolled_layers"] == 16 * h * (l + 1)
        cells.append([row["configuration"], str(row["unrolled_layers"])] + [f"{v:.2f}" for v in row["scores"]])
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(10, 4.35))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=["Configuration", "Unrolled layers"] + data["benchmarks"],
                     cellLoc="center", bbox=[0, 0, 1, 1], colWidths=[.15, .15, .13, .15, .14, .13, .15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CED7DF")
        cell.set_linewidth(.35)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#F1F4F6")
        elif col >= 2:
            value = matrix[row - 1, col - 2]
            if value == best[col - 2]:
                cell.set_facecolor("#D2E6D9")
                cell.set_text_props(weight="bold")
            elif value == second[col - 2]:
                cell.set_facecolor("#EAF3ED")
    fig.subplots_adjust(left=.01, right=.99, top=.98, bottom=.02)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.output / f"table.{suffix}", dpi=240, facecolor="white")
    markdown = ["| " + " | ".join(["Configuration", "Unrolled layers"] + data["benchmarks"]) + " |",
                "| " + " | ".join(["---"] * 7) + " |"]
    for i, row in enumerate(cells):
        values = row[:2] + [f"**{value}**" if matrix[i, j] == best[j] else value for j, value in enumerate(row[2:])]
        markdown.append("| " + " | ".join(values) + " |")
    (args.output / "table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
