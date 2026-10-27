"""Generate the LMEIC-full role-wise error profile figure.

Version: v1.0, 2026-06-04.

This script reads saved LMEIC-full test predictions and creates a publication
figure with per-role confusion matrices using the approved LMEIC colour scheme.
All input and output paths are provided through command-line arguments.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "white": "#FFFFFF",
    "panel_bg": "#F6F9F4",
    "llm_full": "#4A4F7E",
    "d3": "#D19246",
    "co_reader": "#5B8C8A",
    "text": "#2D3436",
    "muted": "#636E72",
    "border": "#B5AF8B",
    "light_gray": "#C8CDD0",
}

ROLE_COLOURS = {
    "Mgr": PALETTE["llm_full"],
    "CS": PALETTE["d3"],
    "Sch": PALETTE["co_reader"],
}

LABELS = ["High", "Medium", "Low"]
ROLES = ["Mgr", "CS", "Sch"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LMEIC-full role-wise error profile.")
    parser.add_argument("--predictions-jsonl", required=True, type=Path)
    parser.add_argument("--output-pdf", required=True, type=Path)
    parser.add_argument("--output-png", required=True, type=Path)
    parser.add_argument("--output-svg", type=Path)
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "axes.edgecolor": PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_predictions(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def blend_hex(foreground: str, background: str = "#FFFFFF", alpha: float = 0.5) -> str:
    fg = np.array(mcolors.to_rgb(foreground))
    bg = np.array(mcolors.to_rgb(background))
    return mcolors.to_hex(alpha * fg + (1.0 - alpha) * bg)


def make_role_cmap(role: str) -> mcolors.LinearSegmentedColormap:
    role_colour = ROLE_COLOURS[role]
    return mcolors.LinearSegmentedColormap.from_list(
        f"{role}_cmap",
        [
            blend_hex(role_colour, PALETTE["panel_bg"], 0.06),
            blend_hex(role_colour, PALETTE["white"], 0.20),
            blend_hex(role_colour, PALETTE["white"], 0.55),
        ],
    )


def build_matrices(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    matrices = {}
    for role in ROLES:
        counter = Counter((row["gold"], row["pred"]) for row in rows if row["role"] == role)
        matrices[role] = np.array(
            [[counter[(gold, pred)] for pred in LABELS] for gold in LABELS],
            dtype=int,
        )
    return matrices


def annotate_matrix(ax: plt.Axes, matrix: np.ndarray, max_count: int) -> None:
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_pct = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)
    for row_idx in range(len(LABELS)):
        for col_idx in range(len(LABELS)):
            count = matrix[row_idx, col_idx]
            pct = row_pct[row_idx, col_idx] * 100
            ax.text(
                col_idx,
                row_idx,
                f"{count}\n{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=8.3,
                color=PALETTE["text"],
            )


def plot_error_profile(rows: list[dict[str, str]]) -> plt.Figure:
    matrices = build_matrices(rows)
    max_count = max(int(matrix.max()) for matrix in matrices.values())

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.65), constrained_layout=True)
    fig.patch.set_facecolor(PALETTE["white"])

    for ax, role in zip(axes, ROLES):
        matrix = matrices[role]
        ax.imshow(matrix, cmap=make_role_cmap(role), vmin=0, vmax=max_count)
        ax.set_title(role, color=ROLE_COLOURS[role], fontweight="bold", pad=8)
        ax.set_xticks(range(len(LABELS)), labels=LABELS, rotation=30, ha="right")
        ax.set_yticks(range(len(LABELS)), labels=LABELS)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Ground-truth label")
        annotate_matrix(ax, matrix, max_count)

        for boundary in np.arange(-0.5, len(LABELS), 1):
            ax.axhline(boundary, color=PALETTE["white"], linewidth=1.0)
            ax.axvline(boundary, color=PALETTE["white"], linewidth=1.0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor(PALETTE["border"])
        ax.tick_params(axis="both", length=0)

    fig.suptitle("LMEIC-full error profile by recipient role", fontsize=12.5, fontweight="bold", color=PALETTE["text"])
    return fig


def save_figure(fig: plt.Figure, output_pdf: Path, output_png: Path, output_svg: Path | None) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", facecolor=PALETTE["white"])
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=PALETTE["white"])
    if output_svg is not None:
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_svg, bbox_inches="tight", facecolor=PALETTE["white"])
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_style()
    rows = read_predictions(args.predictions_jsonl)
    fig = plot_error_profile(rows)
    save_figure(fig, args.output_pdf, args.output_png, args.output_svg)
    print(f"rows={len(rows)}")
    print(f"output_pdf={args.output_pdf}")
    print(f"output_png={args.output_png}")
    if args.output_svg is not None:
        print(f"output_svg={args.output_svg}")


if __name__ == "__main__":
    main()
