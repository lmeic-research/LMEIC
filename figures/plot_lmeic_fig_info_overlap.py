"""Generate the upgraded LMEIC information-diagnostics figure.

Version: v1.1, 2026-10-28 (v1.1: panel (c) labels say increment/overlap instead of unique/shared,
because the values are classifier-based estimates, not a partial-information decomposition).

This script renders a three-panel figure for the LMEIC ESWA manuscript:
(a) predictive information (PI) bars with bootstrap intervals,
(b) interaction information (II) bars with bootstrap intervals,
(c) a quantitative information-overlap diagram that decomposes the joint
predictive information of D1 and D2+D3 into unique and shared components
on the label-entropy scale, using exactly the values reported in the
information-diagnostics table. All input and output paths are provided
through command-line arguments.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


PALETTE = {
    "white": "#FFFFFF",
    "d1": "#71A682",
    "d2": "#7EA4B6",
    "d3": "#D19246",
    "llm_full": "#4A4F7E",
    "co_reader": "#5B8C8A",
    "neutral": "#81989B",
    "text": "#2D3436",
    "muted": "#636E72",
    "light_gray": "#C8CDD0",
    "overlap": "#9C8AA5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LMEIC information overlap figure.")
    parser.add_argument("--info-diagnostics-csv", required=True, type=Path)
    parser.add_argument("--label-entropy", type=float, default=1.3239)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stem", default="fig_information_overlap_v1")
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
            "legend.fontsize": 8,
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, kwargs in [("pdf", {}), ("svg", {}), ("png", {"dpi": 300})]:
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor=PALETTE["white"], **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_pi_panel(ax: plt.Axes, quantities: dict[str, dict[str, str]]) -> None:
    pi_order = ["PI(D1)", "PI(D2)", "PI(D3)", "PI(D2+D3)", "PI(D1+D2+D3)"]
    values = [float(quantities[item]["point_bits"]) for item in pi_order]
    low = [float(quantities[item]["ci95_lo"]) for item in pi_order]
    high = [float(quantities[item]["ci95_hi"]) for item in pi_order]
    errors = np.array(
        [
            [value - lo for value, lo in zip(values, low)],
            [hi - value for value, hi in zip(values, high)],
        ]
    )
    colours = [PALETTE["d1"], PALETTE["d2"], PALETTE["d3"], PALETTE["co_reader"], PALETTE["llm_full"]]

    x = np.arange(len(pi_order))
    ax.bar(x, values, color=colours, edgecolor=PALETTE["text"], linewidth=0.5)
    ax.errorbar(x, values, yerr=errors, fmt="none", ecolor=PALETTE["text"], elinewidth=1.0, capsize=2.5)
    for idx, value in enumerate(values):
        ax.text(idx, high[idx] + 0.018, f"{value:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(pi_order, rotation=35, ha="right")
    ax.set_ylabel("Predictive information (bits)")
    ax.set_ylim(0, max(high) + 0.13)
    ax.set_title("(a) Predictive information")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_ii_panel(ax: plt.Axes, quantities: dict[str, dict[str, str]]) -> None:
    ii_order = ["II(D1;D2)", "II(D1;D3)", "II(D2;D3)", "II(D1;D2+D3)"]
    values = [float(quantities[item]["point_bits"]) for item in ii_order]
    low = [float(quantities[item]["ci95_lo"]) for item in ii_order]
    high = [float(quantities[item]["ci95_hi"]) for item in ii_order]
    errors = np.array(
        [
            [value - lo for value, lo in zip(values, low)],
            [hi - value for value, hi in zip(values, high)],
        ]
    )

    x = np.arange(len(ii_order))
    ax.bar(x, values, color=PALETTE["muted"], edgecolor=PALETTE["text"], linewidth=0.5)
    ax.errorbar(x, values, yerr=errors, fmt="none", ecolor=PALETTE["text"], elinewidth=1.0, capsize=2.5)
    ax.axhline(0, color=PALETTE["text"], linewidth=0.8)
    for idx, value in enumerate(values):
        ax.text(idx, low[idx] - 0.018, f"{value:.3f}", ha="center", va="top", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(ii_order, rotation=35, ha="right")
    ax.set_ylabel("Interaction information (bits)")
    ax.set_ylim(min(low) - 0.12, 0.05)
    ax.set_title("(b) Interaction information")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_overlap_panel(ax: plt.Axes, quantities: dict[str, dict[str, str]], h_y: float) -> dict[str, float]:
    pi_d1 = float(quantities["PI(D1)"]["point_bits"])
    pi_d23 = float(quantities["PI(D2+D3)"]["point_bits"])
    pi_joint = float(quantities["PI(D1+D2+D3)"]["point_bits"])

    shared = pi_d1 + pi_d23 - pi_joint
    d1_unique = pi_joint - pi_d23
    d23_unique = pi_joint - pi_d1
    residual = h_y - pi_joint

    bar_y, bar_h = 0.36, 0.30
    segments = [
        (0.0, d1_unique, PALETTE["d1"], f"D1 increment\n{d1_unique:.3f}"),
        (d1_unique, shared, PALETTE["overlap"], f"overlap\n{shared:.3f}"),
        (pi_d1, d23_unique, PALETTE["d2"], f"D2+D3\n{d23_unique:.3f}"),
        (pi_joint, residual, PALETTE["light_gray"], f"residual label uncertainty\n{residual:.3f}"),
    ]
    for x0, width, colour, label in segments:
        ax.add_patch(
            Rectangle((x0, bar_y), width, bar_h, facecolor=colour, edgecolor=PALETTE["text"], linewidth=0.6)
        )
        ax.text(x0 + width / 2, bar_y + bar_h / 2, label, ha="center", va="center", fontsize=7.2)

    def bracket(x0: float, x1: float, y: float, label: str, above: bool) -> None:
        tick = 0.035 if above else -0.035
        ax.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick], color=PALETTE["text"], linewidth=0.9)
        ax.text((x0 + x1) / 2, y + (0.045 if above else -0.045), label, ha="center",
                va="bottom" if above else "top", fontsize=7.8)

    bracket(0.0, pi_d1, bar_y + bar_h + 0.085, f"PI(D1) = {pi_d1:.3f}", above=True)
    bracket(d1_unique, pi_joint, bar_y - 0.085, f"PI(D2+D3) = {pi_d23:.3f}", above=False)
    bracket(0.0, pi_joint, bar_y + bar_h + 0.275, f"PI(D1+D2+D3) = {pi_joint:.3f}", above=True)

    ax.annotate(
        f"H(Y) = {h_y:.4f} bit",
        xy=(h_y, bar_y + bar_h / 2),
        xytext=(h_y - 0.012, bar_y - 0.28),
        ha="right",
        fontsize=7.8,
        arrowprops={"arrowstyle": "-", "color": PALETTE["muted"], "linewidth": 0.7},
    )

    ax.set_xlim(-0.015, h_y + 0.015)
    ax.set_ylim(-0.32, 1.06)
    ax.set_xlabel("Label information (bits)")
    ax.set_yticks([])
    ax.set_title("(c) Split of the estimated joint predictive information on the label-entropy scale")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    return {
        "d1_unique": d1_unique,
        "shared": shared,
        "d23_unique": d23_unique,
        "residual": residual,
    }


def main() -> None:
    args = parse_args()
    setup_style()
    rows = read_csv_rows(args.info_diagnostics_csv)
    quantities = {row["quantity"]: row for row in rows}

    fig = plt.figure(figsize=(7.1, 6.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85], width_ratios=[1.1, 1.0], hspace=0.62, wspace=0.34)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, :])

    plot_pi_panel(ax1, quantities)
    plot_ii_panel(ax2, quantities)
    decomposition = plot_overlap_panel(ax3, quantities, args.label_entropy)

    files = save_figure(fig, args.output_dir, args.stem)
    manifest = {
        "script": "plot_lmeic_fig_info_overlap.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"info_diagnostics_csv": str(args.info_diagnostics_csv), "label_entropy": args.label_entropy},
        "decomposition_bits": decomposition,
        "files": files,
    }
    manifest_path = args.output_dir / f"{args.stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
