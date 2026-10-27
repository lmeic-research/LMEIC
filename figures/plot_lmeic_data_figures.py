"""Generate LMEIC manuscript data figures.

Version: v1.1, 2026-06-04.

This script creates Fig.3-Fig.6 for the LMEIC ESWA manuscript using
validated analysis artifacts and the approved LMEIC colour scheme.
All input and output paths are provided through command-line arguments.
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


PALETTE = {
    "white": "#FFFFFF",
    "panel_bg": "#F6F9F4",
    "d1": "#71A682",
    "d2": "#7EA4B6",
    "d3": "#D19246",
    "llm_full": "#4A4F7E",
    "co_reader": "#5B8C8A",
    "neutral": "#81989B",
    "text": "#2D3436",
    "muted": "#636E72",
    "border": "#B5AF8B",
    "light_gray": "#C8CDD0",
}


MODEL_COLOURS = {
    "LMEIC-full": PALETTE["llm_full"],
    "LMEIC-explicit": PALETTE["d1"],
    "Supervised PLM": PALETTE["d2"],
    "PLM representation + ML": PALETTE["d2"],
    "Explicit feature ML": PALETTE["border"],
    "LLM direct classification": PALETTE["neutral"],
    "Enhanced LLM": PALETTE["neutral"],
    "Classical text ML": PALETTE["light_gray"],
    "Naive": PALETTE["light_gray"],
}


ROLE_COLOURS = {
    "Mgr": PALETTE["llm_full"],
    "CS": PALETTE["d3"],
    "Sch": PALETTE["co_reader"],
}


DOMAIN_COLOURS = {
    "Text SVD": PALETTE["neutral"],
    "D1 semantic": PALETTE["d1"],
    "D2 relational": PALETTE["d2"],
    "D3 role-context": PALETTE["d3"],
    "Cross-source interactions": PALETTE["llm_full"],
    "Role-conditional interactions": PALETTE["co_reader"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LMEIC manuscript data figures.")
    parser.add_argument("--main-results-csv", required=True, type=Path)
    parser.add_argument("--shap-summary-json", required=True, type=Path)
    parser.add_argument("--info-diagnostics-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--version", default="v1", help="Figure version label.")
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, kwargs in [
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 300}),
    ]:
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor=PALETTE["white"], **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def add_value_labels(ax: plt.Axes, values: list[float], offset: float = 0.008) -> None:
    for idx, value in enumerate(values):
        ax.text(
            value + offset,
            idx,
            f"{value:.4f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color=PALETTE["text"],
        )


def plot_main_results(rows: list[dict[str, str]], output_dir: Path) -> dict[str, Any]:
    selected_models = [
        "LMEIC-full",
        "LMEIC-explicit",
        "XLM-R [CLS] + XGBoost",
        "Fine-tuned XLM-R",
        "D1-only XGBoost",
        "DeepSeek V4 Pro direct classification",
        "GPT-5.5 context-enriched",
        "TF-IDF + logistic regression",
        "Majority-class baseline",
    ]
    by_model = {row["model"]: row for row in rows}
    selected = [by_model[name] for name in selected_models]
    labels = [row["model"] for row in selected][::-1]
    values = [float(row["macro_f1"]) for row in selected][::-1]
    colours = [
        MODEL_COLOURS.get(row["model"], MODEL_COLOURS.get(row["category"], PALETTE["neutral"]))
        for row in selected
    ][::-1]

    fig, ax = plt.subplots(figsize=(7.1, 4.4))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colours, edgecolor=PALETTE["text"], linewidth=0.5)

    add_value_labels(ax, values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test macro-F1")
    ax.set_xlim(0.20, 0.84)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    files = save_figure(fig, output_dir, "fig3_main_results_v1")
    return {
        "figure": "Fig.3",
        "files": files,
        "models": [{"model": label, "macro_f1": value} for label, value in zip(labels, values)],
    }


def plot_per_role_results(rows: list[dict[str, str]], output_dir: Path) -> dict[str, Any]:
    selected_models = [
        "LMEIC-full",
        "LMEIC-explicit",
        "XLM-R [CLS] + XGBoost",
        "DeepSeek V4 Pro direct classification",
    ]
    roles = ["Mgr", "CS", "Sch"]
    role_labels = {
        "Mgr": "Business Manager\n(Mgr)",
        "CS": "Overseas Customer\nService (CS)",
        "Sch": "Shipping Schedule\nCoordinator (Sch)",
    }
    by_model = {row["model"]: row for row in rows}
    values = np.array([[float(by_model[model][role]) for role in roles] for model in selected_models])

    fig, ax = plt.subplots(figsize=(7.1, 4.2))
    x_pos = np.arange(len(roles))
    width = 0.18
    offsets = (np.arange(len(selected_models)) - (len(selected_models) - 1) / 2) * width
    model_styles = [
        (PALETTE["llm_full"], None),
        (PALETTE["d1"], "//"),
        (PALETTE["d2"], None),
        (PALETTE["neutral"], ".."),
    ]

    for idx, model in enumerate(selected_models):
        colour, hatch = model_styles[idx]
        bars = ax.bar(
            x_pos + offsets[idx],
            values[idx],
            width=width,
            label=model,
            color=colour,
            edgecolor=PALETTE["text"],
            linewidth=0.5,
            hatch=hatch,
        )
        for bar, value in zip(bars, values[idx]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
                color=PALETTE["text"],
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([role_labels[role] for role in roles])
    ax.set_ylabel("Per-role macro-F1")
    ax.set_ylim(0.42, 0.86)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    files = save_figure(fig, output_dir, "fig4_per_role_results_v1")
    return {
        "figure": "Fig.4",
        "files": files,
        "roles": roles,
        "models": {
            model: {role: float(by_model[model][role]) for role in roles}
            for model in selected_models
        },
    }


def plot_shap_domains(summary: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    mapping = [
        ("text", "Text SVD"),
        ("D1", "D1 semantic"),
        ("interactions", "Cross-source interactions"),
        ("D2", "D2 relational"),
        ("role_cond", "Role-conditional interactions"),
        ("D3", "D3 role-context"),
    ]
    source = summary["domain_attribution_pct"]
    labels = [label for _, label in mapping][::-1]
    values = [float(source[key]) for key, _ in mapping][::-1]
    colours = [DOMAIN_COLOURS[label] for label in labels]

    fig, ax = plt.subplots(figsize=(7.1, 3.7))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colours, edgecolor=PALETTE["text"], linewidth=0.5)
    for idx, value in enumerate(values):
        ax.text(
            value + 0.8,
            idx,
            f"{value:.1f}%",
            va="center",
            ha="left",
            fontsize=8,
            color=PALETTE["text"],
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean absolute SHAP attribution (%)")
    ax.set_xlim(0, 42)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    files = save_figure(fig, output_dir, "fig5_shap_domains_v1")
    return {
        "figure": "Fig.5",
        "files": files,
        "domain_attribution_pct": {label: value for label, value in zip(labels, values)},
    }


def plot_information_diagnostics(rows: list[dict[str, str]], output_dir: Path) -> dict[str, Any]:
    quantities = {row["quantity"]: row for row in rows}
    pi_order = ["PI(D1)", "PI(D2)", "PI(D3)", "PI(D2+D3)", "PI(D1+D2+D3)"]
    ii_order = ["II(D1;D2)", "II(D1;D3)", "II(D2;D3)", "II(D1;D2+D3)"]

    pi_values = [float(quantities[item]["point_bits"]) for item in pi_order]
    pi_low = [float(quantities[item]["ci95_lo"]) for item in pi_order]
    pi_high = [float(quantities[item]["ci95_hi"]) for item in pi_order]
    pi_errors = np.array(
        [
            [value - low for value, low in zip(pi_values, pi_low)],
            [high - value for value, high in zip(pi_values, pi_high)],
        ]
    )
    pi_colours = [
        PALETTE["d1"],
        PALETTE["d2"],
        PALETTE["d3"],
        PALETTE["co_reader"],
        PALETTE["llm_full"],
    ]

    ii_values = [float(quantities[item]["point_bits"]) for item in ii_order]
    ii_low = [float(quantities[item]["ci95_lo"]) for item in ii_order]
    ii_high = [float(quantities[item]["ci95_hi"]) for item in ii_order]
    ii_errors = np.array(
        [
            [value - low for value, low in zip(ii_values, ii_low)],
            [high - value for value, high in zip(ii_values, ii_high)],
        ]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})

    x1 = np.arange(len(pi_order))
    ax1.bar(x1, pi_values, color=pi_colours, edgecolor=PALETTE["text"], linewidth=0.5)
    ax1.errorbar(x1, pi_values, yerr=pi_errors, fmt="none", ecolor=PALETTE["text"], elinewidth=1.0, capsize=2.5)
    for idx, value in enumerate(pi_values):
        label_y = pi_high[idx] + 0.018
        ax1.text(idx, label_y, f"{value:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(pi_order, rotation=35, ha="right")
    ax1.set_ylabel("Predictive information (bits)")
    ax1.set_ylim(0, max(pi_high) + 0.095)
    ax1.set_title("Predictive information")
    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    x2 = np.arange(len(ii_order))
    ax2.bar(x2, ii_values, color=PALETTE["muted"], edgecolor=PALETTE["text"], linewidth=0.5)
    ax2.errorbar(x2, ii_values, yerr=ii_errors, fmt="none", ecolor=PALETTE["text"], elinewidth=1.0, capsize=2.5)
    ax2.axhline(0, color=PALETTE["text"], linewidth=0.8)
    for idx, value in enumerate(ii_values):
        label_y = ii_low[idx] - 0.018
        ax2.text(idx, label_y, f"{value:.3f}", ha="center", va="top", fontsize=7, rotation=90)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(ii_order, rotation=35, ha="right")
    ax2.set_ylabel("Interaction information (bits)")
    ax2.set_ylim(min(ii_low) - 0.075, 0.04)
    ax2.set_title("Interaction information")
    ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()

    files = save_figure(fig, output_dir, "fig6_information_diagnostics_v1")
    return {
        "figure": "Fig.6",
        "files": files,
        "pi": {name: value for name, value in zip(pi_order, pi_values)},
        "ii": {name: value for name, value in zip(ii_order, ii_values)},
    }


def write_manifest(output_dir: Path, manifest: dict[str, Any]) -> Path:
    path = output_dir / "figure_generation_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    setup_style()

    rows = read_csv_rows(args.main_results_csv)
    shap_summary = read_json(args.shap_summary_json)
    info_rows = read_csv_rows(args.info_diagnostics_csv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "script": "plot_lmeic_data_figures.py",
        "version": args.version,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "main_results_csv": str(args.main_results_csv),
            "shap_summary_json": str(args.shap_summary_json),
            "info_diagnostics_csv": str(args.info_diagnostics_csv),
        },
        "figures": [],
    }

    manifest["figures"].append(plot_main_results(rows, args.output_dir))
    manifest["figures"].append(plot_per_role_results(rows, args.output_dir))
    manifest["figures"].append(plot_shap_domains(shap_summary, args.output_dir))
    manifest["figures"].append(plot_information_diagnostics(info_rows, args.output_dir))

    manifest_path = write_manifest(args.output_dir, manifest)
    print(f"Wrote figure outputs to: {args.output_dir}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
