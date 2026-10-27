"""
Revised main-table numbers for EMAIL-3R (thread-grouped protocol, R2 / R3).
Version: v1 | Date: 2026-10-27

Reads the systems registered in rev1_reference.yaml (five-seed rebuild variants, v1.0 stored
baseline predictions, E11 open-weight baselines) and the paired comparisons written by
analyse_reference.py, then reports for each system: seeds, macro-F1 (seed mean), SD over seeds
(population, ddof = 0), per-class F1 (seed mean), sender-unseen macro-F1, and for the comparisons
of R3 and R2 against every main-text baseline the Holm-adjusted p within each family.

Usage:
  python -m lmeic.reference.main_table \
      --run-dir datasets/email3r_runs/rev1_reference_20261022_191130 --analysis analysis_20261027_100042
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.reference.analyse_reference import CONFIG_NAME, LABELS, load_systems, macro  # noqa: E402
from lmeic.common.rev1_common import git_commit, load_config, norm_mid, source_run_dir, write_json  # noqa: E402
from lmeic.common.data_loader import load_split  # noqa: E402

BASELINES = ["v1_B00_majority", "v1_B01_tfidf_lr", "v1_B02_gpt4o_zs", "v1_B04_gpt55_zs", "v1_B06_deepseek_v4pro_zs",
             "v1_B07_gpt55_fewshot", "v1_B08_gpt55_context_enriched", "E11_zs_qwen25_7b", "E11_zs_qwen3_8b",
             "R4_B11_v3_grouped", "R5_B12_grouped", "v1_B09_xlmr_ft", "v1_B10_xlmr_cls_xgb", "E11_lora_qwen25_role"]
PROPOSED = ["R2_v3_grouped_explicit", "R3_v3_grouped_full"]


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def per_class(y, P):
    out = []
    for p in P:
        cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3).astype(float)
        tp = np.diag(cm); den = cm.sum(0) + cm.sum(1)
        out.append(np.divide(2 * tp, den, out=np.zeros(3), where=den > 0))
    return np.mean(out, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--analysis", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir); ana = run_dir / args.analysis
    paths, cfg = load_config(CONFIG_NAME)
    cfg["_repo_root"] = paths["repo_root"]
    te = load_split("test")
    test_mids = [norm_mid(m) for m in te["message_id"]]
    y = te["label_final"].map({l: i for i, l in enumerate(LABELS)}).values
    systems = load_systems(run_dir, source_run_dir(paths, cfg["source_run_id"]), cfg, test_mids)
    metrics = pd.read_csv(ana / "system_metrics.csv").set_index("system")
    rows = []
    for name in BASELINES + PROPOSED:
        P = systems[name]
        seeds = [macro(y, P[i:i + 1]) for i in range(P.shape[0])]
        pc = per_class(y, P)
        rows.append({"system": name, "n_seeds": P.shape[0], "macro_f1": macro(y, P),
                     "sd": float(np.std(seeds)) if P.shape[0] > 1 else np.nan,
                     "f1_high": pc[0], "f1_medium": pc[1], "f1_low": pc[2],
                     "sender_unseen": metrics.loc[name, "sender_unseen"]})
    tab = pd.DataFrame(rows)
    comp = pd.read_csv(ana / "paired_comparisons.csv")
    fam = []
    for a in PROPOSED:
        c = comp[(comp.a == a) & comp.b.isin(BASELINES)].drop_duplicates(["a", "b"]).copy()
        c["p_holm"] = holm(c["p_two_sided"].to_numpy()); c["family"] = a
        fam.append(c)
    fam = pd.concat(fam)
    extra = comp[(comp.a == "R3_v3_grouped_full") & (comp.b == "R2_v3_grouped_explicit")]
    out = ana / "main_table"
    out.mkdir(exist_ok=True)
    tab.to_csv(out / "system_rows.csv", index=False)
    fam.to_csv(out / "comparisons_holm.csv", index=False)
    write_json(out / "meta.json", {"analysis": args.analysis, "n_test": int(len(y)),
                                   "family": "per proposed system, all main-text baselines; Holm",
                                   "r3_minus_r2": extra.iloc[0].to_dict(), "git_commit": git_commit(_REPO_ROOT)})
    pd.set_option("display.width", 200)
    print(tab.round(4).to_string(index=False))
    f = fam.copy()
    for col in ["delta", "ci95_low", "ci95_high"]:
        f[col] = (f[col] * 100).round(2)
    print(f[["family", "b", "delta", "ci95_low", "ci95_high", "p_two_sided", "p_holm"]].to_string(index=False))


if __name__ == "__main__":
    main()
