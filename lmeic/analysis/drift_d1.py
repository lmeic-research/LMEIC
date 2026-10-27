"""
Extractor drift check: v1.0 D1 descriptors vs re-extraction with gpt-4o-2024-11-20.
Version: v1 | Date: 2026-10-22

Part 1 (descriptor agreement, test split): per-field agreement; Cohen's kappa for the
categorical and boolean fields; exact / within-1 agreement, quadratic-weighted kappa,
Spearman rho and ICC(A,1) for the 0-10 fields.
Part 2 (downstream): the LMEIC-explicit reference model (v1.0 features, stored v1.0
hyper-parameters, five seeds) is fitted once on the development pool and scores the
test split twice, once with v1.0 descriptors and once with re-extracted descriptors.
Reports macro-F1 per seed, the paired difference with a bootstrap CI, and the
prediction flip rate.

Usage:
  python -m lmeic.analysis.drift_d1 --d1-dir <rev1_d1 run dir> [--skip-downstream]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.common.rev1_common import (  # noqa: E402
    git_commit, load_config, read_jsonl, source_run_dir, write_json,
)
from lmeic.features.extract_d1 import BOOL_FIELDS, NUMERIC_FIELDS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
D1_CONFIG = "rev1_d1_extraction.yaml"
REF_CONFIG = "rev1_reference.yaml"
INTENTS = ["request", "inform", "confirm", "inquiry", "reminder", "escalate", "other"]


def icc_a1(a: np.ndarray, b: np.ndarray) -> float:
    """Two-way random-effects, absolute-agreement, single-rater ICC (Shrout-Fleiss ICC(2,1))."""
    x = np.column_stack([a, b]).astype(float)
    n, k = x.shape
    gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def descriptor_agreement(v1: pd.DataFrame, new: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    m = v1.merge(new, on="row_idx", suffixes=("_v1", "_new"))
    rows = []
    a, b = m["intent_type_v1"].astype(str), m["intent_type_new"].astype(str)
    rows.append({"field": "intent_type", "type": "categorical", "agreement": float((a == b).mean()),
                 "kappa": float(cohen_kappa_score(a, b))})
    for f in BOOL_FIELDS:
        a, b = m[f"{f}_v1"].astype(bool), m[f"{f}_new"].astype(bool)
        rows.append({"field": f, "type": "bool", "agreement": float((a == b).mean()),
                     "kappa": float(cohen_kappa_score(a, b)),
                     "rate_v1": float(a.mean()), "rate_new": float(b.mean())})
    for f in NUMERIC_FIELDS:
        a, b = m[f"{f}_v1"].astype(int).values, m[f"{f}_new"].astype(int).values
        rows.append({"field": f, "type": "0-10", "agreement": float((a == b).mean()),
                     "within1": float((np.abs(a - b) <= 1).mean()),
                     "kappa_quadratic": float(cohen_kappa_score(a, b, weights="quadratic", labels=list(range(11)))),
                     "spearman": float(spearmanr(a, b).statistic), "icc_a1": icc_a1(a, b),
                     "mean_v1": float(a.mean()), "mean_new": float(b.mean())})
    tab = pd.DataFrame(rows)
    summary = {"n_aligned": int(len(m)),
               "intent_unknown_new": int((~m["intent_type_new"].isin(INTENTS)).sum()),
               "intent_dist_v1": m["intent_type_v1"].value_counts(normalize=True).round(4).to_dict(),
               "intent_dist_new": m["intent_type_new"].value_counts(normalize=True).round(4).to_dict()}
    return summary, tab


def downstream(new: pd.DataFrame, out_dir: Path, n_boot: int = 2000) -> dict:
    from lmeic.reference import features as F
    from lmeic.reference import models as M

    paths, cfg = load_config(REF_CONFIG)
    repo_root = Path(paths["repo_root"])
    run_dir = source_run_dir(paths, cfg["source_run_id"])
    groups = {"d1", "d2", "d3", "text", "interactions", "role_cond"}
    ov = new[["row_idx"] + ["intent_type"] + NUMERIC_FIELDS + BOOL_FIELDS].copy()
    ov["intent_type"] = ov["intent_type"].where(ov["intent_type"].isin(INTENTS), "other")

    fa, y, meta = F.build(cfg, repo_root, "v2", groups)
    fb, yb, metab = F.build(cfg, repo_root, "v2", groups, d1_override=ov)
    Xa, yy, g, Xt_a = F.pool(fa, y, meta)
    Xb, _, _, Xt_b = F.pool(fb, yb, metab)
    assert np.array_equal(Xa, Xb), "development pool changed; override must touch test rows only"
    assert (meta["test"]["message_id"].values == metab["test"]["message_id"].values).all()
    changed_rows = int((np.abs(Xt_a - Xt_b).sum(1) > 0).sum())

    v1 = json.loads((run_dir / cfg["v1_artefacts"]["tide_v9_config"]).read_text())
    params = {"xgb": v1["best_xgb_params"], "lgbm": v1["best_lgbm_params"]}
    res = M.stacking(Xa, yy, g, np.vstack([Xt_a, Xt_b]), params, "v1_original", cfg["seeds"],
                     cfg["n_folds"], cfg["inner_es_fraction"])
    yt, n = y["test"], len(y["test"])
    rng = np.random.default_rng(20261022)
    per_seed, preds = [], []
    for s in cfg["seeds"]:
        pa, pb = res[s]["pred"][:n], res[s]["pred"][n:]
        preds.append((s, pa, pb))
        per_seed.append({"seed": s, "macro_f1_v1_d1": float(f1_score(yt, pa, average="macro")),
                         "macro_f1_new_d1": float(f1_score(yt, pb, average="macro")),
                         "flip_rate": float((pa != pb).mean())})
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(np.mean([f1_score(yt[idx], pb[idx], average="macro") -
                              f1_score(yt[idx], pa[idx], average="macro") for _, pa, pb in preds]))
    ps = pd.DataFrame(per_seed)
    ps.to_csv(out_dir / "drift_downstream_per_seed.csv", index=False)
    rows = []
    for s, pa, pb in preds:
        for i, mid in enumerate(meta["test"]["message_id"].values):
            rows.append({"seed": s, "message_id": mid, "gold": int(yt[i]), "pred_v1_d1": int(pa[i]),
                         "pred_new_d1": int(pb[i])})
    pd.DataFrame(rows).to_parquet(out_dir / "drift_downstream_predictions.parquet", index=False)
    return {"test_rows_with_changed_features": changed_rows, "n_test": n,
            "macro_f1_v1_d1_mean": float(ps["macro_f1_v1_d1"].mean()),
            "macro_f1_new_d1_mean": float(ps["macro_f1_new_d1"].mean()),
            "delta_mean": float((ps["macro_f1_new_d1"] - ps["macro_f1_v1_d1"]).mean()),
            "delta_ci95_bootstrap": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "flip_rate_mean": float(ps["flip_rate"].mean()), "n_boot": n_boot}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1-dir", required=True)
    ap.add_argument("--skip-downstream", action="store_true")
    args = ap.parse_args()
    d1_dir = Path(args.d1_dir)
    paths, cfg = load_config(D1_CONFIG)
    run_dir = source_run_dir(paths, cfg["source_run_id"])

    new = pd.DataFrame(read_jsonl(d1_dir / "d1_test_drift.jsonl"))
    new["row_idx"] = new["id"].astype(int)
    new = new.drop_duplicates("row_idx")
    v1 = pd.DataFrame(read_jsonl(run_dir / cfg["drift"]["v1_d1_relpath"]))
    v1 = v1.drop_duplicates("row_idx")
    v1 = v1[v1["row_idx"].isin(new["row_idx"])]
    logger.info("aligned %d test rows (new=%d)", len(v1), len(new))

    out_dir = d1_dir / f"drift_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir()
    summary, tab = descriptor_agreement(v1, new)
    tab.to_csv(out_dir / "drift_descriptor_agreement.csv", index=False)
    logger.info("descriptor agreement:\n%s", tab.round(3).to_string(index=False))
    result = {"descriptor": summary, "git_commit": git_commit(Path(paths["repo_root"]))}
    if not args.skip_downstream:
        result["downstream"] = downstream(new, out_dir)
        logger.info("downstream: %s", json.dumps(result["downstream"], indent=1))
    write_json(out_dir / "drift_summary.json", result)


if __name__ == "__main__":
    main()
