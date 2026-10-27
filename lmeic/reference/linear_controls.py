#!/usr/bin/env python3
"""
R2.1 linear controls: logistic regression on the D1 descriptors alone and on the full
explicit feature set, under the thread-grouped protocol of the reference rebuild.
Version: v1 | Date: 2026-10-23

Answers Reviewer 2's question on the relation between "TF-IDF + LR" and "D1-only + LR": the
2x2 of input (sparse TF-IDF text vs D1 descriptors) and learner (LR vs gradient boosting).
  d1_lr        20 D1 columns (one-hot intent, 0-10 scales, Boolean cues)
  explicit_lr  the 127 LMEIC-explicit columns (R2 feature set)
Each: StandardScaler + multinomial LR (lbfgs, class_weight balanced as in the v1.0 TF-IDF + LR
baseline), C chosen by mean macro-F1 over the thread-grouped 5-fold split of the development
pool (fold seed 42), refit on the whole pool, test scored once. LR is deterministic, so one run.
Paired thread-cluster bootstrap (n_boot and seed from rev1_reference.yaml: analysis) against
R5 (D1-only XGBoost, 5 seeds), R2 (LMEIC-explicit, 5 seeds) and the v1.0 TF-IDF + LR (B01).

Usage:
  python -m lmeic.reference.linear_controls --run-dir <rev1_reference run>
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.reference import features as F  # noqa: E402
from lmeic.reference.analyse_reference import LABELS, macro  # noqa: E402
from lmeic.reference.folds import check_group_disjoint, make_folds  # noqa: E402
from lmeic.common.rev1_common import (  # noqa: E402
    git_commit, load_config, norm_mid, read_jsonl, source_run_dir, write_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
ARMS = {"d1_lr": ["d1"], "explicit_lr": ["d1", "d2", "d3", "text", "interactions", "role_cond"]}


def lr(c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000, class_weight="balanced"))


def fit_arm(cfg: dict, groups: list[str]) -> tuple[np.ndarray, list[str], dict]:
    feats, y, meta = F.build(cfg, _REPO_ROOT, "v3", set(groups))
    X, yy, g, Xt = F.pool(feats, y, meta)
    folds = make_folds(yy, g, "grouped_clean", 42, cfg["n_folds"])
    assert check_group_disjoint(folds, g) == 0
    cv = {}
    for c in C_GRID:
        oof = np.zeros(len(yy), dtype=int)
        for tr, va in folds:
            oof[va] = lr(c).fit(X[tr], yy[tr]).predict(X[va])
        cv[c] = float(f1_score(yy, oof, average="macro"))
    best = max(cv, key=cv.get)
    pred = lr(best).fit(X, yy).predict(Xt)
    info = {"n_features": X.shape[1], "columns": list(feats["train"].columns), "cv_macro_f1": cv, "best_C": best,
            "dev_oof_macro_f1": cv[best]}
    return pred, list(meta["test"]["message_id"]), info, y["test"], meta["test"]["thread_id"].values


def aligned(frame: pd.DataFrame, mids: list[str]) -> np.ndarray:
    lab = {l: i for i, l in enumerate(LABELS)}
    frame = frame.assign(message_id=frame["message_id"].map(norm_mid))
    out = []
    for _, g in frame.groupby("seed"):
        m = dict(zip(g["message_id"], g["pred"].map(lab)))
        missing = [x for x in mids if x not in m]
        assert not missing, f"{len(missing)} test ids missing"
        out.append([m[x] for x in mids])
    return np.array(out, dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    paths, cfg = load_config("rev1_reference.yaml")
    src_dir = source_run_dir(paths, cfg["source_run_id"])
    t0 = time.time()

    systems, infos, mids, yt, threads = {}, {}, None, None, None
    for arm, groups in ARMS.items():
        pred, m, info, y_te, th = fit_arm(cfg, groups)
        m = [norm_mid(x) for x in m]
        if mids is None:
            mids, yt, threads = m, y_te, th
        assert m == mids and (y_te == yt).all()
        systems[arm] = pred[None, :]
        infos[arm] = info
        logger.info("%s: %d features, best C %s (dev OOF %.4f), test macro-F1 %.4f",
                    arm, info["n_features"], info["best_C"], info["dev_oof_macro_f1"], macro(yt, systems[arm]))

    systems["R5_d1_xgb"] = aligned(pd.read_parquet(run_dir / "R5_B12_grouped" / "test_predictions_all_seeds.parquet"), mids)
    systems["R2_explicit"] = aligned(pd.read_parquet(run_dir / "R2_v3_grouped_explicit" / "test_predictions_all_seeds.parquet"), mids)
    b01 = pd.DataFrame(read_jsonl(src_dir / "phase4_experiments/baselines/B01_tfidf_lr/predictions.jsonl"))
    systems["B01_tfidf_lr"] = aligned(b01, mids)

    ac = cfg["analysis"]
    rng = np.random.default_rng(ac["boot_seed"])
    uniq = np.unique(threads)
    members = {t: np.where(threads == t)[0] for t in uniq}
    boots = [np.concatenate([members[t] for t in rng.choice(uniq, len(uniq), replace=True)]) for _ in range(ac["n_boot"])]
    comps = []
    for a, b in [("d1_lr", "B01_tfidf_lr"), ("R5_d1_xgb", "d1_lr"), ("R2_explicit", "explicit_lr"),
                 ("explicit_lr", "d1_lr"), ("R5_d1_xgb", "B01_tfidf_lr")]:
        d = np.array([macro(yt, systems[a], i) - macro(yt, systems[b], i) for i in boots])
        comps.append({"a": a, "b": b, "delta": macro(yt, systems[a]) - macro(yt, systems[b]),
                      "ci95_low": float(np.percentile(d, 2.5)), "ci95_high": float(np.percentile(d, 97.5)),
                      "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))})

    out = run_dir / f"r21_linear_controls_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir()
    tab = pd.DataFrame([{"system": k, "n_seeds": v.shape[0], "macro_f1": macro(yt, v)} for k, v in systems.items()])
    tab.to_csv(out / "system_metrics.csv", index=False)
    pd.DataFrame(comps).to_csv(out / "paired_comparisons.csv", index=False)
    pd.DataFrame({"message_id": mids, **{k: v[0] for k, v in systems.items() if k in ARMS}}).to_parquet(
        out / "test_predictions.parquet", index=False)
    write_json(out / "summary.json", {
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "columns"} | {"columns": v["columns"]} for k, v in infos.items()},
        "protocol": "grouped_clean, fold seed 42, 5 folds; C by mean dev OOF macro-F1; one test evaluation",
        "learner": "StandardScaler + LogisticRegression(lbfgs, max_iter=5000, class_weight=balanced)",
        "bootstrap": {"unit": "thread", "n_boot": ac["n_boot"], "seed": ac["boot_seed"]},
        "elapsed_s": round(time.time() - t0, 1), "git_commit": git_commit(_REPO_ROOT)})
    logger.info("\n%s\n%s", tab.round(4).to_string(index=False), pd.DataFrame(comps).round(4).to_string(index=False))
    logger.info("saved %s", out)


if __name__ == "__main__":
    main()
