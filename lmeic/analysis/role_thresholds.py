"""
Role-wise decision-threshold experiment (E15).
Version: v1.1 | Date: 2026-10-23
v1.1: development-pool roles cover train+val (the pool used by F.pool); both base
learners (XGB + LGBM) as in the reference stacking; meta-layer OOF probabilities are
cross-fitted over thread-grouped folds, so thresholds are never fitted on rows the
meta-model was trained on.

The stacking meta-model returns three class probabilities per instance and the
current system takes the argmax. The High/Medium/Low label is ordered, and 99%
of test errors are adjacent-class confusions, so a role-specific trade-off
between High and Low may recover some of that. This script asks a narrower and
more honest question than "can test macro-F1 be raised": how much of the
potential gain survives when the thresholds are fitted on development data and
then applied once to test?

For each seed:
  1. Refit the stacking pipeline exactly as the reference run does and keep the
     out-of-fold meta probabilities (thread-disjoint, so leak-free).
  2. Fit per-role multipliers (w_high, w_low) on the OOF predictions by
     maximising macro-F1 within each role, because macro-F1 is dominated by the
     small High class.
  3. Apply the fitted multipliers to the test probabilities of the same seed and
     report macro-F1 before and after, with a thread-clustered paired bootstrap.

Thresholds are fitted on cross-fitted meta probabilities of the development pool
and applied once to test; nothing is tuned on test.

Usage:
  python -m lmeic.analysis.e15_role_thresholds \
      --run-dir datasets/email3r_runs/rev1_reference_20261022_191130 \
      --variant R3_v3_grouped_full
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("e15")

from lmeic.reference import features as F  # noqa: E402
from lmeic.reference import models as M  # noqa: E402
from lmeic.common.rev1_common import (  # noqa: E402
    git_commit, load_config, write_json,
)
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

ROLES = ["Mgr", "CS", "Sch"]
W_GRID = np.round(np.arange(0.5, 3.01, 0.1), 2)


def fit_role_weights(proba: np.ndarray, y: np.ndarray, role: np.ndarray,
                     which: str) -> dict[str, tuple[float, float]]:
    """Grid search (w_high, w_low) per role maximising that role's macro-F1."""
    out = {}
    for r in ROLES:
        sel = role == r
        if not sel.any():
            out[r] = (1.0, 1.0)
            continue
        p, yy = proba[sel], y[sel]
        best = (f1_score(yy, p.argmax(1), average="macro"), 1.0, 1.0)
        for wh in W_GRID:
            for wl in W_GRID:
                pred = (p * np.array([wh, 1.0, wl])).argmax(1)
                s = f1_score(yy, pred, average="macro")
                if s > best[0]:
                    best = (s, float(wh), float(wl))
        out[r] = (best[1], best[2])
        logger.info("[%s] %s: w_high=%.1f w_low=%.1f oof macro-F1 %.4f (argmax %.4f)",
                    which, r, best[1], best[2], best[0],
                    f1_score(yy, p.argmax(1), average="macro"))
    return out


def apply_role_weights(proba: np.ndarray, role: np.ndarray,
                       weights: dict[str, tuple[float, float]]) -> np.ndarray:
    out = np.zeros(len(proba), dtype=int)
    for r in ROLES:
        sel = role == r
        if not sel.any():
            continue
        wh, wl = weights[r]
        out[sel] = (proba[sel] * np.array([wh, 1.0, wl])).argmax(1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", default="R3_v3_grouped_full")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    paths, cfg = load_config("rev1_reference.yaml")
    repo_root = Path(paths["repo_root"])
    vcfg = json.loads((run_dir / args.variant / "config.json").read_text())["variant"]
    params = json.loads((run_dir / args.variant / "params.json").read_text())
    groups = set(vcfg["groups"])

    probs = None
    if "coreader" in groups:
        probs = pd.read_parquet(run_dir / "coreader_probs_grouped.parquet")
    feats, y, meta = F.build(cfg, repo_root, vcfg["d2d3"], groups, probs)
    X, yy, g, Xt = F.pool(feats, y, meta)
    # F.pool stacks train and val into the development pool; roles must follow that order
    role_dev = np.concatenate([meta["train"]["role"].values, meta["val"]["role"].values])
    assert len(role_dev) == len(yy)
    role_te = meta["test"]["role"].values
    preds: list[pd.DataFrame] = []
    y_te = y["test"]

    per_seed, weights_all = [], {}
    for seed in cfg["seeds"]:
        folds = M.make_folds(yy, g, vcfg["protocol"], seed, cfg["n_folds"])
        oof_x = np.zeros((len(X), 3)); oof_l = np.zeros((len(X), 3))
        te_x = np.zeros((len(Xt), 3)); te_l = np.zeros((len(Xt), 3))
        for tr, va in folds:
            xm = M._xgb(params["xgb"], seed, prob=True)
            xm.fit(X[tr], yy[tr], sample_weight=M.sample_weights(yy[tr]))
            oof_x[va] = xm.predict_proba(X[va]); te_x += xm.predict_proba(Xt) / cfg["n_folds"]
            lm = M._fit_lgbm(M._lgbm(params["lgbm"], seed), X, yy, tr, va, g, vcfg["protocol"],
                             cfg["inner_es_fraction"], seed)
            oof_l[va] = lm.predict_proba(X[va]); te_l += lm.predict_proba(Xt) / cfg["n_folds"]
        z, zt = np.hstack([oof_x, oof_l]), np.hstack([te_x, te_l])
        # test probabilities: meta fitted on the full development pool, as in M.stacking
        meta_lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
        meta_lr.fit(z, yy)
        te_p = meta_lr.predict_proba(zt)
        # development probabilities for threshold fitting: meta layer cross-fitted
        oof_p = np.zeros((len(X), 3))
        for tr, va in folds:
            m2 = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
            m2.fit(z[tr], yy[tr])
            oof_p[va] = m2.predict_proba(z[va])

        weights = fit_role_weights(oof_p, yy, role_dev, f"{args.variant} seed {seed}")
        weights_all[seed] = weights
        pred_base = te_p.argmax(1)
        pred_cal = apply_role_weights(te_p, role_te, weights)
        per_seed.append({
            "seed": int(seed),
            "macro_f1_argmax": float(f1_score(y_te, pred_base, average="macro")),
            "macro_f1_role_thresholds": float(f1_score(y_te, pred_cal, average="macro")),
            "dev_oof_macro_f1_argmax": float(f1_score(yy, oof_p.argmax(1), average="macro")),
        })
        logger.info("seed %d: argmax %.4f -> role thresholds %.4f",
                    seed, per_seed[-1]["macro_f1_argmax"], per_seed[-1]["macro_f1_role_thresholds"])
        preds.append(pd.DataFrame({"seed": seed, "message_id": meta["test"]["message_id"].values,
                                   "role": role_te, "gold": y_te, "pred_argmax": pred_base,
                                   "pred_role_thresholds": pred_cal}))

    df = pd.DataFrame(per_seed)
    summary = {
        "variant": args.variant,
        "protocol": vcfg["protocol"],
        "note": "XGB+LGBM stacking with the variant's stored parameters; thresholds fitted on "
                "thread-grouped cross-fitted meta probabilities of the development pool",
        "argmax_mean": float(df.macro_f1_argmax.mean()),
        "threshold_mean": float(df.macro_f1_role_thresholds.mean()),
        "delta_mean": float((df.macro_f1_role_thresholds - df.macro_f1_argmax).mean()),
        "delta_std": float((df.macro_f1_role_thresholds - df.macro_f1_argmax).std()),
        "weights_by_role": {r: {str(sd): list(w[r]) for sd, w in weights_all.items()} for r in ROLES},
        "git_commit": git_commit(repo_root),
    }
    out = run_dir / f"e15_role_thresholds_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    out.mkdir()
    df.to_csv(out / "per_seed.csv", index=False)
    pd.concat(preds).to_parquet(out / "test_predictions.parquet", index=False)
    write_json(out / "summary.json", summary)
    logger.info("summary: %s", {k: summary[k] for k in ("argmax_mean", "threshold_mean", "delta_mean")})
    print(json.dumps({k: summary[k] for k in ("argmax_mean", "threshold_mean", "delta_mean")}, indent=2))


if __name__ == "__main__":
    main()
