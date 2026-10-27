#!/usr/bin/env python3
"""
E13: rolling-origin replay of model updating on EMAIL-3R.
Version: v1 | Date: 2026-10-23

All labelled instances (train, val, test) are ordered by month. Each update strategy fits the
reference stacking model (fixed tuned parameters of the chosen variant) on the instances dated
before a cutoff and scores the months it serves until the next update:
  static     one model, cutoff at the first evaluation month
  quarterly  refit at the first month of every quarter
  monthly    refit at the start of every evaluation month
Training rows whose thread also appears in a served month are removed. TF-IDF/SVD text features
are refitted on each training window; D1 descriptors need no refit and D2/D3 features are pre-t
by construction. Output: per-month macro-F1 per strategy plus thread-cluster bootstrap CIs of the
pooled differences against the static model.

Usage:
  python -m lmeic.deploy.temporal_replay --run-dir <rev1_reference run>
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
from sklearn.metrics import f1_score

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.reference import features as F  # noqa: E402
from lmeic.reference import models as M  # noqa: E402
from lmeic.common.rev1_common import git_commit, load_config, write_json  # noqa: E402
from lmeic.common.data_loader import load_splits  # noqa: E402
from lmeic.common.tide_features import build_text_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEXT_COLS = [f"word_svd_{i}" for i in range(40)] + [f"char_svd_{i}" for i in range(20)]


def pooled_frame(rcfg: dict, repo_root: Path, variant: dict):
    groups = set(variant["groups"]) - {"text"}
    feats, y, meta = F.build(rcfg, repo_root, variant["d2d3"], groups)
    splits = load_splits(["train", "val", "test"])
    X, metas, ys, texts = [], [], [], []
    for s in ["train", "val", "test"]:
        info = splits[s].set_index("message_id").loc[meta[s]["message_id"], ["date_utc", "latest_text"]]
        m = meta[s].copy()
        m["month"] = pd.to_datetime(info["date_utc"].values, utc=True).strftime("%Y-%m")
        X.append(feats[s]); metas.append(m); ys.append(y[s]); texts.append(info["latest_text"].fillna("").values)
    return (pd.concat(X, ignore_index=True), np.concatenate(ys), pd.concat(metas, ignore_index=True),
            np.concatenate(texts), "text" in variant["groups"])


def schedules(months: list[str], first_eval: str) -> dict[str, list[tuple[str, list[str]]]]:
    ev = [m for m in months if m >= first_eval]
    quarter_starts = [m for m in ev if m == ev[0] or int(m[5:]) in (1, 4, 7, 10)]
    q = [(c, [m for m in ev if m >= c and (i + 1 == len(quarter_starts) or m < quarter_starts[i + 1])])
         for i, c in enumerate(quarter_starts)]
    return {"static": [(ev[0], ev)], "quarterly": q, "monthly": [(m, [m]) for m in ev]}


def fit_and_score(X, y, meta, texts, use_text, cutoff, served, params, rcfg, seeds):
    te = meta["month"].isin(served).values
    tr = (meta["month"] < cutoff).values & ~meta["thread_id"].isin(set(meta.loc[te, "thread_id"])).values
    Xtr, Xte = X.loc[tr].reset_index(drop=True), X.loc[te].reset_index(drop=True)
    if use_text:
        a, _, b = build_text_features(texts[tr], texts[te][:1], texts[te])
        Xtr = pd.concat([Xtr, pd.DataFrame(a, columns=TEXT_COLS)], axis=1)
        Xte = pd.concat([Xte, pd.DataFrame(b, columns=TEXT_COLS)], axis=1)
    A = np.nan_to_num(Xtr.values.astype(np.float32)); B = np.nan_to_num(Xte.values.astype(np.float32))
    res = M.stacking(A, y[tr], meta.loc[tr, "thread_id"].values, B, params, "grouped_clean", seeds,
                     rcfg["n_folds"], rcfg["inner_es_fraction"])
    proba = np.mean([r["proba"] for r in res.values()], axis=0)
    return np.where(te)[0], proba, int(tr.sum())


def cluster_boot(y, pa, pb, groups, n_boot, seed=42):
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    idx_by = [np.where(inv == k)[0] for k in range(len(ug))]
    obs = f1_score(y, pa, average="macro") - f1_score(y, pb, average="macro")
    d = []
    for _ in range(n_boot):
        ii = np.concatenate([idx_by[k] for k in rng.integers(0, len(ug), len(ug))])
        d.append(f1_score(y[ii], pa[ii], average="macro") - f1_score(y[ii], pb[ii], average="macro"))
    return obs, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(np.array(d) <= 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    paths, cfg = load_config("rev1_e13_deploy.yaml")
    _, rcfg = load_config("rev1_reference.yaml")
    tcfg = cfg["temporal"]
    repo_root = Path(paths["repo_root"])
    run_dir = Path(args.run_dir)
    out = run_dir / f"e13_temporal_{tcfg['variant']}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir()
    variant = next(v for v in rcfg["variants"] if v["name"] == tcfg["variant"])
    params = json.loads((run_dir / tcfg["variant"] / "params.json").read_text())

    X, y, meta, texts, use_text = pooled_frame(rcfg, repo_root, variant)
    months = sorted(meta["month"].unique())
    sched = schedules(months, tcfg["first_eval_month"])
    proba = {}
    n_train = []
    for name, plan in sched.items():
        P = np.full((len(meta), 3), np.nan)
        for cutoff, served in plan:
            idx, p, ntr = fit_and_score(X, y, meta, texts, use_text, cutoff, served, params, rcfg, tcfg["seeds"])
            P[idx] = p
            n_train.append({"strategy": name, "cutoff": cutoff, "served": ",".join(served), "n_train": ntr})
            logger.info("%s cutoff %s -> %s (n_train %d)", name, cutoff, served, ntr)
        proba[name] = P

    ev = meta["month"] >= tcfg["first_eval_month"]
    rows = []
    for m in sorted(meta.loc[ev, "month"].unique()):
        mask = (meta["month"] == m).values
        r = {"month": m, "n": int(mask.sum())}
        r.update({f"share_{c}": float(np.mean(y[mask] == k)) for k, c in enumerate(["high", "medium", "low"])})
        for name, P in proba.items():
            r[name] = f1_score(y[mask], P[mask].argmax(1), average="macro")
        rows.append(r)
    per_month = pd.DataFrame(rows)
    per_month.to_csv(out / "per_month.csv", index=False)

    e = ev.values
    pred = {k: P[e].argmax(1) for k, P in proba.items()}
    pooled = {k: float(f1_score(y[e], p, average="macro")) for k, p in pred.items()}
    comps = []
    for k in ["quarterly", "monthly"]:
        obs, lo, hi, p = cluster_boot(y[e], pred[k], pred["static"], meta.loc[e, "thread_id"].values, tcfg["n_boot"])
        comps.append({"a": k, "b": "static", "delta": obs, "ci_low": lo, "ci_high": hi, "p_one_sided": p})
    pd.DataFrame(n_train).to_csv(out / "training_windows.csv", index=False)
    pr = pd.DataFrame({"message_id": meta.loc[e, "message_id"].values, "month": meta.loc[e, "month"].values,
                       "y": y[e], **{f"pred_{k}": v for k, v in pred.items()}})
    pr.to_parquet(out / "predictions.parquet", index=False)
    write_json(out / "summary.json", {"pooled_macro_f1": pooled, "comparisons": comps, "config": tcfg,
                                      "params_from": tcfg["variant"], "git_commit": git_commit(repo_root)})
    logger.info("per month:\n%s", per_month.round(4).to_string(index=False))
    logger.info("pooled: %s", pooled)
    logger.info("comparisons: %s", comps)


if __name__ == "__main__":
    main()
