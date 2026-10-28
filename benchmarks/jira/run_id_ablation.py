#!/usr/bin/env python3
"""
E18c: Apache Jira — how much of the signal is reporter/project identity rather than content?
Version: v1.2 | Date: 2026-10-28
v1.1: paired bootstrap on seed-averaged macro-F1 (see run_d1_fusion v1.2); adds full+d1 - full.
v1.2: --n-orders repeats every arm under that many column orders (order 0 = original), because
      XGBoost column subsampling depends on column order: the same features in another order moved the
      S2 D1 gain from +4.09 to +2.74 points (run_d1_fusion). Estimates average over orders x seeds.
      --device cuda runs XGBoost on the GPU, but on these sparse TF-IDF inputs the GPU build gives
      different and much worse fits (S1 one seed: 0.359 GPU vs 0.511 CPU), so E26 uses the
      CPU (default) with --n-jobs threads. (E26)

Same sample, rows (valid D1), time split and XGBoost settings as run_d1_fusion v1.1. The concern
(experiment card, E18): a model may learn "this reporter / this project tends to file Major" instead
of importance. Arms, with and without the D1 descriptors:
  full          TF-IDF + project/type one-hot + all history columns            (= run_d1_fusion base)
  no_reporter   drop the reporter history columns (reporter_prior_n / _high_share / _low_share)
  no_project    drop the project one-hot (issue type kept)
  no_identity   drop both
  text_only     TF-IDF only
Each arm: five seeds; paired issue-level bootstrap (2,000 draws) of the seed-majority predictions
against `full` of the same D1 condition. Settings S1 (three classes) and S2 (non-Major, two classes).

Usage:
  python -m benchmarks.jira.run_id_ablation \
      --d1 datasets/public/apache_jira/d1/d1_jira_qwen25_7b.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.jira.run_baselines import HIST_COLS, history_features, load, macro  # noqa: E402
from benchmarks.jira.run_baselines import SEEDS  # noqa: E402
from benchmarks.jira.run_d1_fusion import d1_frame, paired_boot, paired_boot_seeds  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402
from lmeic.common.rev1_common import git_commit, write_json  # noqa: E402

from benchmarks.jira.sample_ids import DEFAULT_SAMPLE_IDS, read_sample_ids  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

REPORTER_COLS = ["reporter_prior_n", "reporter_prior_high_share", "reporter_prior_low_share"]


def xgb_runs_orders(Xtr, ytr, Xte, yte, k, n_orders: int, device: str, n_jobs: int = 8):
    """Five seeds under each of n_orders column orders; returns per-run F1, majority vote, all predictions."""
    fs, preds = [], []
    for o in range(n_orders):
        perm = np.arange(Xtr.shape[1]) if o == 0 else np.random.default_rng(1000 + o).permutation(Xtr.shape[1])
        A, B = Xtr[:, perm], Xte[:, perm]
        for s in SEEDS:
            m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8,
                              random_state=s, n_jobs=n_jobs, tree_method="hist", device=device,
                              **({"objective": "multi:softprob", "num_class": k, "eval_metric": "mlogloss"} if k > 2
                                 else {"objective": "binary:logistic", "eval_metric": "logloss"}))
            m.fit(A, ytr)
            p = m.predict(B)
            fs.append(macro(yte, p, k)); preds.append(p)
    P = np.vstack(preds)
    vote = np.apply_along_axis(lambda c: np.bincount(c, minlength=k).argmax(), 0, P)
    return fs, vote, P


def run_setting(df: pd.DataFrame, d1c: list[str], setting: str, n_orders: int = 1, device: str = "cpu",
                n_jobs: int = 8) -> dict:
    if setting == "S2":
        df = df[df["priority_creation"] != "Major"].reset_index(drop=True)
    present = sorted(df["y"].unique()); k = len(present)
    df = df.assign(y=df["y"].map({c: i for i, c in enumerate(present)}))
    train = df[df["created"] < "2022-01-01"]; test = df[df["created"] >= "2023-01-01"]
    ytr, yte = train["y"].values, test["y"].values
    logger.info("%s | train %d / test %d | classes %s", setting, len(train), len(test), present)

    tf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=3)
    T = (tf.fit_transform(train["text"]), tf.transform(test["text"]))
    cp = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    P = (cp.fit_transform(train[["project"]].astype(str)), cp.transform(test[["project"]].astype(str)))
    ct = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    Ty = (ct.fit_transform(train[["type"]].astype(str)), ct.transform(test[["type"]].astype(str)))
    other = [c for c in HIST_COLS if c not in REPORTER_COLS]
    Hr = tuple(csr_matrix(x[REPORTER_COLS].values) for x in (train, test))
    Ho = tuple(csr_matrix(x[other].values) for x in (train, test))
    D = tuple(csr_matrix(x[d1c].values) for x in (train, test))
    blocks = {"full": [T, P, Ty, Ho, Hr], "no_reporter": [T, P, Ty, Ho], "no_project": [T, Ty, Ho, Hr],
              "no_identity": [T, Ty, Ho], "text_only": [T]}
    res, votes, allp = {}, {}, {}
    for d1 in (False, True):
        for name, bl in blocks.items():
            parts = bl + ([D] if d1 else [])
            Xa = hstack([b[0] for b in parts]).tocsr(); Xb = hstack([b[1] for b in parts]).tocsr()
            arm = f"{name}{'+d1' if d1 else ''}"
            fs, vote, allp[arm] = xgb_runs_orders(Xa, ytr, Xb, yte, k, n_orders, device, n_jobs)
            res[arm] = {"macro_f1_mean": float(np.mean(fs)), "macro_f1_sd": float(np.std(fs)), "by_seed": fs,
                        "vote_macro_f1": macro(yte, vote, k),
                        "vote_per_class_f1": f1_score(yte, vote, average=None, labels=list(range(k)),
                                                      zero_division=0).round(4).tolist()}
            votes[arm] = vote
            logger.info("%s %-16s %.4f ± %.4f", setting, arm, np.mean(fs), np.std(fs))
    pairs = [(f"full{d1}", f"{name}{d1}") for d1 in ("", "+d1")
             for name in ("no_reporter", "no_project", "no_identity", "text_only")]
    pairs += [("full", "full+d1"), ("no_identity", "no_identity+d1"), ("text_only", "text_only+d1")]
    comps = {f"{b} - {a}": paired_boot_seeds(yte, allp[a], allp[b], k) for a, b in pairs}
    comps_vote = {f"{b} - {a}": paired_boot(yte, votes[a], votes[b], k) for a, b in pairs}
    for kk, v in comps.items():
        logger.info("%s %s: %+.4f CI %s p=%.3f", setting, kk, v["diff"], v["ci95"], v["p_two_sided"])
    return {"setting": setting, "n_train": len(train), "n_test": len(test), "classes": [int(c) for c in present],
            "n_orders": n_orders, "runs_per_arm": n_orders * len(SEEDS), "device": device, "n_jobs": n_jobs,
            "arms": res, "comparisons": comps, "comparisons_majority_vote": comps_vote}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="datasets/public/apache_jira/processed")
    ap.add_argument("--sample-ids", default=DEFAULT_SAMPLE_IDS)
    ap.add_argument("--d1", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-orders", type=int, default=1)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time(); proc = Path(args.processed)
    df = load(proc)
    df = df.join(history_features(df))
    df["issue_id"] = df["issue_id"].astype(str)
    ids = read_sample_ids(proc, args.sample_ids)
    df = df.set_index("issue_id").loc[ids].reset_index()
    d1 = d1_frame(Path(args.d1))
    df = df.merge(d1, on="issue_id", how="inner").sort_values("created").reset_index(drop=True)
    d1c = [c for c in d1.columns if c != "issue_id"]
    out_dir = Path(args.out) if args.out else proc / f"id_ablation_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {"n_rows": len(df), "reporter_cols": REPORTER_COLS, "d1_file": args.d1, "git_commit": git_commit(_REPO_ROOT)}
    for s in ("S1", "S2"):
        out[s] = run_setting(df, d1c, s, args.n_orders, args.device, args.n_jobs)
        write_json(out_dir / "results.json", {**out, "elapsed_s": round(time.time() - t0, 1)})
    logger.info("saved %s", out_dir)


if __name__ == "__main__":
    main()
