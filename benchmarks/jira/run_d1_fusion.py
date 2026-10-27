#!/usr/bin/env python3
"""
E18b: Apache Jira — does adding LLM descriptors (D1) to the creation-time features help?
Version: v1.2 | Date: 2026-10-26
v1.1: --setting S2 (issues whose creation priority is not Major; High/Low remapped to 0/1, as in
      run_baselines). S1 is the default.
v1.2: the paired bootstrap now compares seed-averaged macro-F1 (all five seeds of each arm, as in
      the EMAIL-3R analysis), so the reported difference matches the arm means; the earlier
      majority-vote comparison is kept under comparisons_majority_vote.

Same sample, splits, features and XGBoost settings as run_baselines v1.1 (S1_all or S2_non_default).
Arms, each XGBoost over five seeds:
  base       TF-IDF + project/type one-hot + reporter/component history (run_baselines features)
  base+d1    base + the 13 D1 descriptors (intent one-hot, 6 ordinal, 6 boolean)
  d1_only    D1 descriptors + project/type one-hot (no text, no history)
Issues without a valid D1 record are dropped from every arm, so all arms use identical rows.
Paired test on test macro-F1: per-seed differences plus an issue-level paired bootstrap (2,000 draws)
of the seed-averaged predictions (majority vote across seeds).

Usage:
  python -m benchmarks.jira.run_d1_fusion \
      --d1 datasets/public/apache_jira/d1/d1_jira_qwen25_7b.jsonl
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
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.jira.run_baselines import (  # noqa: E402
    HIST_COLS, SEEDS, history_features, load, macro,
)
from lmeic.common.rev1_common import git_commit, write_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

INTENTS = ["request", "inform", "confirm", "inquiry", "reminder", "escalate", "other"]
ORD = ["urgency", "time_concreteness", "request_specificity", "consequence_severity", "role_relevance"]
BOOL = ["action_required", "decision_required", "deadline_mentioned", "external_party_waiting",
        "financial_mention", "direct_question"]


def d1_frame(path: Path) -> pd.DataFrame:
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    d = pd.DataFrame(recs).drop_duplicates("issue_id", keep="last")
    d["issue_id"] = d["issue_id"].astype(str)
    out = pd.DataFrame({"issue_id": d["issue_id"]})
    for c in ORD:
        out[f"d1_{c}"] = d[c].astype(float).values
    for c in BOOL:
        out[f"d1_{c}"] = d[c].astype(bool).astype(float).values
    it = d["intent_type"].where(d["intent_type"].isin(INTENTS), "other")
    for c in INTENTS:
        out[f"d1_intent_{c}"] = (it == c).astype(float).values
    return out


def xgb_runs(Xtr, ytr, Xte, yte, k):
    fs, preds = [], []
    for s in SEEDS:
        m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8,
                          random_state=s, n_jobs=8,
                          **({"objective": "multi:softprob", "num_class": k, "eval_metric": "mlogloss"} if k > 2
                             else {"objective": "binary:logistic", "eval_metric": "logloss"}))
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        fs.append(macro(yte, p, k)); preds.append(p)
    P = np.vstack(preds)
    vote = np.apply_along_axis(lambda c: np.bincount(c, minlength=k).argmax(), 0, P)
    return fs, vote, P


def paired_boot_seeds(y, Pa, Pb, k, n=2000, seed=20261024):
    """Paired bootstrap of the seed-averaged macro-F1 difference (rows of Pa / Pb are seeds).

    Same estimand as the EMAIL-3R analysis (analyse_reference.py): resample test items, average
    macro-F1 over the five seeds of each arm, take the difference. Primary from v1.2 on."""
    def m(P, b):
        return float(np.mean([macro(y[b], r[b], k) for r in P]))
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); d = []
    for _ in range(n):
        b = rng.choice(idx, len(idx))
        d.append(m(Pb, b) - m(Pa, b))
    d = np.array(d)
    return {"diff": m(Pb, idx) - m(Pa, idx), "ci95": np.percentile(d, [2.5, 97.5]).round(4).tolist(),
            "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
            "estimand": "seed-averaged macro-F1"}


def paired_boot(y, pa, pb, k, n=2000, seed=20261024):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); d = []
    for _ in range(n):
        b = rng.choice(idx, len(idx))
        d.append(macro(y[b], pb[b], k) - macro(y[b], pa[b], k))
    d = np.array(d)
    return {"diff": macro(y, pb, k) - macro(y, pa, k), "ci95": np.percentile(d, [2.5, 97.5]).round(4).tolist(),
            "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="datasets/public/apache_jira/processed")
    ap.add_argument("--sample-ids", default="sample_30000_issue_ids.txt")
    ap.add_argument("--d1", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--setting", choices=["S1", "S2"], default="S1")
    args = ap.parse_args()
    t0 = time.time(); proc = Path(args.processed)

    df = load(proc)
    df = df.join(history_features(df))
    df["issue_id"] = df["issue_id"].astype(str)
    ids = [l.strip() for l in (proc / args.sample_ids).read_text().splitlines() if l.strip()]
    df = df.set_index("issue_id").loc[ids].reset_index()
    d1 = d1_frame(Path(args.d1))
    n_before = len(df)
    df = df.merge(d1, on="issue_id", how="inner").sort_values("created").reset_index(drop=True)
    logger.info("sample %d | with valid D1 %d", n_before, len(df))
    D1C = [c for c in d1.columns if c != "issue_id"]
    if args.setting == "S2":
        df = df[df["priority_creation"] != "Major"].reset_index(drop=True)
    present = sorted(df["y"].unique()); k = len(present)
    df["y"] = df["y"].map({c: i for i, c in enumerate(present)})
    logger.info("setting %s | rows %d | classes %s", args.setting, len(df), present)

    train = df[df["created"] < "2022-01-01"]; test = df[df["created"] >= "2023-01-01"]
    ytr, yte = train["y"].values, test["y"].values
    tf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=3)
    Ttr, Tte = tf.fit_transform(train["text"]), tf.transform(test["text"])
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    Ctr = cat.fit_transform(train[["project", "type"]].astype(str)); Cte = cat.transform(test[["project", "type"]].astype(str))
    Htr, Hte = csr_matrix(train[HIST_COLS].values), csr_matrix(test[HIST_COLS].values)
    Dtr, Dte = csr_matrix(train[D1C].values), csr_matrix(test[D1C].values)
    arms = {"base": (hstack([Ttr, Ctr, Htr]), hstack([Tte, Cte, Hte])),
            "base+d1": (hstack([Ttr, Ctr, Htr, Dtr]), hstack([Tte, Cte, Hte, Dte])),
            "d1_only": (hstack([Ctr, Dtr]), hstack([Cte, Dte]))}
    res, votes, allp = {}, {}, {}
    for name, (Xa, Xb) in arms.items():
        fs, vote, allp[name] = xgb_runs(Xa.tocsr(), ytr, Xb.tocsr(), yte, k)
        res[name] = {"macro_f1_mean": float(np.mean(fs)), "macro_f1_sd": float(np.std(fs)), "by_seed": fs,
                     "vote_macro_f1": macro(yte, vote, k),
                     "vote_per_class_f1": f1_score(yte, vote, average=None, labels=list(range(k)), zero_division=0).round(4).tolist()}
        votes[name] = vote
        logger.info("%-8s macro-F1 %.4f ± %.4f", name, np.mean(fs), np.std(fs))
    comps = {
        "base+d1 - base": {"per_seed_diff": (np.array(res["base+d1"]["by_seed"]) - np.array(res["base"]["by_seed"])).round(4).tolist(),
                           **paired_boot_seeds(yte, allp["base"], allp["base+d1"], k)},
        "d1_only - base": paired_boot_seeds(yte, allp["base"], allp["d1_only"], k),
    }
    comps_vote = {"base+d1 - base": paired_boot(yte, votes["base"], votes["base+d1"], k),
                  "d1_only - base": paired_boot(yte, votes["base"], votes["d1_only"], k)}
    for k, v in comps.items():
        logger.info("%s: %+.4f CI %s p=%.3f", k, v["diff"], v["ci95"], v["p_two_sided"])
    out_dir = Path(args.out) if args.out else proc / f"d1_fusion_{args.setting}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "results.json", {
        "setting": args.setting, "n_sample": n_before, "n_with_d1": len(df), "n_train": len(train), "n_test": len(test),
        "class_share_test": pd.Series(yte).value_counts(normalize=True).sort_index().round(4).tolist(),
        "arms": res, "comparisons": comps, "comparisons_majority_vote": comps_vote, "d1_file": args.d1, "elapsed_s": round(time.time() - t0, 1),
        "git_commit": git_commit(_REPO_ROOT)})
    logger.info("saved %s", out_dir)


if __name__ == "__main__":
    main()
