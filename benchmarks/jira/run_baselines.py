#!/usr/bin/env python3
"""
E18: Apache Jira creation-time priority benchmark — dataset and text-only baselines.
Version: v1.1 | Date: 2026-10-24
v1.1: history features computed once on the full table (strictly earlier issues), before sampling
      and splitting; v1 restarted history at each split. S2 labels remapped to the classes present.
      Each setting is written as soon as it finishes.

Reads processed/issues.parquet (build_issues.py), applies the pre-registered design:
  - projects: selected by the criteria in build_issues (five levels, >5 years, >150 changes)
  - label: priority_creation mapped 5 -> 3 (Blocker+Critical=High, Major=Medium, Minor+Trivial=Low)
  - main setting S1: all issues; sensitivity S2: issues whose creation priority is not "Major"
  - split by `created` (time), not at random: train <= TRAIN_END, val = the next window, test after
  - majority-class baseline, TF-IDF + logistic regression, XGBoost on the same features
  - features use only information at creation (text, components, type, reporter history < created)

Usage:
  python -m benchmarks.jira.run_baselines --processed datasets/public/apache_jira/processed
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
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.common.rev1_common import git_commit, write_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

MAP5 = {"Blocker": "High", "Critical": "High", "Major": "Medium", "Minor": "Low", "Trivial": "Low"}
LABELS = ["High", "Medium", "Low"]
SEEDS = [42, 123, 456, 789, 2026]


def macro(y, p, k=3):
    return float(f1_score(y, p, average="macro", labels=list(range(k)), zero_division=0))


def load(processed: Path) -> pd.DataFrame:
    df = pd.read_parquet(processed / "issues.parquet")
    df["created"] = pd.to_datetime(df["created"], utc=True)
    df = df[df["priority_creation"].isin(MAP5) & df["project_selected"]].copy()
    df["created"] = pd.to_datetime(df["created"], utc=True, errors="coerce")
    df = df[df["created"] >= "2000-01-01"].copy()   # 289 rows carry corrupt years 0010-0012
    df["label3"] = df["priority_creation"].map(MAP5)
    df = df[df["label3"].notna()].copy()
    df["y"] = df["label3"].map({k: i for i, k in enumerate(LABELS)})
    df["text"] = (df["summary"].fillna("") + " . " + df["description"].fillna("")).str.slice(0, 4000)
    return df.sort_values("created").reset_index(drop=True)


def history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reporter and component history strictly before each issue's creation."""
    df = df.sort_values("created")
    out = pd.DataFrame(index=df.index)
    out["reporter_prior_n"] = df.groupby("reporter").cumcount()
    out["comp_prior_n"] = df.groupby("components").cumcount()
    # running share of High / Low by reporter, using only earlier rows
    hi = (df["y"] == 0).astype(float); lo = (df["y"] == 2).astype(float)
    out["reporter_prior_high_share"] = hi.groupby(df["reporter"]).transform(
        lambda s: s.shift().expanding().mean())
    out["reporter_prior_low_share"] = lo.groupby(df["reporter"]).transform(
        lambda s: s.shift().expanding().mean())
    out["reporter_prior_high_share"] = out["reporter_prior_high_share"].fillna(0.0)
    out["reporter_prior_low_share"] = out["reporter_prior_low_share"].fillna(0.0)
    out["desc_chars"] = df["description"].fillna("").str.len()
    out["summary_chars"] = df["summary"].fillna("").str.len()
    out["h_n_components"] = df["n_components"].astype(float)
    out["h_has_attachment"] = df["has_attachment"].astype(float)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


HIST_COLS = ["reporter_prior_n", "comp_prior_n", "reporter_prior_high_share", "reporter_prior_low_share",
             "desc_chars", "summary_chars", "h_n_components", "h_has_attachment"]


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 20261024) -> pd.DataFrame:
    """Proportional sample by project, keeping the time order intact."""
    frac = min(1.0, n / len(df))
    return df.groupby("project", group_keys=False)[df.columns.tolist()].apply(
        lambda g: g.sample(max(1, int(round(len(g) * frac))), random_state=seed))


def run(df: pd.DataFrame, name: str) -> dict:
    train = df[df["created"] < "2022-01-01"]
    val = df[(df["created"] >= "2022-01-01") & (df["created"] < "2023-01-01")]
    test = df[df["created"] >= "2023-01-01"]
    logger.info("%s | train %d / val %d / test %d", name, len(train), len(val), len(test))
    if min(len(train), len(val), len(test)) < 100:
        return {"setting": name, "skipped": "too few rows"}
    present = sorted(df["y"].unique())                      # S2 has no Medium
    remap = {c: i for i, c in enumerate(present)}; k = len(present)
    names = [LABELS[c] for c in present]
    ytr, yva, yte = (x["y"].map(remap).values for x in (train, val, test))

    tf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=3)
    Xtr_t, Xva_t, Xte_t = tf.fit_transform(train["text"]), tf.transform(val["text"]), tf.transform(test["text"])
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    C = cat.fit_transform(train[["project", "type"]].astype(str))
    Xtr_c, Xva_c, Xte_c = C, cat.transform(val[["project", "type"]].astype(str)), cat.transform(test[["project", "type"]].astype(str))
    Htr, Hva, Hte = (x[HIST_COLS] for x in (train, val, test))
    Xtr = hstack([Xtr_t, Xtr_c, Htr.values]).tocsr()
    Xva = hstack([Xva_t, Xva_c, Hva.values]).tocsr()
    Xte = hstack([Xte_t, Xte_c, Hte.values]).tocsr()

    rows = []
    # majority class
    maj = np.bincount(ytr).argmax()
    rows.append({"setting": name, "model": "majority", "macro_f1": macro(yte, np.full_like(yte, maj), k),
                 "per_class_f1": f1_score(yte, np.full_like(yte, maj), average=None, labels=list(range(k)), zero_division=0).round(4).tolist()})
    # TF-IDF + LR (C by val)
    best = (None, -1)
    for c in [0.1, 0.5, 2.0]:
        lr = LogisticRegression(C=c, max_iter=1000, solver="lbfgs", class_weight="balanced", n_jobs=8).fit(Xtr, ytr)
        f = macro(yva, lr.predict(Xva), k)
        if f > best[1]:
            best = (c, f)
    lr = LogisticRegression(C=best[0], max_iter=1000, solver="lbfgs", class_weight="balanced", n_jobs=8).fit(Xtr, ytr)
    p = lr.predict(Xte)
    rows.append({"setting": name, "model": f"tfidf_lr(C={best[0]})", "val_macro_f1": best[1], "macro_f1": macro(yte, p, k),
                 "per_class_f1": f1_score(yte, p, average=None, labels=list(range(k)), zero_division=0).round(4).tolist()})
    # XGBoost, five seeds
    fs, pcs = [], []
    for s in SEEDS:
        xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9,
                            colsample_bytree=0.8, random_state=s, n_jobs=8,
                            **({"objective": "multi:softprob", "num_class": k, "eval_metric": "mlogloss"} if k > 2
                               else {"objective": "binary:logistic", "eval_metric": "logloss"}))
        xgb.fit(Xtr, ytr)
        px = xgb.predict(Xte)
        fs.append(macro(yte, px, k)); pcs.append(f1_score(yte, px, average=None, labels=list(range(k)), zero_division=0))
    rows.append({"setting": name, "model": "xgboost(5 seeds)", "macro_f1": float(np.mean(fs)),
                 "macro_f1_sd": float(np.std(fs)), "macro_f1_by_seed": fs,
                 "per_class_f1": np.mean(pcs, axis=0).round(4).tolist()})
    out = {"setting": name, "classes": names, "n_train": len(train), "n_val": len(val), "n_test": len(test),
           "class_share_train": (pd.Series(ytr).value_counts(normalize=True).sort_index().round(4).tolist()),
           "class_share_test": (pd.Series(yte).value_counts(normalize=True).sort_index().round(4).tolist()),
           "rows": rows}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="datasets/public/apache_jira/processed")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sample", type=int, default=30000, help="proportional sample by project (0 = all)")
    args = ap.parse_args()
    proc = Path(args.processed)
    t0 = time.time()
    df = load(proc)
    df = df.join(history_features(df))           # full table, strictly earlier issues, before sampling
    logger.info("rows %d | projects %d | date %s .. %s", len(df), df["project"].nunique(),
                df["created"].min().date(), df["created"].max().date())
    if args.sample:
        df = stratified_sample(df, args.sample).sort_values("created").reset_index(drop=True)
        logger.info("sampled %d rows for the baseline run", len(df))
    out_dir = Path(args.out) if args.out else proc / f"baselines_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = []
    for nm, sub in (("S1_all", df), ("S2_non_default", df[df["priority_creation"] != "Major"])):
        res.append(run(sub, nm))
        write_json(out_dir / f"setting_{nm}.json", res[-1])
    comp = []
    for r in res:
        for row in r.get("rows", []):
            comp.append({"setting": r["setting"], "model": row["model"],
                         "macro_f1": round(row["macro_f1"], 4),
                         "sd": round(row.get("macro_f1_sd", float("nan")), 4)})
    pd.DataFrame(comp).to_csv(out_dir / "summary.csv", index=False)
    write_json(out_dir / "results.json", {"settings": res, "elapsed_s": round(time.time() - t0, 1),
                                          "git_commit": git_commit(_REPO_ROOT),
                                          "design": {"map5": MAP5, "split": "train < 2022, val 2022, test >= 2023",
                                                     "seeds": SEEDS, "metric": "macro-F1 over 3 classes"}})
    logger.info("\n%s", pd.DataFrame(comp).to_string(index=False))
    logger.info("saved %s", out_dir)


if __name__ == "__main__":
    main()
