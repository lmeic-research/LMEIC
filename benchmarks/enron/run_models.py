#!/usr/bin/env python3
"""
E12 step 5: LMEIC variants, domain ablations and baselines on the Enron benchmark.
Version: v1.1 | Date: 2026-10-23 (v1.1: --keep-instances, --tag)

Protocol as in the EMAIL-3R reference rebuild (grouped_clean): Optuna tuning with
thread-grouped folds, XGB+LGBM stacking with a logistic meta-learner, five seeds; ablations
reuse the tuned parameters of their parent variant. Comparisons use a thread-cluster paired
bootstrap of the seed-averaged macro-F1 with Holm correction within each family. The
cross-role analysis uses test emails received by owners of two or more roles.

Usage:
  python -m benchmarks.enron.run_models --run-dir <build_instances dir> \
      --labels <agent_labels_e12_enron.jsonl> --d1 <d1 jsonl> --d1-name gpt4o \
      [--coreader <coreader dir>] [--variants explicit,full,...]
"""

from __future__ import annotations

import argparse
import glob
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.analysis.context_analysis import holm  # noqa: E402
from benchmarks.enron import data as D  # noqa: E402
from benchmarks.enron.enron_common import CONFIG_NAME  # noqa: E402
from lmeic.reference import models as M  # noqa: E402
from lmeic.reference.analyse_reference import macro  # noqa: E402
from lmeic.reference.coreader import PROB_COLS  # noqa: E402
from lmeic.common.rev1_common import git_commit, load_config, write_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXPLICIT = {"d1", "text", "d2", "d3", "interactions", "role_cond"}
VARIANTS = {  # name: (groups, parameters: "tune" or parent variant)
    "explicit": (EXPLICIT, "tune"),
    "full": (EXPLICIT | {"coreader"}, "tune"),
    "explicit_minus_d2": (EXPLICIT - {"d2"}, "explicit"),
    "explicit_minus_d3": (EXPLICIT - {"d3"}, "explicit"),
    "explicit_minus_d2d3": (EXPLICIT - {"d2", "d3"}, "explicit"),
    "explicit_minus_rolecond": (EXPLICIT - {"role_cond"}, "explicit"),
    "d1_only": ({"d1"}, "explicit"),
    "context_only": ({"d2", "d3"}, "explicit"),
    "full_minus_d2d3": (EXPLICIT - {"d2", "d3"} | {"coreader"}, "full"),
}
FAMILIES = {
    "domain_explicit": [("explicit", v) for v in ["explicit_minus_d2", "explicit_minus_d3", "explicit_minus_d2d3",
                                                  "explicit_minus_rolecond", "d1_only"]],
    "coreader": [("full", "explicit"), ("full", "full_minus_d2d3")],
    "baselines": [("explicit", "tfidf_lr"), ("explicit", "xlmr"), ("full", "tfidf_lr"), ("full", "xlmr")],
}


def thread_boot(threads: np.ndarray, n_boot: int, seed: int = 20261023) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(threads)
    members = {t: np.where(threads == t)[0] for t in uniq}
    return [np.concatenate([members[t] for t in rng.choice(uniq, len(uniq), replace=True)]) for _ in range(n_boot)]


def select(base: pd.DataFrame, cols: dict[str, list[str]], roles: np.ndarray, groups: set[str]) -> pd.DataFrame:
    X = base[[c for g in ["d1", "text", "d2", "d3", "coreader"] if g in groups for c in cols[g]]]
    return pd.concat([X, D.products(X, roles, groups)], axis=1)


def role_sensitivity(te: pd.DataFrame, proba: np.ndarray, pred: np.ndarray) -> dict:
    t = te.assign(e_imp=proba[:, 0] * 2 + proba[:, 1], pred=pred)
    t = t[t["cross_role"] == 1]
    rows = []
    for _, g in t.groupby("email_key"):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g.iloc[i], g.iloc[j]
                gd = int(np.sign(b["y"] - a["y"]))
                rows.append({"gold_differs": gd != 0, "pred_differs": a["pred"] != b["pred"],
                             "concordant": gd != 0 and np.sign(a["e_imp"] - b["e_imp"]) == gd})
    r = pd.DataFrame(rows, columns=["gold_differs", "pred_differs", "concordant"])
    if r.empty:
        return {"n_pairs": 0}
    gdiff = r[r["gold_differs"]]
    return {"n_pairs": int(len(r)), "gold_differs": float(r["gold_differs"].mean()),
            "pred_differs": float(r["pred_differs"].mean()), "n_gold_differing": int(len(gdiff)),
            "concordance_on_gold_differing": float(gdiff["concordant"].mean()) if len(gdiff) else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--d1", required=True)
    ap.add_argument("--d1-name", required=True)
    ap.add_argument("--coreader", default=None)
    ap.add_argument("--context", default=None, help="context features parquet (default: latest in run dir)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--keep-instances", default=None, help="text file of instance ids to keep (e.g. position audit subset)")
    ap.add_argument("--tag", default="", help="suffix of the output directory name")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    paths, ecfg = load_config(CONFIG_NAME)
    _, rcfg = load_config("rev1_reference.yaml")
    repo = Path(paths["repo_root"])
    out = run_dir / f"models_{args.d1_name}{'_' + args.tag if args.tag else ''}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir()
    t0 = time.time()

    df = D.load_labelled(run_dir, Path(args.labels))
    if args.keep_instances:
        keep = set(Path(args.keep_instances).read_text().split())
        df = df[df["instance_id"].isin(keep)].reset_index(drop=True)
    d1 = D.load_d1(Path(args.d1), df)
    ctx_path = Path(args.context) if args.context else Path(sorted(glob.glob(str(run_dir / "context_features_*.parquet")))[-1])
    ctx = pd.read_parquet(ctx_path).set_index("instance_id").loc[df["instance_id"]].reset_index(drop=True)
    mined = D.mined_features(df, ecfg["mined_ngrams"])
    text = D.text_features(df)
    parts = {"d1": d1, "text": text, "d2": ctx[[c for c in ctx.columns if c.startswith("d2_")]],
             "d3": pd.concat([ctx[[c for c in ctx.columns if c.startswith("d3_")]], mined], axis=1)}
    variants = [v for v in args.variants.split(",")]
    if args.coreader:
        cp = pd.read_parquet(Path(args.coreader) / "coreader_probs.parquet")
        cp = cp[(cp["split"] == "dev") | (cp["seed"] == 42)].set_index("instance_id")
        parts["coreader"] = cp.loc[df["instance_id"], PROB_COLS].reset_index(drop=True)
    else:
        variants = [v for v in variants if "coreader" not in VARIANTS[v][0]]
    base = pd.concat(parts.values(), axis=1)
    cols = {k: list(v.columns) for k, v in parts.items()}
    roles = df["role"].values
    dev, te = (df["split"] == "dev").values, (df["split"] == "test").values
    y, yt = df.loc[dev, "y"].values, df.loc[te, "y"].values
    g = df.loc[dev, "thread_id"].values
    tdf = df[te].reset_index(drop=True)
    logger.info("labelled %d (dev %d, test %d); D1 missing %d; feature columns %d", len(df), dev.sum(), te.sum(),
                len(d1.attrs["missing"]), base.shape[1])

    seeds, preds, probas, params, rows = rcfg["seeds"], {}, {}, {}, []
    for v in variants:
        groups, pfrom = VARIANTS[v]
        X = select(base, cols, roles, groups)
        Xd, Xt = X[dev].values.astype(np.float32), X[te].values.astype(np.float32)
        if pfrom == "tune":
            params[v] = M.tune(Xd, y, g, "grouped_clean", rcfg["n_folds"], rcfg["optuna_trials"], rcfg["inner_es_fraction"])
            write_json(out / f"params_{v}.json", params[v])
        res = M.stacking(Xd, y, g, Xt, params[v if pfrom == "tune" else pfrom], "grouped_clean", seeds,
                         rcfg["n_folds"], rcfg["inner_es_fraction"])
        preds[v] = np.vstack([res[s]["pred"] for s in seeds])
        probas[v] = np.mean([res[s]["proba"] for s in seeds], axis=0)
        f1s = [macro(yt, preds[v][[i]]) for i in range(len(seeds))]
        rows.append({"variant": v, "groups": "+".join(sorted(groups)), "n_features": X.shape[1],
                     "macro_f1_mean": float(np.mean(f1s)), "macro_f1_sd": float(np.std(f1s, ddof=1)),
                     **{f"macro_f1_{r}": macro(yt[(tdf["role"] == r).values], preds[v][:, (tdf["role"] == r).values])
                        for r in D.ROLES},
                     "macro_f1_non_cross_role": macro(yt[(tdf["cross_role"] == 0).values], preds[v][:, (tdf["cross_role"] == 0).values]),
                     "role_sensitivity": json.dumps(role_sensitivity(tdf, probas[v], np.round(preds[v].mean(0)).astype(int)))})
        logger.info("[%s] macro-F1 %.4f +/- %.4f (%d features, %.0fs)", v, rows[-1]["macro_f1_mean"],
                    rows[-1]["macro_f1_sd"], X.shape[1], time.time() - t0)
        pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)

    text_all = (df["subject"].fillna("") + "\n" + df["latest_text"].fillna("")).values
    wv = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=3, max_df=0.9)
    cv = TfidfVectorizer(max_features=30000, ngram_range=(2, 4), sublinear_tf=True, min_df=5, max_df=0.9, analyzer="char_wb")
    Zd = hstack([wv.fit_transform(text_all[dev]), cv.fit_transform(text_all[dev])]).tocsr()
    Zt = hstack([wv.transform(text_all[te]), cv.transform(text_all[te])]).tocsr()
    lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(Zd, y)
    preds["tfidf_lr"] = np.tile(lr.predict(Zt), (len(seeds), 1))
    probas["tfidf_lr"] = lr.predict_proba(Zt)
    if args.coreader:
        cpt = pd.read_parquet(Path(args.coreader) / "coreader_probs.parquet")
        cpt = cpt[cpt["split"] == "test"]
        P = [cpt[cpt["seed"] == s].set_index("instance_id").loc[tdf["instance_id"], PROB_COLS].values for s in seeds]
        preds["xlmr"] = np.vstack([p.argmax(1) for p in P])
        probas["xlmr"] = np.mean(P, axis=0)
    for b in [k for k in ("tfidf_lr", "xlmr") if k in preds]:
        f1s = [macro(yt, preds[b][[i]]) for i in range(len(seeds))]
        rows.append({"variant": b, "groups": "baseline", "macro_f1_mean": float(np.mean(f1s)),
                     "macro_f1_sd": float(np.std(f1s, ddof=1)),
                     **{f"macro_f1_{r}": macro(yt[(tdf["role"] == r).values], preds[b][:, (tdf["role"] == r).values]) for r in D.ROLES},
                     "macro_f1_non_cross_role": macro(yt[(tdf["cross_role"] == 0).values], preds[b][:, (tdf["cross_role"] == 0).values]),
                     "role_sensitivity": json.dumps(role_sensitivity(tdf, probas[b], np.round(preds[b].mean(0)).astype(int)))})
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)

    boots = thread_boot(tdf["thread_id"].values, args.n_boot)
    comp = []
    for fam, pairs in FAMILIES.items():
        fam_rows = []
        for a, b in pairs:
            if a not in preds or b not in preds:
                continue
            d = np.array([macro(yt, preds[a], i) - macro(yt, preds[b], i) for i in boots])
            fam_rows.append({"family": fam, "a": a, "b": b, "delta": macro(yt, preds[a]) - macro(yt, preds[b]),
                             "ci95_low": float(np.percentile(d, 2.5)), "ci95_high": float(np.percentile(d, 97.5)),
                             "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))})
        if fam_rows:
            ph = holm(np.array([r["p_two_sided"] for r in fam_rows]))
            for r, p in zip(fam_rows, ph):
                r["p_holm"] = float(p)
            comp += fam_rows
    pd.DataFrame(comp).to_csv(out / "comparisons.csv", index=False)

    pr = []
    for k in preds:
        for i, s in enumerate(seeds):
            pr.append(pd.DataFrame({"variant": k, "seed": s, "instance_id": tdf["instance_id"], "gold": yt, "pred": preds[k][i]}))
    pd.concat(pr).to_parquet(out / "predictions.parquet", index=False)
    write_json(out / "meta.json", {
        "labels": args.labels, "d1": args.d1, "d1_name": args.d1_name, "d1_missing": d1.attrs["missing"],
        "keep_instances": args.keep_instances,
        "coreader": args.coreader, "context": str(ctx_path), "n_dev": int(dev.sum()), "n_test": int(te.sum()),
        "label_dist_dev": np.bincount(y, minlength=3).tolist(), "label_dist_test": np.bincount(yt, minlength=3).tolist(),
        "columns": cols, "variants": {v: sorted(VARIANTS[v][0]) for v in variants}, "seeds": seeds,
        "protocol": "grouped_clean", "optuna_trials": rcfg["optuna_trials"], "n_boot": args.n_boot,
        "elapsed_s": round(time.time() - t0, 1), "git_commit": git_commit(repo)})
    logger.info("summary:\n%s", pd.DataFrame(rows).drop(columns=["role_sensitivity"]).round(4).to_string(index=False))
    logger.info("comparisons:\n%s", pd.DataFrame(comp).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
