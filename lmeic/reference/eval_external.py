"""
Evaluation of external-set predictions against agent labels and the manager's filing.
Version: v1.1 | Date: 2026-10-22

Fresh hold-out: macro-F1 against the revision-1 agent labels, overall, per role and on
the sender-unseen subset (sender absent from the development pool).
Cross-role: macro-F1 by tier and by real/counterfactual instance; pairwise role
concordance - for two role instances of the same email whose reference labels differ,
the share where the model's expected importance (2*p_high + p_med) orders them the
same way (ties count one half); and the share of such pairs where the predicted
labels differ at all.
E1b: ROC-AUC of p_high and of the expected importance for important vs not-important
(scorable instances only), and the agent panel's High rate per manager label.
All model metrics are averaged over seeds. Fresh hold-out macro-F1 and the E1b AUCs carry
95% percentile intervals from a sender-clustered bootstrap (senders resampled with
replacement; the seed average is recomputed in every replicate).

Usage:
  python -m lmeic.reference.eval_external --pred <external_predictions_all_seeds.parquet>
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.common.rev1_common import (  # noqa: E402
    git_commit, load_config, norm_mid, read_jsonl, source_run_dir, write_json,
)
from lmeic.common.data_loader import load_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
CONFIG_NAME = "rev1_reference.yaml"
ORD = {"High": 2, "Medium": 1, "Low": 0}


def mf1(df: pd.DataFrame, ref: str = "ref") -> float:
    if df.empty:
        return float("nan")
    return float(np.mean([f1_score(g[ref], g["pred"], average="macro", labels=list(ORD))
                          for _, g in df.groupby("seed")]))


def cluster_boot(df: pd.DataFrame, cluster: str, stat, n_boot: int, seed: int = 2026) -> dict:
    """Percentile interval of stat(df) when clusters (e.g. senders) are resampled."""
    keys = df[cluster].fillna("").to_numpy()
    groups = {k: np.flatnonzero(keys == k) for k in np.unique(keys)}
    names = list(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(names), size=len(names), replace=True)
        b = df.iloc[np.concatenate([groups[names[i]] for i in pick])].reset_index(drop=True)
        try:
            v = stat(b)
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"), float("nan"))
    return {"point": float(stat(df)), "ci95": [float(lo), float(hi)], "n_clusters": len(names),
            "n_boot_valid": len(vals)}


def auc_by_seed(col: str):
    def f(d: pd.DataFrame) -> float:
        y = (d["manager_label"] == "important").astype(int)
        return float(np.mean([roc_auc_score(y[g.index], g[col]) for _, g in d.groupby("seed")]))
    return f


def role_concordance(df: pd.DataFrame) -> dict:
    conc, diff_pred, n = [], [], 0
    for (_, _), g in df.groupby(["seed", "message_id"]):
        if len(g) < 2:
            continue
        for a, b in itertools.combinations(g.itertuples(), 2):
            if a.ref == b.ref:
                continue
            ra, rb = ORD[a.ref] - ORD[b.ref], a.score - b.score
            conc.append(0.5 if rb == 0 else float(np.sign(ra) == np.sign(rb)))
            diff_pred.append(float(a.pred != b.pred))
            n += 1
    n_seeds = df["seed"].nunique()
    return {"n_pairs": n // max(n_seeds, 1), "concordance": float(np.mean(conc)) if conc else float("nan"),
            "pred_label_differs": float(np.mean(diff_pred)) if diff_pred else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    pred_path = Path(args.pred)
    paths, cfg = load_config(CONFIG_NAME)
    repo = Path(paths["repo_root"])
    es = cfg["external_sets"]
    src = source_run_dir(paths, cfg["source_run_id"])

    P = pd.read_parquet(pred_path)
    P["score"] = 2 * P["p_high"] + P["p_med"]
    agent = {}
    for name in es["sets"]:
        f = repo / es["agent_dir"] / f"agent_labels_{name}.jsonl"
        if f.exists():
            latest = {}
            for r in read_jsonl(f):
                latest[str(r["id"])] = r
            agent[name] = {i: r["label_final"] for i, r in latest.items() if r.get("n_votes_round0") == 3}
    inst = {name: pd.DataFrame(read_jsonl(repo / rel)).assign(id=lambda d: d["instance_id"].astype(str))
            for name, rel in es["sets"].items()}
    res: dict = {"pred": str(pred_path), "git_commit": git_commit(repo)}

    dev = pd.concat([load_split("train"), load_split("val")])
    sender = {norm_mid(r["message_id"]): (r.get("from_addr") or "").strip().lower()
              for r in read_jsonl(src / cfg["analysis"]["sampled_relpath"])}
    dev_senders = {sender[norm_mid(m)] for m in dev["message_id"]}

    if "fresh_holdout" in agent:
        d = P[P["set"] == "fresh_holdout"].merge(inst["fresh_holdout"][["id", "role", "from_addr"]], on="id")
        d["ref"] = d["id"].map(agent["fresh_holdout"])
        d = d.dropna(subset=["ref"])
        d["unseen"] = ~d["from_addr"].fillna("").str.strip().str.lower().isin(dev_senders)
        res["fresh_holdout"] = {"n_labelled": int(d["id"].nunique()), "macro_f1": mf1(d),
                                "per_role": {r: mf1(d[d["role"] == r]) for r in ["Mgr", "CS", "Sch"]},
                                "sender_unseen": {"n": int(d[d["unseen"]]["id"].nunique()),
                                                  "macro_f1": mf1(d[d["unseen"]])},
                                "sender_seen": mf1(d[~d["unseen"]]),
                                "ref_dist": d.drop_duplicates("id")["ref"].value_counts().to_dict()}
        d = d.reset_index(drop=True)
        res["fresh_holdout"]["macro_f1_sender_boot"] = cluster_boot(d, "from_addr", mf1, args.n_boot)
        res["fresh_holdout"]["sender_unseen"]["macro_f1_sender_boot"] = cluster_boot(
            d[d["unseen"]].reset_index(drop=True), "from_addr", mf1, args.n_boot)

    if "cross_role" in agent:
        cols = ["id", "role", "message_id", "eval_tier", "is_counterfactual"]
        d = P[P["set"] == "cross_role"].merge(inst["cross_role"][cols], on="id")
        d["ref"] = d["id"].map(agent["cross_role"])
        d = d.dropna(subset=["ref"])
        res["cross_role"] = {
            "n_labelled": int(d["id"].nunique()), "macro_f1": mf1(d),
            "by_tier": {t: mf1(g) for t, g in d.groupby("eval_tier")},
            "real": mf1(d[~d["is_counterfactual"]]), "counterfactual": mf1(d[d["is_counterfactual"]]),
            "role_concordance_all": role_concordance(d),
            "role_concordance_real_only": role_concordance(d[~d["is_counterfactual"]]),
            "role_concordance_by_tier": {t: role_concordance(g) for t, g in d.groupby("eval_tier")},
        }

    d = P[P["set"] == "e1b"].merge(inst["e1b"][["id", "manager_label", "scorable", "from_addr"]], on="id")
    d = d[d["scorable"]].reset_index(drop=True)
    yb = (d["manager_label"] == "important").astype(int)
    by_seed = [(roc_auc_score(yb[g.index], g["p_high"]), roc_auc_score(yb[g.index], g["score"]),
                float(((g["pred"] == "High") == (yb[g.index] == 1)).mean()))
               for _, g in d.groupby("seed")]
    res["e1b"] = {"n_scorable": int(d["id"].nunique()),
                  "n_important": int(d.drop_duplicates("id")["manager_label"].eq("important").sum()),
                  "auc_p_high": float(np.mean([b[0] for b in by_seed])),
                  "auc_expected_importance": float(np.mean([b[1] for b in by_seed])),
                  "binary_accuracy_high_vs_rest": float(np.mean([b[2] for b in by_seed])),
                  "pred_dist_by_manager_label": d.groupby("manager_label")["pred"].value_counts(normalize=True)
                  .round(4).unstack(fill_value=0).to_dict(orient="index")}
    res["e1b"]["n_senders_by_manager_label"] = (d.drop_duplicates("id").groupby("manager_label")["from_addr"]
                                                .nunique().to_dict())
    res["e1b"]["auc_p_high_sender_boot"] = cluster_boot(d, "from_addr", auc_by_seed("p_high"), args.n_boot)
    res["e1b"]["auc_expected_importance_sender_boot"] = cluster_boot(
        d, "from_addr", auc_by_seed("score"), args.n_boot)
    if "e1b" in agent:
        e = inst["e1b"].assign(agent=lambda x: x["id"].map(agent["e1b"])).dropna(subset=["agent"])
        res["e1b"]["agent_label_dist_by_manager_label"] = (
            e.groupby("manager_label")["agent"].value_counts(normalize=True).round(4)
            .unstack(fill_value=0).to_dict(orient="index"))

    out = pred_path.parent / f"eval_external_{time.strftime('%Y%m%d_%H%M%S')}.json"
    write_json(out, res)
    logger.info("saved %s", out)
    for k in ("fresh_holdout", "cross_role", "e1b"):
        if k in res:
            logger.info("%s: %s", k, {kk: vv for kk, vv in res[k].items() if not isinstance(vv, dict)})


if __name__ == "__main__":
    main()
