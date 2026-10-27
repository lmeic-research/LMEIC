"""E1b: supplementary analyses on the closed human round (post hoc unless noted).

Reads the E1 output (items_with_labels.jsonl) and the round database, and adds:
  1. alt-test (Calderon, Reichart & Dror 2025) for the agent consensus and each round-0
     agent, epsilon 0.10 / 0.15 / 0.20 (literature note A, item 3 pre-registered 0.15 with 0.10-0.20 as sensitivity; D063/D065);
  2. per-agent and agent-consensus scores against humans, with a human ceiling
     (each annotator vs the other two when they agree);
  3. sensitivity without the 24 pilot items (planned) and on the clean subset;
  4. human-agent confusion by source and role, and the criteria humans ticked on the
     two main disagreement cells; agreement by attachment / body length.

Usage: python -m lmeic.human_study.supplement --e1-dir <E1 run dir>
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.metrics import cohen_kappa_score

from lmeic.human_study.annotation_store.export import package_paths
from lmeic.human_study.annotation_store.store import Store
from lmeic.human_study.agreement import (
    BOOT_SEED, LAB, LABELS, ci, macro_f1, mean_pairwise_kappa, mean_vs_reference_kappa)
from lmeic.common.rev1_common import git_commit, load_config, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("e1b")
_REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_IDS = "pilot_r2_item_ids.txt"
EPSILONS = (0.10, 0.15, 0.20)
N_BOOT = 2000


def alt_test(H: np.ndarray, f: np.ndarray, eps: float, q: float = 0.05) -> dict:
    """Alt-test with accuracy alignment; H items x 3 annotators, f items (LLM labels).

    For held-out annotator j, S(.) = share of the remaining annotators matching the label.
    W_f = 1[S(f) >= S(h_j)], W_h = 1[S(h_j) >= S(f)]; one-sided paired t-test of
    H0: mean(W_f - W_h) <= -eps. omega = share of annotators rejected after BY-FDR.
    """
    m = H.shape[1]
    pvals, rho_f, rho_h = [], [], []
    for j in range(m):
        others = [k for k in range(m) if k != j]
        s_f = np.mean([f == H[:, k] for k in others], axis=0)
        s_h = np.mean([H[:, j] == H[:, k] for k in others], axis=0)
        wf, wh = (s_f >= s_h).astype(float), (s_h >= s_f).astype(float)
        d = wf - wh
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        tstat = (d.mean() + eps) / se if se > 0 else np.inf
        pvals.append(float(1 - t_dist.cdf(tstat, n - 1)))
        rho_f.append(float(wf.mean()))
        rho_h.append(float(wh.mean()))
    # Benjamini-Yekutieli
    order = np.argsort(pvals)
    c_m = sum(1 / i for i in range(1, m + 1))
    rejected = np.zeros(m, bool)
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= rank * q / (m * c_m):
            rejected[order[:rank]] = True
    return {"epsilon": eps, "omega": float(rejected.mean()), "passes": bool(rejected.mean() >= 0.5),
            "rho_f_mean": float(np.mean(rho_f)), "rho_f": rho_f, "rho_h": rho_h, "p": pvals}


def kappa_ci(y, p, boots) -> list[float]:
    return ci(np.array([cohen_kappa_score(y[b], p[b], labels=[0, 1, 2]) for b in boots]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-dir", required=True)
    args = ap.parse_args()
    e1 = Path(args.e1_dir)
    if not e1.is_absolute():
        e1 = _REPO_ROOT / e1
    s1 = json.loads((e1 / "summary.json").read_text())
    if s1.get("smoke_test"):
        raise SystemExit("E1 run is a smoke test")
    round_id = s1["round"]
    paths, cfg = load_config("rev1_annotation_web.yaml")
    pkg, _, db_path = package_paths(paths, cfg, round_id)
    out = e1 / "e1b_supplement"
    out.mkdir(exist_ok=True)

    it = pd.read_json(e1 / "items_with_labels.jsonl", lines=True)
    store = Store(db_path, read_only=True)
    ann = pd.DataFrame([r for r in store.all_rows() if r["phase"] == "main"])
    annotators = list(cfg["annotators"])
    wide = ann.pivot_table(index="item_id", columns="annotator", values="label", aggfunc="first")
    it = it.set_index("item_id").loc[wide.index.intersection(it["item_id"])].reset_index()
    H = np.stack([wide.loc[it["item_id"], a].map(LAB).to_numpy(int) for a in annotators], axis=1)
    counts = np.stack([(H == c).sum(1) for c in range(3)], axis=1)
    maj = np.where(counts.max(1) >= 2, counts.argmax(1), -1)
    ok = it["agent_label"].notna().to_numpy() & np.array([v is not None for v in it["agent_votes_r0"]])
    agent = it["agent_label"].map(lambda x: LAB.get(x, -1)).to_numpy(int)
    votes = np.array([[LAB[x] for x in v] if v is not None else [-1, -1, -1] for v in it["agent_votes_r0"]])
    threads = it["thread_id"].to_numpy()

    rng = np.random.default_rng(BOOT_SEED)

    def boots_for(idx):
        th = threads[idx]
        uniq = np.unique(th)
        mem = {t: np.where(th == t)[0] for t in uniq}
        return [np.concatenate([mem[t] for t in rng.choice(uniq, len(uniq))]) for _ in range(N_BOOT)]

    res = {"round": round_id, "e1_dir": str(e1.relative_to(_REPO_ROOT)), "git_commit": git_commit(_REPO_ROOT),
           "status": {"alt_test": "planned (literature note A)", "no_pilot": "planned",
                      "others": "post hoc, descriptive"}}

    # 1. alt-test
    idx = np.where(ok)[0]
    res["alt_test"] = {}
    for name, f in [("agent_consensus", agent)] + [(f"agent_{k}", votes[:, k]) for k in range(3)]:
        res["alt_test"][name] = [alt_test(H[idx], f[idx], e) for e in EPSILONS]

    # 2. scores vs humans: agents, consensus, human ceiling
    sel = ok & (maj >= 0)
    idx = np.where(sel)[0]
    b = boots_for(idx)
    y = maj[idx]
    rows = []
    for name, f in [("agent_consensus", agent)] + [(f"agent_{k}", votes[:, k]) for k in range(3)]:
        p = f[idx]
        rows.append({"system": name, "n": len(idx), "kappa_vs_human_majority": cohen_kappa_score(y, p),
                     "kappa_ci": kappa_ci(y, p, b), "macro_f1_vs_human_majority": macro_f1(y, p),
                     "accuracy": float((y == p).mean()),
                     "mean_kappa_vs_each_human": mean_vs_reference_kappa(H, f, idx)})
    # human ceiling: annotator j vs the other two where they agree
    for j, a in enumerate(annotators):
        o = [k for k in range(3) if k != j]
        s = np.where(ok & (H[:, o[0]] == H[:, o[1]]))[0]
        yy, hj, ag = H[s, o[0]], H[s, j], agent[s]
        rows.append({"system": f"human_{a}_vs_other_two_agreeing", "n": len(s),
                     "kappa_vs_human_majority": cohen_kappa_score(yy, hj), "macro_f1_vs_human_majority": macro_f1(yy, hj),
                     "accuracy": float((yy == hj).mean()),
                     "agent_same_items_kappa": cohen_kappa_score(yy, ag), "agent_same_items_macro_f1": macro_f1(yy, ag)})
    res["scores_vs_humans"] = rows

    # 3. sensitivity subsets
    pilot = set((pkg / PILOT_IDS).read_text().split())
    subsets = {"all": np.ones(len(it), bool), "no_pilot24": ~it["item_id"].isin(pilot).to_numpy(),
               "clean_tier": (it["eval_tier"] == "clean").to_numpy() if "eval_tier" in it else None}
    for src in ("core_test", "cross_role", "fresh_holdout"):
        subsets[src] = (it["source"] == src).to_numpy()
    for r in ("Mgr", "CS", "Sch"):
        subsets[f"role_{r}"] = (it["role"] == r).to_numpy()
    res["subsets"] = {}
    for name, m in subsets.items():
        if m is None or m.sum() < 30:
            continue
        s = np.where(m & ok)[0]
        hh, ha = mean_pairwise_kappa(H, s), mean_vs_reference_kappa(H, agent, s)
        bb = boots_for(s)
        diff = np.array([mean_vs_reference_kappa(H, agent, s[x]) - mean_pairwise_kappa(H, s[x]) for x in bb[:1000]])
        res["subsets"][name] = {"n": int(len(s)), "human_human_kappa": hh, "human_agent_kappa": ha,
                                "difference": ha - hh, "difference_ci95_1000boot": ci(diff),
                                "majority_agent_agreement": float((maj[s] == agent[s]).mean()),
                                "human_dist": {LABELS[k]: float((maj[s] == k).mean()) for k in range(3)},
                                "agent_dist": {LABELS[k]: float((agent[s] == k).mean()) for k in range(3)}}

    # 4. confusion and criteria on the disagreement cells
    d = it.assign(maj=[LABELS[m] if m >= 0 else None for m in maj])[sel]
    conf = {}
    for src, g in [("all", d)] + list(d.groupby("source")):
        conf[src] = pd.crosstab(g["maj"], g["agent_label"]).reindex(index=LABELS, columns=LABELS, fill_value=0).to_dict("index")
    res["confusion_human_rows_agent_cols"] = conf
    checks = ann.set_index(["item_id", "annotator"])
    cells = {}
    for hlab, alab in [("Medium", "High"), ("Medium", "Low"), ("High", "Medium"), ("Low", "Medium")]:
        ids = d[(d["maj"] == hlab) & (d["agent_label"] == alab)]["item_id"]
        c, n_ann = Counter(), 0
        for i in ids:
            for a in annotators:
                if (i, a) in checks.index:
                    r = checks.loc[(i, a)]
                    if r["label"] == hlab:
                        n_ann += 1
                        c.update(r["high_checks"] + r["medium_checks"])
        cells[f"human_{hlab}__agent_{alab}"] = {"n_items": int(len(ids)), "n_majority_labels": n_ann,
                                                 "criteria_share": {k: round(v / max(n_ann, 1), 3) for k, v in c.most_common()}}
    res["disagreement_cells"] = cells
    body_len = it["item_id"].map(dict(zip(it["item_id"], it.get("body_len", pd.Series(dtype=float))))) if "body_len" in it else None
    res["by_attachment"] = {str(k): {"n": int(len(g)), "agree": float((g["maj"] == g["agent_label"]).mean())}
                            for k, g in d.groupby("has_attachment")}
    conf_mean = ann.groupby("item_id")["confidence"].mean()
    d = d.assign(conf=d["item_id"].map(conf_mean))
    d["conf_bin"] = pd.cut(d["conf"], [0, 3, 4, 4.67, 5], labels=["<=3", "3-4", "4-4.67", "5"])
    res["by_mean_human_confidence"] = {str(k): {"n": int(len(g)), "agree": float((g["maj"] == g["agent_label"]).mean())}
                                       for k, g in d.groupby("conf_bin", observed=True)}

    write_json(out / "summary.json", res)
    pd.DataFrame(rows).to_csv(out / "scores_vs_humans.csv", index=False)
    logger.info("output: %s", out)
    print(json.dumps({k: res[k] for k in ("alt_test",)}, indent=1)[:4000])
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    for k, v in res["subsets"].items():
        print(f"{k:14s} n={v['n']:4d} hh={v['human_human_kappa']:.3f} ha={v['human_agent_kappa']:.3f} "
              f"diff={v['difference']:+.3f} {np.round(v['difference_ci95_1000boot'], 3)} agree={v['majority_agent_agreement']:.3f}")
    print(json.dumps(cells, ensure_ascii=False, indent=1))
    print(res["by_attachment"], res["by_mean_human_confidence"])


if __name__ == "__main__":
    main()
