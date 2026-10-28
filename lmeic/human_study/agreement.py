"""
Human-annotation analysis for E1 (inter-annotator agreement and model re-scoring).
Version: v1 | Date: 2026-10-23

Reads one annotation round from the web tool database and reports:

  1. Agreement among the three human annotators (main block only): pairwise Cohen's
     kappa, quadratic-weighted kappa, Fleiss' kappa, Krippendorff's alpha (nominal and
     ordinal, Low < Medium < High), Gwet's AC1, per-class specific agreement.
  2. Human-agent comparison in the observational-equivalence framing: mean pairwise
     human-human kappa versus mean human-vs-agent-consensus kappa, the difference with a
     thread-clustered bootstrap CI, and Fleiss' kappa of the three humans versus the
     three round-0 agent votes on the same items.
  3. Model re-scoring: macro-F1 of each system against the human majority label and
     against the agent consensus label on the same items, per source, and Kendall's tau
     between the two system rankings (core test only, where every system is scored).
  4. Role sensitivity with human labels: on cross-role emails labelled for two or more
     roles, the share of role pairs whose human majority labels differ, against the
     share for agent labels and for model predictions on the same pairs.

Everything below is fixed before round r2 starts (see CARD); the round whose labels
enter the paper is fixed in the decision journal, not chosen from these outputs.
Frozen or pilot rounds can be read only with --smoke, which writes to tmp/ and marks
the output as a code-path test.

Usage:
  python -m lmeic.human_study.agreement --round r2
  python -m lmeic.human_study.agreement --round r1 --smoke
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.human_study.annotation_store.export import package_paths  # noqa: E402
from lmeic.human_study.annotation_store.store import Store  # noqa: E402
from lmeic.common.rev1_common import (  # noqa: E402
    git_commit, load_config, new_output_dir, norm_mid, read_jsonl, write_json,
)
from lmeic.common.data_loader import load_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("e1")

LABELS = ["High", "Medium", "Low"]
LAB = {l: i for i, l in enumerate(LABELS)}
ORD = {"Low": 0, "Medium": 1, "High": 2}
EVAL = "private/runs/rev1_eval_sets_20261022_161427"
V1_VOTES = "private/runs/run_20260526_123858/phase1_annotation/annotations_full.jsonl"
AGENT_DIR = "private/runs/rev1_agent_labels_20261022_192927"
REF = "private/runs/rev1_reference_20261022_191130"
N_BOOT = 2000
BOOT_SEED = 20261024
EQUIV_MARGIN = 0.10

# Systems scored against human labels. "ext": predictions with set/id for test and the
# external sets; "test": test-split predictions only (scored on core-test items).
SYSTEMS = {
    "R2_explicit": ("ext", f"{REF}/R2_v3_grouped_explicit/external_20261022_211532/external_predictions_all_seeds.parquet"),
    "R3_full": ("ext", f"{REF}/R3_v3_grouped_full/external_20261022_225607/external_predictions_all_seeds.parquet"),
    "E11_lora_qwen25_role": ("ext", "private/runs/rev1_e11_lora_role_external_20261023/qwen25_7b_lora_role_s42/external_predictions.parquet"),
    "E14_both_explicit": ("test", f"{REF}/E14_both_extractors_explicit_R2params/test_predictions_all_seeds.parquet"),
    "E14_both_full": ("test", f"{REF}/E14_both_extractors_full_R3params/test_predictions_all_seeds.parquet"),
    "E9_explicit_d1_only": ("test", f"{REF}/E9_explicit_d1_only/test_predictions_all_seeds.parquet"),
}

CARD = {
    "experiment": "E1 human annotation analysis",
    "answers": ["R2.4", "R3.2", "R1.2", "R3.5", "E"],
    "items": "main block of the annotation package: core test 601, cross-role 429 (229 emails), "
             "fresh hold-out 150; calibration items excluded",
    "human_reference": "majority of three annotators; 1-1-1 items excluded from hard-label metrics",
    "agent_reference": "core test: v1.0 label_final; cross-role and hold-out: revision-1 panel, "
                       "latest record with three round-0 votes (same rule as eval_external)",
    "primary": ["mean pairwise human Cohen's kappa", "Krippendorff's alpha (ordinal) among humans",
                "human-agent minus human-human kappa, thread-clustered 95% CI"],
    "equivalence_rule": f"the agent consensus is described as agreeing with humans about as well as "
                        f"humans agree with each other if the 95% CI of (human-agent kappa - "
                        f"human-human kappa) lies within +/-{EQUIV_MARGIN}; otherwise the direction "
                        f"and size of the gap are reported",
    "model_rescoring": "macro-F1 against human majority vs agent consensus on the same items; "
                       "Kendall's tau of system rankings on core test",
    "role_sensitivity": "share of cross-role role pairs with differing labels: humans, agents, models",
    "bootstrap": {"unit": "thread", "n_boot": N_BOOT, "seed": BOOT_SEED},
    "fixed_before": "round r2 annotation (2026-10-23)",
}


# ---- agreement statistics ------------------------------------------------------------

def fleiss_kappa(counts: np.ndarray) -> float:
    """counts: (n_items, n_classes) rater counts per item, equal raters per item."""
    n = counts.sum(1)[0]
    p_j = counts.sum(0) / counts.sum()
    p_i = ((counts * (counts - 1)).sum(1)) / (n * (n - 1))
    pe = (p_j ** 2).sum()
    return float((p_i.mean() - pe) / (1 - pe)) if pe < 1 else float("nan")


def gwet_ac1(counts: np.ndarray) -> float:
    n = counts.sum(1)[0]
    q = counts.shape[1]
    pa = ((counts * (counts - 1)).sum(1) / (n * (n - 1))).mean()
    pi = (counts / n).mean(0)
    pe = (pi * (1 - pi)).sum() / (q - 1)
    return float((pa - pe) / (1 - pe))


def specific_agreement(labels: np.ndarray) -> dict[str, float]:
    """Per-class specific agreement over all rater pairs (labels: items x raters)."""
    out = {}
    for c, name in enumerate(LABELS):
        agree = total = 0
        for a, b in itertools.combinations(range(labels.shape[1]), 2):
            x, y = labels[:, a] == c, labels[:, b] == c
            agree += 2 * (x & y).sum()
            total += x.sum() + y.sum()
        out[name] = float(agree / total) if total else float("nan")
    return out


def to_counts(labels: np.ndarray) -> np.ndarray:
    return np.stack([(labels == c).sum(1) for c in range(3)], axis=1)


def mean_pairwise_kappa(labels: np.ndarray, idx: np.ndarray, weights=None) -> float:
    ks = [cohen_kappa_score(labels[idx, a], labels[idx, b], labels=[0, 1, 2], weights=weights)
          for a, b in itertools.combinations(range(labels.shape[1]), 2)]
    return float(np.mean(ks))


def mean_vs_reference_kappa(labels: np.ndarray, ref: np.ndarray, idx: np.ndarray) -> float:
    return float(np.mean([cohen_kappa_score(labels[idx, a], ref[idx], labels=[0, 1, 2])
                          for a in range(labels.shape[1])]))


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3).astype(float)
    tp = np.diag(cm)
    prec, rec = tp / np.maximum(cm.sum(0), 1), tp / np.maximum(cm.sum(1), 1)
    return float(np.mean(2 * prec * rec / np.maximum(prec + rec, 1e-12)))


def ci(values: np.ndarray) -> list[float]:
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


# ---- data assembly -------------------------------------------------------------------

def load_items(key_path: Path) -> pd.DataFrame:
    """Main-block items with source, thread, instance id and agent references."""
    key = [r for r in read_jsonl(key_path) if r["source"] != "calibration_train"]
    df = pd.DataFrame(key)
    df["mid"] = df["message_id"].map(norm_mid)
    te = load_split("test")
    test_thread = dict(zip(te["message_id"].map(norm_mid), te["thread_id"]))
    inst = {}
    for name in ("cross_role", "fresh_holdout"):
        for r in read_jsonl(_REPO_ROOT / EVAL / f"{name}_instances.jsonl.gz"):
            inst[(name, norm_mid(r["message_id"]), r["role"])] = (str(r["instance_id"]), r["thread_id"])
    ids, threads = [], []
    for r in df.itertuples():
        if r.source == "core_test":
            ids.append(r.mid)
            threads.append(test_thread[r.mid])
        else:
            iid, th = inst[(r.source, r.mid, r.role)]
            ids.append(iid)
            threads.append(th)
    df["pred_set"] = df["source"].map({"core_test": "test", "cross_role": "cross_role",
                                       "fresh_holdout": "fresh_holdout"})
    df["pred_id"], df["thread_id"] = ids, threads

    # agent consensus and round-0 votes
    v1 = {}
    for r in read_jsonl(_REPO_ROOT / V1_VOTES):
        v1[(norm_mid(r["email_id"]), r["role"])] = r
    rev = {}
    for name in ("cross_role", "fresh_holdout"):
        latest = {}
        for r in read_jsonl(_REPO_ROOT / AGENT_DIR / f"agent_labels_{name}.jsonl"):
            latest[str(r["id"])] = r
        rev[name] = latest
    cons, votes, panel = [], [], []
    for r in df.itertuples():
        if r.source == "core_test":
            a = v1[(r.mid, r.role)]
            cons.append(a["label_final"])
            panel.append("v1.0")
        else:
            a = rev[r.source].get(r.pred_id)
            ok = a is not None and a.get("n_votes_round0") == 3
            cons.append(a["label_final"] if ok else None)
            panel.append("rev1")
        v0 = [v["label"] for v in (a or {}).get("agent_votes", []) if v.get("round", 0) == 0]
        votes.append(v0 if len(v0) == 3 else None)
    df["agent_label"], df["agent_votes_r0"], df["agent_panel"] = cons, votes, panel
    return df


def load_system(kind: str, path: Path) -> dict[tuple[str, str], int]:
    """(set, id) -> argmax of seed-averaged probabilities."""
    df = pd.read_parquet(path)
    if kind == "test":
        df = df.assign(set="test", id=df["message_id"])
    df["id"] = df["id"].map(lambda x: norm_mid(str(x)))
    g = df.groupby(["set", "id"])[["p_high", "p_med", "p_low"]].mean()
    return dict(zip(g.index, g.to_numpy().argmax(1)))


# ---- analysis ------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--smoke", action="store_true", help="code-path test on a frozen/pilot round")
    args = ap.parse_args()
    paths, cfg = load_config("rev1_annotation_web.yaml")
    rcfg = cfg["rounds"][args.round]
    analysable = rcfg["status"] in ("open", "closed") and not args.round.endswith("_pilot")
    if not analysable and not args.smoke:
        raise SystemExit(f"round {args.round} ({rcfg['status']}) is not analysable; use --smoke for a code test")
    pkg, key_path, db_path = package_paths(paths, cfg, args.round)
    if args.smoke:
        out = _REPO_ROOT / "tmp" / f"e1_smoke_{args.round}_{time.strftime('%Y%m%d_%H%M%S')}"
        out.mkdir(parents=True)
    else:
        out = new_output_dir(paths, f"rev1_e1_human_{args.round}")
    write_json(out / "experiment_card.json", {**CARD, "round": args.round, "smoke_test": args.smoke})

    store = Store(db_path, read_only=True)
    ann = pd.DataFrame([r for r in store.all_rows() if r["phase"] == "main"])
    annotators = list(cfg["annotators"])
    wide = ann.pivot_table(index="item_id", columns="annotator", values="label", aggfunc="first")
    items = load_items(key_path).set_index("item_id")
    done = [i for i in wide.index if wide.loc[i, annotators].notna().all()]
    logger.info("round %s: %d main items with all three annotators (of %d in package)",
                args.round, len(done), len(items))
    if not done:
        raise SystemExit("no fully annotated items yet")
    it = items.loc[done].copy()
    H = np.stack([wide.loc[done, a].map(LAB).to_numpy(int) for a in annotators], axis=1)
    counts = to_counts(H)
    maj = np.where(counts.max(1) >= 2, counts.argmax(1), -1)
    it["human_majority"] = [LABELS[m] if m >= 0 else None for m in maj]
    it["human_pattern"] = ["3-0" if c.max() == 3 else ("2-1" if c.max() == 2 else "1-1-1") for c in counts]
    has_agent = it["agent_label"].notna().to_numpy()
    agent = it["agent_label"].map(lambda x: LAB.get(x, -1)).to_numpy(int)
    threads = it["thread_id"].to_numpy()

    rng = np.random.default_rng(BOOT_SEED)
    uniq = np.unique(threads)
    members = {t: np.where(threads == t)[0] for t in uniq}
    boots = [np.concatenate([members[t] for t in rng.choice(uniq, len(uniq), replace=True)])
             for _ in range(N_BOOT)]

    # 1. human agreement, overall and per source
    def agreement_block(sel: np.ndarray) -> dict:
        idx = np.where(sel)[0]
        h = H[idx]
        c = to_counts(h)
        return {
            "n_items": int(len(idx)),
            "pairwise_cohen_kappa": {f"{annotators[a]}-{annotators[b]}": float(cohen_kappa_score(h[:, a], h[:, b]))
                                     for a, b in itertools.combinations(range(3), 2)},
            "mean_pairwise_kappa": mean_pairwise_kappa(H, idx),
            "mean_pairwise_quadratic_kappa": mean_pairwise_kappa(H, idx, weights="quadratic"),
            "fleiss_kappa": fleiss_kappa(c),
            "krippendorff_alpha_nominal": float(krippendorff.alpha(h.T, level_of_measurement="nominal")),
            "krippendorff_alpha_ordinal": float(krippendorff.alpha(
                np.vectorize(lambda v: ORD[LABELS[v]])(h).T, level_of_measurement="ordinal")),
            "gwet_ac1": gwet_ac1(c),
            "specific_agreement": specific_agreement(h),
            "pattern": dict(Counter(it["human_pattern"].to_numpy()[idx].tolist())),
            "label_share_by_annotator": {annotators[a]: {LABELS[k]: float((h[:, a] == k).mean()) for k in range(3)}
                                         for a in range(3)},
        }

    agreement = {"all": agreement_block(np.ones(len(it), bool))}
    for src in ("core_test", "cross_role", "fresh_holdout"):
        sel = (it["source"] == src).to_numpy()
        if sel.sum() >= 10:
            agreement[src] = agreement_block(sel)
    b = np.array([mean_pairwise_kappa(H, i) for i in boots])
    agreement["all"]["mean_pairwise_kappa_ci"] = ci(b)
    b = np.array([krippendorff.alpha(np.vectorize(lambda v: ORD[LABELS[v]])(H[i]).T,
                                     level_of_measurement="ordinal") for i in boots[:500]])
    agreement["all"]["krippendorff_alpha_ordinal_ci_500boot"] = ci(b)

    # 2. human-agent comparison
    ia = np.where(has_agent)[0]
    hh = mean_pairwise_kappa(H, ia)
    ha = mean_vs_reference_kappa(H, agent, ia)
    diffs = []
    for i in boots:
        j = i[has_agent[i]]
        diffs.append(mean_vs_reference_kappa(H, agent, j) - mean_pairwise_kappa(H, j))
    diffs = np.array(diffs)
    lo, hi = ci(diffs)
    av = it["agent_votes_r0"].to_numpy()
    iv = np.array([v is not None for v in av]) & has_agent
    A = np.array([[LAB[x] for x in v] for v in av[iv]]) if iv.any() else np.zeros((0, 3), int)
    human_agent = {
        "n_items": int(len(ia)),
        "mean_human_human_kappa": hh,
        "mean_human_vs_agent_consensus_kappa": ha,
        "difference": ha - hh, "difference_ci95": [lo, hi],
        "within_margin": bool(-EQUIV_MARGIN <= lo and hi <= EQUIV_MARGIN), "margin": EQUIV_MARGIN,
        "human_majority_vs_agent_agreement": float((maj[ia] == agent[ia]).mean()),
        "fleiss_humans_on_agent_vote_items": fleiss_kappa(to_counts(H[iv])) if iv.any() else None,
        "fleiss_agents_round0_same_items": fleiss_kappa(to_counts(A)) if iv.any() else None,
        "n_items_with_agent_votes": int(iv.sum()),
        "by_agent_pattern": {},
    }
    if iv.any():
        apat = np.array(["3-0" if c.max() == 3 else "split" for c in to_counts(A)])
        sub = np.where(iv)[0]
        for p in ("3-0", "split"):
            s = sub[apat == p]
            if len(s) >= 10:
                human_agent["by_agent_pattern"][p] = {
                    "n": int(len(s)), "mean_human_human_kappa": mean_pairwise_kappa(H, s),
                    "human_majority_vs_agent": float((maj[s] == agent[s]).mean()),
                    "human_unanimous_share": float((to_counts(H[s]).max(1) == 3).mean())}

    # 3. model re-scoring
    sys_preds = {}
    for name, (kind, rel) in SYSTEMS.items():
        p = _REPO_ROOT / rel
        if p.exists():
            sys_preds[name] = load_system(kind, p)
        else:
            logger.warning("skip %s: %s missing", name, rel)
    scored = []
    keys = list(zip(it["pred_set"], it["pred_id"]))
    for src in ("core_test", "cross_role", "fresh_holdout", "all"):
        sel = np.ones(len(it), bool) if src == "all" else (it["source"] == src).to_numpy()
        sel = sel & has_agent & (maj >= 0)
        for name, lut in sys_preds.items():
            have = np.array([k in lut for k in keys]) & sel
            if have.sum() < 30 or (src == "all" and not have[sel].all()):
                continue
            idx = np.where(have)[0]
            pr = np.array([lut[keys[i]] for i in idx])
            scored.append({"source": src, "system": name, "n": int(len(idx)),
                           "macro_f1_vs_human_majority": macro_f1(maj[idx], pr),
                           "macro_f1_vs_agent_consensus": macro_f1(agent[idx], pr),
                           "accuracy_vs_human_majority": float((maj[idx] == pr).mean())})
    sc = pd.DataFrame(scored)
    tau = None
    core = sc[sc.source == "core_test"] if len(sc) else sc
    if len(core) >= 3:
        t = kendalltau(core["macro_f1_vs_human_majority"], core["macro_f1_vs_agent_consensus"])
        tau = {"tau": float(t.statistic), "p": float(t.pvalue), "n_systems": int(len(core)),
               "note": "few systems; descriptive only"}

    # 4. role sensitivity on cross-role emails
    cr = it[(it["source"] == "cross_role") & (maj >= 0)].copy()
    cr["maj"] = maj[(it["source"] == "cross_role").to_numpy() & (maj >= 0)]
    pairs = []
    for mid, g in cr.groupby("mid"):
        if len(g) < 2:
            continue
        for (ia_, ra), (ib_, rb) in itertools.combinations(g.iterrows(), 2):
            row = {"mid": mid, "roles": f"{ra.role}|{rb.role}", "tier": ra.eval_tier,
                   "human_differ": int(ra.maj != rb.maj)}
            if ra.agent_label and rb.agent_label:
                row["agent_differ"] = int(ra.agent_label != rb.agent_label)
            for name, lut in sys_preds.items():
                ka, kb = (ra.pred_set, ra.pred_id), (rb.pred_set, rb.pred_id)
                if ka in lut and kb in lut:
                    row[f"{name}_differ"] = int(lut[ka] != lut[kb])
            pairs.append(row)
    pr_df = pd.DataFrame(pairs)
    role_sens = {}
    if len(pr_df):
        cols = [c for c in pr_df.columns if c.endswith("_differ")]
        role_sens["all"] = {"n_pairs": int(len(pr_df)), **{c: float(pr_df[c].mean()) for c in cols}}
        for tier, g in pr_df.groupby("tier"):
            role_sens[tier] = {"n_pairs": int(len(g)), **{c: float(g[c].mean()) for c in cols}}
        hd = pr_df[pr_df["human_differ"] == 1]
        role_sens["on_pairs_where_humans_differ"] = {
            "n_pairs": int(len(hd)), **{c: float(hd[c].mean()) for c in cols if c != "human_differ"}}

    # confidence
    conf = ann.groupby("annotator")["confidence"].agg(["mean", "median"]).round(3).to_dict("index")

    it.reset_index().drop(columns=["email_body", "subject"], errors="ignore").to_json(
        out / "items_with_labels.jsonl", orient="records", lines=True, force_ascii=False)
    if len(sc):
        sc.to_csv(out / "model_rescoring.csv", index=False)
    if len(pr_df):
        pr_df.to_csv(out / "role_pairs.csv", index=False)
    summary = {"round": args.round, "smoke_test": args.smoke, "n_items_complete": int(len(it)),
               "agreement": agreement, "human_agent": human_agent, "kendall_tau_core": tau,
               "role_sensitivity": role_sens, "confidence_by_annotator": conf,
               "git_commit": git_commit(_REPO_ROOT)}
    write_json(out / "summary.json", summary)
    logger.info("output: %s", out)
    a = agreement["all"]
    print(f"[{'SMOKE TEST' if args.smoke else 'ANALYSIS'} round {args.round}] items {len(it)}")
    print(f"  human mean pairwise kappa {a['mean_pairwise_kappa']:.3f} {a['mean_pairwise_kappa_ci']} | "
          f"Fleiss {a['fleiss_kappa']:.3f} | alpha ord {a['krippendorff_alpha_ordinal']:.3f} | AC1 {a['gwet_ac1']:.3f}")
    print(f"  human-agent kappa {ha:.3f} vs human-human {hh:.3f} | diff {ha - hh:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    if len(sc):
        print(sc.round(4).to_string(index=False))
    for k, v in role_sens.items():
        print(f"  role sensitivity {k}: {v}")


if __name__ == "__main__":
    main()
