"""
Soft-label evaluation against the agent panel's vote distribution (E16).
Version: v1 | Date: 2026-10-23

Hard macro-F1 counts every disagreement with label_final as an error, including
items on which the three-agent panel itself split. Following the perspectivist
and unreliable-label evaluation literature, this script reports, for each system
with stored test probabilities:

  - macro-F1 separately on unanimous and split (majority / adjudicated) items;
  - Jensen-Shannon divergence (base 2) between the model's class probabilities
    and the panel's round-0 vote distribution;
  - soft cross-entropy of the model probabilities against that distribution;
  - acceptable-label rate: the predicted label received at least one panel vote;
  - AUROC of predictive entropy for detecting split items (does the model know
    which items are ambiguous?).

Round-0 votes are the three independent agent judgements; votes cast after the
debate round (174 of 15,000 instances) have already converged and are not used
for the distribution. "Split" means the three round-0 votes were not identical;
the panel's status field is post-debate and is reported only for reference. Thread-clustered bootstrap gives CIs and paired
differences; Holm correction within each metric family.

Usage:
  python -m lmeic.analysis.e16_soft_label_eval
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.common.rev1_common import (  # noqa: E402
    git_commit, load_config, new_output_dir, norm_mid, write_json,
)
from lmeic.common.data_loader import load_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("e16")

LABELS = ["High", "Medium", "Low"]
LAB = {l: i for i, l in enumerate(LABELS)}
REF = "private/runs/rev1_reference_20261022_191130"
VOTES = "private/runs/run_20260526_123858/phase1_annotation/annotations_full.jsonl"
N_BOOT = 2000
BOOT_SEED = 20261023

# kind "seeds": parquet with seed/message_id/p_high/p_med/p_low; kind "plm": parquet with
# message_id/split/plm_p_* (single model). Fixed before running; nothing selected on test.
SYSTEMS = {
    "R2_explicit": ("seeds", f"{REF}/R2_v3_grouped_explicit/test_predictions_all_seeds.parquet"),
    "R3_full": ("seeds", f"{REF}/R3_v3_grouped_full/test_predictions_all_seeds.parquet"),
    "E14_both_explicit": ("seeds", f"{REF}/E14_both_extractors_explicit_R2params/test_predictions_all_seeds.parquet"),
    "E14_both_full": ("seeds", f"{REF}/E14_both_extractors_full_R3params/test_predictions_all_seeds.parquet"),
    "E11_lora_qwen25_role": ("plm", "private/runs/rev1_e11_llm_baselines_20261023/qwen25_7b_lora_role_s42/test_probs.parquet"),
    "text_xlmr_base_s42": ("plm", f"{REF}/coreader_bank/xlmr_base_s42/probs.parquet"),
    "text_xlmr_large_s42": ("plm", f"{REF}/coreader_bank/xlmr_large_s42/probs.parquet"),
    "text_lora_qwen25_s42": ("plm", f"{REF}/coreader_bank/qwen25_7b_lora_text_s42/probs.parquet"),
}
COMPARISONS = [
    ("R3_full", "E11_lora_qwen25_role"),
    ("E14_both_full", "R3_full"),
    ("E14_both_full", "E11_lora_qwen25_role"),
    ("R3_full", "R2_explicit"),
    ("R3_full", "text_xlmr_base_s42"),
]

CARD = {
    "experiment": "E16 soft-label evaluation",
    "question": "How much of the hard-label error falls on items the agent panel itself disputed, "
                "and how do systems compare when the panel's vote distribution is the target?",
    "answers": ["R3.2", "R2.4", "R3.3", "E (accurate reporting)"],
    "test_set": "EMAIL-3R v1.0 test split, 2,489 instances (unchanged)",
    "soft_target": "round-0 votes of the v1.0 three-agent panel (DeepSeek-V3, GPT-4o-mini, Claude 3.5 Sonnet)",
    "systems": list(SYSTEMS),
    "primary_metrics": ["JSD to vote distribution (lower is better)", "macro-F1 on split items"],
    "secondary_metrics": ["macro-F1 on unanimous items", "soft cross-entropy",
                          "acceptable-label rate", "entropy AUROC for split items"],
    "statistics": f"thread-clustered bootstrap, {N_BOOT} resamples, seed {BOOT_SEED}; "
                  "paired differences with Holm correction within each metric",
    "acceptance": "descriptive analysis; no system or threshold is selected on test",
    "compute": "CPU only, existing predictions",
}


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3).astype(float)
    tp = np.diag(cm)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    return float(np.mean(2 * prec * rec / np.maximum(prec + rec, 1e-12)))


def jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)

    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(a > 0, a * np.log2(a / np.maximum(b, 1e-12)), 0.0)
        return t.sum(1)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def load_votes(test_keys: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Vote distribution (n, 3) from round-0 votes and panel status per test instance."""
    idx = {}
    for line in open(_REPO_ROOT / VOTES, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            idx[(norm_mid(r["email_id"]), r["role"])] = r
    dist = np.zeros((len(test_keys), 3))
    status = []
    for i, k in enumerate(test_keys):
        r = idx[k]
        votes = [v["label"] for v in r["agent_votes"] if v.get("round", 0) == 0]
        assert len(votes) == 3, (k, len(votes))
        for v in votes:
            dist[i, LAB[v]] += 1 / 3
        status.append(r["status"])
    return dist, np.array(status)


def load_system(kind: str, path: Path, mids: list[str]) -> np.ndarray:
    """Return probabilities (n_seeds, n_test, 3) aligned to mids."""
    df = pd.read_parquet(path)
    if kind == "seeds":
        out = []
        for s in sorted(df["seed"].unique()):
            m = df[df["seed"] == s].copy()
            m["k"] = m["message_id"].map(norm_mid)
            m = m.drop_duplicates("k").set_index("k")
            out.append(m.loc[mids, ["p_high", "p_med", "p_low"]].to_numpy(float))
        return np.stack(out)
    m = df[df["split"] == "test"].copy()
    m["k"] = m["message_id"].map(norm_mid)
    m = m.drop_duplicates("k").set_index("k")
    return m.loc[mids, ["plm_p_high", "plm_p_med", "plm_p_low"]].to_numpy(float)[None]


def per_item(P: np.ndarray, dist: np.ndarray) -> dict[str, np.ndarray]:
    """Per-instance values averaged over seeds."""
    Pn = P / P.sum(-1, keepdims=True)
    j = np.mean([jsd(p, dist) for p in Pn], axis=0)
    ce = np.mean([-(dist * np.log(np.clip(p, 1e-7, 1))).sum(1) for p in Pn], axis=0)
    acc_ok = np.mean([dist[np.arange(len(dist)), p.argmax(1)] > 0 for p in Pn], axis=0)
    return {"jsd": j, "soft_ce": ce, "acceptable": acc_ok}


def main() -> None:
    paths, _ = load_config("rev1_reference.yaml")
    out = new_output_dir(paths, "rev1_soft_label_eval")
    write_json(out / "experiment_card.json", CARD)
    t0 = time.time()

    te = load_split("test")
    mids = [norm_mid(m) for m in te["message_id"]]
    keys = list(zip(mids, te["role"]))
    y = te["label_final"].map(LAB).to_numpy(int)
    threads = te["thread_id"].to_numpy()
    roles = te["role"].to_numpy()
    dist, status = load_votes(keys)
    # Split is defined on the three independent round-0 votes. The panel's status field
    # reflects the post-debate outcome: 12 test items with a 1-1-1 round-0 split are
    # recorded as "unanimous" after the debate round, so status is not used here.
    n_max = (dist * 3).round().max(1)
    split = n_max < 3
    assert (dist.argmax(1)[~split] == y[~split]).all(), "round-0 unanimous items must match label_final"
    logger.info("test: %d, unanimous %d, split %d (2-1: %d, 1-1-1: %d)", len(y), (~split).sum(),
                split.sum(), (n_max == 2).sum(), (n_max == 1).sum())

    rng = np.random.default_rng(BOOT_SEED)
    uniq = np.unique(threads)
    members = {t: np.where(threads == t)[0] for t in uniq}
    boots = [np.concatenate([members[t] for t in rng.choice(uniq, len(uniq), replace=True)])
             for _ in range(N_BOOT)]

    def f1_split(P, idx):
        sel = idx[split[idx]]
        return float(np.mean([macro_f1(y[sel], p[sel].argmax(1)) for p in P]))

    def f1_unan(P, idx):
        sel = idx[~split[idx]]
        return float(np.mean([macro_f1(y[sel], p[sel].argmax(1)) for p in P]))

    def f1_all(P, idx):
        return float(np.mean([macro_f1(y[idx], p[idx].argmax(1)) for p in P]))

    probs, items, rows = {}, {}, []
    all_idx = np.arange(len(y))
    for name, (kind, rel) in SYSTEMS.items():
        path = _REPO_ROOT / rel
        if not path.exists():
            logger.warning("skip %s: %s missing", name, rel)
            continue
        P = load_system(kind, path, mids)
        probs[name], items[name] = P, per_item(P, dist)
        Pm = P.mean(0)
        ent = -(Pm * np.log(np.clip(Pm, 1e-12, 1))).sum(1)
        r = {"system": name, "n_seeds": P.shape[0],
             "macro_f1_all": f1_all(P, all_idx),
             "macro_f1_unanimous": f1_unan(P, all_idx),
             "macro_f1_split": f1_split(P, all_idx),
             "jsd_mean": float(items[name]["jsd"].mean()),
             "soft_ce_mean": float(items[name]["soft_ce"].mean()),
             "acceptable_rate_all": float(items[name]["acceptable"].mean()),
             "acceptable_rate_split": float(items[name]["acceptable"][split].mean()),
             "entropy_auroc_split": float(roc_auc_score(split.astype(int), ent))}
        for metric, fn in (("macro_f1_split", f1_split), ("macro_f1_unanimous", f1_unan)):
            b = np.array([fn(P, i) for i in boots])
            r[f"{metric}_ci"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        bj = np.array([items[name]["jsd"][i].mean() for i in boots])
        r["jsd_ci"] = [float(np.percentile(bj, 2.5)), float(np.percentile(bj, 97.5))]
        for role in ("Mgr", "CS", "Sch"):
            r[f"jsd_{role}"] = float(items[name]["jsd"][roles == role].mean())
        rows.append(r)
        logger.info("%s: F1 all %.4f | unanimous %.4f | split %.4f | JSD %.4f | acceptable(split) %.3f | AUROC %.3f",
                    name, r["macro_f1_all"], r["macro_f1_unanimous"], r["macro_f1_split"],
                    r["jsd_mean"], r["acceptable_rate_split"], r["entropy_auroc_split"])
    tab = pd.DataFrame(rows)

    comps = []
    for a, b in COMPARISONS:
        if a not in probs or b not in probs:
            continue
        for metric in ("jsd", "acceptable"):
            va, vb = items[a][metric], items[b][metric]
            d = np.array([va[i].mean() - vb[i].mean() for i in boots])
            comps.append({"a": a, "b": b, "metric": metric, "delta": float(va.mean() - vb.mean()),
                          "ci95_low": float(np.percentile(d, 2.5)), "ci95_high": float(np.percentile(d, 97.5)),
                          "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))})
        for metric, fn in (("macro_f1_split", f1_split), ("macro_f1_unanimous", f1_unan)):
            d = np.array([fn(probs[a], i) - fn(probs[b], i) for i in boots])
            comps.append({"a": a, "b": b, "metric": metric,
                          "delta": fn(probs[a], all_idx) - fn(probs[b], all_idx),
                          "ci95_low": float(np.percentile(d, 2.5)), "ci95_high": float(np.percentile(d, 97.5)),
                          "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))})
    comp = pd.DataFrame(comps)
    if len(comp):
        comp["p_holm"] = np.nan
        for metric, grp in comp.groupby("metric"):
            order = grp["p_two_sided"].sort_values().index
            m = len(order)
            running = 0.0
            for rank, ix in enumerate(order):
                running = max(running, min(1.0, (m - rank) * comp.at[ix, "p_two_sided"]))
                comp.at[ix, "p_holm"] = running

    tab.to_csv(out / "system_soft_metrics.csv", index=False)
    comp.to_csv(out / "paired_comparisons.csv", index=False)
    write_json(out / "summary.json", {
        "n_test": int(len(y)), "n_unanimous": int((~split).sum()), "n_split": int(split.sum()),
        "split_pattern": {"2-1": int((n_max == 2).sum()), "1-1-1": int((n_max == 1).sum())},
        "panel_status_counts_post_debate": dict(Counter(status.tolist())),
        "bootstrap": {"unit": "thread", "n_boot": N_BOOT, "seed": BOOT_SEED},
        "elapsed_s": round(time.time() - t0, 1), "git_commit": git_commit(_REPO_ROOT)})
    logger.info("output: %s", out)
    print(tab.drop(columns=[c for c in tab.columns if c.endswith("_ci")]).round(4).to_string(index=False))
    print(comp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
