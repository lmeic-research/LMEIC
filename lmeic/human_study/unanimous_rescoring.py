"""E29b: human-label re-scoring on the instances where all three annotators agree (revision 1).

Version: v1 | Date: 2026-10-29

Same estimand and systems as E25 (every run scored separately, macro-F1 averaged over runs;
LMEIC-explicit 5 runs, LMEIC-full 25 runs, LoRA 5 runs) on the 1,178 instances with a human majority
and an agent label, split by the annotation pattern: unanimous (3-0) and not unanimous. Also reported:
the agent consensus scored against the human majority, and the mean pairwise macro-F1 between the
three annotators (both directions of each pair) as a reference. Paired thread-clustered bootstrap
(2,000, seed 20261028) for the three system pairs, Holm within subset.

Output: <E1 run dir>/e29_unanimous_<timestamp>/ (rescoring.csv, paired.csv, summary.json)

Usage: python -m lmeic.human_study.unanimous_rescoring --e1-dir <E1 run dir> --lora-root <E22 output root>

Requires the confidential EMAIL-3R human-study files (items_with_labels.jsonl and the
annotation database); it is included to document the analysis and does not run without them.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.human_study.annotation_store.store import Store  # noqa: E402
from lmeic.human_study.crossrole_multiseed import (  # noqa: E402
    BOOT_SEED, LAB, LABELS, N_BOOT, load_systems, macro_f1, score,
)
from lmeic.common.rev1_common import git_commit, norm_mid, write_json  # noqa: E402

DB = _REPO_ROOT / "private/runs/rev1_eval_sets_20261022_161427/human_annotation/web_annotations_r2.sqlite"
SYSTEMS = ("LMEIC_explicit", "LMEIC_full", "LoRA")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-dir", required=True)
    ap.add_argument("--lora-root", required=True, help="E22 output root (five-seed LoRA runs)")
    args = ap.parse_args()
    e1 = Path(args.e1_dir)
    lora_root = Path(args.lora_root)
    it = pd.read_json(e1 / "items_with_labels.jsonl", lines=True)
    it["pred_id"] = it["pred_id"].map(lambda x: norm_mid(str(x)))
    S = {k: v for k, v in load_systems(lora_root).items() if k in SYSTEMS}

    ann = pd.DataFrame([r for r in Store(str(DB), read_only=True).all_rows() if r["phase"] == "main"])
    w = ann.pivot_table(index="item_id", columns="annotator", values="label", aggfunc="first")

    d = it[it["human_majority"].isin(LABELS) & it["agent_label"].isin(LABELS)].reset_index(drop=True)
    H = np.stack([w.loc[d["item_id"], a].map(LAB).to_numpy() for a in ("A1", "A2", "A3")], 1)
    unanimous = (H[:, 0] == H[:, 1]) & (H[:, 1] == H[:, 2])
    assert (unanimous == (d["human_pattern"] == "3-0").to_numpy()).all()
    yh, ya = d["human_majority"].map(LAB).to_numpy(), d["agent_label"].map(LAB).to_numpy()
    keys = list(zip(d["pred_set"], d["pred_id"]))
    core = (d["source"] == "core_test").to_numpy()
    subsets = {"all": np.ones(len(d), bool), "unanimous": unanimous, "not_unanimous": ~unanimous,
               "core_test_unanimous": core & unanimous, "core_test_not_unanimous": core & ~unanimous}

    rows, paired = [], []
    rng = np.random.default_rng(BOOT_SEED)
    for sub, m in subsets.items():
        idx = np.where(m)[0]; k = [keys[i] for i in idx]
        pair_f1 = [macro_f1(H[idx, a], H[idx, b]) for a, b in itertools.permutations(range(3), 2)]
        rows.append({"subset": sub, "system": "agent_consensus", "n": int(len(idx)), "n_runs": 1,
                     "macro_f1_vs_human": macro_f1(yh[idx], ya[idx])})
        rows.append({"subset": sub, "system": "annotator_pairwise_mean", "n": int(len(idx)), "n_runs": 6,
                     "macro_f1_vs_human": float(np.mean(pair_f1))})
        for name, runs in S.items():
            rows.append({"subset": sub, "system": name, "n": int(len(idx)), "n_runs": len(runs),
                         "macro_f1_vs_human": score(runs, k, yh[idx]), "macro_f1_vs_agent": score(runs, k, ya[idx])})
        th = d["thread_id"].to_numpy()[idx]; u = np.unique(th); mem = {t: np.where(th == t)[0] for t in u}
        boots = [np.concatenate([mem[t] for t in rng.choice(u, len(u))]) for _ in range(N_BOOT)]
        mats = {n: np.array([[r[kk] for kk in k] for r in S[n]]) for n in SYSTEMS}
        f = lambda P, bi: float(np.mean([macro_f1(yh[idx][bi], p[bi]) for p in P]))  # noqa: E731
        fam = []
        for a, b in [("LMEIC_full", "LoRA"), ("LMEIC_full", "LMEIC_explicit"), ("LoRA", "LMEIC_explicit")]:
            allidx = np.arange(len(idx))
            dd = np.array([f(mats[a], bi) - f(mats[b], bi) for bi in boots])
            fam.append({"subset": sub, "a": a, "b": b, "n": int(len(idx)),
                        "delta": f(mats[a], allidx) - f(mats[b], allidx),
                        "ci95_low": float(np.percentile(dd, 2.5)), "ci95_high": float(np.percentile(dd, 97.5)),
                        "p_two_sided": float(min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean())))})
        p = np.array([r["p_two_sided"] for r in fam]); order = np.argsort(p); run = 0.0
        for j, i in enumerate(order):
            run = max(run, (len(p) - j) * p[i]); fam[i]["p_holm"] = min(1.0, run)
        paired += fam

    out = e1 / f"e29_unanimous_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir()
    sc = pd.DataFrame(rows)
    sc.to_csv(out / "rescoring.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)
    write_json(out / "summary.json", {"e1_dir": str(e1), "lora_root": str(lora_root), "git_commit": git_commit(_REPO_ROOT),
                                      "n": {k: int(v.sum()) for k, v in subsets.items()},
                                      "n_runs": {k: len(v) for k, v in S.items()},
                                      "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED}})
    print(sc.round(4).to_string(index=False)); print(pd.DataFrame(paired).round(4).to_string(index=False))
    print(json.dumps({k: int(v.sum()) for k, v in subsets.items()}))


if __name__ == "__main__":
    main()
