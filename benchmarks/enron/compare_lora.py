#!/usr/bin/env python3
"""
E12: paired comparison of the Qwen2.5-7B LoRA classifier (lora_classifier --set e12) with the
LMEIC variants of a run_models output, on the same Enron test instances.
Version: v1 | Date: 2026-10-23

LoRA has one seed (42); LMEIC rows use the seed-averaged macro-F1 of run_models (all five
seeds) and, separately, the aligned seed-42 predictions. Thread-cluster paired bootstrap with
the run_models resampling (thread_boot, seed 20261023); Holm within the comparison family.
Role sensitivity of the LoRA predictions uses the run_models definition.

Usage:
  python -m benchmarks.enron.compare_lora --run-dir <rev1_e12_enron run> \
      --models-dir <run>/models_gpt4o_<ts> --lora-dir <e11 out-root>/qwen25_7b_lora_role_e12_s42 \
      --labels <agent_labels_e12_enron.jsonl> [--keep-instances FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.analysis.context_analysis import holm  # noqa: E402
from benchmarks.enron import data as D  # noqa: E402
from benchmarks.enron.run_models import role_sensitivity, thread_boot  # noqa: E402
from lmeic.reference.analyse_reference import macro  # noqa: E402
from lmeic.common.rev1_common import git_commit, write_json  # noqa: E402

LMEIC = ["explicit", "full", "d1_only", "tfidf_lr", "xlmr"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--lora-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--keep-instances", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    df = D.load_labelled(Path(args.run_dir), Path(args.labels))
    te = df[df["split"] == "test"].reset_index(drop=True)
    if args.keep_instances:
        keep = set(Path(args.keep_instances).read_text().split())
        te = te[te["instance_id"].isin(keep)].reset_index(drop=True)
    ids = te["instance_id"].values
    yt = te["y"].values

    pr = pd.read_parquet(Path(args.models_dir) / "predictions.parquet")
    preds = {}
    for v in LMEIC:
        p = pr[pr["variant"] == v].pivot(index="seed", columns="instance_id", values="pred")
        missing = set(ids) - set(p.columns)
        assert not missing, f"{v}: {len(missing)} test instances without predictions"
        preds[v] = p[ids].values.astype(int)
        g = pr[pr["variant"] == v].drop_duplicates("instance_id").set_index("instance_id").loc[ids, "gold"].values
        assert (g == yt).all(), f"{v}: gold labels differ from the label file"
    lo = pd.read_parquet(Path(args.lora_dir) / "test_probs.parquet").set_index("message_id").loc[ids]
    lp = lo[["plm_p_high", "plm_p_med", "plm_p_low"]].values
    preds["lora"] = lp.argmax(1)[None, :]
    preds_s42 = {v: (preds[v][[0]] if v == "lora" else
                     pr[(pr["variant"] == v) & (pr["seed"] == 42)].set_index("instance_id").loc[ids, "pred"]
                     .values.astype(int)[None, :]) for v in preds}

    boots = thread_boot(te["thread_id"].values, args.n_boot)
    rows = []
    for label, P in (("five_seed_mean", preds), ("seed42_aligned", preds_s42)):
        fam = []
        for a in ("explicit", "full"):
            for b in ("lora",):
                d = np.array([macro(yt, P[a], i) - macro(yt, P[b], i) for i in boots])
                fam.append({"protocol": label, "a": a, "b": b, "f1_a": macro(yt, P[a]), "f1_b": macro(yt, P[b]),
                            "delta": macro(yt, P[a]) - macro(yt, P[b]),
                            "ci95_low": float(np.percentile(d, 2.5)), "ci95_high": float(np.percentile(d, 97.5)),
                            "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))})
        for r, p in zip(fam, holm(np.array([r["p_two_sided"] for r in fam]))):
            r["p_holm"] = float(p)
        rows += fam
    out = {
        "n_test": int(len(te)), "n_threads": int(te["thread_id"].nunique()),
        "subset": args.keep_instances,
        "macro_f1": {v: macro(yt, P) for v, P in preds.items()},
        "macro_f1_by_role": {v: {r: macro(yt[(te["role"] == r).values], P[:, (te["role"] == r).values])
                                 for r in D.ROLES} for v, P in preds.items()},
        "lora_role_sensitivity": role_sensitivity(te, lp, lp.argmax(1)),
        "comparisons": rows,
        "bootstrap": {"unit": "thread", "n_boot": args.n_boot, "seed": 20261023},
        "models_dir": args.models_dir, "lora_dir": args.lora_dir, "git_commit": git_commit(_REPO_ROOT),
    }
    tag = "_isi" if args.keep_instances else ""
    path = Path(args.lora_dir) / f"compare_vs_{Path(args.models_dir).name}{tag}.json"
    write_json(path, out)
    print(json.dumps({k: out[k] for k in ("n_test", "macro_f1", "lora_role_sensitivity")}, indent=1))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("saved", path)


if __name__ == "__main__":
    main()
