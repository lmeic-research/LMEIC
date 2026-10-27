#!/usr/bin/env python3
"""
E18e: Apache Jira — fine-tuned XLM-R text-only baseline (GPU).
Version: v1 | Date: 2026-10-26

Same rows and time split as run_d1_fusion / run_id_ablation (sample_30000_issue_ids.txt, valid D1
record; train created < 2022, test >= 2023). Input: summary + description. Hyper-parameters are the
reference co-reader settings (rev1_reference.yaml: coreader: xlm-roberta-base, max_len 256, 4 epochs,
lr 2e-5, class-weighted cross-entropy). Five seeds per setting; S2 uses the classes present (High/Low),
argmax restricted to them. Writes per-seed test probabilities so they can be paired with other arms.

Usage:
  python -m benchmarks.jira.xlmr_baseline
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.jira.run_baselines import SEEDS, load  # noqa: E402
from benchmarks.jira.run_d1_fusion import d1_frame  # noqa: E402
from lmeic.reference.coreader import predict, train  # noqa: E402
from lmeic.common.rev1_common import git_commit, load_config, write_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="datasets/public/apache_jira/processed")
    ap.add_argument("--sample-ids", default="sample_30000_issue_ids.txt")
    ap.add_argument("--d1", default="datasets/public/apache_jira/d1/d1_jira_qwen25_7b.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    t0 = time.time(); proc = Path(args.processed)
    _, rcfg = load_config("rev1_reference.yaml")
    cc = rcfg["coreader"]
    df = load(proc)
    df["issue_id"] = df["issue_id"].astype(str)
    ids = [l.strip() for l in (proc / args.sample_ids).read_text().splitlines() if l.strip()]
    df = df.set_index("issue_id").loc[ids].reset_index()
    df = df[df["issue_id"].isin(set(d1_frame(Path(args.d1))["issue_id"]))].sort_values("created").reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(cc["model_name"])
    out_dir = Path(args.out) if args.out else proc / f"xlmr_baseline_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    res, probs = {"coreader_config": cc, "git_commit": git_commit(_REPO_ROOT)}, []
    for setting in ("S1", "S2"):
        d = df if setting == "S1" else df[df["priority_creation"] != "Major"].reset_index(drop=True)
        present = sorted(d["y"].unique()); k = len(present)
        y = d["y"].map({c: i for i, c in enumerate(present)}).to_numpy()
        tr = (d["created"] < "2022-01-01").to_numpy(); te = (d["created"] >= "2023-01-01").to_numpy()
        x = d["text"].to_numpy()
        fs, per = [], []
        for s in SEEDS:
            m = train(tok, x[tr], y[tr], cc, seed=s)
            p = predict(m, tok, x[te], cc["max_len"])[:, :k]
            pred = p.argmax(1)
            fs.append(float(f1_score(y[te], pred, average="macro", labels=list(range(k)), zero_division=0)))
            per.append(f1_score(y[te], pred, average=None, labels=list(range(k)), zero_division=0).tolist())
            probs.append(pd.DataFrame(p, columns=[f"p{c}" for c in range(k)]).assign(
                setting=setting, seed=s, issue_id=d.loc[te, "issue_id"].values, y=y[te]))
            logger.info("%s seed %d macro-F1 %.4f (%.0fs)", setting, s, fs[-1], time.time() - t0)
            del m; torch.cuda.empty_cache()
        res[setting] = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "classes": [int(c) for c in present],
                        "macro_f1_mean": float(np.mean(fs)), "macro_f1_sd": float(np.std(fs)), "by_seed": fs,
                        "per_class_f1_mean": np.mean(per, axis=0).round(4).tolist()}
        write_json(out_dir / "results.json", {**res, "elapsed_s": round(time.time() - t0, 1)})
        pd.concat(probs).to_parquet(out_dir / "test_probs.parquet", index=False)
    logger.info("saved %s", out_dir)


if __name__ == "__main__":
    main()
