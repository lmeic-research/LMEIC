#!/usr/bin/env python3
"""
E12 step 4: XLM-R co-reader scores for the Enron instances (GPU).
Version: v1 | Date: 2026-10-23

Development instances are scored by models fine-tuned on thread-grouped folds that share no
thread with them (fold seed 42, as in the EMAIL-3R reference rebuild); test instances by models
fine-tuned on the full development pool. The co-reader feature uses the full-pool model with
seed 42; the fine-tuned XLM-R baseline reports the five full-pool seeds. Hyper-parameters are
the reference co-reader settings (rev1_reference.yaml: coreader).

Usage:
  python -m benchmarks.enron.coreader --run-dir <build_instances dir> \
      --labels <agent_labels_e12_enron.jsonl>
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

from benchmarks.enron.data import load_labelled  # noqa: E402
from lmeic.reference.coreader import PROB_COLS, predict, train  # noqa: E402
from lmeic.reference.folds import check_group_disjoint, make_folds  # noqa: E402
from lmeic.common.rev1_common import git_commit, load_config, write_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
SEEDS = [42, 123, 456, 789, 2026]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    paths, rcfg = load_config("rev1_reference.yaml")
    cc = rcfg["coreader"]
    assert torch.cuda.is_available(), "CUDA required"
    df = load_labelled(run_dir, Path(args.labels))
    text = lambda d: (d["subject"].fillna("") + "\n" + d["latest_text"].fillna("")).values  # noqa: E731
    dev, te = df[df["split"] == "dev"].reset_index(drop=True), df[df["split"] == "test"].reset_index(drop=True)
    x, y, g = text(dev), dev["y"].values, dev["thread_id"].values
    folds = make_folds(y, g, "grouped_clean", 42, rcfg["n_folds"])
    assert check_group_disjoint(folds, g) == 0
    tok = AutoTokenizer.from_pretrained(cc["model_name"])
    t0 = time.time()

    oof = np.zeros((len(dev), 3), dtype=np.float32)
    for k, (a, b) in enumerate(folds):
        m = train(tok, x[a], y[a], cc, seed=cc["seed"] + k)
        oof[b] = predict(m, tok, x[b], cc["max_len"])
        logger.info("fold %d OOF macro-F1 %.4f (%.0fs)", k + 1, f1_score(y[b], oof[b].argmax(1), average="macro"),
                    time.time() - t0)
        del m; torch.cuda.empty_cache()

    test_probs, test_f1 = {}, {}
    for s in SEEDS:
        m = train(tok, x, y, cc, seed=s)
        test_probs[s] = predict(m, tok, text(te), cc["max_len"])
        test_f1[s] = float(f1_score(te["y"].values, test_probs[s].argmax(1), average="macro"))
        logger.info("full-pool seed %d test macro-F1 %.4f (%.0fs)", s, test_f1[s], time.time() - t0)
        del m; torch.cuda.empty_cache()

    rows = [pd.DataFrame(oof, columns=PROB_COLS).assign(instance_id=dev["instance_id"].values, split="dev", seed=42)]
    for s in SEEDS:
        rows.append(pd.DataFrame(test_probs[s], columns=PROB_COLS).assign(
            instance_id=te["instance_id"].values, split="test", seed=s))
    out = run_dir / f"coreader_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir()
    pd.concat(rows, ignore_index=True).to_parquet(out / "coreader_probs.parquet", index=False)
    write_json(out / "coreader_stats.json", {
        "labels": args.labels, "oof_macro_f1": float(f1_score(y, oof.argmax(1), average="macro")),
        "test_macro_f1_by_seed": test_f1, "test_macro_f1_mean": float(np.mean(list(test_f1.values()))),
        "test_macro_f1_sd": float(np.std(list(test_f1.values()), ddof=1)), "coreader": cc,
        "fold_seed": 42, "elapsed_s": round(time.time() - t0, 1), "git_commit": git_commit(Path(paths["repo_root"]))})
    logger.info("saved %s", out)


if __name__ == "__main__":
    main()
