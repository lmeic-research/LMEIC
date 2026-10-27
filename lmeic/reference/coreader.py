"""
Deep-text co-reader scores with thread-grouped cross-fitting.
Version: v1 | Date: 2026-10-22

Development-pool (train+val) scores come from a 5-fold StratifiedGroupKFold over
threads: each message is scored by an XLM-R model fine-tuned on folds that share no
thread with it. Test scores do not depend on the fold scheme (they come from the
model fine-tuned on the full development pool), so the v1.0 test scores are reused
and the v1.0 cache is checked for coverage.

With --external, one model is fitted on the full development pool (as for the v1.0
test scores) and scores the test split, for a consistency check against the v1.0
cache, and the external evaluation sets.

Usage:
  python -m lmeic.reference.coreader --out-dir <rev1_reference run dir> [--external]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.common.rev1_common import load_config, read_jsonl, source_run_dir, write_json  # noqa: E402
from lmeic.reference.folds import check_group_disjoint, make_folds  # noqa: E402
from lmeic.common.data_loader import LABEL_MAP, load_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
CONFIG_NAME = "rev1_reference.yaml"
PROB_COLS = ["plm_p_high", "plm_p_med", "plm_p_low"]


class TextDS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(list(texts), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i], "attention_mask": self.enc["attention_mask"][i],
                "labels": self.labels[i]}


def class_weights(y: np.ndarray) -> torch.Tensor:
    cls, cnt = np.unique(y, return_counts=True)
    w = np.zeros(3)
    for c, n in zip(cls, cnt):
        w[c] = len(y) / (3 * n)
    return torch.tensor(w, dtype=torch.float32, device="cuda")


def train(tok, x, y, cc: dict, seed: int):
    torch.manual_seed(seed); np.random.seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(cc["model_name"], num_labels=3).cuda()
    dl = DataLoader(TextDS(x, y, tok, cc["max_len"]), batch_size=cc["batch_size"], shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cc["lr"], weight_decay=0.01)
    total = len(dl) * cc["epochs"]
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    lossf = torch.nn.CrossEntropyLoss(weight=class_weights(y))
    model.train()
    for _ in range(cc["epochs"]):
        for b in dl:
            opt.zero_grad()
            logits = model(input_ids=b["input_ids"].cuda(), attention_mask=b["attention_mask"].cuda()).logits
            loss = lossf(logits, b["labels"].cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
    return model


@torch.no_grad()
def predict(model, tok, x, max_len):
    model.eval()
    dl = DataLoader(TextDS(x, np.zeros(len(x), dtype=int), tok, max_len), batch_size=64)
    out = []
    for b in dl:
        logits = model(input_ids=b["input_ids"].cuda(), attention_mask=b["attention_mask"].cuda()).logits
        out.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(out)


def score_external(out_dir: Path, paths: dict, cfg: dict) -> None:
    """Fit one model on the full development pool; score test and the external sets."""
    cc = cfg["coreader"]
    run_dir = source_run_dir(paths, cfg["source_run_id"])
    repo = Path(paths["repo_root"])
    dev = pd.concat([load_split("train"), load_split("val")]).reset_index(drop=True)
    te = load_split("test")
    tok = AutoTokenizer.from_pretrained(cc["model_name"])
    t0 = time.time()
    m = train(tok, dev["latest_text"].fillna("").values, dev["label_final"].map(LABEL_MAP).values,
              cc, seed=cc["seed"])
    logger.info("full-pool model trained (%.0fs)", time.time() - t0)

    p_te = predict(m, tok, te["latest_text"].fillna("").values, cc["max_len"])
    v1 = pd.read_parquet(run_dir / cfg["v1_artefacts"]["plm_probs"])
    v1_te = v1[v1["split"] == "test"].set_index("message_id").loc[te["message_id"], PROB_COLS].values
    y_te = te["label_final"].map(LABEL_MAP).values
    check = {"test_macro_f1_refit": float(f1_score(y_te, p_te.argmax(1), average="macro")),
             "test_macro_f1_v1": float(f1_score(y_te, v1_te.argmax(1), average="macro")),
             "test_argmax_agreement_vs_v1": float((p_te.argmax(1) == v1_te.argmax(1)).mean()),
             "test_prob_mae_vs_v1": float(np.abs(p_te - v1_te).mean())}
    logger.info("refit vs v1 on test: %s", check)

    frames = []
    for name, rel in cfg["external_sets"]["sets"].items():
        rows = pd.DataFrame(read_jsonl(repo / rel))
        p = predict(m, tok, rows["latest_text"].fillna("").values, cc["max_len"])
        f = pd.DataFrame(p, columns=PROB_COLS)
        f.insert(0, "set", name)
        f.insert(0, "instance_id", rows["instance_id"].astype(str).values)
        frames.append(f)
        logger.info("scored %s: %d", name, len(f))
    pd.concat(frames, ignore_index=True).to_parquet(out_dir / "coreader_probs_external.parquet", index=False)
    write_json(out_dir / "coreader_external_stats.json",
               {**check, "seed": cc["seed"], "elapsed_s": round(time.time() - t0, 1), "coreader": cc})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--external", action="store_true", help="score external sets with a full-pool model")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    paths, cfg = load_config(CONFIG_NAME)
    cc = cfg["coreader"]
    run_dir = source_run_dir(paths, cfg["source_run_id"])
    assert torch.cuda.is_available(), "CUDA required"
    if args.external:
        score_external(out_dir, paths, cfg)
        return

    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    dev = pd.concat([tr, va]).reset_index(drop=True)
    x = dev["latest_text"].fillna("").values
    y = dev["label_final"].map(LABEL_MAP).values
    g = dev["thread_id"].values
    folds = make_folds(y, g, "grouped_clean", cc["seed"], cfg["n_folds"])
    assert check_group_disjoint(folds, g) == 0

    tok = AutoTokenizer.from_pretrained(cc["model_name"])
    oof = np.zeros((len(dev), 3), dtype=np.float32)
    t0 = time.time()
    fold_f1 = []
    for k, (a, b) in enumerate(folds):
        m = train(tok, x[a], y[a], cc, seed=cc["seed"] + k)
        oof[b] = predict(m, tok, x[b], cc["max_len"])
        fold_f1.append(float(f1_score(y[b], oof[b].argmax(1), average="macro")))
        logger.info("fold %d/%d OOF macro-F1 %.4f (%.0fs)", k + 1, len(folds), fold_f1[-1], time.time() - t0)
        del m; torch.cuda.empty_cache()

    v1 = pd.read_parquet(run_dir / cfg["v1_artefacts"]["plm_probs"])
    v1_test = v1[v1["split"] == "test"].set_index("message_id")
    assert set(te["message_id"]) <= set(v1_test.index), "v1 test co-reader scores incomplete"
    v1_dev = v1[v1["split"] == "trainval"].set_index("message_id").loc[dev["message_id"], PROB_COLS].values

    rows = pd.DataFrame(oof, columns=PROB_COLS)
    rows.insert(0, "split", "trainval")
    rows.insert(0, "message_id", dev["message_id"].values)
    test_rows = v1_test.loc[te["message_id"], PROB_COLS].reset_index()
    test_rows.insert(1, "split", "test")
    probs = pd.concat([rows, test_rows], ignore_index=True)
    probs.to_parquet(out_dir / "coreader_probs_grouped.parquet", index=False)

    stats = {
        "folds": "StratifiedGroupKFold(thread_id)", "fold_oof_macro_f1": fold_f1,
        "oof_macro_f1_grouped": float(f1_score(y, oof.argmax(1), average="macro")),
        "oof_macro_f1_v1_stratified": float(f1_score(y, v1_dev.argmax(1), average="macro")),
        "test_scores": "reused from v1.0 full-pool model",
        "elapsed_s": round(time.time() - t0, 1), "coreader": cc,
        "hf_home": os.environ.get("HF_HOME"),
    }
    write_json(out_dir / "coreader_stats.json", stats)
    logger.info("OOF macro-F1 grouped %.4f vs v1 stratified %.4f",
                stats["oof_macro_f1_grouped"], stats["oof_macro_f1_v1_stratified"])


if __name__ == "__main__":
    main()
