"""
Co-reader bank for the robustness study (revision 1, E10).
Version: v1.1 | Date: 2026-10-28
  v1.1: the full-pool model also scores the external sets (rev1_reference.yaml: external_sets)
        -> probs_external.parquet (instance_id, set, plm_p_*), for E21 multi-seed scoring.

For each configured candidate encoder and seed: development-pool (train+val) scores from
thread-grouped cross-fitting (shared folds, fold seed from the config) and test scores
from a model fine-tuned on the full development pool. Training follows the reference
co-reader recipe (AdamW, weight decay 0.01, 10% linear warm-up, class-weighted cross
entropy, gradient clipping 1.0), optionally under bf16 autocast. Each candidate-seed is
written to its own directory and skipped if already complete.

Usage:
  python -m lmeic.analysis.coreader_bank --out-dir <rev1_reference run dir> \
      [--candidates xlmr_large,bge_m3] [--seeds 42]
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
import torch.nn.functional as TF
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.reference.coreader import PROB_COLS, TextDS, class_weights  # noqa: E402
from lmeic.reference.folds import check_group_disjoint, make_folds  # noqa: E402
from lmeic.common.rev1_common import load_config, read_jsonl, write_json  # noqa: E402
from lmeic.common.data_loader import LABEL_MAP, load_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)
CONFIG_NAME = "rev1_reference.yaml"


def _amp(cc: dict):
    return torch.autocast("cuda", dtype=torch.bfloat16) if cc.get("bf16") else torch.autocast("cuda", enabled=False)


def train(tok, x, y, cc: dict, seed: int):
    torch.manual_seed(seed); np.random.seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(cc["model_name"], num_labels=3).cuda()
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TextDS(x, y, tok, cc["max_len"]), batch_size=cc["batch_size"], shuffle=True, generator=g)
    opt = torch.optim.AdamW(model.parameters(), lr=cc["lr"], weight_decay=0.01)
    total = len(dl) * cc["epochs"]
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    lossf = torch.nn.CrossEntropyLoss(weight=class_weights(y))
    model.train()
    for _ in range(cc["epochs"]):
        for b in dl:
            opt.zero_grad()
            with _amp(cc):
                logits = model(input_ids=b["input_ids"].cuda(), attention_mask=b["attention_mask"].cuda()).logits
            loss = lossf(logits.float(), b["labels"].cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
    return model


@torch.no_grad()
def predict(model, tok, x, cc: dict) -> np.ndarray:
    model.eval()
    dl = DataLoader(TextDS(x, np.zeros(len(x), dtype=int), tok, cc["max_len"]), batch_size=64)
    out = []
    for b in dl:
        with _amp(cc):
            logits = model(input_ids=b["input_ids"].cuda(), attention_mask=b["attention_mask"].cuda()).logits
        out.append(TF.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.vstack(out)


def run_one(name: str, cc: dict, seed: int, fold_seed: int, n_folds: int, out_root: Path,
            ext_sets: dict | None = None, repo: Path | None = None) -> None:
    cdir = out_root / f"{name}_s{seed}"
    if (cdir / "probs.parquet").exists() and (ext_sets is None or (cdir / "probs_external.parquet").exists()):
        logger.info("%s seed %d already complete", name, seed)
        return
    cdir.mkdir(parents=True, exist_ok=True)
    dev = pd.concat([load_split("train"), load_split("val")]).reset_index(drop=True)
    te = load_split("test")
    x, y, g = dev["latest_text"].fillna("").values, dev["label_final"].map(LABEL_MAP).values, dev["thread_id"].values
    folds = make_folds(y, g, "grouped_clean", fold_seed, n_folds)
    assert check_group_disjoint(folds, g) == 0
    tok = AutoTokenizer.from_pretrained(cc["model_name"])
    t0 = time.time()
    oof = np.zeros((len(dev), 3), dtype=np.float32)
    fold_f1 = []
    for k, (a, b) in enumerate(folds):
        m = train(tok, x[a], y[a], cc, seed=seed + k)
        oof[b] = predict(m, tok, x[b], cc)
        fold_f1.append(float(f1_score(y[b], oof[b].argmax(1), average="macro")))
        logger.info("%s s%d fold %d/%d OOF macro-F1 %.4f (%.0fs)", name, seed, k + 1, n_folds, fold_f1[-1],
                    time.time() - t0)
        del m; torch.cuda.empty_cache()
    m = train(tok, x, y, cc, seed=seed + 100)
    p_te = predict(m, tok, te["latest_text"].fillna("").values, cc)
    if ext_sets:
        frames = []
        for set_name, rel in ext_sets.items():
            rows_e = pd.DataFrame(read_jsonl(repo / rel))
            f = pd.DataFrame(predict(m, tok, rows_e["latest_text"].fillna("").values, cc), columns=PROB_COLS)
            f.insert(0, "set", set_name)
            f.insert(0, "instance_id", rows_e["instance_id"].astype(str).values)
            frames.append(f)
        pd.concat(frames, ignore_index=True).to_parquet(cdir / "probs_external.parquet", index=False)
    del m; torch.cuda.empty_cache()
    y_te = te["label_final"].map(LABEL_MAP).values

    rows = pd.DataFrame(oof, columns=PROB_COLS)
    rows.insert(0, "split", "trainval")
    rows.insert(0, "message_id", dev["message_id"].values)
    trows = pd.DataFrame(p_te, columns=PROB_COLS)
    trows.insert(0, "split", "test")
    trows.insert(0, "message_id", te["message_id"].values)
    stats = {"candidate": name, "seed": seed, "fold_seed": fold_seed, "coreader": cc,
             "fold_oof_macro_f1": fold_f1, "oof_macro_f1": float(f1_score(y, oof.argmax(1), average="macro")),
             "test_macro_f1": float(f1_score(y_te, p_te.argmax(1), average="macro")),
             "elapsed_s": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name(0),
             "hf_home": os.environ.get("HF_HOME")}
    write_json(cdir / "stats.json", stats)
    pd.concat([rows, trows], ignore_index=True).to_parquet(cdir / "probs.parquet", index=False)
    logger.info("%s s%d done: OOF %.4f test %.4f (%.0fs)", name, seed, stats["oof_macro_f1"],
                stats["test_macro_f1"], stats["elapsed_s"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--external", action="store_true", help="also score the external sets with the full-pool model")
    args = ap.parse_args()
    paths, cfg = load_config(CONFIG_NAME)
    assert torch.cuda.is_available(), "CUDA required"
    bank = cfg["coreader_bank"]
    names = args.candidates.split(",") if args.candidates else list(bank["candidates"])
    out_root = Path(args.out_dir) / "coreader_bank"
    for name in names:
        spec = dict(bank["candidates"][name])
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else spec.pop("seeds")
        spec.pop("seeds", None)
        cc = {**bank["common"], **spec}
        for seed in seeds:
            run_one(name, cc, seed, bank["fold_seed"], cfg["n_folds"], out_root,
                    cfg["external_sets"]["sets"] if args.external else None, Path(paths["repo_root"]))


if __name__ == "__main__":
    main()
