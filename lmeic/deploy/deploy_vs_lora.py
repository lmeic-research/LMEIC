#!/usr/bin/env python3
"""
E20: deployment cost of LMEIC versus a LoRA-tuned 7B classifier (same GPU, measured idle).

Measured here:
  lora_infer_b1 / _b16   Qwen2.5-7B + LoRA sequence classifier (E11 settings, max_len 512, bf16),
                         per-email latency and throughput on n test emails; adapter weights are
                         freshly initialised (inference cost does not depend on the trained values)
  lora_train_1k          one epoch on 1,000 development emails (E11 batch, grad accumulation,
                         gradient checkpointing); seconds per email, peak memory
  stack_refit            LMEIC monthly update: refit XGB + LGBM + logistic meta-learner on the full
                         development pool with the stored R2 parameters (CPU), seconds
Taken from records (not re-measured): LMEIC per-email latency outside D1 (E13 latency run) and the D1
throughput of the local Qwen2.5-7B extractor (E4 run).

Usage (GPU idle):
  python -m lmeic.deploy.deploy_vs_lora --n 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.baselines.lora_classifier import batches, build_model, inputs  # noqa: E402
from lmeic.reference import features as F  # noqa: E402
from lmeic.reference import models as M  # noqa: E402
from lmeic.common.rev1_common import git_commit, load_config, write_json  # noqa: E402
from lmeic.common.data_loader import LABEL_MAP, load_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("e20")
REF = _REPO_ROOT / "private/runs/rev1_reference_20261022_191130"
E13_LATENCY = REF / "e13_latency_20261023_200830/summary.json"


def pct(secs, q):
    return float(np.percentile(np.array(secs) * 1000, q))


@torch.no_grad()
def time_lora_infer(model, tok, enc, bs: int) -> dict:
    model.eval()
    idx = np.arange(len(enc))
    for _, ids, att in batches(enc, idx[:bs], bs, tok.pad_token_id):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=ids, attention_mask=att)
    torch.cuda.synchronize()
    secs = []
    t_all = time.perf_counter()
    for _, ids, att in batches(enc, idx, bs, tok.pad_token_id):
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=ids, attention_mask=att)
        torch.cuda.synchronize()
        secs.append((time.perf_counter() - t) / ids.shape[0])
    el = time.perf_counter() - t_all
    return {"component": f"lora_infer_b{bs}", "p50_ms_per_email": pct(secs, 50), "p95_ms_per_email": pct(secs, 95),
            "throughput_per_s": len(enc) / el, "n": len(enc)}


def time_lora_train(lc, tok, enc, y) -> dict:
    import math
    import torch.nn.functional as Fn
    from transformers import get_linear_schedule_with_warmup
    torch.cuda.reset_peak_memory_stats()
    model = build_model(lc, tok)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lc["lr"])
    steps = math.ceil(len(y) / (lc["batch_size"] * lc["grad_accum"]))
    sch = get_linear_schedule_with_warmup(opt, 1, steps)
    yt = torch.tensor(y, dtype=torch.long, device="cuda")
    model.train()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for i, (b, ids, att) in enumerate(batches(enc, np.random.default_rng(0).permutation(len(y)), lc["batch_size"], tok.pad_token_id)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = Fn.cross_entropy(model(input_ids=ids, attention_mask=att).logits.float(), yt[b]) / lc["grad_accum"]
        loss.backward()
        if (i + 1) % lc["grad_accum"] == 0:
            opt.step(); sch.step(); opt.zero_grad()
    torch.cuda.synchronize(); el = time.perf_counter() - t0
    out = {"component": "lora_train_1epoch", "n_emails": len(y), "seconds": el, "seconds_per_email_epoch": el / len(y),
           "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "epochs_in_e11": lc["epochs"]}
    del model, opt; torch.cuda.empty_cache()
    return out


def time_stack_refit(rcfg, paths) -> dict:
    v = next(x for x in rcfg["variants"] if x["name"] == "R2_v3_grouped_explicit")
    params = json.loads((REF / "R2_v3_grouped_explicit" / "params.json").read_text())
    feats, y, meta = F.build(rcfg, Path(paths["repo_root"]), v["d2d3"], set(v["groups"]))
    X, yy, g, Xt = F.pool(feats, y, meta)
    from sklearn.linear_model import LogisticRegression
    t0 = time.perf_counter()
    xm = M._xgb(params["xgb"], 42, prob=True).fit(X, yy, sample_weight=M.sample_weights(yy))
    lm = M._lgbm(params["lgbm"], 42).fit(X, yy, sample_weight=M.sample_weights(yy))
    LogisticRegression(max_iter=1000).fit(np.hstack([xm.predict_proba(X), lm.predict_proba(X)]), yy)
    el = time.perf_counter() - t0
    return {"component": "lmeic_stack_refit_full_pool", "n_train": int(len(yy)), "seconds": el,
            "hardware": "CPU (XGBoost device as in params)", "note": "base learners on the full pool; the reference "
            "stacking also fits 5-fold OOF base learners for the meta-learner, about 6x this time"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=1000)
    args = ap.parse_args()
    assert torch.cuda.is_available()
    paths, cfg = load_config("rev1_e11_llm_baselines.yaml")
    _, rcfg = load_config("rev1_reference.yaml")
    lc = cfg["lora"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(lc["model_name"]); tok.padding_side = "right"
    te = load_split("test").head(args.n)
    dev = load_split("train").sample(args.n_train, random_state=0).reset_index(drop=True)
    enc_t = tok(inputs(te, "role"), truncation=True, max_length=lc["max_len"])["input_ids"]
    enc_d = tok(inputs(dev, "role"), truncation=True, max_length=lc["max_len"])["input_ids"]
    rows = [{"component": "lora_input_tokens_mean", "value": float(np.mean([len(e) for e in enc_t]))}]

    torch.cuda.reset_peak_memory_stats()
    model = build_model(lc, tok)
    for bs in (1, 16):
        rows.append(time_lora_infer(model, tok, enc_t, bs)); logger.info("%s", rows[-1])
    rows.append({"component": "lora_infer_peak_gpu_mem_gb", "value": torch.cuda.max_memory_allocated() / 1e9})
    del model; torch.cuda.empty_cache()
    rows.append(time_lora_train(lc, tok, enc_d, dev["label_final"].map(LABEL_MAP).values)); logger.info("%s", rows[-1])
    rows.append(time_stack_refit(rcfg, paths)); logger.info("%s", rows[-1])

    e13 = json.loads(E13_LATENCY.read_text())
    out = REF / f"e20_deploy_vs_lora_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir()
    write_json(out / "summary.json", {"experiment": "E20", "rows": rows,
                                      "e13_latency_rows": e13["rows"], "gpu": torch.cuda.get_device_name(0),
                                      "lora_config": lc, "git_commit": git_commit(_REPO_ROOT)})
    logger.info("saved %s", out)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
