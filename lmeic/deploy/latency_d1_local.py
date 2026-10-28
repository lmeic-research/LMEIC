#!/usr/bin/env python3
"""
E27: per-email latency of local D1 extraction (one request at a time) on an idle GPU.
Version: v1 | Date: 2026-10-28

The E13 latency table timed every inference component except D1 extraction, whose cost was taken
from batch throughput. This script serves the same model, prompt, one-shot example and JSON
schema as d1_extract_local and sends test emails one by one, so the latency of a single email
(the online case) is measured directly. It also records the batch throughput on the same emails.

Usage:
  python -m lmeic.deploy.latency_d1_local --model qwen3_8b \
      --out-dir <dir> [--n 200] [--warmup 10]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.common.data_loader import load_split  # noqa: E402
from lmeic.features.d1_extract_local import CONFIG_PATH, sampling_params  # noqa: E402
from lmeic.features.extract_d1 import (  # noqa: E402
    ONE_SHOT_ASSISTANT, ONE_SHOT_USER, REQUIRED_FIELDS, SYSTEM_PROMPT, build_user_msg,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3_8b")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    mc = cfg["models"][args.model]
    gpu_before = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv"],
                                capture_output=True, text=True).stdout

    te = load_split("test").sample(n=args.n + args.warmup, random_state=20261028)
    recs = te.to_dict("records")
    convs = [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ONE_SHOT_USER},
              {"role": "assistant", "content": ONE_SHOT_ASSISTANT}, {"role": "user", "content": build_user_msg(r)}]
             for r in recs]

    from vllm import LLM
    llm = LLM(model=mc["hf_id"], dtype=mc["dtype"], max_model_len=mc["max_model_len"],
              gpu_memory_utilization=mc["gpu_memory_utilization"], seed=cfg["sampling"]["seed"])
    sp = sampling_params(cfg["sampling"])
    kw = {"chat_template_kwargs": mc["chat_template_kwargs"]} if mc.get("chat_template_kwargs") else {}

    for c in convs[: args.warmup]:
        llm.chat([c], sp, use_tqdm=False, **kw)
    secs, ok, toks = [], 0, []
    for c in convs[args.warmup:]:
        t = time.perf_counter()
        o = llm.chat([c], sp, use_tqdm=False, **kw)[0]
        secs.append(time.perf_counter() - t)
        toks.append(len(o.outputs[0].token_ids))
        try:
            ok += all(k in json.loads(o.outputs[0].text) for k in REQUIRED_FIELDS)
        except json.JSONDecodeError:
            pass
    t = time.perf_counter()
    llm.chat(convs[args.warmup:], sp, use_tqdm=False, **kw)
    batch_s = time.perf_counter() - t

    a = np.array(secs) * 1000
    res = {"model": mc["hf_id"], "n_timed": len(secs), "warmup": args.warmup, "valid_json": ok,
           "batch1_p50_ms": float(np.percentile(a, 50)), "batch1_p95_ms": float(np.percentile(a, 95)),
           "batch1_mean_ms": float(a.mean()), "completion_tokens_mean": float(np.mean(toks)),
           "batch_all_throughput_per_s": float(len(secs) / batch_s), "sampling": cfg["sampling"],
           "gpu_state_before": gpu_before, "sample_seed": 20261028}
    (out / "d1_local_latency.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "gpu_state_before"}, indent=2))


if __name__ == "__main__":
    main()
