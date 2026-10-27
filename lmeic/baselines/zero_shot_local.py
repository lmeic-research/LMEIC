#!/usr/bin/env python3
"""
E11: zero-shot importance classification with locally served open-weight LLMs (vLLM).
Version: v1 | Date: 2026-10-23

The prompt template and role descriptions are read verbatim from the v1.0 direct-LLM baseline
runner (experiments/baselines/run_wave2_llm_zs.py) without importing it; decoding is
constrained to the labels High / Medium / Low. --set e12 scores the Enron test split with the
Enron role descriptions (rev1_e12_enron.yaml: role_desc); the template itself is domain-neutral.

Usage (inside the vLLM environment):
  python -m lmeic.baselines.zero_shot_local --model qwen3_8b \
      --out-dir DIR [--set email3r|e12] [--e12-labels FILE]
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.common.rev1_common import load_config, release_data_dir, write_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
V1_RUNNER = _REPO_ROOT / "email3r" / "code" / "experiments" / "baselines" / "run_wave2_llm_zs.py"
LABELS = ["High", "Medium", "Low"]


def v1_prompt() -> tuple[str, dict]:
    tree = ast.parse(V1_RUNNER.read_text())
    vals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and \
                node.targets[0].id in ("PROMPT_TEMPLATE", "ROLE_DESCRIPTIONS"):
            vals[node.targets[0].id] = ast.literal_eval(node.value)
    return vals["PROMPT_TEMPLATE"], vals["ROLE_DESCRIPTIONS"]


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    f = []
    for c in range(3):
        tp = ((y == c) & (p == c)).sum()
        d = (y == c).sum() + (p == c).sum()
        f.append(2 * tp / d if d else 0.0)
    return float(np.mean(f))


def load_records(args, paths) -> tuple[list[dict], dict, str]:
    template, role_desc = v1_prompt()
    if args.set == "email3r":
        rows = [json.loads(l) for l in open(release_data_dir(paths) / "test.jsonl")]
        return [{"id": r["message_id"], "role": r["role"], "gold": r["label_final"], **r} for r in rows], role_desc, template
    _, ecfg = load_config("rev1_e12_enron.yaml")
    _, acfg = load_config("rev1_agent_labelling.yaml")
    inst = [json.loads(l) for l in gzip.open(_REPO_ROOT / acfg["sets"]["e12_enron"]["path"], "rt")]
    gold = {}
    for l in open(args.e12_labels):
        r = json.loads(l)
        if r.get("label_final"):
            gold[str(r["id"])] = r["label_final"]
    rows = [dict(r, id=r["instance_id"], gold=gold[r["instance_id"]], date_local=r["date_utc"])
            for r in inst if r["split"] == "test" and r["instance_id"] in gold]
    return rows, ecfg["role_desc"], template


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--set", default="email3r", choices=["email3r", "e12"])
    ap.add_argument("--e12-labels", default=None)
    args = ap.parse_args()
    paths, cfg = load_config("rev1_e11_llm_baselines.yaml")
    _, lcfg = load_config("rev1_d1_local.yaml")
    mc, zc = lcfg["models"][args.model], cfg["zero_shot"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, role_desc, template = load_records(args, paths)
    prompts = [template.format(role_desc=role_desc.get(r["role"], "an office worker"), subject=r.get("subject", ""),
                               from_display=r.get("from_display", ""), date_local=r.get("date_local", ""),
                               body=(r.get("latest_text") or "")[:4000]) for r in rows]

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    llm = LLM(model=mc["hf_id"], dtype=mc["dtype"], max_model_len=mc["max_model_len"],
              gpu_memory_utilization=mc["gpu_memory_utilization"], seed=zc["seed"])
    sp = SamplingParams(temperature=zc["temperature"], max_tokens=zc["max_tokens"], seed=zc["seed"],
                        structured_outputs=StructuredOutputsParams(choice=LABELS))
    chat_kw = {"chat_template_kwargs": mc["chat_template_kwargs"]} if mc.get("chat_template_kwargs") else {}
    t0 = time.time()
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp, use_tqdm=False, **chat_kw)
    elapsed = time.time() - t0
    pred = [o.outputs[0].text.strip() for o in outs]
    tag = f"{args.set}_{args.model}"
    with open(out_dir / f"zero_shot_{tag}.jsonl", "w") as f:
        for r, p in zip(rows, pred):
            f.write(json.dumps({"id": r["id"], "role": r["role"], "gold": r["gold"], "pred": p}) + "\n")
    idx = {l: i for i, l in enumerate(LABELS)}
    y = np.array([idx[r["gold"]] for r in rows])
    p = np.array([idx.get(x, 1) for x in pred])
    roles = np.array([r["role"] for r in rows])
    res = {"set": args.set, "model": args.model, "hf_id": mc["hf_id"], "n": len(rows),
           "macro_f1": macro_f1(y, p), "per_role": {r: macro_f1(y[roles == r], p[roles == r]) for r in sorted(set(roles))},
           "invalid_outputs": int(sum(x not in idx for x in pred)), "elapsed_s": round(elapsed, 1),
           "throughput_per_s": round(len(rows) / elapsed, 2), "prompt_source": str(V1_RUNNER.relative_to(_REPO_ROOT)),
           "config": zc}
    write_json(out_dir / f"zero_shot_{tag}_metrics.json", res)
    logger.info("%s", res)


if __name__ == "__main__":
    main()
