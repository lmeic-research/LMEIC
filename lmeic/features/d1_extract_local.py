"""
D1 descriptor extraction with a locally served open-weight LLM (vLLM, offline).
Version: v1.1 | Date: 2026-10-22 (v1.1: --input for a row-aligned masked copy of sampled_all)

Same system prompt, one-shot example, input formatting and type coercion as
extract_d1_v2; decoding is constrained to the descriptor JSON schema. Rows are keyed
by their position in sampled_all (id = row index), matching the v1.0 D1 file, so the
output can replace the GPT-4o descriptors for all 15K instances. Resumable.

Usage (inside the vLLM environment):
  python -m lmeic.features.d1_extract_local --model qwen25_7b [--out-dir DIR] [--limit N]
      [--input datasets/.../sampled_all_masked.jsonl.gz --tag masked]
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]

from lmeic.features.extract_d1 import (  # noqa: E402
    BOOL_FIELDS, NUMERIC_FIELDS, ONE_SHOT_ASSISTANT, ONE_SHOT_USER, REQUIRED_FIELDS, SYSTEM_PROMPT,
    build_user_msg, coerce,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
CONFIG_PATH = _REPO_ROOT / "configs" / "rev1_d1_local.yaml"
PATHS_PATH = _REPO_ROOT / "configs" / "paths.yaml"
INTENTS = ["request", "inform", "confirm", "inquiry", "reminder", "escalate", "other"]

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 400},
        "intent_type": {"type": "string", "enum": INTENTS},
        **{f: {"type": "integer", "minimum": 0, "maximum": 10} for f in NUMERIC_FIELDS},
        **{f: {"type": "boolean"} for f in BOOL_FIELDS},
    },
    "required": ["reasoning"] + REQUIRED_FIELDS,
    "additionalProperties": False,
}


def runs_base() -> Path:
    raw = PATHS_PATH.read_text().replace("${repo_root}", str(_REPO_ROOT))
    return Path(yaml.safe_load(raw)["output"]["runs_base_dir"])


def sampling_params(sp_cfg: dict):
    from vllm import SamplingParams
    kw = {"temperature": sp_cfg["temperature"], "max_tokens": sp_cfg["max_tokens"], "seed": sp_cfg["seed"]}
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(structured_outputs=StructuredOutputsParams(json=SCHEMA), **kw)
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(guided_decoding=GuidedDecodingParams(json=SCHEMA), **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--input", default=None,
                    help="row-aligned replacement for sampled_all (jsonl.gz, relative to repo root)")
    ap.add_argument("--tag", default="", help="suffix of the default output directory name")
    args = ap.parse_args()
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    mc = cfg["models"][args.model]
    base = runs_base()
    tag = f"_{args.tag}" if args.tag else ""
    out_dir = Path(args.out_dir) if args.out_dir else \
        base / f"{cfg['output_prefix']}_{args.model}{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "d1_all.jsonl"

    src = _REPO_ROOT / args.input if args.input else base / cfg["source_run_id"] / cfg["sampled_relpath"]
    with gzip.open(src, "rt", encoding="utf-8") as f:
        records = [json.loads(l) for l in f]
    items = list(enumerate(records))[: args.limit] if args.limit else list(enumerate(records))
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text().splitlines() if l.strip()}
    todo = [(i, r) for i, r in items if str(i) not in done]
    logger.info("model=%s total=%d done=%d todo=%d -> %s", args.model, len(items), len(done), len(todo), out_path)
    if not todo:
        return

    from vllm import LLM
    llm = LLM(model=mc["hf_id"], dtype=mc["dtype"], max_model_len=mc["max_model_len"],
              gpu_memory_utilization=mc["gpu_memory_utilization"], seed=cfg["sampling"]["seed"])
    sp = sampling_params(cfg["sampling"])
    convs = [[{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": ONE_SHOT_USER},
              {"role": "assistant", "content": ONE_SHOT_ASSISTANT},
              {"role": "user", "content": build_user_msg(r)}] for _, r in todo]
    t0 = time.time()
    chat_kw = {"chat_template_kwargs": mc["chat_template_kwargs"]} if mc.get("chat_template_kwargs") else {}
    ok = bad = 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for bs in range(0, len(todo), 2000):
            outs = llm.chat(convs[bs:bs + 2000], sp, use_tqdm=False, **chat_kw)
            for (i, r), o in zip(todo[bs:bs + 2000], outs):
                text = o.outputs[0].text
                try:
                    parsed = json.loads(text)
                    assert all(k in parsed for k in REQUIRED_FIELDS)
                except (json.JSONDecodeError, AssertionError):
                    bad += 1
                    continue
                fout.write(json.dumps({"id": str(i), "row_idx": i, "message_id": r.get("message_id"),
                                       "role": r.get("role", ""), **coerce(parsed),
                                       "model": mc["hf_id"], "prompt_tokens": len(o.prompt_token_ids),
                                       "completion_tokens": len(o.outputs[0].token_ids)},
                                      ensure_ascii=False) + "\n")
                ok += 1
            fout.flush()
            logger.info("progress %d/%d ok=%d invalid=%d (%.0fs)", min(bs + 2000, len(todo)), len(todo), ok, bad,
                        time.time() - t0)
    stats = {"model": args.model, "hf_id": mc["hf_id"], "n_items": len(items), "previously_done": len(done),
             "ok": ok, "invalid_json": bad, "elapsed_s": round(time.time() - t0, 1),
             "throughput_per_s": round(ok / max(time.time() - t0, 1e-9), 2), "input": str(src), "config": cfg}
    (out_dir / f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(stats, indent=2))
    logger.info("done %s", stats)


if __name__ == "__main__":
    main()
