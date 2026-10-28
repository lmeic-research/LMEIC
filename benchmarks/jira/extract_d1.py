#!/usr/bin/env python3
"""
E18: D1 descriptor extraction for the Apache Jira issues with a local vLLM model.
Version: v1.1 | Date: 2026-10-24
v1.1: reads the sample from the shipped data/jira/sample_issue_ids.txt (the stratified sample drawn
      with run_baselines.load + stratified_sample) so it runs in the vLLM environment, which has no scikit-learn.

Same descriptor schema, one-shot example and coercion as the EMAIL-3R extractor
(extract_d1_v2 / d1_extract_local); only the domain framing and the role line change:
  - system prompt says "software issue-tracking system" instead of the logistics company
  - the role line is the issue's component (the nearest analogue of a recipient role in Jira),
    with the assignee-at-creation as a fallback; recorded in the output as role_basis
The model never sees the priority field. Resumable by row index.

Usage (GPU, vLLM environment):
  python -m benchmarks.jira.extract_d1 --model qwen25_7b \
      --processed datasets/public/apache_jira/processed [--limit N] [--sample-ids sample_issue_ids.txt]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.features.d1_extract_local import SCHEMA, sampling_params  # noqa: E402
from lmeic.features.extract_d1 import (  # noqa: E402
    BODY_MAX_CHARS, ONE_SHOT_ASSISTANT, ONE_SHOT_USER, REQUIRED_FIELDS, coerce,
)

from benchmarks.jira.sample_ids import DEFAULT_SAMPLE_IDS, read_sample_ids  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CONFIG_PATH = _REPO_ROOT / "configs" / "rev1_d1_local.yaml"

SYSTEM_PROMPT = """You are a text feature extraction assistant for a software issue-tracking system. \
Given an issue report and the component or team that owns it, you extract OBJECTIVE, structured semantic features.

CRITICAL RULES:
- You extract descriptive features ONLY. Do NOT output or infer a priority label (no Blocker/Critical/Major/Minor/Trivial, no High/Medium/Low).
- All features describe what the text objectively contains or conveys, read through the lens of the owning component or team.
- Be objective, consistent, and grounded strictly in the issue text.

PROCEDURE (Chain-of-Structured-Thought):
1. First, in "reasoning", briefly note (1-2 sentences) the objective handling-relevant factors present in the text: \
who is waiting or affected, what concrete action/decision is requested, what consequences are stated, and how specific any timing is.
2. Then emit the structured features below, grounded in that reasoning.

FEATURES (return as JSON):
- reasoning (string): your brief objective analysis (will be discarded)
- intent_type (string): primary communicative intent - one of: request, inform, confirm, inquiry, reminder, escalate, other
- urgency (int 0-10): time sensitivity conveyed by the language (0=none, 10=immediate)
- action_required (bool): does the text explicitly ask someone to do something
- decision_required (bool): does it require a decision or approval (not just an action)
- deadline_mentioned (bool): is a specific date/time deadline mentioned
- time_concreteness (int 0-10): how concrete the timing is (0=no timing, 5=vague 'soon/asap', 10=exact date+time)
- request_specificity (int 0-10): how concrete and actionable the request is (0=none, 10=fully specified)
- consequence_severity (int 0-10): magnitude of negative consequence stated or implied if not handled (0=none stated, 10=severe operational/security/financial)
- external_party_waiting (bool): is an external user, customer or downstream system affected or awaiting a response
- financial_mention (bool): are monetary amounts, costs, billing or licensing charges mentioned
- direct_question (bool): does the text contain a direct question that expects an answer
- role_relevance (int 0-10): how relevant the content is to the owning component's core duties (0=unrelated, 10=core duty)

Return ONLY a JSON object with these fields."""


def build_user_msg(rec: dict) -> str:
    body = rec.get("text", "") or ""
    if len(body) > BODY_MAX_CHARS:
        body = body[:BODY_MAX_CHARS] + "\n[TRUNCATED]"
    return f"Owning component: {rec['role_basis']}\n\nIssue type: {rec['type']}\n\n{body}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--processed", default="datasets/public/apache_jira/processed")
    ap.add_argument("--out-dir", default="datasets/public/apache_jira/d1")
    ap.add_argument("--sample-ids", default=DEFAULT_SAMPLE_IDS)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    mc = cfg["models"][args.model]
    proc = Path(args.processed)
    df = pd.read_parquet(proc / "issues.parquet")
    ids = read_sample_ids(proc, args.sample_ids)
    df["issue_id"] = df["issue_id"].astype(str)
    df = df.set_index("issue_id").loc[ids].reset_index()      # keeps the sample's time order
    if args.limit:
        df = df.head(args.limit)
    df["text"] = (df["summary"].fillna("") + " . " + df["description"].fillna("")).str.slice(0, 4000)
    df["role_basis"] = df["components"].fillna("").where(df["components"].fillna("") != "",
                                                         df["assignee_at_creation"].fillna("unassigned"))
    df["idx"] = range(len(df))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"d1_jira_{args.model}.jsonl"
    done = {json.loads(l)["id"] for l in out_path.read_text().splitlines() if l.strip()} if out_path.exists() else set()
    todo = [r for _, r in df.iterrows() if str(r["idx"]) not in done]
    logger.info("model=%s total=%d done=%d todo=%d", args.model, len(df), len(done), len(todo))
    if not todo:
        return

    from vllm import LLM
    llm = LLM(model=mc["hf_id"], dtype=mc["dtype"], max_model_len=mc["max_model_len"],
              gpu_memory_utilization=mc["gpu_memory_utilization"], seed=cfg["sampling"]["seed"])
    sp = sampling_params(cfg["sampling"])
    convs = [[{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": ONE_SHOT_USER},
              {"role": "assistant", "content": ONE_SHOT_ASSISTANT},
              {"role": "user", "content": build_user_msg(r)}] for r in todo]
    chat_kw = {"chat_template_kwargs": mc["chat_template_kwargs"]} if mc.get("chat_template_kwargs") else {}
    t0 = time.time(); ok = bad = 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for bs in range(0, len(todo), 2000):
            outs = llm.chat(convs[bs:bs + 2000], sp, use_tqdm=False, **chat_kw)
            for r, o in zip(todo[bs:bs + 2000], outs):
                try:
                    parsed = json.loads(o.outputs[0].text)
                    assert all(k in parsed for k in REQUIRED_FIELDS)
                except (json.JSONDecodeError, AssertionError):
                    bad += 1
                    continue
                fout.write(json.dumps({"id": str(r["idx"]), "issue_id": r["issue_id"], "project": r["project"],
                                       "role_basis": r["role_basis"][:120], **coerce(parsed),
                                       "model": mc["hf_id"]}, ensure_ascii=False) + "\n")
                ok += 1
            fout.flush()
            logger.info("progress %d/%d ok=%d invalid=%d (%.0fs)", min(bs + 2000, len(todo)), len(todo), ok, bad,
                        time.time() - t0)
    (out_dir / f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(
        {"model": args.model, "hf_id": mc["hf_id"], "n": len(df), "ok": ok, "invalid": bad,
         "elapsed_s": round(time.time() - t0, 1), "throughput_per_s": round(ok / max(time.time() - t0, 1e-9), 2),
         "sample_ids": args.sample_ids, "domain_prompt": "software issue-tracking"}, indent=2))
    logger.info("done ok=%d invalid=%d", ok, bad)


if __name__ == "__main__":
    main()
