#!/usr/bin/env python3
"""
LMEIC D1 v2 Extraction: Reasoning-augmented, role-aware semantic feature extraction.

Version: v2.0 | Date: 2026-05-29

Motivation (see decision-journal D017 / literature S29):
  D1 v1 (8 generic fields, single-step JSON) suffered severe mode collapse:
  sentiment 97% "neutral" (dead), 6 of 13 intent categories ~0% (dead one-hot).
  Only ~4 of 8 fields carried discriminative signal.

D1 v2 redesign:
  1. Reasoning-augmented (CoST-inspired): the model first reasons briefly about
     OBJECTIVE handling-relevant factors, then emits structured features. The
     reasoning grounds extraction quality; it is discarded (only features kept),
     so the LLM still does NOT classify importance (respects design principle P1).
  2. Higher-entropy objective dimensions: dead fields (sentiment/formality/
     topic_category/relationship_tone) replaced by importance-discriminative
     objective descriptors (consequence_severity, request_specificity,
     decision_required, external_party_waiting, financial_mention, ...).
  3. Continuous 0-10 scales (vs 1-5) for resolution.
  4. Role-aware focusing: the recipient's role is provided so extraction attends
     to role-relevant content (role_relevance). This is attention focusing, not
     knowledge injection for classification (distinct from the failed OrgCond).

Input:  private/runs/run_20260526_123858/phase0_preprocessing/06_sampled/sampled_all.jsonl.gz
Output: private/runs/run_20260526_123858/lmeic_features_v2/d1_features_v2.jsonl
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "private/runs/run_20260526_123858"
INPUT_PATH = RUN_DIR / "phase0_preprocessing/06_sampled/sampled_all.jsonl.gz"
OUTPUT_DIR = RUN_DIR / "lmeic_features_v2"
OUTPUT_PATH = OUTPUT_DIR / "d1_features_v2.jsonl"

SCI_ENV = Path(__file__).resolve().parents[2] / ".env"

MAX_CONCURRENT = 16
MAX_RETRIES = 3
RETRY_DELAY = 2.0
REQUEST_TIMEOUT = 45.0
BATCH_SIZE = 50
BODY_MAX_CHARS = 3000
MODEL = "gpt-4o"

ROLE_DESC = {
    "Mgr": "a business manager who oversees client relationships, contracts, and exception handling",
    "CS": "an overseas customer service representative handling customer inquiries, complaints, and order follow-ups",
    "Sch": "a shipping schedule coordinator managing vessel schedules, booking slots, and cut-off times",
}

SYSTEM_PROMPT = """You are an email feature extraction assistant for a logistics/shipping company. \
Given an email and the recipient's role, you extract OBJECTIVE, structured semantic features.

CRITICAL RULES:
- You extract descriptive features ONLY. Do NOT output or imply an importance/priority label (no High/Medium/Low).
- All features describe what the text objectively contains or conveys, read through the lens of the recipient's role.
- Be objective, consistent, and grounded strictly in the email text.

PROCEDURE (Chain-of-Structured-Thought):
1. First, in "reasoning", briefly note (1-2 sentences) the objective handling-relevant factors present in the text: \
who is waiting or affected, what concrete action/decision is requested, what consequences are stated, and how specific any timing is.
2. Then emit the structured features below, grounded in that reasoning.

FEATURES (return as JSON):
- reasoning (string): your brief objective analysis (will be discarded)
- intent_type (string): primary communicative intent — one of: request, inform, confirm, inquiry, reminder, escalate, other
- urgency (int 0-10): time sensitivity conveyed by the language (0=none, 10=immediate)
- action_required (bool): does the email explicitly ask the recipient to do something
- decision_required (bool): does it require the recipient to make a decision/approval (not just an action)
- deadline_mentioned (bool): is a specific date/time deadline mentioned
- time_concreteness (int 0-10): how concrete the timing is (0=no timing, 5=vague 'soon/asap', 10=exact date+time)
- request_specificity (int 0-10): how concrete and actionable the request is (0=none, 10=fully specified)
- consequence_severity (int 0-10): magnitude of negative consequence stated/implied if not handled (0=none stated, 10=severe financial/legal/operational)
- external_party_waiting (bool): is an external party (customer, client, carrier, partner) affected or awaiting response
- financial_mention (bool): are monetary amounts, costs, payments, or charges mentioned
- direct_question (bool): does the email contain a direct question that expects an answer
- role_relevance (int 0-10): how relevant the email content is to the recipient role's core duties (0=unrelated, 10=core duty)

Return ONLY a JSON object with these fields."""

ONE_SHOT_USER = """Recipient role: a business manager who oversees client relationships, contracts, and exception handling

Subject: Re: URGENT - Container Damage Claim #CLM-2025-0892

Dear Mr. Johnson,

Following our call this morning, I need to escalate this matter. The damaged containers from voyage V2503E have not been addressed despite three previous emails. Our client is threatening legal action if we don't resolve this by Friday EOD.

Please authorize the $12,500 compensation immediately and confirm receipt of the survey report attached.

Best regards,
Sarah Chen
Claims Department"""

ONE_SHOT_ASSISTANT = json.dumps({
    "reasoning": "An external client is threatening legal action and awaits resolution; a specific monetary authorization ($12,500) and confirmation are requested by a concrete deadline (Friday EOD).",
    "intent_type": "escalate",
    "urgency": 9,
    "action_required": True,
    "decision_required": True,
    "deadline_mentioned": True,
    "time_concreteness": 8,
    "request_specificity": 9,
    "consequence_severity": 9,
    "external_party_waiting": True,
    "financial_mention": True,
    "direct_question": False,
    "role_relevance": 9,
}, ensure_ascii=False)

NUMERIC_FIELDS = ["urgency", "time_concreteness", "request_specificity",
                  "consequence_severity", "role_relevance"]
BOOL_FIELDS = ["action_required", "decision_required", "deadline_mentioned",
               "external_party_waiting", "financial_mention", "direct_question"]
REQUIRED_FIELDS = ["intent_type"] + NUMERIC_FIELDS + BOOL_FIELDS


def load_data() -> list[dict]:
    records = []
    with gzip.open(INPUT_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), INPUT_PATH)
    return records


def get_email_id(rec: dict) -> str:
    return rec.get("email_id") or f"{rec.get('source_id', '')}_{rec.get('original_uid', '')}"


def build_user_msg(rec: dict) -> str:
    role = rec.get("role", "")
    role_desc = ROLE_DESC.get(role, "an email recipient")
    subject = rec.get("subject", "(no subject)")
    body = rec.get("latest_text", "") or ""
    if len(body) > BODY_MAX_CHARS:
        body = body[:BODY_MAX_CHARS] + "\n[TRUNCATED]"
    return f"Recipient role: {role_desc}\n\nSubject: {subject}\n\n{body}"


def load_done_indices() -> set[int]:
    """Set of row_idx already extracted. row_idx makes alignment bulletproof
    against API failures, out-of-order completion and email_id collisions."""
    done: set[int] = set()
    if not OUTPUT_PATH.exists():
        return done
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "row_idx" in rec:
                done.add(int(rec["row_idx"]))
    return done


def coerce(parsed: dict) -> dict:
    """Coerce types and clamp numeric ranges."""
    out = {"intent_type": str(parsed.get("intent_type", "other")).lower().strip()}
    for f in NUMERIC_FIELDS:
        try:
            v = float(parsed.get(f, 0))
        except (TypeError, ValueError):
            v = 0.0
        out[f] = int(max(0, min(10, round(v))))
    for f in BOOL_FIELDS:
        v = parsed.get(f, False)
        if isinstance(v, str):
            v = v.strip().lower() in ("true", "yes", "1")
        out[f] = bool(v)
    return out


async def extract_one(client, rec: dict, row_idx: int,
                      semaphore: asyncio.Semaphore) -> dict | None:
    email_id = get_email_id(rec)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ONE_SHOT_USER},
        {"role": "assistant", "content": ONE_SHOT_ASSISTANT},
        {"role": "user", "content": build_user_msg(rec)},
    ]
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                resp = await asyncio.wait_for(
                    asyncio.to_thread(
                        client.chat.completions.create,
                        model=MODEL,
                        messages=messages,
                        temperature=0,
                        max_tokens=400,
                        response_format={"type": "json_object"},
                        timeout=REQUEST_TIMEOUT,
                    ),
                    timeout=REQUEST_TIMEOUT + 15,
                )
            parsed = json.loads(resp.choices[0].message.content.strip())
            missing = [f for f in REQUIRED_FIELDS if f not in parsed]
            if missing:
                logger.warning("Missing %s for %s", missing, email_id)
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                return None
            return {
                "row_idx": row_idx,
                "email_id": email_id,
                "role": rec.get("role", ""),
                "split": rec.get("split", ""),
                **coerce(parsed),
            }
        except json.JSONDecodeError as e:
            logger.warning("JSON error %s (attempt %d): %s", email_id, attempt + 1, e)
        except Exception as e:
            logger.warning("API error %s (attempt %d): %s", email_id, attempt + 1, e)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
    return None


async def main_async() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    done = load_done_indices()
    logger.info("Checkpoint: %d already extracted (row_idx resume)", len(done))
    records = load_data()
    remaining = [(i, r) for i, r in enumerate(records) if i not in done]
    logger.info("Total %d, done %d, remaining %d", len(records), len(done), len(remaining))
    if not remaining:
        logger.info("All done.")
        return

    load_dotenv(SCI_ENV)
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        timeout=REQUEST_TIMEOUT,
        max_retries=2,
    )
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    t0 = time.time()
    success = fail = 0

    with open(OUTPUT_PATH, "a", encoding="utf-8") as fout:
        for bs in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[bs:bs + BATCH_SIZE]
            results = await asyncio.gather(
                *[extract_one(client, r, idx, semaphore) for idx, r in batch],
                return_exceptions=True,
            )
            for res in results:
                if isinstance(res, Exception):
                    fail += 1
                    logger.error("Unhandled: %s", res)
                elif res is not None:
                    fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                    success += 1
                else:
                    fail += 1
            fout.flush()
            elapsed = time.time() - t0
            processed = success + fail
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - processed) / rate if rate > 0 else 0
            logger.info("Progress %d/%d (%.1f%%) | ok=%d fail=%d | %.1f/s | ETA=%.1f min",
                        processed, len(remaining), processed / len(remaining) * 100,
                        success, fail, rate, eta / 60)

    elapsed = time.time() - t0
    total_done = len(done) + success
    logger.info("D1 v2 done: %d ok, %d fail in %.1f min", success, fail, elapsed / 60)
    stats = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": MODEL,
        "schema_version": "v2_reasoning_augmented_role_aware",
        "total_records": len(records),
        "extracted": total_done,
        "this_run_success": success,
        "this_run_fail": fail,
        "elapsed_s": round(elapsed, 1),
    }
    with open(OUTPUT_DIR / "d1_v2_extraction_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main_async())
