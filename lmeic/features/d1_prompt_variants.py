"""
Prompt and output-structure variants of the D1 v2 extraction prompt (revision 1, E5).
Version: v1.0 | Date: 2026-10-22

Every variant asks for the same twelve descriptors with the same value ranges, so that
outputs are directly comparable with the original prompt:
  original       the v2 system prompt, one-shot example, JSON mode
  paraphrase_a   reworded rules and field definitions, same structure
  paraphrase_b   analyst framing; definitions rephrased as questions
  paraphrase_c   field definitions first, rules last; scale anchors restated
  zero_shot      original system prompt without the one-shot example
  field_order    original prompt with the twelve fields listed in a fixed shuffled order
  unconstrained  original prompt without JSON mode; JSON is parsed from free text
  no_reasoning   original prompt without the reasoning step
"""

from __future__ import annotations

import json
import random
import re

from lmeic.features.extract_d1 import (
    BOOL_FIELDS, NUMERIC_FIELDS, ONE_SHOT_ASSISTANT, ONE_SHOT_USER, SYSTEM_PROMPT,
)

_ONE_SHOT = json.loads(ONE_SHOT_ASSISTANT)
FIELD_ORDER = [k for k in _ONE_SHOT if k != "reasoning"]
assert sorted(FIELD_ORDER) == sorted(["intent_type"] + NUMERIC_FIELDS + BOOL_FIELDS)

_FIELD_LINES = {
    "intent_type": "- intent_type (string): primary communicative intent — one of: request, inform, confirm, inquiry, reminder, escalate, other",
    "urgency": "- urgency (int 0-10): time sensitivity conveyed by the language (0=none, 10=immediate)",
    "action_required": "- action_required (bool): does the email explicitly ask the recipient to do something",
    "decision_required": "- decision_required (bool): does it require the recipient to make a decision/approval (not just an action)",
    "deadline_mentioned": "- deadline_mentioned (bool): is a specific date/time deadline mentioned",
    "time_concreteness": "- time_concreteness (int 0-10): how concrete the timing is (0=no timing, 5=vague 'soon/asap', 10=exact date+time)",
    "request_specificity": "- request_specificity (int 0-10): how concrete and actionable the request is (0=none, 10=fully specified)",
    "consequence_severity": "- consequence_severity (int 0-10): magnitude of negative consequence stated/implied if not handled (0=none stated, 10=severe financial/legal/operational)",
    "external_party_waiting": "- external_party_waiting (bool): is an external party (customer, client, carrier, partner) affected or awaiting response",
    "financial_mention": "- financial_mention (bool): are monetary amounts, costs, payments, or charges mentioned",
    "direct_question": "- direct_question (bool): does the email contain a direct question that expects an answer",
    "role_relevance": "- role_relevance (int 0-10): how relevant the email content is to the recipient role's core duties (0=unrelated, 10=core duty)",
}

PARAPHRASE_A = """You annotate emails for a shipping and logistics firm. For each email you receive, \
together with the role of the person who received it, you describe the content with objective structured features.

RULES:
- Only describe the email. Never give, suggest or hint at a priority or importance level (no High/Medium/Low).
- Read the email from the recipient role's point of view, but report only what the text actually says or conveys.
- Stay consistent and base every value on the email text alone.

STEPS:
1. In "reasoning", write one or two sentences on the objective facts that matter for handling the email: \
whether someone is waiting or affected, which concrete action or decision is asked for, which consequences are mentioned, and how precise any timing is.
2. Fill in the features below so that they agree with that reasoning.

FEATURES (JSON output):
- reasoning (string): short objective analysis (discarded later)
- intent_type (string): the main communicative purpose; choose one of: request, inform, confirm, inquiry, reminder, escalate, other
- urgency (int 0-10): how time-sensitive the wording is (0 = not at all, 10 = needs immediate attention)
- action_required (bool): true if the recipient is explicitly asked to do something
- decision_required (bool): true if the recipient must decide or approve something, beyond simply acting
- deadline_mentioned (bool): true if a specific deadline date or time appears
- time_concreteness (int 0-10): precision of any timing (0 = none, 5 = vague such as 'soon' or 'asap', 10 = exact date and time)
- request_specificity (int 0-10): how specific and actionable the request is (0 = no request, 10 = completely specified)
- consequence_severity (int 0-10): size of the negative outcome stated or implied if nothing is done (0 = none, 10 = severe financial, legal or operational harm)
- external_party_waiting (bool): true if an outside party (customer, client, carrier, partner) is affected or waiting for a reply
- financial_mention (bool): true if amounts of money, costs, payments or charges are mentioned
- direct_question (bool): true if the email asks a direct question that expects an answer
- role_relevance (int 0-10): how closely the content relates to the recipient role's core duties (0 = unrelated, 10 = a core duty)

Output nothing but the JSON object with these fields."""

PARAPHRASE_B = """Act as an operations analyst at a logistics and shipping company. You will see one email \
and a short description of the recipient's job. Your task is to record factual properties of the email in a fixed JSON format.

Do not judge how important the email is and do not output any priority label such as High, Medium or Low. \
Record only properties that can be read off the text, interpreted from the recipient's perspective.

Begin with a brief "reasoning" note (at most two sentences) covering: who is waiting or affected, what action \
or decision is requested, what consequences are mentioned, and how specific the timing is. Then answer each item below.

- reasoning (string): the brief note described above; it will not be used
- intent_type (string): What is the email mainly doing? One of: request, inform, confirm, inquiry, reminder, escalate, other
- urgency (int 0-10): How urgent does the language sound? 0 means no time pressure; 10 means it must be handled immediately.
- action_required (bool): Is the recipient explicitly asked to carry out some action?
- decision_required (bool): Must the recipient make a decision or give an approval, rather than only perform a task?
- deadline_mentioned (bool): Does the email state a specific deadline (date or time)?
- time_concreteness (int 0-10): How exact is the timing information? 0 means no timing; 5 means vague wording like 'soon' or 'asap'; 10 means an exact date and time.
- request_specificity (int 0-10): How clear and actionable is what is being requested? 0 means nothing is requested; 10 means fully specified.
- consequence_severity (int 0-10): How serious would the stated or implied consequences be if the email were not handled? 0 means none mentioned; 10 means severe financial, legal or operational damage.
- external_party_waiting (bool): Is a party outside the company (customer, client, carrier, partner) affected or waiting for an answer?
- financial_mention (bool): Does the email mention monetary amounts, costs, payments or charges?
- direct_question (bool): Does the email ask a direct question that expects a reply?
- role_relevance (int 0-10): How relevant is the content to the core duties of the recipient's role? 0 means unrelated; 10 means a core duty.

Reply with a single JSON object containing exactly these keys."""

PARAPHRASE_C = """Return one JSON object with the following keys describing the email shown by the user. \
The email was received by an employee of a logistics/shipping company whose role is given.

KEYS:
- reasoning (string): one or two sentences noting objective handling factors (affected or waiting parties, requested action or decision, stated consequences, timing precision); discarded after extraction
- intent_type (string): request | inform | confirm | inquiry | reminder | escalate | other — the dominant intent
- urgency (int 0-10): time pressure expressed in the text; 0 = none, 10 = immediate
- action_required (bool): an explicit request for the recipient to act is present
- decision_required (bool): the recipient has to decide or approve (more than just perform an action)
- deadline_mentioned (bool): a concrete date or time limit is given
- time_concreteness (int 0-10): 0 = no time reference, 5 = vague ('soon', 'asap'), 10 = exact date and time
- request_specificity (int 0-10): 0 = no request, 10 = request fully specified and actionable
- consequence_severity (int 0-10): severity of stated or implied harm if not handled; 0 = none stated, 10 = severe financial/legal/operational harm
- external_party_waiting (bool): a customer, client, carrier or partner is affected or awaiting a response
- financial_mention (bool): money amounts, costs, payments or charges appear
- direct_question (bool): there is a direct question awaiting an answer
- role_relevance (int 0-10): relevance to the recipient role's core duties; 0 = unrelated, 10 = core duty

GUIDELINES:
- Write "reasoning" first, then derive the remaining keys from it.
- These are descriptive features of the text read from the recipient role's perspective; they are not an importance rating, \
and no High/Medium/Low label may be given.
- Ground every value strictly in the email text and apply the scales consistently.

Output only the JSON object."""


def _shuffled_fields(seed: int = 20261022) -> list[str]:
    order = FIELD_ORDER.copy()
    random.Random(seed).shuffle(order)
    return order


def _with_field_list(fields: list[str], include_reasoning: bool) -> str:
    head, _, _ = SYSTEM_PROMPT.partition("FEATURES (return as JSON):")
    lines = (["- reasoning (string): your brief objective analysis (will be discarded)"] if include_reasoning else [])
    lines += [_FIELD_LINES[f] for f in fields]
    return head + "FEATURES (return as JSON):\n" + "\n".join(lines) + "\n\nReturn ONLY a JSON object with these fields."


def _one_shot(fields: list[str], include_reasoning: bool) -> str:
    out = {"reasoning": _ONE_SHOT["reasoning"]} if include_reasoning else {}
    out.update({f: _ONE_SHOT[f] for f in fields})
    return json.dumps(out, ensure_ascii=False)


_NO_REASONING_PROMPT = _with_field_list(FIELD_ORDER, include_reasoning=False).replace(
    "PROCEDURE (Chain-of-Structured-Thought):\n1. First, in \"reasoning\", briefly note (1-2 sentences) the objective "
    "handling-relevant factors present in the text: who is waiting or affected, what concrete action/decision is "
    "requested, what consequences are stated, and how specific any timing is.\n2. Then emit the structured features "
    "below, grounded in that reasoning.\n\n", "")


def build_messages(variant: str, user_msg: str) -> tuple[list[dict], bool]:
    """Return (chat messages, use_json_mode) for one email under the given variant."""
    system, shot, json_mode = SYSTEM_PROMPT, ONE_SHOT_ASSISTANT, True
    if variant == "original":
        pass
    elif variant == "paraphrase_a":
        system = PARAPHRASE_A
    elif variant == "paraphrase_b":
        system = PARAPHRASE_B
    elif variant == "paraphrase_c":
        system = PARAPHRASE_C
    elif variant == "zero_shot":
        shot = None
    elif variant == "field_order":
        fields = _shuffled_fields()
        system, shot = _with_field_list(fields, True), _one_shot(fields, True)
    elif variant == "unconstrained":
        json_mode = False
    elif variant == "no_reasoning":
        system, shot = _NO_REASONING_PROMPT, _one_shot(FIELD_ORDER, False)
    else:
        raise ValueError(f"unknown variant {variant}")
    msgs = [{"role": "system", "content": system}]
    if shot is not None:
        msgs += [{"role": "user", "content": ONE_SHOT_USER}, {"role": "assistant", "content": shot}]
    msgs.append({"role": "user", "content": user_msg})
    return msgs, json_mode


def parse_output(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences and surrounding prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return json.loads(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    raise json.JSONDecodeError("no JSON object found", text, 0)


VARIANTS = ["original", "paraphrase_a", "paraphrase_b", "paraphrase_c", "zero_shot", "field_order",
            "unconstrained", "no_reasoning"]
