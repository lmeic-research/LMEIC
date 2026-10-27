"""Consensus rule and JSON parsing for the multi-agent labelling protocol.

Extracted unchanged from the original labelling pipeline (two-of-three majority; tolerant JSON
parsing of agent responses) so that the labelling script has no other dependency.
"""
from __future__ import annotations

import json
import re
from collections import Counter


def extract_json_from_text(text: str) -> dict | None:
    """Extract JSON from LLM response that may contain markdown, extra text, or truncation."""
    import re
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        inner = m.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            text = inner
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break
    if start >= 0:
        fragment = text[start:]
        m2 = re.search(r'"label"\s*:\s*"(High|Medium|Low|high|medium|low|HIGH|MEDIUM|LOW)"', fragment, re.IGNORECASE)
        if m2:
            label = m2.group(1).capitalize()
            conf_m = re.search(r'"confidence"\s*:\s*(\d)', fragment)
            confidence = int(conf_m.group(1)) if conf_m else 3
            rat_m = re.search(r'"rationale"\s*:\s*"([^"]*)', fragment)
            rationale = rat_m.group(1) if rat_m else ""
            return {"label": label, "confidence": confidence, "rationale": rationale}
    return None


def check_consensus(votes: list[dict]) -> tuple[str | None, str]:
    """Check if consensus is reached (3-agent: 2/3 majority). Returns (final_label, status)."""
    if len(votes) < 2:
        return None, "insufficient_votes"

    labels = [v["label"] for v in votes]
    counter = Counter(labels)
    most_common_label, most_common_count = counter.most_common(1)[0]
    n_agents = len(votes)

    if most_common_count == n_agents:
        return most_common_label, "unanimous"
    elif most_common_count >= 2:
        return most_common_label, "majority"
    else:
        return None, "no_consensus"
