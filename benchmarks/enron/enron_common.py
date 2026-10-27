"""
Shared helpers for the E12 Enron recipient-aware benchmark.
Version: v1 | Date: 2026-10-23
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

CONFIG_NAME = "rev1_e12_enron.yaml"

_SUBJ_PREFIX = re.compile(r"^\s*((re|fw|fwd)\s*(\[\d+\])?\s*:\s*)+", re.I)
_QUOTE_MARKERS = [
    re.compile(r"-{2,}\s*Original Message\s*-{2,}", re.I),
    re.compile(r"-{2,}\s*Forwarded by", re.I),
    re.compile(r"\n\s*From:\s[^\n]*\n\s*(Sent|Date):", re.I),
    re.compile(r"\n[^\n]*\bwrote:\s*\n", re.I),
    re.compile(r"\n[^\n]{0,80}\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*[AP]M\s*\n\s*(To|Please respond to):", re.I),
    re.compile(r"\n\s*To:\s[^\n]*\n\s*cc:", re.I),
]
_ATTACH = re.compile(r"<<[^>\n]*>>|\b[\w\-]+\.(?:doc|xls|xlsx|ppt|pdf|zip)\b", re.I)


def norm_addr(a) -> str:
    return str(a or "").strip().strip("<>\"' ").lower()


def norm_subject(s) -> str:
    s = _SUBJ_PREFIX.sub("", str(s or ""))
    return re.sub(r"\s+", " ", s).strip().lower()


def split_latest(body: str) -> tuple[str, bool]:
    """Return the newest message part and whether the email is a pure forward/reply shell."""
    body = (body or "").replace("\r\n", "\n")
    body = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith(">"))
    cut = min((m.start() for p in _QUOTE_MARKERS for m in [p.search(body)] if m), default=len(body))
    latest = body[:cut].strip()
    return latest, cut < len(body)


def has_attachment(body: str) -> bool:
    return bool(_ATTACH.search(body or ""))


def ascii_ratio(text: str) -> float:
    return sum(ord(c) < 128 for c in text) / max(len(text), 1)


def load_positions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", encoding="latin-1", dtype=str).fillna("")
    df = df.rename(columns={"Ordner": "mailbox"})
    df["status"] = df["status"].str.strip().replace({"NULL": "N/A", "": "N/A"})
    df["addrs"] = df[["E-Mail1", "E-Mail2", "E-Mail3"]].apply(
        lambda r: sorted({norm_addr(x) for x in r if "@" in str(x)}), axis=1)
    return df[["eid", "firstName", "lastName", "mailbox", "status", "addrs"]]
