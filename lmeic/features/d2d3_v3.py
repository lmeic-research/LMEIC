#!/usr/bin/env python3
"""
D2/D3 feature extraction v3: strictly pre-timestamp relational and role-context features.

Version: v3.0 | Date: 2026-10-22

Change vs v2 (extract_d2d3_v2.py):
  v2 defined d2_thread_position = thread_depth / (thread_size - 1), where
  thread_size is the size of the whole thread component, including messages
  sent after the target email. v3 removes d2_thread_position and adds
  d2_thread_prior_count = number of messages in the same thread dated strictly
  before the target email. All other features keep the v2 definitions; the
  domain lists and role keywords are read from configuration.

The module exposes HistoryIndex and compute_features() so that the same
definitions apply to EMAIL-3R v1.0 and to the revision evaluation sets.

Usage:
  python -m lmeic.features.d2d3_v3
"""

from __future__ import annotations

import bisect
import gzip
import json
import logging
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.common.rev1_common import (  # noqa: E402
    git_commit, load_config, new_output_dir, source_run_dir, write_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_NAME = "rev1_features_d2d3_v3.yaml"
POS_MAP = {"to": 2, "cc": 1, "bcc": 0, "none": 0}


def parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def read_jsonl_gz(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class HistoryIndex:
    """Sorted per-key timelines built from the redacted history pool."""

    def __init__(self, history: list[dict]):
        self.sender_role: dict[str, list[datetime]] = defaultdict(list)
        self.thread: dict[str, list[datetime]] = defaultdict(list)
        sent: dict[str, list[datetime]] = defaultdict(list)
        replied: dict[str, list[datetime]] = defaultdict(list)
        thread_msgs: dict[str, list[tuple[datetime, str, bool]]] = defaultdict(list)

        for rec in history:
            dt = parse_date(rec.get("date_utc"))
            if dt is None:
                continue
            sender = rec.get("from_addr", "") or ""
            tid = rec.get("thread_id", "") or ""
            self.sender_role[f"{sender}|{rec.get('role', '')}"].append(dt)
            self.thread[tid].append(dt)
            thread_msgs[tid].append((dt, sender, bool(rec.get("in_reply_to"))))

        for msgs in thread_msgs.values():
            msgs.sort(key=lambda x: x[0])
            for i, (dt, sender, is_reply) in enumerate(msgs):
                sent[sender].append(dt)
                if i > 0 and is_reply:
                    for prev_dt, prev_sender, _ in reversed(msgs[:i]):
                        if prev_sender != sender:
                            replied[prev_sender].append(dt)
                            break

        for d in (self.sender_role, self.thread):
            for k in d:
                d[k].sort()
        self.sent = {k: sorted(v) for k, v in sent.items()}
        self.replied = {k: sorted(v) for k, v in replied.items()}

    @classmethod
    def from_file(cls, path: Path) -> "HistoryIndex":
        keep = ("from_addr", "date_utc", "role", "thread_id", "in_reply_to")
        history = [{k: r.get(k) for k in keep} for r in read_jsonl_gz(path)]
        logger.info("History pool: %d records", len(history))
        return cls(history)

    def sender_freq(self, sender: str, role: str, t: datetime | None, days: int) -> int:
        if t is None:
            return 0
        dates = self.sender_role.get(f"{sender}|{role}", [])
        return max(0, bisect.bisect_left(dates, t) - bisect.bisect_left(dates, t - timedelta(days=days)))

    def sender_role_count(self, sender: str, role: str, t: datetime | None) -> int:
        if t is None:
            return 0
        return bisect.bisect_left(self.sender_role.get(f"{sender}|{role}", []), t)

    def reply_rate(self, sender: str, t: datetime | None) -> float:
        if t is None or sender not in self.sent:
            return 0.0
        n_sent = bisect.bisect_left(self.sent[sender], t)
        if n_sent == 0:
            return 0.0
        return bisect.bisect_left(self.replied.get(sender, []), t) / n_sent

    def thread_prior_count(self, thread_id: str, t: datetime | None) -> int:
        if t is None or not thread_id:
            return 0
        return bisect.bisect_left(self.thread.get(thread_id, []), t)


class FeatureSpec:
    def __init__(self, cfg: dict[str, Any], repo_root: Path):
        with (repo_root / cfg["partner_identifiers"]).open("r", encoding="utf-8") as f:
            partner = yaml.safe_load(f)
        self.internal = {d.lower() for d in partner.get("internal_domains", [])}
        self.carrier = {d.lower() for d in cfg["carrier_domains"]}
        self.kw = {
            role: re.compile(r"\b(" + "|".join(kws) + r")", re.IGNORECASE)
            for role, kws in cfg["role_keywords"].items()
        }

    def domain_type(self, addr: str) -> str:
        if not addr or "@" not in addr:
            return "other"
        dom = addr.split("@")[-1].lower().strip()
        if dom in self.internal:
            return "internal"
        if dom in self.carrier:
            return "carrier"
        return "external"


def compute_features(records: list[dict], idx: HistoryIndex, spec: FeatureSpec) -> pd.DataFrame:
    rows = []
    for rec in records:
        role = rec.get("role", "") or ""
        sender = rec.get("from_addr", "") or ""
        t = parse_date(rec.get("date_utc"))
        hour = int(rec.get("hour_of_day", 12) or 12)
        is_weekend = 1 if (rec.get("weekday") or "Monday") in ("Saturday", "Sunday") else 0
        dom = spec.domain_type(sender)
        lang = rec.get("language", "other") or "other"
        text = ((rec.get("subject") or "") + " " + (rec.get("latest_text") or "")).lower()
        pat = spec.kw.get(role)
        rows.append({
            "message_id": (rec.get("message_id") or "").strip(),
            "role": role,
            "d2_sender_freq_30d": idx.sender_freq(sender, role, t, 30),
            "d2_sender_freq_90d": idx.sender_freq(sender, role, t, 90),
            "d2_sender_reply_rate": round(idx.reply_rate(sender, t), 4),
            "d2_thread_depth": int(rec.get("thread_depth", 0) or 0),
            "d2_thread_prior_count": idx.thread_prior_count(rec.get("thread_id") or "", t),
            "d2_is_reply": 1 if rec.get("in_reply_to") else 0,
            "d2_to_count": int(rec.get("to_addrs_count", 0) or 0),
            "d2_cc_count": int(rec.get("cc_addrs_count", 0) or 0),
            "d2_has_attachment": 1 if rec.get("has_attachment") else 0,
            "d2_attachment_count": int(rec.get("attachment_count", 0) or 0),
            "d2_body_length_log": round(math.log1p(int(rec.get("latest_chars", 0) or 0)), 4),
            "d2_subject_length": len(rec.get("subject", "") or ""),
            "d2_hour_of_day": hour,
            "d2_is_weekend": is_weekend,
            "d2_is_business_hour": 1 if (not is_weekend and 9 <= hour <= 18) else 0,
            "d2_sender_domain_internal": int(dom == "internal"),
            "d2_sender_domain_carrier": int(dom == "carrier"),
            "d2_sender_domain_external": int(dom == "external"),
            "d2_target_user_position": POS_MAP.get((rec.get("target_user_position") or "none").lower(), 0),
            "d2_language_en": int(lang == "en"),
            "d2_language_zh": int(lang == "zh"),
            "d2_language_mixed": int(lang == "mixed"),
            "d2_references_count": int(rec.get("references_count", 0) or 0),
            "d3_role_Mgr": int(role == "Mgr"),
            "d3_role_CS": int(role == "CS"),
            "d3_role_Sch": int(role == "Sch"),
            "d3_role_keyword_count": len(pat.findall(text)) if pat else 0,
            "d3_role_sender_freq": idx.sender_role_count(sender, role, t),
        })
    return pd.DataFrame(rows)


def compare_with_v2(v3: pd.DataFrame, v2_path: Path) -> dict[str, Any]:
    v2 = pd.read_parquet(v2_path)
    assert len(v2) == len(v3), f"row mismatch v2={len(v2)} v3={len(v3)}"
    out: dict[str, Any] = {}
    for c in [c for c in v2.columns if c.startswith(("d2_", "d3_"))]:
        if c not in v3.columns:
            out[c] = "removed_in_v3"
            continue
        a, b = v2[c].astype(float).values, v3[c].astype(float).values
        out[c] = int(np.sum(~np.isclose(a, b, atol=1e-4)))
    for c in [c for c in v3.columns if c.startswith(("d2_", "d3_")) and c not in v2.columns]:
        out[c] = "added_in_v3"
    return out


def main() -> None:
    paths, cfg = load_config(CONFIG_NAME)
    repo_root = Path(paths["repo_root"])
    run_dir = source_run_dir(paths, cfg["source_run_id"])
    out_dir = new_output_dir(paths, cfg["output_prefix"])
    logger.info("Output: %s", out_dir)

    idx = HistoryIndex.from_file(run_dir / cfg["history_relpath"])
    spec = FeatureSpec(cfg, repo_root)
    sampled = list(read_jsonl_gz(run_dir / cfg["sampled_relpath"]))
    feats = compute_features(sampled, idx, spec)
    feats.to_parquet(out_dir / "d2d3_features_v3.parquet", index=False)

    diff = compare_with_v2(feats, run_dir / cfg["v2_features_relpath"])
    stats = {
        "config": cfg, "git_commit": git_commit(repo_root), "n_records": len(feats),
        "n_mismatch_vs_v2_per_feature": diff,
        "thread_prior_count": feats["d2_thread_prior_count"].describe().to_dict(),
    }
    write_json(out_dir / "build_stats.json", stats)
    logger.info("Mismatch vs v2: %s", {k: v for k, v in diff.items() if v})
    logger.info("Saved %s", out_dir / "d2d3_features_v3.parquet")


if __name__ == "__main__":
    main()
