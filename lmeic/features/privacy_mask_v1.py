#!/usr/bin/env python3
"""
E6: mask business identifiers in subject and latest_text before D1 extraction.

Version: v1 | Date: 2026-10-22

Applied in this order on top of the existing redaction:
  [CONTAINER]  ISO 6346 container numbers (4 letters + 7 digits)
  [VESSEL]     names after M/V, MV, VSL, vessel; "<NAME> V.<voyage>" patterns
  [COMPANY]    capitalised name sequences ending in a legal or industry suffix
               (Co. Ltd, Ltd, Limited, Inc, Corp, LLC, GmbH, Pte Ltd, Logistics, Shipping, Line(s));
               CJK names ending in the company / group suffixes
  [CARRIER]    configured public ocean-carrier names
  [NAME]       the name following a greeting (Dear / Hi / Hello / Attn)
  [REF]        upper-case alphanumeric references with both letters and digits (>= 6 chars):
               booking, B/L, order and invoice numbers
  [NUM]        digit runs of 6 or more
The output keeps row order and every other field of sampled_all, so descriptors can
be keyed by row index. Counts per mask type are written to mask_stats.json.

Usage:
  python -m lmeic.features.privacy_mask_v1 [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.common.rev1_common import (  # noqa: E402
    git_commit, load_config, new_output_dir, source_run_dir, write_json,
)
from lmeic.features.d2d3_v3 import read_jsonl_gz  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
CONFIG_NAME = "rev1_privacy_mask.yaml"

CJK_COMPANY_SUFFIX = "(?:\u6709\u9650\u516c\u53f8|\u80a1\u4efd\u516c\u53f8|\u96c6\u56e2|\u516c\u53f8)"
SUFFIX = (r"(?:Co\.?,?\s*Ltd\.?|Pte\.?\s*Ltd\.?|Ltd\.?|Limited|Inc\.?|Corp\.?|Corporation|LLC|GmbH|"
          r"Logistics|Shipping|Lines?)")
CAP = r"[A-Z][A-Za-z0-9&'\-]*"


def build_patterns(carriers: list[str]) -> list[tuple[str, re.Pattern, str]]:
    car = "|".join(re.escape(c) for c in sorted(carriers, key=len, reverse=True))
    return [
        ("CONTAINER", re.compile(r"\b[A-Z]{4}\d{7}\b"), "[CONTAINER]"),
        ("VESSEL", re.compile(r"\b(M/?V|MV|VSL|VESSEL|[Vv]essel)([:.\s]+)" + CAP + r"(?:\s+" + CAP + r"){0,3}"),
         r"\1\2[VESSEL]"),
        ("VESSEL", re.compile(r"\b" + CAP + r"(?:\s+" + CAP + r"){0,3}\s+(?:V\.|VOY\.?\s?)\s?\d{2,4}[A-Z]?\b"),
         "[VESSEL]"),
        ("COMPANY", re.compile(r"\b(?:" + CAP + r"\s+){1,5}" + SUFFIX + r"(?![A-Za-z])"), "[COMPANY]"),
        ("COMPANY", re.compile(r"[\u4e00-\u9fff]{2,20}" + CJK_COMPANY_SUFFIX), "[COMPANY]"),
        ("CARRIER", re.compile(r"(?<![A-Za-z])(?:" + car + r")(?![A-Za-z])"), "[CARRIER]"),
        ("NAME", re.compile(r"\b(Dear|Hi|Hello|Attn:?)(\s+)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"), r"\1\2[NAME]"),
        ("REF", re.compile(r"\b(?=[A-Z0-9/\-]*\d)(?=[A-Z0-9/\-]*[A-Z])[A-Z0-9][A-Z0-9/\-]{5,}\b"), "[REF]"),
        ("NUM", re.compile(r"(?<![\w\[])\d{6,}(?![\w\]])"), "[NUM]"),
    ]


def mask(text: str, patterns, counts: Counter) -> str:
    for name, pat, repl in patterns:
        text, n = pat.subn(repl, text)
        counts[name] += n
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    paths, cfg = load_config(CONFIG_NAME)
    out_dir = Path(args.out_dir) if args.out_dir else new_output_dir(paths, cfg["output_prefix"])
    out_dir.mkdir(parents=True, exist_ok=True)
    patterns = build_patterns(cfg["carrier_names"])

    counts, n, changed = Counter(), 0, 0
    chars_before = chars_after = 0
    out_path = out_dir / "sampled_all_masked.jsonl.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as fout:
        for r in read_jsonl_gz(source_run_dir(paths, cfg["source_run_id"]) / cfg["sampled_relpath"]):
            before = (r.get("subject") or "", r.get("latest_text") or "")
            r["subject"] = mask(before[0], patterns, counts)
            r["latest_text"] = mask(before[1], patterns, counts)
            changed += int((r["subject"], r["latest_text"]) != before)
            chars_before += sum(len(x) for x in before)
            chars_after += len(r["subject"]) + len(r["latest_text"])
            n += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(out_dir / "mask_stats.json", {
        "config": cfg, "git_commit": git_commit(Path(paths["repo_root"])), "n_records": n,
        "n_records_changed": changed, "share_changed": round(changed / max(n, 1), 4),
        "replacements": dict(counts), "chars_before": chars_before, "chars_after": chars_after})
    logger.info("masked %d records (%d changed): %s -> %s", n, changed, dict(counts), out_path)


if __name__ == "__main__":
    main()
