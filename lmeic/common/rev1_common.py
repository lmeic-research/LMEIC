"""
Shared helpers for revision-1 experiments: config loading, split indexes and
redacted-pool access.
Version: v1 | Date: 2026-10-22
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.common.paths import load_paths  # noqa: E402

SPLITS = ["train", "val", "test", "test_overlap_sci", "test_recent"]

POOL_FIELDS = [
    "message_id", "role", "subject", "from_display", "from_addr", "date_utc",
    "to_addrs_count", "cc_addrs_count", "has_attachment", "attachment_count",
    "thread_depth", "thread_id", "thread_size", "target_user_position",
    "language", "latest_text", "latest_chars", "is_auto_generated",
    "cross_role", "references_count",
]


def norm_mid(mid: Any) -> str:
    return (mid or "").strip()


def load_config(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = load_paths()
    cfg_path = Path(paths["repo_root"]) / "configs" / name
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    return paths, cfg


def release_data_dir(paths: dict[str, Any], version: str = "v1.0") -> Path:
    return Path(paths["output"]["release_dir"]) / version / "data"


def source_run_dir(paths: dict[str, Any], run_id: str) -> Path:
    return Path(paths["output"]["runs_base_dir"]) / run_id


def new_output_dir(paths: dict[str, Any], prefix: str) -> Path:
    out = Path(paths["output"]["runs_base_dir"]) / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def read_jsonl(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, records: list[dict]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_split_index(paths: dict[str, Any]) -> tuple[dict[str, list[dict]], dict[str, set[str]]]:
    """Return message_id -> list of split records and thread_id -> set of split names."""
    by_mid: dict[str, list[dict]] = {}
    threads: dict[str, set[str]] = {}
    data_dir = release_data_dir(paths)
    for split in SPLITS:
        for d in read_jsonl(data_dir / f"{split}.jsonl"):
            rec = {
                "split": split,
                "role": d.get("role"),
                "label_final": d.get("label_final"),
                "annotation_status": d.get("annotation_status"),
                "thread_id": d.get("thread_id"),
                "date_utc": d.get("date_utc"),
            }
            by_mid.setdefault(norm_mid(d.get("message_id")), []).append(rec)
            threads.setdefault(d.get("thread_id"), set()).add(split)
    return by_mid, threads


def load_pool(run_dir: Path, paths: dict[str, Any]) -> list[dict]:
    """Load the redacted, thread-tagged pool restricted to POOL_FIELDS."""
    src = run_dir / paths["output"]["phase0_subdirs"]["redacted"] / "with_threads.jsonl.gz"
    out = []
    for d in read_jsonl(src):
        r = {k: d.get(k) for k in POOL_FIELDS}
        r["message_id"] = norm_mid(r["message_id"])
        out.append(r)
    return out


def is_eligible(rec: dict, elig_cfg: dict) -> bool:
    if elig_cfg.get("exclude_auto_generated", True) and rec.get("is_auto_generated"):
        return False
    if int(rec.get("latest_chars") or 0) < int(elig_cfg.get("min_latest_chars", 30)):
        return False
    if not rec.get("date_utc"):
        return False
    return rec.get("role") in set(elig_cfg.get("roles", ["Mgr", "CS", "Sch"]))


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"
