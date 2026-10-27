"""
Export human annotations from the web tool database.
Version: v1.1 | Date: 2026-10-23 (v1.1: per-round export)

Writes a timestamped directory under <package_dir>/<export_subdir>/ containing
the current annotations of one round (jsonl and csv), the full revision history,
calibration discussion notes, the event log and export metadata. Earlier exports
are never overwritten.

Usage:
  python -m lmeic.human_study.annotation_store.export [--round r1]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.human_study.annotation_store.store import Store  # noqa: E402
from lmeic.common.rev1_common import (  # noqa: E402
    git_commit, load_config, write_json, write_jsonl,
)

CSV_COLUMNS = [
    "annotator", "item_id", "phase", "label", "suggested_label", "override_reason",
    "confidence", "note", "high_checks", "medium_checks", "active_seconds",
    "opened_at", "submitted_at", "revision", "server_elapsed_s", "step_seconds",
]


def package_paths(paths: dict, cfg: dict, round_id: str | None = None) -> tuple[Path, Path, Path]:
    pkg = Path(paths["output"]["runs_base_dir"]) / cfg["package_dir"]
    rnd = cfg["rounds"][round_id or cfg["active_round"]]
    return pkg, pkg / cfg["answer_key_relpath"], pkg / rnd["database_filename"]


def export_all(store: Store, out_base: Path, config_path: str, repo_root: Path, round_id: str) -> Path:
    out = out_base / f"export_{round_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    rows = sorted(store.all_rows(), key=lambda r: (r["annotator"], r["phase"], r["item_id"]))
    write_jsonl(out / "annotations.jsonl", rows)
    with (out / "annotations.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["high_checks"] = ";".join(r["high_checks"])
            r2["medium_checks"] = ";".join(r["medium_checks"])
            if isinstance(r.get("step_seconds"), dict):
                r2["step_seconds"] = json.dumps(r["step_seconds"])
            w.writerow({k: r2.get(k) for k in CSV_COLUMNS})
    write_jsonl(out / "annotation_history.jsonl", store.history_rows())
    write_json(out / "calibration_review_notes.json", store.review_notes())
    write_jsonl(out / "calibration_review_opinions.jsonl", store.review_opinion_history())
    write_jsonl(out / "events.jsonl", store.events())
    write_json(out / "export_meta.json", {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "round": round_id,
        "git_commit": git_commit(repo_root),
        "config_path": config_path,
        "database": str(store.db_path),
        "n_rows": len(rows),
        "rows_by_annotator_phase": dict(Counter(f"{r['annotator']}|{r['phase']}" for r in rows)),
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default=None, help="round id (default: active_round)")
    args = ap.parse_args()
    paths, cfg = load_config("rev1_annotation_web.yaml")
    round_id = args.round or cfg["active_round"]
    pkg, _, db = package_paths(paths, cfg, round_id)
    read_only = cfg["rounds"][round_id]["status"] == "frozen"
    out = export_all(Store(db, read_only=read_only), pkg / cfg["export_subdir"], cfg["_config_path"],
                     Path(paths["repo_root"]), round_id)
    print(f"exported to {out}")


if __name__ == "__main__":
    main()
