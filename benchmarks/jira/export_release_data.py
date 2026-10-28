#!/usr/bin/env python3
"""
Jira benchmark: export (or verify) the shipped labels, time split and context features.
Version: v1.0

Writes data/jira/labels_splits_features.csv.gz with one row per sampled issue (the identifiers in
data/jira/sample_issue_ids.txt) and no issue text:

  issue_id, issue_key, project, type, created           identifiers and creation-time metadata
  priority_creation, label3, s2_non_default              creation-time priority, 3-class label
                                                         (Blocker/Critical = High, Major = Medium,
                                                         Minor/Trivial = Low), S2 membership (not Major)
  split                                                  time split by `created`: train < 2022,
                                                         val 2022, test >= 2023
  reporter_prior_n ... h_has_attachment                  the eight context features of the baselines
                                                         (run_baselines.HIST_COLS), computed on the full
                                                         project-selected table before sampling

Reporter and component names, issue text and the other raw fields are not exported; rebuild them
from the Zenodo dump with benchmarks.jira.build_issues.

Usage:
  python -m benchmarks.jira.export_release_data --processed datasets/public/apache_jira/processed          # write
  python -m benchmarks.jira.export_release_data --processed datasets/public/apache_jira/processed --check  # compare with shipped file
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.jira.run_baselines import HIST_COLS, MAP5, history_features, load  # noqa: E402
from benchmarks.jira.sample_ids import DEFAULT_SAMPLE_IDS, read_sample_ids  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DEFAULT_OUT = _REPO_ROOT / "data" / "jira" / "labels_splits_features.csv.gz"
ID_COLS = ["issue_id", "issue_key", "project", "type", "created"]
LABEL_COLS = ["priority_creation", "label3", "s2_non_default", "split"]


def split_of(created: pd.Series) -> pd.Series:
    """Time split of run_baselines.run (train < 2022, val 2022, test >= 2023)."""
    return pd.Series(np.select([created < pd.Timestamp("2022-01-01", tz="UTC"),
                                created < pd.Timestamp("2023-01-01", tz="UTC")], ["train", "val"], "test"),
                     index=created.index)


def build_table(processed: Path, sample_ids: str) -> pd.DataFrame:
    df = load(processed)
    df = df.join(history_features(df))              # full table, strictly earlier issues, before sampling
    df["issue_id"] = df["issue_id"].astype(str)
    ids = read_sample_ids(processed, sample_ids)
    df = df.set_index("issue_id").loc[ids].reset_index()
    df["split"] = split_of(df["created"])
    df["s2_non_default"] = (df["priority_creation"] != "Major").astype(int)
    df["created"] = df["created"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert set(df["priority_creation"]) <= set(MAP5)
    return df[ID_COLS + LABEL_COLS + HIST_COLS].reset_index(drop=True)


def write_gz(df: pd.DataFrame, path: Path) -> None:
    """Gzip with mtime 0 and no stored file name, so the bytes depend only on the content."""
    raw = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, compresslevel=9, mtime=0) as g:
        g.write(raw)
    path.write_bytes(buf.getvalue())


def check(new: pd.DataFrame, path: Path) -> bool:
    old = pd.read_csv(path, dtype={"issue_id": str})
    ok = list(old.columns) == list(new.columns) and len(old) == len(new)
    if ok:
        for c in ID_COLS + LABEL_COLS:
            if not (old[c].astype(str).values == new[c].astype(str).values).all():
                logger.error("column differs: %s", c); ok = False
        for c in HIST_COLS:
            if not np.allclose(old[c].values, new[c].values, rtol=0, atol=1e-9):
                logger.error("column differs: %s", c); ok = False
    logger.info("check against %s: %s (%d rows)", path, "IDENTICAL" if ok else "DIFFERENT", len(new))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="datasets/public/apache_jira/processed")
    ap.add_argument("--sample-ids", default=DEFAULT_SAMPLE_IDS)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--check", action="store_true", help="compare the rebuilt table with --out instead of writing it")
    args = ap.parse_args()
    tab = build_table(Path(args.processed), args.sample_ids)
    logger.info("rows %d | split %s", len(tab), tab["split"].value_counts().to_dict())
    out = Path(args.out)
    if args.check:
        sys.exit(0 if check(tab, out) else 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_gz(tab, out)
    logger.info("wrote %s (%d bytes)", out, out.stat().st_size)


if __name__ == "__main__":
    main()
