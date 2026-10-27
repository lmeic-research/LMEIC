"""
Unified data loader for Phase 4 experiments.
Version: 1.1 | 2026-05-28

Loads EMAIL-3R v1.0 splits and merges with D1/D2D3 features.
All baselines use the same loader to ensure data consistency.

v1.1 fix: positional message_id injection replaces email_id join.
  mgr_v2 source used folder names as original_uid, collapsing 2,209
  records (14.7%) into 2 unique email_ids. Feature files are row-aligned
  with sampled_all.jsonl.gz, so message_id is injected by row position.
"""
import json
import gzip
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "datasets" / "EMAIL-3R" / "v1.0" / "data"
RUN_ROOT = REPO_ROOT / "datasets" / "email3r_runs" / "run_20260526_123858"
FEATURE_ROOT = RUN_ROOT / "tide_features"
SAMPLED_ALL = RUN_ROOT / "phase0_preprocessing" / "06_sampled" / "sampled_all.jsonl.gz"

LABEL_MAP = {"High": 0, "Medium": 1, "Low": 2}
LABEL_NAMES = ["High", "Medium", "Low"]

D1_CAT_COLS = ["intent_type", "sentiment", "topic_category", "relationship_tone"]
D1_NUM_COLS = ["urgency_score", "formality_level"]
D1_BOOL_COLS = ["action_required", "deadline_mentioned"]

# --- D1 v2 (reasoning-augmented, role-aware) schema ---
FEATURE_ROOT_V2 = RUN_ROOT / "tide_features_v2"
D1_V2_CAT_COLS = ["intent_type"]
D1_V2_NUM_COLS = ["urgency", "time_concreteness", "request_specificity",
                  "consequence_severity", "role_relevance"]
D1_V2_BOOL_COLS = ["action_required", "decision_required", "deadline_mentioned",
                   "external_party_waiting", "financial_mention", "direct_question"]

_MESSAGE_ID_CACHE: Optional[List[str]] = None


def _load_positional_message_ids() -> List[str]:
    """Load message_ids from sampled_all in row order (cached)."""
    global _MESSAGE_ID_CACHE
    if _MESSAGE_ID_CACHE is not None:
        return _MESSAGE_ID_CACHE
    mids = []
    with gzip.open(SAMPLED_ALL, "rt") as f:
        for line in f:
            mids.append(json.loads(line)["message_id"])
    _MESSAGE_ID_CACHE = mids
    logger.info(f"Positional message_id cache loaded: {len(mids)} entries")
    return mids


def load_split(split: str) -> pd.DataFrame:
    """Load a single split from EMAIL-3R v1.0."""
    path = DATA_ROOT / f"{split}.jsonl"
    records = [json.loads(line) for line in open(path)]
    df = pd.DataFrame(records)
    logger.info(f"Loaded {split}: {len(df)} records")
    return df


def load_splits(splits: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Load multiple splits."""
    if splits is None:
        splits = ["train", "val", "test"]
    return {s: load_split(s) for s in splits}


def load_d1_features() -> pd.DataFrame:
    """Load D1 features with positional message_id injection."""
    path = FEATURE_ROOT / "d1_features.jsonl"
    records = [json.loads(line) for line in open(path)]
    df = pd.DataFrame(records)

    mids = _load_positional_message_ids()
    assert len(df) == len(mids), (
        f"D1 rows ({len(df)}) != sampled_all rows ({len(mids)})")
    df["message_id"] = mids
    logger.info(f"D1 loaded: {len(df)} records, {df['message_id'].nunique()} unique message_ids")
    return df


def load_d2d3_features() -> pd.DataFrame:
    """Load D2+D3 features with positional message_id injection."""
    df = pd.read_parquet(FEATURE_ROOT / "d2d3_features.parquet")

    mids = _load_positional_message_ids()
    assert len(df) == len(mids), (
        f"D2D3 rows ({len(df)}) != sampled_all rows ({len(mids)})")
    df["message_id"] = mids
    logger.info(f"D2D3 loaded: {len(df)} records, {df['message_id'].nunique()} unique message_ids")
    return df


def encode_d1(df_d1: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Encode D1 raw features into numeric matrix. Returns (encoded_df, feature_names)."""
    encoded = pd.get_dummies(df_d1[D1_CAT_COLS], prefix_sep="_")
    for c in D1_NUM_COLS:
        encoded[c] = df_d1[c].astype(float)
    for c in D1_BOOL_COLS:
        encoded[c] = df_d1[c].astype(int)
    encoded["message_id"] = df_d1["message_id"].values
    feat_cols = [c for c in encoded.columns if c != "message_id"]
    return encoded, feat_cols


def load_d1_features_v2() -> pd.DataFrame:
    """Load D1 v2 (reasoning-augmented) features by email_id, then position-align.

    The async extractor preserves input order; we verify the email_id sequence
    matches sampled_all row-by-row before injecting positional message_id. If a
    mismatch is found (e.g. dropped failures), we fall back to a sequential
    alignment that skips the missing sampled rows.
    """
    path = FEATURE_ROOT_V2 / "d1_features_v2.jsonl"
    records = [json.loads(line) for line in open(path)]
    df = pd.DataFrame(records)
    mids = _load_positional_message_ids()

    # Align by row_idx (bulletproof against API fails / reordering / id collisions).
    assert "row_idx" in df.columns, "D1 v2 file missing row_idx"
    df = df.drop_duplicates(subset="row_idx", keep="first").reset_index(drop=True)
    df["message_id"] = df["row_idx"].map(lambda i: mids[int(i)])
    logger.info(f"D1 v2 loaded: {len(df)} records (row_idx aligned)")
    return df


def load_d2d3_features_v2() -> pd.DataFrame:
    """Load leakage-corrected D2+D3 v2 features with positional message_id."""
    df = pd.read_parquet(FEATURE_ROOT_V2 / "d2d3_features_v2.parquet")
    mids = _load_positional_message_ids()
    assert len(df) == len(mids), f"D2D3 v2 rows ({len(df)}) != sampled ({len(mids)})"
    df["message_id"] = mids
    logger.info(f"D2D3 v2 loaded: {len(df)} records")
    return df


def encode_d1_v2(df_d1: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Encode D1 v2 raw features into numeric matrix."""
    encoded = pd.get_dummies(df_d1[D1_V2_CAT_COLS], prefix_sep="_")
    for c in D1_V2_NUM_COLS:
        encoded[c] = pd.to_numeric(df_d1[c], errors="coerce").fillna(0).astype(float)
    for c in D1_V2_BOOL_COLS:
        encoded[c] = df_d1[c].astype(int)
    encoded["message_id"] = df_d1["message_id"].values
    feat_cols = [c for c in encoded.columns if c != "message_id"]
    return encoded, feat_cols


def get_d2d3_feature_cols(df: pd.DataFrame) -> List[str]:
    """Get D2+D3 feature column names."""
    return [c for c in df.columns if c.startswith("d2_") or c.startswith("d3_")]


def prepare_feature_matrix(
    split_df: pd.DataFrame,
    d1_encoded: Optional[pd.DataFrame] = None,
    d1_feat_cols: Optional[List[str]] = None,
    d2d3_df: Optional[pd.DataFrame] = None,
    d2d3_feat_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Build feature matrix X and label vector y for a given split.

    Returns:
        X, y, feature_names, aligned_df (with message_id, label_final, role)
    """
    keep_cols = ["message_id", "label_final"]
    if "role" in split_df.columns:
        keep_cols.append("role")
    df = split_df[keep_cols].copy()

    all_feat_cols = []

    if d1_encoded is not None and d1_feat_cols is not None:
        df = df.merge(d1_encoded[["message_id"] + d1_feat_cols], on="message_id", how="inner")
        all_feat_cols.extend(d1_feat_cols)

    if d2d3_df is not None and d2d3_feat_cols is not None:
        df = df.merge(d2d3_df[["message_id"] + d2d3_feat_cols], on="message_id", how="inner")
        all_feat_cols.extend(d2d3_feat_cols)

    df = df.dropna(subset=["label_final"])
    X = df[all_feat_cols].astype(float).fillna(0).values
    y = df["label_final"].map(LABEL_MAP).values

    logger.info(f"Feature matrix: {X.shape}, labels: {np.bincount(y)}")
    return X, y, all_feat_cols, df
