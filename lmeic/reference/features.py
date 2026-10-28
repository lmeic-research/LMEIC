"""
Feature assembly for the reference rebuild.
Version: v1.2 | Date: 2026-10-22
v1.1: the role-conditional filter no longer drops the D1 field role_relevance.
v1.3 (2026-10-28): group flag no_role_d1 drops the D1 field role_relevance and the interaction
      inter_conseq_role (E23: ablation without recipient-role information).
v1.2: groups d2meta / d2hist (D2 header metadata vs organisational history) and the E7
context blocks d2x / d2x_noprior / d3x read from features.context_rich_dir.

Reproduces the EMAIL-3R v1.0 LMEIC feature set (D1 v2, D2/D3, TF-IDF+SVD text,
interactions, role-conditional products) with a selectable D2/D3 version and an
optional co-reader probability block. Feature construction functions are
imported from the v1.0 code so that definitions stay identical.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]

from lmeic.common.data_loader import (  # noqa: E402
    _load_positional_message_ids, encode_d1_v2, get_d2d3_feature_cols,
    load_d1_features_v2, load_d2d3_features_v2, load_splits, prepare_feature_matrix,
)
from lmeic.common.lmeic_features import build_interactions, build_text_features  # noqa: E402

SPLIT_NAMES = ["train", "val", "test"]
PROB_COLS = ["plm_p_high", "plm_p_med", "plm_p_low"]
ROLE_COND_PREFIXES = ("role_Mgr_", "role_CS_", "role_Sch_")
D2_HIST = ["d2_sender_freq_30d", "d2_sender_freq_90d", "d2_sender_reply_rate", "d2_thread_prior_count"]
RICH_PARTS = ["history", "semantic", "mined"]


def _d2d3_selection(all_cols: list[str], groups: set[str]) -> list[str]:
    d2 = [c for c in all_cols if c.startswith("d2_")]
    sel = d2 if "d2" in groups else (
        ([c for c in d2 if c not in D2_HIST] if "d2meta" in groups else []) +
        ([c for c in d2 if c in D2_HIST] if "d2hist" in groups else []))
    return sel + ([c for c in all_cols if c.startswith("d3_")] if "d3" in groups else [])


def load_rich(cfg: dict[str, Any], repo_root: Path, groups: set[str], sets: list[str]) -> pd.DataFrame | None:
    """E7 context features (d2x_*, d3x_*) keyed by message_id (sampled) or instance_id."""
    prefixes = tuple(p for p, gname in (("d2x_", "d2x"), ("d2x_", "d2x_noprior"), ("d3x_", "d3x"))
                     if gname in groups)
    if not prefixes:
        return None
    drop_prior = "d2x_noprior" in groups and "d2x" not in groups
    base = repo_root / cfg["features"]["context_rich_dir"]
    frames = []
    for s in sets:
        parts = [pd.read_parquet(base / f"{p}_{s}.parquet").set_index("key") for p in RICH_PARTS]
        frames.append(pd.concat(parts, axis=1))
    df = pd.concat(frames)
    assert df.index.is_unique, "duplicate keys in context features"
    return df[[c for c in df.columns if c.startswith(prefixes) and not (drop_prior and c.startswith("d2x_prior_"))]]


def load_d2d3(version: str, cfg: dict[str, Any], repo_root: Path) -> pd.DataFrame:
    if version == "v2":
        return load_d2d3_features_v2()
    if version == "v3" or version.startswith("file:"):
        path = repo_root / cfg["features"]["d2d3_v3_path"] if version == "v3" else Path(version[5:])
        df = pd.read_parquet(path)
        mids = _load_positional_message_ids()
        assert len(df) == len(mids)
        assert all(a == (b or "").strip() for a, b in zip(df["message_id"], mids))
        df["message_id"] = mids
        return df
    raise ValueError(version)


def load_d1_extra(paths: list[str], repo_root: Path) -> dict[str, tuple[pd.DataFrame, list[str]]]:
    """Additional extractor views of D1, each encoded with the v1.0 D1 encoding.

    Each file is keyed by message_id and yields its own numeric block with a
    per-file prefix, so the model sees two independently extracted descriptions
    of the same email. Extractor errors are only partly correlated (pairwise
    kappa around 0.65 between GPT-4o and Qwen3), so the block adds information
    rather than duplicating the base D1 columns.
    """
    out: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    for spec in paths:
        name, _, relpath = spec.partition("=")
        assert relpath, f"d1_extra expects name=path, got {spec!r}"
        df = pd.read_json(repo_root / relpath, lines=True)
        for col in ("intent_type",):
            df[col] = df[col].where(df[col].isin(INTENTS), "other")
        enc, cols = encode_d1_v2(df)
        enc = enc.rename(columns={c: f"d1x_{name}_{c}" for c in cols})
        enc_cols = [f"d1x_{name}_{c}" for c in cols]
        assert enc["message_id"].is_unique, f"duplicate message_id in {relpath}"
        out[name] = (enc[["message_id"] + enc_cols], enc_cols)
    return out


def load_d1(d1_override: pd.DataFrame | None = None):
    """Encoded D1 v2; rows in d1_override (keyed by row_idx) replace the v1.0 values.

    The column layout is fixed to the v1.0 encoding so that models fitted on v1.0
    descriptors can score overridden rows.
    """
    d1 = load_d1_features_v2()
    base_enc, base_cols = encode_d1_v2(d1)
    if d1_override is None:
        return base_enc, base_cols
    d1 = d1.set_index("row_idx")
    ov = d1_override.set_index("row_idx")
    assert ov.index.is_unique and set(ov.index) <= set(d1.index)
    fields = [c for c in ov.columns if c in d1.columns and c not in ("message_id", "email_id", "role", "split")]
    for c in fields:
        d1[c] = d1[c].astype(object)
    d1.loc[ov.index, fields] = ov[fields].values
    enc, _ = encode_d1_v2(d1.reset_index())
    return enc.reindex(columns=base_cols + ["message_id"], fill_value=0), base_cols


def build(cfg: dict[str, Any], repo_root: Path, d2d3_version: str, groups: set[str],
          coreader_probs: pd.DataFrame | None = None, d1_override: pd.DataFrame | None = None,
          d1_extra: dict[str, tuple[pd.DataFrame, list[str]]] | None = None):
    """Return feats{split: DataFrame}, y{split: array}, meta{split: DataFrame}."""
    splits = load_splits(SPLIT_NAMES)
    d1_enc, d1_cols = load_d1(d1_override)
    d2d3 = load_d2d3(d2d3_version, cfg, repo_root)
    all_cols = get_d2d3_feature_cols(d2d3)
    sel = _d2d3_selection(all_cols, groups)
    want_d1 = "d1" in groups

    mats = {}
    for s in SPLIT_NAMES:
        X, y, cols, df = prepare_feature_matrix(
            splits[s], d1_encoded=d1_enc if want_d1 else None,
            d1_feat_cols=d1_cols if want_d1 else None,
            d2d3_df=d2d3 if sel else None, d2d3_feat_cols=sel if sel else None)
        assert splits[s]["message_id"].is_unique, f"{s}: duplicate message_id"
        df = df.merge(splits[s][["message_id", "thread_id"]], on="message_id", how="left")
        assert len(df) == len(X) == len(splits[s]), f"{s}: row mismatch"
        mats[s] = (X, y, cols, df)
    cols = mats["train"][2]
    feats = {s: pd.DataFrame(mats[s][0], columns=cols) for s in SPLIT_NAMES}


    if "text" in groups:
        text = {s: splits[s].set_index("message_id").loc[mats[s][3]["message_id"], "latest_text"]
                .fillna("").values for s in SPLIT_NAMES}
        tr_t, va_t, te_t = build_text_features(text["train"], text["val"], text["test"])
        tcols = [f"word_svd_{i}" for i in range(40)] + [f"char_svd_{i}" for i in range(20)]
        for s, arr in zip(SPLIT_NAMES, [tr_t, va_t, te_t]):
            feats[s] = pd.concat([feats[s], pd.DataFrame(arr, columns=tcols)], axis=1)

    for s in SPLIT_NAMES:
        feats[s] = build_interactions(feats[s], mats[s][3], "v2")
        rm = []
        if "interactions" not in groups:
            rm += [c for c in feats[s].columns if c.startswith("inter_")]
        if "role_cond" not in groups:
            rm += [c for c in feats[s].columns if c.startswith(ROLE_COND_PREFIXES)]
        if "no_role_d1" in groups:
            rm += [c for c in ("role_relevance", "inter_conseq_role") if c in feats[s].columns]
        if rm:
            feats[s] = feats[s].drop(columns=rm)

    rich = load_rich(cfg, repo_root, groups, ["sampled"])
    if rich is not None:
        for s in SPLIT_NAMES:
            keys = [str(m).strip() for m in mats[s][3]["message_id"].values]
            missing = [k for k in keys if k not in rich.index]
            assert not missing, f"{s}: {len(missing)} rows without context features"
            feats[s] = pd.concat([feats[s], rich.loc[keys].reset_index(drop=True).astype(float)], axis=1)

    if "coreader" in groups:
        assert coreader_probs is not None, "co-reader probabilities required"
        lut = coreader_probs.set_index("message_id")[PROB_COLS]
        for s in SPLIT_NAMES:
            mids = mats[s][3]["message_id"].values
            missing = [m for m in mids if m not in lut.index]
            assert not missing, f"{s}: {len(missing)} message_ids without co-reader scores"
            feats[s][PROB_COLS] = lut.loc[mids].values

    if d1_extra:
        for s in SPLIT_NAMES:
            mids = mats[s][3]["message_id"].values
            for name, (enc, enc_cols) in d1_extra.items():
                lut = enc.set_index("message_id")
                missing = [m for m in mids if m not in lut.index]
                assert not missing, f"{s}: {len(missing)} rows without d1x_{name} descriptors"
                block = lut.loc[mids, enc_cols].reset_index(drop=True).astype(float)
                feats[s] = pd.concat([feats[s], block], axis=1)

    y = {s: mats[s][1] for s in SPLIT_NAMES}
    meta = {s: mats[s][3][["message_id", "role", "thread_id"]].reset_index(drop=True)
            for s in SPLIT_NAMES}
    return feats, y, meta


INTENTS = ["request", "inform", "confirm", "inquiry", "reminder", "escalate", "other"]


def build_external(cfg: dict[str, Any], repo_root: Path, groups: set[str], train_cols: list[str],
                   ext: pd.DataFrame, d1_rows: pd.DataFrame, d2d3_ext: pd.DataFrame,
                   coreader_ext: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature matrix for instances outside EMAIL-3R, column-identical to the training matrix.

    ext: one row per instance with instance_id, role, latest_text.
    d1_rows: raw D1 descriptors keyed by id (= instance_id); d2d3_ext: v3 features keyed by
    instance_id; coreader_ext: co-reader probabilities keyed by instance_id.
    """
    ids = ext["instance_id"].astype(str).values
    feats = pd.DataFrame(index=range(len(ext)))
    if "d1" in groups:
        _, d1_cols = load_d1()
        raw = d1_rows.assign(id=d1_rows["id"].astype(str)).drop_duplicates("id").set_index("id")
        missing = set(ids) - set(raw.index)
        assert not missing, f"{len(missing)} instances without D1 descriptors"
        raw = raw.loc[ids].reset_index(drop=True)
        raw["intent_type"] = raw["intent_type"].where(raw["intent_type"].isin(INTENTS), "other")
        raw["message_id"] = ids
        enc, _ = encode_d1_v2(raw)
        feats = pd.concat([feats, enc.reindex(columns=d1_cols, fill_value=0).astype(float)], axis=1)
    d2d3_cols = [c for c in train_cols if c.startswith(("d2_", "d3_")) and c in d2d3_ext.columns]
    if d2d3_cols:
        dd = d2d3_ext.assign(instance_id=d2d3_ext["instance_id"].astype(str)).set_index("instance_id")
        feats = pd.concat([feats, dd.loc[ids, d2d3_cols].reset_index(drop=True).astype(float)], axis=1)

    if "text" in groups:
        splits = load_splits(["train", "val"])
        _, _, ext_t = build_text_features(splits["train"]["latest_text"].fillna("").values,
                                          splits["val"]["latest_text"].fillna("").values,
                                          ext["latest_text"].fillna("").values)
        tcols = [f"word_svd_{i}" for i in range(40)] + [f"char_svd_{i}" for i in range(20)]
        feats = pd.concat([feats, pd.DataFrame(ext_t, columns=tcols)], axis=1)

    feats = build_interactions(feats, ext[["role"]].reset_index(drop=True), "v2")
    rich = load_rich(cfg, repo_root, groups, list(cfg["external_sets"]["sets"]))
    if rich is not None:
        feats = pd.concat([feats, rich.loc[ids].reset_index(drop=True).astype(float)], axis=1)
    if "coreader" in groups:
        assert coreader_ext is not None, "co-reader probabilities required"
        lut = coreader_ext.assign(instance_id=coreader_ext["instance_id"].astype(str)).set_index("instance_id")
        feats[PROB_COLS] = lut.loc[ids, PROB_COLS].values
    feats = feats[[c for c in feats.columns if c in train_cols]]
    assert list(feats.columns) == list(train_cols), "external feature columns differ from training columns"
    return feats


def pool(feats, y, meta):
    """Stack train and val into the development pool used for CV and final fitting."""
    X_full = np.vstack([feats["train"].values, feats["val"].values]).astype(np.float32)
    y_full = np.concatenate([y["train"], y["val"]])
    g_full = np.concatenate([meta["train"]["thread_id"].values, meta["val"]["thread_id"].values])
    X_test = feats["test"].values.astype(np.float32)
    np.nan_to_num(X_full, copy=False)
    np.nan_to_num(X_test, copy=False)
    return X_full, y_full, g_full, X_test
