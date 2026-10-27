"""
E12 data assembly: labelled instances, encoded D1, context features, mined role n-grams,
interaction and role-conditional products.
Version: v1 | Date: 2026-10-23

Feature groups mirror the EMAIL-3R reference rebuild:
  d1          intent one-hot + numeric + boolean descriptors
  text        word/char TF-IDF + SVD (60), fitted on the development pool
  d2          header metadata and pre-t mailbox history (d2_*)
  d3          role one-hot, role-statement similarity, cross-fitted mined role n-grams (d3_*)
  interactions, role_cond   products built as in run_tide_v9.build_interactions (v2) with the
              Enron roles; a product is formed only when all of its factors are in the model
  coreader    XLM-R probabilities (plm_p_*)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lmeic.reference.folds import make_folds

LABEL_MAP = {"High": 0, "Medium": 1, "Low": 2}
INTENTS = ["request", "inform", "confirm", "inquiry", "reminder", "escalate", "other"]
NUMERIC = ["urgency", "time_concreteness", "request_specificity", "consequence_severity", "role_relevance"]
BOOLS = ["action_required", "decision_required", "deadline_mentioned", "external_party_waiting",
         "financial_mention", "direct_question"]
ROLES = ["EXE", "TRD", "STF"]


def load_labelled(run_dir: Path, labels_path: Path) -> pd.DataFrame:
    inst = pd.read_json(run_dir / "instances.jsonl.gz", lines=True, dtype={"instance_id": str})
    inst["row_idx"] = np.arange(len(inst))
    latest = {}
    for line in labels_path.read_text().splitlines():
        r = json.loads(line)
        if r.get("label_final"):
            latest[str(r["id"])] = r
    lab = pd.DataFrame([{"instance_id": k, "label_final": v["label_final"], "label_status": v["status"]}
                        for k, v in latest.items()])
    df = inst.merge(lab, on="instance_id", how="inner")
    df["y"] = df["label_final"].map(LABEL_MAP).astype(int)
    return df.reset_index(drop=True)


def load_d1(path: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Encoded descriptors aligned to df; local-extractor files are keyed by instance row position."""
    raw = pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])
    if raw["id"].astype(str).str.startswith("e12_").all():
        raw["instance_id"] = raw["id"].astype(str)
    else:
        raw["row_idx"] = raw["id"].astype(int)
        raw = raw.merge(df[["row_idx", "instance_id"]], on="row_idx", how="inner")
    raw = raw.drop_duplicates("instance_id", keep="last").set_index("instance_id")
    missing = sorted(set(df["instance_id"]) - set(raw.index))
    raw = raw.reindex(df["instance_id"])
    enc = pd.DataFrame(index=raw.index)
    intent = raw["intent_type"].where(raw["intent_type"].isin(INTENTS), "other")
    for it in INTENTS:
        enc[f"intent_type_{it}"] = (intent == it).astype(int)
    for c in NUMERIC:
        enc[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0).astype(float)
    for c in BOOLS:
        enc[c] = raw[c].fillna(False).astype(bool).astype(int)
    enc.attrs["missing"] = missing
    return enc.reset_index(drop=True)


def mined_features(df: pd.DataFrame, mcfg: dict) -> pd.DataFrame:
    """Cross-fitted on the development pool (thread-grouped folds); test from a full-dev miner."""
    import lmeic.features.context_mined_v1 as cm
    cm.ROLES = ROLES
    text = (df["subject"].fillna("") + "\n" + df["latest_text"].fillna("")).values
    roles, y = df["role"].values, df["y"].values
    dev = np.where(df["split"] == "dev")[0]
    te = np.where(df["split"] == "test")[0]
    out = np.zeros((len(df), 3))
    folds = make_folds(y[dev], df["thread_id"].values[dev], "grouped_clean", mcfg["fold_seed"], mcfg["n_folds"])
    for a, b in folds:
        out[dev[b]] = cm.RoleMiner(mcfg).fit(text[dev[a]], roles[dev[a]], y[dev[a]]).score(text[dev[b]], roles[dev[b]])
    out[te] = cm.RoleMiner(mcfg).fit(text[dev], roles[dev], y[dev]).score(text[te], roles[te])
    return pd.DataFrame({"d3_mined_high": out[:, 0], "d3_mined_low": out[:, 1], "d3_role_keyword_count": out[:, 2]})


def products(X: pd.DataFrame, roles: np.ndarray, groups: set[str]) -> pd.DataFrame:
    """Interaction and role-conditional products of run_tide_v9.build_interactions (v2)."""
    P = {}
    has = X.columns.__contains__
    if "interactions" in groups:
        pairs = [("inter_urg_body", "urgency", "d2_body_length_log"), ("inter_urg_sfreq", "urgency", "d2_sender_freq_30d"),
                 ("inter_urg_rolekw", "urgency", "d3_role_keyword_count"), ("inter_urg_thread", "urgency", "d2_thread_depth"),
                 ("inter_action_thread", "action_required", "d2_thread_depth"),
                 ("inter_deadline_reply", "deadline_mentioned", "d2_sender_reply_rate"),
                 ("inter_conseq_external", "consequence_severity", "external_party_waiting"),
                 ("inter_spec_action", "request_specificity", "action_required"),
                 ("inter_conseq_role", "consequence_severity", "role_relevance"),
                 ("inter_decision_fin", "decision_required", "financial_mention")]
        for name, a, b in pairs:
            if has(a) and has(b):
                P[name] = X[a].values * X[b].values
    if "role_cond" in groups:
        for r in ROLES:
            m = (roles == r).astype(float)
            for suffix, col in (("urgency", "urgency"), ("sfreq", "d2_sender_freq_30d"), ("conseq", "consequence_severity")):
                if has(col):
                    P[f"role_{r}_{suffix}"] = X[col].values * m
    return pd.DataFrame(P, index=X.index)


def text_features(df: pd.DataFrame) -> pd.DataFrame:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from lmeic.common.tide_features import build_text_features
    text = (df["subject"].fillna("") + "\n" + df["latest_text"].fillna("")).values
    dev, te = (df["split"] == "dev").values, (df["split"] == "test").values
    tr, _, tt = build_text_features(text[dev], text[dev][:1], text[te])
    cols = [f"word_svd_{i}" for i in range(40)] + [f"char_svd_{i}" for i in range(20)]
    out = pd.DataFrame(np.zeros((len(df), 60)), columns=cols)
    out.loc[dev, cols] = tr
    out.loc[te, cols] = tt
    return out
