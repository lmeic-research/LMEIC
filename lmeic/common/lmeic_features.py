"""
LMEIC v9: Clean re-baseline on a corrected data pipeline.
Version: 1.0 | 2026-05-29

Purpose
-------
Re-establish LMEIC on a corrected pipeline and isolate two upstream changes via
a controlled ablation that holds the (proven v6) model architecture fixed:

  1. D2/D3 temporal-leakage fix (sender_reply_rate + role_sender_freq now
     strictly before-date; see extract_d2d3_v2.py).
  2. D1 redesign: v1 (8 generic fields, mode-collapsed) vs v2 (reasoning-
     augmented, role-aware, higher-entropy schema; see extract_d1_v2.py).

Architecture (identical to v6, the 0.7611 baseline):
  XGBoost + LightGBM base learners (Optuna 5-fold CV tuned)
  → out-of-fold stacking → LogisticRegression meta-learner
  + interaction & role-conditional features.

Usage:
  python -m lmeic.common.lmeic_features v1   # old D1 + corrected D2/D3 (isolates leakage fix)
  python -m lmeic.common.lmeic_features v2   # new D1 v2 + corrected D2/D3 (full new pipeline)
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score as _f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import lightgbm as lgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lmeic.common.data_loader import (
    LABEL_NAMES,
    load_splits, load_d1_features, load_d2d3_features_v2,
    load_d1_features_v2, encode_d1, encode_d1_v2,
    get_d2d3_feature_cols, prepare_feature_matrix,
)
from lmeic.common.evaluator import (
    compute_metrics, aggregate_seed_metrics, save_results,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (REPO_ROOT / "private" / "runs" / "run_20260526_123858"
               / "v1_experiments" / "lmeic")
SEEDS = [42, 123, 456, 789, 2026]
N_TRIALS = 60

# Dead one-hot fields in v1 D1 (near-zero frequency); v2 has none by design.
DEAD_FEATURES_V1 = [
    "intent_type_booking", "intent_type_collaborative", "intent_type_complaint",
    "intent_type_schedule", "intent_type_suggest", "sentiment_mixed",
    "topic_category_complaint", "topic_category_inquiry", "topic_category_operational",
    "topic_category_process", "topic_category_shipment", "topic_category_logistics",
    "intent_type_escalate", "topic_category_shipping", "relationship_tone_escalatory",
]


def compute_sample_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    w = np.zeros(len(y))
    for c, cnt in zip(classes, counts):
        w[y == c] = total / (len(classes) * cnt)
    return w


def build_text_features(train_text, val_text, test_text):
    word_tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                 sublinear_tf=True, min_df=3, max_df=0.90, analyzer="word")
    word_svd = TruncatedSVD(n_components=40, random_state=42)
    tr_w = word_svd.fit_transform(word_tfidf.fit_transform(train_text))
    va_w = word_svd.transform(word_tfidf.transform(val_text))
    te_w = word_svd.transform(word_tfidf.transform(test_text))
    char_tfidf = TfidfVectorizer(max_features=30000, ngram_range=(2, 4),
                                 sublinear_tf=True, min_df=5, max_df=0.90, analyzer="char_wb")
    char_svd = TruncatedSVD(n_components=20, random_state=42)
    tr_c = char_svd.fit_transform(char_tfidf.fit_transform(train_text))
    va_c = char_svd.transform(char_tfidf.transform(val_text))
    te_c = char_svd.transform(char_tfidf.transform(test_text))
    return (np.hstack([tr_w, tr_c]), np.hstack([va_w, va_c]), np.hstack([te_w, te_c]))


def build_interactions(df, split_df, d1_version):
    """Interaction + role-conditional features (schema-aware)."""
    # primary continuous urgency signal differs by D1 version
    urg = "urgency" if d1_version == "v2" else "urgency_score"
    if urg in df.columns:
        if "d2_body_length_log" in df.columns:
            df["inter_urg_body"] = df[urg] * df["d2_body_length_log"]
        if "d2_sender_freq_30d" in df.columns:
            df["inter_urg_sfreq"] = df[urg] * df["d2_sender_freq_30d"]
        if "d3_role_keyword_count" in df.columns:
            df["inter_urg_rolekw"] = df[urg] * df["d3_role_keyword_count"]
        if "d2_thread_depth" in df.columns:
            df["inter_urg_thread"] = df[urg] * df["d2_thread_depth"]
    if "action_required" in df.columns and "d2_thread_depth" in df.columns:
        df["inter_action_thread"] = df["action_required"] * df["d2_thread_depth"]
    if "deadline_mentioned" in df.columns and "d2_sender_reply_rate" in df.columns:
        df["inter_deadline_reply"] = df["deadline_mentioned"] * df["d2_sender_reply_rate"]

    # v2-only richer interactions among new high-entropy fields
    if d1_version == "v2":
        if "consequence_severity" in df.columns and "external_party_waiting" in df.columns:
            df["inter_conseq_external"] = df["consequence_severity"] * df["external_party_waiting"]
        if "request_specificity" in df.columns and "action_required" in df.columns:
            df["inter_spec_action"] = df["request_specificity"] * df["action_required"]
        if "consequence_severity" in df.columns and "role_relevance" in df.columns:
            df["inter_conseq_role"] = df["consequence_severity"] * df["role_relevance"]
        if "decision_required" in df.columns and "financial_mention" in df.columns:
            df["inter_decision_fin"] = df["decision_required"] * df["financial_mention"]

    roles = split_df["role"].values if "role" in split_df.columns else None
    if roles is not None and urg in df.columns:
        for r in ["Mgr", "CS", "Sch"]:
            mask = (roles == r).astype(float)
            df[f"role_{r}_urgency"] = df[urg].values * mask
            if "d2_sender_freq_30d" in df.columns:
                df[f"role_{r}_sfreq"] = df["d2_sender_freq_30d"].values * mask
            if d1_version == "v2" and "consequence_severity" in df.columns:
                df[f"role_{r}_conseq"] = df["consequence_severity"].values * mask
    return df


def optuna_search(X_full, y_full, skf):
    def xgb_obj(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        }
        scores = []
        for tr, va in skf.split(X_full, y_full):
            m = XGBClassifier(**params, objective="multi:softmax", num_class=3,
                              eval_metric="mlogloss", tree_method="hist", device="cuda",
                              verbosity=0, random_state=42)
            m.fit(X_full[tr], y_full[tr], sample_weight=compute_sample_weights(y_full[tr]))
            scores.append(_f1_score(y_full[va], m.predict(X_full[va]), average="macro"))
        return float(np.mean(scores))

    def lgbm_obj(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 1500),
            "num_leaves": trial.suggest_int("num_leaves", 31, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 3.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 15.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.3),
        }
        scores = []
        for tr, va in skf.split(X_full, y_full):
            m = lgb.LGBMClassifier(**params, objective="multiclass", num_class=3,
                                   verbosity=-1, random_state=42, n_jobs=-1, is_unbalance=True)
            m.fit(X_full[tr], y_full[tr], sample_weight=compute_sample_weights(y_full[tr]),
                  eval_set=[(X_full[va], y_full[va])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            scores.append(_f1_score(y_full[va], m.predict(X_full[va]), average="macro"))
        return float(np.mean(scores))

    logger.info("Optuna XGB search (%d trials)...", N_TRIALS)
    xgb_study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    xgb_study.optimize(xgb_obj, n_trials=N_TRIALS, show_progress_bar=False)
    logger.info("  Best XGB CV: %.4f", xgb_study.best_value)
    logger.info("Optuna LGBM search (%d trials)...", N_TRIALS)
    lgbm_study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=123))
    lgbm_study.optimize(lgbm_obj, n_trials=N_TRIALS, show_progress_bar=False)
    logger.info("  Best LGBM CV: %.4f", lgbm_study.best_value)
    return xgb_study, lgbm_study


def run(d1_version: str):
    logger.info("=" * 70)
    logger.info("LMEIC v9 — d1_version=%s + leakage-corrected D2/D3", d1_version)
    logger.info("=" * 70)
    t0 = time.time()

    splits = load_splits(["train", "val", "test"])
    if d1_version == "v2":
        d1_raw = load_d1_features_v2()
        d1_encoded, d1_cols = encode_d1_v2(d1_raw)
    else:
        d1_raw = load_d1_features()
        d1_encoded, d1_cols = encode_d1(d1_raw)
    d2d3_raw = load_d2d3_features_v2()
    d2d3_cols = get_d2d3_feature_cols(d2d3_raw)

    mats = {}
    for s in ["train", "val", "test"]:
        X, y, base_cols, df = prepare_feature_matrix(
            splits[s], d1_encoded=d1_encoded, d1_feat_cols=d1_cols,
            d2d3_df=d2d3_raw, d2d3_feat_cols=d2d3_cols)
        mats[s] = (X, y, base_cols, df)
    base_cols = mats["train"][2]

    dead = DEAD_FEATURES_V1 if d1_version == "v1" else []
    keep = [c for c in base_cols if c not in dead]
    feats = {s: pd.DataFrame(mats[s][0], columns=base_cols)[keep].copy()
             for s in ["train", "val", "test"]}
    logger.info("Base features: %d (d1=%d, d2d3=%d)", len(keep), len(d1_cols), len(d2d3_cols))

    text = {s: splits[s]["latest_text"].fillna("").values for s in ["train", "val", "test"]}
    tr_t, va_t, te_t = build_text_features(text["train"], text["val"], text["test"])
    tcols = [f"word_svd_{i}" for i in range(40)] + [f"char_svd_{i}" for i in range(20)]
    for s, arr in zip(["train", "val", "test"], [tr_t, va_t, te_t]):
        feats[s] = pd.concat([feats[s].reset_index(drop=True),
                              pd.DataFrame(arr, columns=tcols)], axis=1)

    for s in ["train", "val", "test"]:
        feats[s] = build_interactions(feats[s], mats[s][3], d1_version)
    n_total = feats["train"].shape[1]
    logger.info("Final features: %d", n_total)

    y_train, y_val, y_test = mats["train"][1], mats["val"][1], mats["test"][1]
    X_full = np.vstack([feats["train"].values, feats["val"].values]).astype(np.float32)
    y_full = np.concatenate([y_train, y_val])
    X_test = feats["test"].values.astype(np.float32)
    np.nan_to_num(X_full, copy=False)
    np.nan_to_num(X_test, copy=False)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_study, lgbm_study = optuna_search(X_full, y_full, skf)
    best_xgb, best_lgbm = xgb_study.best_params, lgbm_study.best_params

    test_df = mats["test"][3]
    roles_test = test_df["role"].values
    mids_test = test_df["message_id"].values

    seed_metrics, all_preds = [], []
    for seed in SEEDS:
        oof_x = np.zeros((len(X_full), 3)); oof_l = np.zeros((len(X_full), 3))
        te_x = np.zeros((len(X_test), 3)); te_l = np.zeros((len(X_test), 3))
        skf_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, va in skf_s.split(X_full, y_full):
            sw = compute_sample_weights(y_full[tr])
            xm = XGBClassifier(**best_xgb, objective="multi:softprob", num_class=3,
                               eval_metric="mlogloss", tree_method="hist", device="cuda",
                               verbosity=0, random_state=seed)
            xm.fit(X_full[tr], y_full[tr], sample_weight=sw)
            oof_x[va] = xm.predict_proba(X_full[va]); te_x += xm.predict_proba(X_test) / 5
            lm = lgb.LGBMClassifier(**best_lgbm, objective="multiclass", num_class=3,
                                    verbosity=-1, random_state=seed, n_jobs=-1, is_unbalance=True)
            lm.fit(X_full[tr], y_full[tr], sample_weight=sw,
                   eval_set=[(X_full[va], y_full[va])],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
            oof_l[va] = lm.predict_proba(X_full[va]); te_l += lm.predict_proba(X_test) / 5
        meta = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
        meta.fit(np.hstack([oof_x, oof_l]), y_full)
        y_pred = meta.predict(np.hstack([te_x, te_l]))
        m = compute_metrics(y_test, y_pred, roles_test)
        seed_metrics.append(m)
        logger.info("  Seed %d: Macro-F1=%.4f | Mgr=%.3f CS=%.3f Sch=%.3f", seed,
                    m["overall"]["macro_f1"],
                    m["per_role"].get("Mgr", {}).get("macro_f1", 0),
                    m["per_role"].get("CS", {}).get("macro_f1", 0),
                    m["per_role"].get("Sch", {}).get("macro_f1", 0))
        if seed == SEEDS[0]:
            all_preds = [{"message_id": mid, "role": r, "gold": LABEL_NAMES[g],
                          "pred": LABEL_NAMES[p], "seed": seed}
                         for mid, r, g, p in zip(mids_test, roles_test, y_test, y_pred)]

    elapsed = time.time() - t0
    agg = aggregate_seed_metrics(seed_metrics)
    agg["n_test"] = len(y_test)
    agg["cost"] = {"usd_total": 0, "ms_per_sample": round(elapsed * 1000 / len(y_test) / len(SEEDS), 1)}
    config = {
        "experiment_id": f"LMEIC_v9_d1{d1_version}",
        "experiment_name": f"LMEIC v9 (D1 {d1_version} + leakage-corrected D2/D3)",
        "d1_version": d1_version,
        "d2d3_version": "v2_leakage_corrected",
        "architecture": "XGB+LGBM stacking -> LR meta (v6 proven)",
        "n_features_total": n_total,
        "xgb_cv_f1": xgb_study.best_value, "lgbm_cv_f1": lgbm_study.best_value,
        "best_xgb_params": best_xgb, "best_lgbm_params": best_lgbm,
        "random_seeds": SEEDS, "elapsed_s": round(elapsed, 1),
    }
    out_dir = OUTPUT_ROOT / f"LMEIC_v9_d1{d1_version}"
    save_results(out_dir, config["experiment_id"], config["experiment_name"], config, agg, all_preds)

    mean_f1, std_f1 = agg["overall"]["macro_f1"]["mean"], agg["overall"]["macro_f1"]["std"]
    logger.info("=" * 70)
    logger.info("LMEIC v9 (d1=%s) Macro-F1: %.4f +/- %.4f", d1_version, mean_f1, std_f1)
    logger.info("  Per-role: Mgr=%.4f CS=%.4f Sch=%.4f",
                agg["per_role"]["Mgr"]["macro_f1"]["mean"],
                agg["per_role"]["CS"]["macro_f1"]["mean"],
                agg["per_role"]["Sch"]["macro_f1"]["mean"])
    logger.info("  vs v6 0.7611: %+.2fpp | vs v7.0 0.7649: %+.2fpp",
                (mean_f1 - 0.7611) * 100, (mean_f1 - 0.7649) * 100)
    logger.info("=" * 70)
    return mean_f1, std_f1


if __name__ == "__main__":
    d1v = sys.argv[1] if len(sys.argv) > 1 else "v2"
    run(d1v)
