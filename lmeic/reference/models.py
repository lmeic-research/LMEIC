"""
Model fitting for the reference rebuild: Optuna tuning, XGB+LGBM stacking and single XGB.
Version: v1 | Date: 2026-10-22

Search spaces, base learners and the logistic-regression meta-learner follow the
EMAIL-3R v1.0 implementation (run_tide_v9.optuna_search, run_ablation.run_stacking).
The only protocol differences are the fold construction and, under grouped_clean,
LightGBM early stopping on an inner thread-disjoint split instead of the scored fold.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from .folds import inner_es_split, make_folds

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def sample_weights(y: np.ndarray) -> np.ndarray:
    cls, cnt = np.unique(y, return_counts=True)
    w = np.zeros(len(y))
    for c, n in zip(cls, cnt):
        w[y == c] = len(y) / (len(cls) * n)
    return w


def _xgb(params: dict, seed: int, prob: bool) -> XGBClassifier:
    return XGBClassifier(**params, objective="multi:softprob" if prob else "multi:softmax",
                         num_class=3, eval_metric="mlogloss", tree_method="hist", device="cuda",
                         verbosity=0, random_state=seed)


def _lgbm(params: dict, seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**params, objective="multiclass", num_class=3, verbosity=-1,
                              random_state=seed, n_jobs=-1, is_unbalance=True)


def _fit_lgbm(m, X, y, tr, va, groups, protocol, es_frac, seed):
    if protocol == "grouped_clean":
        fit, es = inner_es_split(tr, groups, es_frac, seed)
    else:
        fit, es = tr, va
    m.fit(X[fit], y[fit], sample_weight=sample_weights(y[fit]), eval_set=[(X[es], y[es])],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def tune(X, y, groups, protocol: str, n_folds: int, n_trials: int, es_frac: float,
         tune_lgbm: bool = True) -> dict:
    folds = make_folds(y, groups, protocol, 42, n_folds)

    def xgb_obj(trial):
        p = {
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
        s = []
        for tr, va in folds:
            m = _xgb(p, 42, prob=False)
            m.fit(X[tr], y[tr], sample_weight=sample_weights(y[tr]))
            s.append(f1_score(y[va], m.predict(X[va]), average="macro"))
        return float(np.mean(s))

    def lgbm_obj(trial):
        p = {
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
        s = []
        for tr, va in folds:
            m = _fit_lgbm(_lgbm(p, 42), X, y, tr, va, groups, protocol, es_frac, 42)
            s.append(f1_score(y[va], m.predict(X[va]), average="macro"))
        return float(np.mean(s))

    xs = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    xs.optimize(xgb_obj, n_trials=n_trials, show_progress_bar=False)
    logger.info("XGB best CV macro-F1 %.4f", xs.best_value)
    out = {"xgb": xs.best_params, "xgb_cv": xs.best_value}
    if tune_lgbm:
        ls = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=123))
        ls.optimize(lgbm_obj, n_trials=n_trials, show_progress_bar=False)
        logger.info("LGBM best CV macro-F1 %.4f", ls.best_value)
        out.update({"lgbm": ls.best_params, "lgbm_cv": ls.best_value})
    return out


def stacking(X, y, groups, X_test, params: dict, protocol: str, seeds: list[int],
             n_folds: int, es_frac: float) -> dict[int, dict]:
    """Return {seed: {"pred": test labels, "proba": meta test probabilities, "oof_f1": float}}."""
    out = {}
    for seed in seeds:
        folds = make_folds(y, groups, protocol, seed, n_folds)
        oof_x = np.zeros((len(X), 3)); oof_l = np.zeros((len(X), 3))
        te_x = np.zeros((len(X_test), 3)); te_l = np.zeros((len(X_test), 3))
        for tr, va in folds:
            xm = _xgb(params["xgb"], seed, prob=True)
            xm.fit(X[tr], y[tr], sample_weight=sample_weights(y[tr]))
            oof_x[va] = xm.predict_proba(X[va]); te_x += xm.predict_proba(X_test) / n_folds
            lm = _fit_lgbm(_lgbm(params["lgbm"], seed), X, y, tr, va, groups, protocol, es_frac, seed)
            oof_l[va] = lm.predict_proba(X[va]); te_l += lm.predict_proba(X_test) / n_folds
        meta = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
        z = np.hstack([oof_x, oof_l])
        meta.fit(z, y)
        zt = np.hstack([te_x, te_l])
        out[seed] = {"pred": meta.predict(zt), "proba": meta.predict_proba(zt),
                     "oof_f1": float(f1_score(y, meta.predict(z), average="macro"))}
        logger.info("seed %d done (OOF meta macro-F1 %.4f)", seed, out[seed]["oof_f1"])
    return out


def single_xgb(X, y, X_test, params: dict, seeds: list[int]) -> dict[int, dict]:
    out = {}
    for seed in seeds:
        m = _xgb(params["xgb"], seed, prob=True)
        m.fit(X, y, sample_weight=sample_weights(y))
        p = m.predict_proba(X_test)
        out[seed] = {"pred": p.argmax(1), "proba": p}
    return out
