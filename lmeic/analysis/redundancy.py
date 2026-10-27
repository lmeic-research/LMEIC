"""
E8: redundancy-aware architectures compared with early fusion (a reference variant).
Version: v1 | Date: 2026-10-22

Columns of the reference feature matrix are split into a content block (D1 descriptors,
lexical SVD components, D1-only interactions and, for the full variant, co-reader
probabilities), a pure-context block (D2, D3, role x sender-frequency) and the
content-by-context products (urgency/consequence x role, D1 x D2/D3 interactions). The
context block used by the fusion arms is pure context plus products, so every arm sees
the reference feature set. Every arm uses the reference variant's tuned parameters, the
thread-grouped protocol and the same seeds, so test predictions pair with the stored
reference predictions.

Arms:
  residual : content stacking gives cross-fitted class probabilities; their logarithm is
             the XGBoost base_margin / LightGBM init_score of boosters fitted on the context
             block only (early stopping on an inner thread-disjoint split), so context
             features model only what the content model leaves unexplained.
  late     : separate content and context stackings, merged by a logistic-regression meta
             learner on the six base-learner OOF probabilities of each block.
  gated    : logistic regression on log p_content, log p_context and
             max(p_content) * log p_context (cross-fitted meta probabilities).
  orth     : continuous D2/D3 columns replaced by residuals of a ridge regression on the
             D1 columns (fitted on the development pool); interactions keep raw values.
  select   : JMI and mRMR rankings on the development pool; stacking on the top-k
             features for each k; k is chosen by development-pool OOF macro-F1 only.
Paired comparisons with the reference use a thread-clustered bootstrap on test.

Usage:
  python -m lmeic.analysis.e8_redundancy --run-dir <rev1_reference run dir>
      [--variant R2_v3_grouped_explicit] [--arms residual,late,gated,orth,select]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.analysis import mi_selection as S  # noqa: E402
from lmeic.reference import features as F  # noqa: E402
from lmeic.reference import models as M  # noqa: E402
from lmeic.reference.analyse_reference import macro  # noqa: E402
from lmeic.reference.folds import inner_es_split, make_folds  # noqa: E402
from lmeic.common.rev1_common import git_commit, load_config, write_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
CONFIG_NAME = "rev1_reference.yaml"
LABELS = ["High", "Medium", "Low"]
CONTEXT_INTER = {"inter_urg_body", "inter_urg_sfreq", "inter_urg_rolekw", "inter_urg_thread",
                 "inter_action_thread", "inter_deadline_reply"}
CONTENT_ONLY_VARIANT = {"R2_v3_grouped_explicit": "E9_explicit_no_context",
                        "R3_v3_grouped_full": "E9_full_no_context"}
SELECT_K = [30, 50, 70, 90, 110]
EPS = 1e-6


def split_columns(cols: list[str]) -> tuple[list[int], list[int], list[int]]:
    """Content, pure-context (D2, D3, role x D2) and content-by-context product columns."""
    role_cond = [c.startswith(F.ROLE_COND_PREFIXES) for c in cols]
    pure = [i for i, c in enumerate(cols) if c.startswith(("d2_", "d3_")) or
            (role_cond[i] and c.endswith("_sfreq"))]
    hyb = [i for i, c in enumerate(cols) if c in CONTEXT_INTER or
           (role_cond[i] and not c.endswith("_sfreq"))]
    con = [i for i in range(len(cols)) if i not in set(pure) | set(hyb)]
    return con, pure, hyb


def base_stack(X, y, g, Xt, params, protocol, seed, folds, es_frac):
    """Six-column OOF (XGB, LGBM probabilities) on the pool and fold-averaged test scores."""
    oof = np.zeros((len(X), 6))
    te = np.zeros((len(Xt), 6))
    for tr, va in folds:
        xm = M._xgb(params["xgb"], seed, prob=True)
        xm.fit(X[tr], y[tr], sample_weight=M.sample_weights(y[tr]))
        oof[va, :3] = xm.predict_proba(X[va])
        te[:, :3] += xm.predict_proba(Xt) / len(folds)
        lm = M._fit_lgbm(M._lgbm(params["lgbm"], seed), X, y, tr, va, g, protocol, es_frac, seed)
        oof[va, 3:] = lm.predict_proba(X[va])
        te[:, 3:] += lm.predict_proba(Xt) / len(folds)
    return oof, te


def meta_probs(Z, Zt, y, folds, seed):
    """Cross-fitted meta probabilities on the pool and full-pool meta probabilities on test."""
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
    p_dev = cross_val_predict(lr, Z, y, cv=folds, method="predict_proba")
    lr.fit(Z, y)
    return p_dev, lr.predict_proba(Zt)


def residual_fit(Xk, y, g, m_dev, Xkt, m_test, params, seed, es_frac):
    fit, es = inner_es_split(np.arange(len(y)), g, es_frac, seed)
    xp = {**params["xgb"], "n_estimators": 2000}
    xm = XGBClassifier(**xp, objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                       tree_method="hist", device="cuda", verbosity=0, random_state=seed,
                       early_stopping_rounds=50)
    xm.fit(Xk[fit], y[fit], base_margin=m_dev[fit], eval_set=[(Xk[es], y[es])],
           base_margin_eval_set=[m_dev[es]], verbose=False)
    px = xm.predict_proba(Xkt, base_margin=m_test)
    lp = {**params["lgbm"], "n_estimators": 3000}
    lm = lgb.LGBMClassifier(**lp, objective="multiclass", num_class=3, verbosity=-1,
                            random_state=seed, n_jobs=-1)
    lm.fit(Xk[fit], y[fit], init_score=m_dev[fit], eval_set=[(Xk[es], y[es])],
           eval_init_score=[m_dev[es]], callbacks=[lgb.early_stopping(50, verbose=False)])
    pl = softmax(lm.predict(Xkt, raw_score=True) + m_test, axis=1)
    return (px + pl) / 2, {"xgb_best_iteration": int(xm.best_iteration),
                           "lgbm_best_iteration": int(lm.best_iteration_ or 0)}


def orthogonalise(X, Xt, cols, d1_cols):
    reg = [i for i, c in enumerate(cols) if c in set(d1_cols)]
    tgt = [i for i, c in enumerate(cols) if c.startswith(("d2_", "d3_")) and len(np.unique(X[:, i])) > 2]
    sc = StandardScaler().fit(X[:, reg])
    R, Rt = sc.transform(X[:, reg]), sc.transform(Xt[:, reg])
    Xo, Xto = X.copy(), Xt.copy()
    r2 = {}
    for j in tgt:
        m = Ridge(alpha=1.0).fit(R, X[:, j])
        pred = m.predict(R)
        var = X[:, j].var()
        r2[cols[j]] = float(1 - ((X[:, j] - pred) ** 2).mean() / var) if var > 0 else 0.0
        Xo[:, j] = X[:, j] - pred
        Xto[:, j] = Xt[:, j] - m.predict(Rt)
    return Xo, Xto, r2


def compare(y, P_arm, P_ref, threads, n_boot, seed) -> dict:
    rng = np.random.default_rng(seed)
    uniq = np.unique(threads)
    members = {t: np.where(threads == t)[0] for t in uniq}
    d_obs = macro(y, P_arm) - macro(y, P_ref)
    d = []
    for _ in range(n_boot):
        i = np.concatenate([members[t] for t in rng.choice(uniq, len(uniq), replace=True)])
        d.append(macro(y, P_arm, i) - macro(y, P_ref, i))
    d = np.array(d)
    return {"delta_vs_ref": d_obs, "ci95_low": float(np.percentile(d, 2.5)),
            "ci95_high": float(np.percentile(d, 97.5)),
            "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def save_preds(path: Path, proba: dict[int, np.ndarray], mids, roles, y) -> np.ndarray:
    rows, mats = [], []
    for s, P in proba.items():
        pred = P.argmax(1)
        mats.append(pred)
        for i in range(len(mids)):
            rows.append({"seed": s, "message_id": mids[i], "role": roles[i], "gold": LABELS[y[i]],
                         "pred": LABELS[int(pred[i])], "p_high": float(P[i, 0]),
                         "p_med": float(P[i, 1]), "p_low": float(P[i, 2])})
    pd.DataFrame(rows).to_parquet(path, index=False)
    return np.vstack(mats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", default="R2_v3_grouped_explicit")
    ap.add_argument("--arms", default="residual,late,gated,orth,select")
    ap.add_argument("--seeds", default=None, help="comma-separated; default: config seeds")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    arms = set(args.arms.split(","))
    run_dir = Path(args.run_dir)
    paths, cfg = load_config(CONFIG_NAME)
    repo = Path(paths["repo_root"])
    v = next(x for x in cfg["variants"] if x["name"] == args.variant)
    params = json.loads((run_dir / v["name"] / "params.json").read_text())
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["seeds"]
    protocol, nf, es_frac = v["protocol"], cfg["n_folds"], cfg["inner_es_fraction"]

    groups = set(v["groups"])
    probs = pd.read_parquet(run_dir / "coreader_probs_grouped.parquet") if "coreader" in groups else None
    feats, y, meta = F.build(cfg, repo, v["d2d3"], groups, probs)
    X, yy, g, Xt = F.pool(feats, y, meta)
    cols = list(feats["train"].columns)
    con, pure, hyb = split_columns(cols)
    ctx = pure + hyb
    _, d1_cols = F.load_d1()
    logger.info("%s: %d features (%d content, %d pure context, %d content x context)",
                v["name"], len(cols), len(con), len(pure), len(hyb))
    ref_cfg = run_dir / CONTENT_ONLY_VARIANT.get(v["name"], "") / "config.json"
    if ref_cfg.is_file():
        expected = set(json.loads(ref_cfg.read_text())["feature_columns"])
        assert expected == {cols[i] for i in con}, "content block differs from the content-only variant"

    yt, mids, roles = y["test"], meta["test"]["message_id"].values, meta["test"]["role"].values
    threads = meta["test"]["thread_id"].values
    ref = pd.read_parquet(run_dir / v["name"] / "test_predictions_all_seeds.parquet")
    lab = {l: i for i, l in enumerate(LABELS)}
    P_ref = np.vstack([ref[ref["seed"] == s].set_index("message_id").loc[mids, "pred"].map(lab).values
                       for s in seeds])

    out = run_dir / v["name"] / f"e8_redundancy_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir()
    summary, info = [], {"variant": v, "params_source": v["name"], "seeds": seeds,
                         "n_content": len(con), "n_pure_context": len(pure), "n_product": len(hyb),
                         "pure_context_columns": [cols[i] for i in pure],
                         "product_columns": [cols[i] for i in hyb], "git_commit": git_commit(repo)}

    def record(arm, proba, n_feat, n_models, extra=None):
        P = save_preds(out / f"pred_{arm}.parquet", proba, mids, roles, yt)
        per_seed = [macro(yt, P[k:k + 1]) for k in range(len(P))]
        row = {"arm": arm, "n_features": n_feat, "n_base_models": n_models,
               "macro_f1": macro(yt, P), "macro_f1_sd": float(np.std(per_seed)),
               **compare(yt, P, P_ref, threads, args.n_boot, 20261022), **(extra or {})}
        summary.append(row)
        pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
        logger.info("[%s] macro-F1 %.4f (delta %+.4f [%.4f, %.4f])", arm, row["macro_f1"],
                    row["delta_vs_ref"], row["ci95_low"], row["ci95_high"])

    if arms & {"residual", "late", "gated"}:
        res = {a: {} for a in ("content_only", "pure_context_only", "context_block_only",
                               "residual", "late", "gated")}
        iters = {}
        for seed in seeds:
            t0 = time.time()
            folds = make_folds(yy, g, protocol, seed, nf)
            oc, tc = base_stack(X[:, con], yy, g, Xt[:, con], params, protocol, seed, folds, es_frac)
            ok, tk = base_stack(X[:, ctx], yy, g, Xt[:, ctx], params, protocol, seed, folds, es_frac)
            op, tp = base_stack(X[:, pure], yy, g, Xt[:, pure], params, protocol, seed, folds, es_frac)
            pc_dev, pc_te = meta_probs(oc, tc, yy, folds, seed)
            pk_dev, pk_te = meta_probs(ok, tk, yy, folds, seed)
            res["pure_context_only"][seed] = meta_probs(op, tp, yy, folds, seed)[1]
            res["content_only"][seed], res["context_block_only"][seed] = pc_te, pk_te
            m_dev, m_te = np.log(np.clip(pc_dev, EPS, 1)), np.log(np.clip(pc_te, EPS, 1))
            res["residual"][seed], iters[seed] = residual_fit(X[:, ctx], yy, g, m_dev, Xt[:, ctx], m_te,
                                                              params, seed, es_frac)
            lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
            res["late"][seed] = lr.fit(np.hstack([oc, ok]), yy).predict_proba(np.hstack([tc, tk]))
            lk_dev, lk_te = np.log(np.clip(pk_dev, EPS, 1)), np.log(np.clip(pk_te, EPS, 1))
            G = np.hstack([m_dev, lk_dev, pc_dev.max(1, keepdims=True) * lk_dev])
            Gt = np.hstack([m_te, lk_te, pc_te.max(1, keepdims=True) * lk_te])
            lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
            res["gated"][seed] = lr.fit(G, yy).predict_proba(Gt)
            logger.info("seed %d: residual/late/gated done (%.0fs, residual iterations %s)",
                        seed, time.time() - t0, iters[seed])
        info["residual_iterations"] = iters
        record("content_only", res["content_only"], len(con), 2)
        record("pure_context_only", res["pure_context_only"], len(pure), 2)
        record("context_block_only", res["context_block_only"], len(ctx), 2)
        for a in ("residual", "late", "gated"):
            if a in arms:
                record(a, res[a], len(cols), 4)

    if "orth" in arms:
        Xo, Xto, r2 = orthogonalise(X, Xt, cols, d1_cols)
        info["orth_r2_on_d1"] = r2
        r = M.stacking(Xo, yy, g, Xto, params, protocol, seeds, nf, es_frac)
        record("orth", {s: r[s]["proba"] for s in seeds}, len(cols), 2,
               {"dev_oof_f1": float(np.mean([r[s]["oof_f1"] for s in seeds])),
                "mean_r2_context_on_d1": float(np.mean(list(r2.values())))})

    if "select" in arms:
        D = S.discretise(X)
        info["selection"] = {}
        for method, fn in (("jmi", S.rank_jmi), ("mrmr", S.rank_mrmr)):
            t0 = time.time()
            order = fn(D, yy, max(SELECT_K))
            info["selection"][method] = {"order": [cols[i] for i in order], "rank_s": round(time.time() - t0, 1)}
            best = None
            for k in SELECT_K:
                idx = order[:k]
                r = M.stacking(X[:, idx], yy, g, Xt[:, idx], params, protocol, seeds, nf, es_frac)
                dev = float(np.mean([r[s]["oof_f1"] for s in seeds]))
                n_ctx = sum(1 for i in idx if i in set(ctx))
                record(f"{method}_k{k}", {s: r[s]["proba"] for s in seeds}, k, 2,
                       {"dev_oof_f1": dev, "n_context_selected": n_ctx})
                if best is None or dev > best[1]:
                    best = (k, dev)
            info["selection"][method]["k_by_dev_oof"] = best[0]
            write_json(out / "info.json", info)

    write_json(out / "info.json", info)
    logger.info("saved %s", out)


if __name__ == "__main__":
    main()
