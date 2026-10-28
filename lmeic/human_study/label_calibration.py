"""E19: can a few human labels correct the systematic gap between agent-trained models and humans?

Design (written before the run, experiment E19). Only the output layer is adjusted; no model is retrained.

  C0  no correction (reference)
  C1  class-prior shift: per-class log offset fitted on calibration folds (3 parameters)
  C2  vector scaling: per-class scale + bias on log-probabilities, multinomial LR with L2 (6 parameters)
  C3  decision thresholds for High and Low, grid search on calibration folds for human macro-F1

Protocol: 5 thread-grouped folds (StratifiedGroupKFold, seed 20261026); fit on 4 folds, predict the
fifth; macro-F1 against the human majority over the pooled out-of-fold predictions. Thread-cluster
bootstrap (2,000) for differences against C0. Primary comparison: R3 C2 - C0; others Holm-corrected.
Learning curve: C2 fitted on 100 / 200 / 400 human labels drawn by thread inside the training folds
(20 repeats).

Usage: python -m lmeic.human_study.label_calibration --e1-dir <E1 run dir>
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from lmeic.human_study.agreement import LAB, SYSTEMS, load_system, macro_f1
from lmeic.common.rev1_common import git_commit, norm_mid, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("e19")
_REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 20261026
N_BOOT = 2000
MODELS = ("R3_full", "R2_explicit", "E11_lora_qwen25_role")
EPS = 1e-6


def load_probs(kind: str, path: Path) -> dict[tuple[str, str], np.ndarray]:
    df = pd.read_parquet(path)
    if kind == "test":
        df = df.assign(set="test", id=df["message_id"])
    df["id"] = df["id"].map(lambda x: norm_mid(str(x)))
    g = df.groupby(["set", "id"])[["p_high", "p_med", "p_low"]].mean()
    return dict(zip(g.index, g.to_numpy()))


def fit_c1(P, y):
    prior_h = np.bincount(y, minlength=3) / len(y)
    prior_m = P.mean(0)
    return np.log(prior_h + EPS) - np.log(prior_m + EPS)


def apply_c1(P, b):
    return np.argmax(np.log(P + EPS) + b, axis=1)


def fit_c2(P, y):
    return LogisticRegression(C=1.0, max_iter=2000).fit(np.log(P + EPS), y)


def apply_c2(P, m):
    return m.predict(np.log(P + EPS))


def thr_predict(P, th, tl):
    pred = np.ones(len(P), int)
    pred[P[:, 2] >= tl] = 2
    pred[P[:, 0] >= th] = 0
    return pred


def fit_c3(P, y):
    grid = np.linspace(0.05, 0.95, 19)
    best = (-1, 0.5, 0.5)
    for th in grid:
        for tl in grid:
            f = macro_f1(y, thr_predict(P, th, tl))
            if f > best[0]:
                best = (f, th, tl)
    return best[1:]


def oof_predict(P, y, groups, method, rng=None, n_lab=None):
    pred = np.zeros(len(y), int)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(P, y, groups):
        if n_lab is not None:
            th = np.unique(groups[tr]); rng.shuffle(th)
            keep, n = [], 0
            for t in th:
                idx = tr[groups[tr] == t]
                keep.extend(idx); n += len(idx)
                if n >= n_lab:
                    break
            tr = np.array(keep)
        if method == "C0":
            pred[te] = P[te].argmax(1)
        elif method == "C1":
            pred[te] = apply_c1(P[te], fit_c1(P[tr], y[tr]))
        elif method == "C2":
            pred[te] = apply_c2(P[te], fit_c2(P[tr], y[tr]))
        elif method == "C3":
            pred[te] = thr_predict(P[te], *fit_c3(P[tr], y[tr]))
    return pred


def holm(p: np.ndarray) -> np.ndarray:
    o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-dir", required=True)
    args = ap.parse_args()
    e1 = Path(args.e1_dir) if Path(args.e1_dir).is_absolute() else _REPO_ROOT / args.e1_dir
    out = e1 / "e19_label_calibration"; out.mkdir(exist_ok=True)
    it = pd.read_json(e1 / "items_with_labels.jsonl", lines=True)
    it["pred_id"] = it["pred_id"].astype(str)
    it = it[it["human_majority"].notna() & it["agent_label"].notna()].reset_index(drop=True)
    y = it["human_majority"].map(LAB).to_numpy()
    ya = it["agent_label"].map(LAB).to_numpy()
    groups = it["thread_id"].astype(str).to_numpy()
    keys = list(zip(it["pred_set"], it["pred_id"]))

    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups); mem = {t: np.where(groups == t)[0] for t in uniq}
    boots = [np.concatenate([mem[t] for t in rng.choice(uniq, len(uniq))]) for _ in range(N_BOOT)]

    rows, comps, curve = [], [], []
    for name in MODELS:
        kind, rel = SYSTEMS[name]
        lut = load_probs(kind, _REPO_ROOT / rel)
        assert all(k in lut for k in keys), f"{name}: missing items"
        P = np.vstack([lut[k] for k in keys])
        preds = {m: oof_predict(P, y, groups, m) for m in ("C0", "C1", "C2", "C3")}
        for m, p in preds.items():
            rows.append({"model": name, "method": m, "n": len(y), "macro_f1_vs_human": macro_f1(y, p),
                         "accuracy_vs_human": float((p == y).mean()), "macro_f1_vs_agent": macro_f1(ya, p),
                         "pred_share": {k: float((p == LAB[k]).mean()) for k in LAB}})
        for m in ("C1", "C2", "C3"):
            d = np.array([macro_f1(y[b], preds[m][b]) - macro_f1(y[b], preds["C0"][b]) for b in boots])
            comps.append({"model": name, "comparison": f"{m} - C0", "diff": macro_f1(y, preds[m]) - macro_f1(y, preds["C0"]),
                          "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                          "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
                          "primary": name == "R3_full" and m == "C2"})
        for n_lab in (100, 200, 400):
            fs = [macro_f1(y, oof_predict(P, y, groups, "C2", rng=np.random.default_rng(SEED + r), n_lab=n_lab))
                  for r in range(20)]
            curve.append({"model": name, "n_labels": n_lab, "macro_f1_mean": float(np.mean(fs)),
                          "macro_f1_sd": float(np.std(fs))})
        logger.info("%s done", name)

    sec = [c for c in comps if not c["primary"]]
    for c, p in zip(sec, holm(np.array([c["p_two_sided"] for c in sec]))):
        c["p_holm"] = float(p)
    res = {"experiment": "E19", "n_items": len(y), "n_threads": len(uniq),
           "folds": "StratifiedGroupKFold(5) by thread, seed 20261026", "rows": rows, "comparisons": comps,
           "learning_curve_C2": curve, "git_commit": git_commit(_REPO_ROOT)}
    write_json(out / "summary.json", res)
    print(pd.DataFrame([{k: v for k, v in r.items() if k != "pred_share"} for r in rows]).round(4).to_string(index=False))
    for c in comps:
        print(f"{c['model']:22s} {c['comparison']}: {c['diff']*100:+.2f} pp [{c['ci95'][0]*100:+.2f}, {c['ci95'][1]*100:+.2f}] "
              f"p={c['p_two_sided']:.3f}{' PRIMARY' if c['primary'] else ''} {('holm=%.3f' % c['p_holm']) if 'p_holm' in c else ''}")
    print(pd.DataFrame(curve).round(4).to_string(index=False))
    for r in rows:
        print(r["model"], r["method"], {k: round(v, 3) for k, v in r["pred_share"].items()})


if __name__ == "__main__":
    main()
