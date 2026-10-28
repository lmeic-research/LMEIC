"""Predictive / interaction information of D1, D2, D3 recomputed on the revision-1 features.

v1.0 (framework/analysis/info_decomposition.py) used the v2 D2/D3 features, whose
d2_thread_position used the full thread size (future information), and stratified folds that
can put messages of one thread on both sides. This rerun keeps the v1.0 estimator
(HistGradientBoosting, out-of-fold cross-entropy, bits) and changes two things only:
  - D2/D3 v3 features (strictly pre-timestamp, rev1_features_d2d3_v3), D1 as in the reference
  - thread-grouped folds (StratifiedGroupKFold, 5 folds, seed 42), training split only
Thread-cluster bootstrap of the out-of-fold log-losses (1,000 draws) gives 95% intervals.

Usage: python -m lmeic.analysis.info_decomposition
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

from lmeic.reference import features as F
from lmeic.common.rev1_common import git_commit, load_config, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pi_v3")
_REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
N_BOOT = 1000


def entropy_bits(y):
    p = np.bincount(y) / len(y)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main() -> None:
    paths, rcfg = load_config("rev1_reference.yaml")
    feats, y, meta = F.build(rcfg, Path(paths["repo_root"]), "v3", {"d1", "d2", "d3"})
    X, yy = feats["train"], y["train"]
    groups = meta["train"]["thread_id"].astype(str).to_numpy()
    cols = {"D1": [c for c in X.columns if not c.startswith(("d2_", "d3_"))],
            "D2": [c for c in X.columns if c.startswith("d2_")],
            "D3": [c for c in X.columns if c.startswith("d3_")]}
    sets = {"D1": ["D1"], "D2": ["D2"], "D3": ["D3"], "D2+D3": ["D2", "D3"], "D1+D2": ["D1", "D2"],
            "D1+D3": ["D1", "D3"], "D1+D2+D3": ["D1", "D2", "D3"]}
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, yy, groups))
    h = entropy_bits(yy)
    ll = {}
    for name, parts in sets.items():
        t0 = time.time()
        Z = X[[c for p in parts for c in cols[p]]].astype(float).fillna(0.0).to_numpy()
        oof = np.zeros((len(yy), 3))
        for tr, te in folds:
            clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.08,
                                                 l2_regularization=1.0, random_state=SEED).fit(Z[tr], yy[tr])
            oof[te] = clf.predict_proba(Z[te])
        ll[name] = -np.log2(np.clip(oof[np.arange(len(yy)), yy], 1e-12, 1))
        logger.info("PI(%s) = %.4f bits (%.0fs)", name, h - ll[name].mean(), time.time() - t0)

    def stats(idx):
        hb = entropy_bits(yy[idx])
        pi = {k: hb - v[idx].mean() for k, v in ll.items()}
        ii = {"II(D1;D2)": pi["D1+D2"] - pi["D1"] - pi["D2"], "II(D1;D3)": pi["D1+D3"] - pi["D1"] - pi["D3"],
              "II(D2;D3)": pi["D2+D3"] - pi["D2"] - pi["D3"], "II(D1;D2+D3)": pi["D1+D2+D3"] - pi["D1"] - pi["D2+D3"]}
        return pi, ii

    pi, ii = stats(np.arange(len(yy)))
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups); mem = {t: np.where(groups == t)[0] for t in uniq}
    B = [stats(np.concatenate([mem[t] for t in rng.choice(uniq, len(uniq))])) for _ in range(N_BOOT)]
    ci = {k: [float(np.percentile([b[0][k] for b in B], 2.5)), float(np.percentile([b[0][k] for b in B], 97.5))] for k in pi}
    ci.update({k: [float(np.percentile([b[1][k] for b in B], 2.5)), float(np.percentile([b[1][k] for b in B], 97.5))] for k in ii})
    out = Path(paths["output"]["runs_base_dir"]) / f"rev1_info_decomposition_v3_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir()
    write_json(out / "summary.json", {"split": "train", "n": int(len(yy)), "n_threads": int(len(uniq)), "H_Y_bits": h,
                                      "PI": pi, "II": ii, "ci95": ci, "n_columns": {k: len(v) for k, v in cols.items()},
                                      "folds": "StratifiedGroupKFold(5) by thread, seed 42",
                                      "estimator": "HistGradientBoosting(max_depth=4, max_iter=200, lr=0.08, l2=1.0)",
                                      "v1_reference": "v1_experiments/analysis/info_decomp_20260529_184337",
                                      "git_commit": git_commit(_REPO_ROOT)})
    for k, v in {**pi, **ii}.items():
        print(f"{k:14s} {v:+.4f}  [{ci[k][0]:+.4f}, {ci[k][1]:+.4f}]")
    print("saved", out)


if __name__ == "__main__":
    main()
