"""
E3 robustness checks on the reference model (leakage and label-proximity controls).
Version: v1 | Date: 2026-10-22

Variants, all with the reference variant's tuned parameters and protocol:
  label_shuffle    development-pool labels permuted within role (class marginals kept);
                   test macro-F1 against true labels should fall to chance level.
  drop_near_label  removes the descriptors closest to the labelling criteria (urgency,
                   role_relevance, consequence_severity) and every interaction or
                   role-conditional product built from them.
  history_lag<k>   every history query (sender frequency, reply rate, sender-role count,
                   prior thread count) uses only history older than t - k days.

Usage:
  python -m lmeic.analysis.robustness_e3 --run-dir <rev1_reference run dir> \
      --variant R2_v3_grouped_explicit [--checks label_shuffle,drop_near_label,history_lag7]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.common.rev1_common import git_commit, load_config, source_run_dir, write_json  # noqa: E402
from lmeic.reference import features as F  # noqa: E402
from lmeic.reference import models as M  # noqa: E402
from lmeic.common.evaluator import aggregate_seed_metrics, compute_metrics  # noqa: E402
from lmeic.features.d2d3_v3 import FeatureSpec, HistoryIndex, compute_features, read_jsonl_gz  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
REF_CONFIG = "rev1_reference.yaml"
FEAT_CONFIG = "rev1_features_d2d3_v3.yaml"
NEAR_LABEL = ["urgency", "role_relevance", "consequence_severity"]
NEAR_LABEL_DERIVED = ["inter_urg_", "inter_conseq_", "_urgency", "_conseq"]


class LaggedIndex(HistoryIndex):
    def __init__(self, base: HistoryIndex, days: int):
        self.__dict__.update(base.__dict__)
        self.lag = timedelta(days=days)

    def _t(self, t):
        return None if t is None else t - self.lag

    def sender_freq(self, sender, role, t, days):
        return super().sender_freq(sender, role, self._t(t), days)

    def sender_role_count(self, sender, role, t):
        return super().sender_role_count(sender, role, self._t(t))

    def reply_rate(self, sender, t):
        return super().reply_rate(sender, self._t(t))

    def thread_prior_count(self, thread_id, t):
        return super().thread_prior_count(thread_id, self._t(t))


def lagged_features(days: int, out_dir: Path) -> Path:
    paths, fcfg = load_config(FEAT_CONFIG)
    repo = Path(paths["repo_root"])
    run_dir = source_run_dir(paths, fcfg["source_run_id"])
    idx = LaggedIndex(HistoryIndex.from_file(run_dir / fcfg["history_relpath"]), days)
    feats = compute_features(list(read_jsonl_gz(run_dir / fcfg["sampled_relpath"])), idx, FeatureSpec(fcfg, repo))
    p = out_dir / f"d2d3_v3_lag{days}.parquet"
    feats.to_parquet(p, index=False)
    return p


def evaluate(X, y_dev, g, Xt, yt, roles, params, v, cfg) -> dict:
    if v["arch"] == "stacking":
        res = M.stacking(X, y_dev, g, Xt, params, v["protocol"], cfg["seeds"], cfg["n_folds"],
                         cfg["inner_es_fraction"])
    else:
        res = M.single_xgb(X, y_dev, Xt, params, cfg["seeds"])
    return aggregate_seed_metrics([compute_metrics(yt, res[s]["pred"], roles) for s in cfg["seeds"]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--checks", default="label_shuffle,drop_near_label,history_lag7")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    paths, cfg = load_config(REF_CONFIG)
    repo = Path(paths["repo_root"])
    v = next(x for x in cfg["variants"] if x["name"] == args.variant)
    assert "coreader" not in v["groups"], "checks are defined on the explicit variants"
    params = json.loads((run_dir / v["name"] / "params.json").read_text())
    out_dir = run_dir / v["name"] / f"robustness_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir()
    groups = set(v["groups"])

    feats, y, meta = F.build(cfg, repo, v["d2d3"], groups)
    X, yy, g, Xt = F.pool(feats, y, meta)
    cols = list(feats["train"].columns)
    yt, roles = y["test"], meta["test"]["role"].values
    results = {"variant": v, "params_from": str(run_dir / v["name"] / "params.json"),
               "git_commit": git_commit(repo), "checks": {}}

    for check in args.checks.split(","):
        t0 = time.time()
        if check == "label_shuffle":
            rng = np.random.default_rng(20261022)
            dev_roles = np.concatenate([meta["train"]["role"].values, meta["val"]["role"].values])
            ys = yy.copy()
            for r in np.unique(dev_roles):
                idx = np.where(dev_roles == r)[0]
                ys[idx] = rng.permutation(ys[idx])
            agg = evaluate(X, ys, g, Xt, yt, roles, params, v, cfg)
            info = {"permutation": "within role", "seed": 20261022}
        elif check == "drop_near_label":
            keep = [i for i, c in enumerate(cols) if c not in NEAR_LABEL and not any(k in c for k in NEAR_LABEL_DERIVED)]
            dropped = [c for i, c in enumerate(cols) if i not in set(keep)]
            agg = evaluate(X[:, keep], yy, g, Xt[:, keep], yt, roles, params, v, cfg)
            info = {"dropped": dropped, "n_features": len(keep)}
        elif check.startswith("history_lag"):
            days = int(check[len("history_lag"):])
            p = lagged_features(days, out_dir)
            fl, yl, ml = F.build(cfg, repo, f"file:{p}", groups)
            Xl, yyl, gl, Xtl = F.pool(fl, yl, ml)
            assert np.array_equal(yyl, yy) and list(fl["train"].columns) == cols
            agg = evaluate(Xl, yyl, gl, Xtl, yt, roles, params, v, cfg)
            info = {"lag_days": days, "features_file": str(p)}
        else:
            raise ValueError(check)
        mf = agg["overall"]["macro_f1"]
        results["checks"][check] = {**info, "macro_f1_mean": mf["mean"], "macro_f1_std": mf["std"],
                                    "per_role": {r: agg["per_role"][r]["macro_f1"]["mean"] for r in ["Mgr", "CS", "Sch"]},
                                    "elapsed_s": round(time.time() - t0, 1)}
        write_json(out_dir / "robustness.json", results)
        logger.info("[%s] macro-F1 %.4f +/- %.4f", check, mf["mean"], mf["std"])


if __name__ == "__main__":
    main()
