"""
Score the revision-1 external sets (cross-role, fresh hold-out, E1b) with a reference variant.
Version: v1 | Date: 2026-10-22

Fits the variant's model on the development pool with its tuned parameters (five
seeds) and predicts the EMAIL-3R test split together with every external set. The
test predictions are compared with the variant's stored test predictions as an
assembly check. External D1 descriptors come from the revision-1 extraction
(gpt-4o-2024-11-20); the model is fitted on v1.0 descriptors (drift quantified
separately by drift_d1).

Usage:
  python -m lmeic.reference.score_external --run-dir <rev1_reference run dir> --variant R3_v3_grouped_full
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.common.rev1_common import git_commit, load_config, read_jsonl, write_json  # noqa: E402
from lmeic.reference import features as F  # noqa: E402
from lmeic.reference import models as M  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
CONFIG_NAME = "rev1_reference.yaml"
LABELS = ["High", "Medium", "Low"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    paths, cfg = load_config(CONFIG_NAME)
    repo = Path(paths["repo_root"])
    v = next(x for x in cfg["variants"] if x["name"] == args.variant)
    assert v["d2d3"] == "v3", "external sets carry v3 D2/D3 features only"
    groups = set(v["groups"])
    es = cfg["external_sets"]
    params = json.loads((run_dir / v["name"] / "params.json").read_text())

    probs_dev = pd.read_parquet(run_dir / "coreader_probs_grouped.parquet") if "coreader" in groups else None
    probs_ext = pd.read_parquet(run_dir / "coreader_probs_external.parquet") if "coreader" in groups else None
    feats, y, meta = F.build(cfg, repo, "v3", groups, probs_dev)
    X, yy, g, Xt = F.pool(feats, y, meta)
    cols = list(feats["train"].columns)

    blocks, index = [Xt], [("test", m) for m in meta["test"]["message_id"]]
    ext_meta = {}
    for name, rel in es["sets"].items():
        ext = pd.DataFrame(read_jsonl(repo / rel))
        d1 = pd.DataFrame(read_jsonl(repo / es["d1_dir"] / f"d1_{name}.jsonl"))
        dd = pd.read_parquet(repo / es["d2d3_dir"] / f"d2d3_v3_{name}.parquet")
        pe = probs_ext[probs_ext["set"] == name] if probs_ext is not None else None
        fx = F.build_external(cfg, repo, groups, cols, ext, d1, dd, pe)
        blocks.append(np.nan_to_num(fx.values.astype(np.float32)))
        index += [(name, i) for i in ext["instance_id"].astype(str)]
        ext_meta[name] = len(ext)
        logger.info("%s: %d instances", name, len(ext))
    X_all = np.vstack(blocks)

    if v["arch"] == "stacking":
        res = M.stacking(X, yy, g, X_all, params, v["protocol"], cfg["seeds"], cfg["n_folds"],
                         cfg["inner_es_fraction"])
    else:
        res = M.single_xgb(X, yy, X_all, params, cfg["seeds"])

    rows = []
    for s in cfg["seeds"]:
        for (set_name, iid), pred, pr in zip(index, res[s]["pred"], res[s]["proba"]):
            rows.append({"seed": s, "set": set_name, "id": iid, "pred": LABELS[int(pred)],
                         "p_high": float(pr[0]), "p_med": float(pr[1]), "p_low": float(pr[2])})
    out = pd.DataFrame(rows)
    out_dir = run_dir / v["name"] / f"external_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir()
    out.to_parquet(out_dir / "external_predictions_all_seeds.parquet", index=False)

    stored = pd.read_parquet(run_dir / v["name"] / "test_predictions_all_seeds.parquet")
    t = out[out["set"] == "test"].merge(stored[["seed", "message_id", "pred"]],
                                        left_on=["seed", "id"], right_on=["seed", "message_id"],
                                        suffixes=("", "_stored"))
    agree = float((t["pred"] == t["pred_stored"]).mean())
    write_json(out_dir / "external_meta.json", {
        "variant": v, "n_external": ext_meta, "test_agreement_with_stored_predictions": agree,
        "d1_dir": es["d1_dir"], "d2d3_dir": es["d2d3_dir"], "git_commit": git_commit(repo)})
    logger.info("test agreement with stored predictions: %.4f -> %s", agree, out_dir)


if __name__ == "__main__":
    main()
