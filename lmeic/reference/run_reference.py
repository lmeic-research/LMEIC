"""
Reference rebuild runner: evaluates the configured variants one factor at a time.
Version: v1.2 | Date: 2026-10-22

For each variant: build features, tune (unless v1.0 parameters are reused), fit the
five-seed stacking (or single XGB), and save metrics, all-seed test predictions and
the tuned parameters. The summary table is re-read and rewritten after every variant,
so concurrent runs on the same output directory keep each other's rows.
v1.1 optional variant keys: d1_file (descriptor file, relative to repo_root, replacing
the GPT-4o descriptors for all instances; keyed by row_idx) and params_from (reuse the
tuned parameters of another variant in the same output directory).

Usage:
  python -m lmeic.reference.run_reference [--out-dir DIR] [--only R1,R2]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.common.rev1_common import (  # noqa: E402
    git_commit, load_config, new_output_dir, read_jsonl, source_run_dir, write_json,
)
from lmeic.common.data_loader import load_d1_features_v2  # noqa: E402
from lmeic.features.extract_d1 import BOOL_FIELDS, NUMERIC_FIELDS, coerce  # noqa: E402
from lmeic.reference import features as F  # noqa: E402
from lmeic.reference import models as M  # noqa: E402
from lmeic.reference.folds import check_group_disjoint, make_folds  # noqa: E402
from lmeic.common.evaluator import aggregate_seed_metrics, compute_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
CONFIG_NAME = "rev1_reference.yaml"
LABELS = ["High", "Medium", "Low"]


def load_d1_file(path: Path) -> tuple[pd.DataFrame, dict]:
    """Raw descriptors for every instance, keyed by row_idx; unparsed rows get the extractor default."""
    df = pd.DataFrame(read_jsonl(path))
    df["row_idx"] = df["row_idx"].astype(int)
    df = df.drop_duplicates("row_idx", keep="last")
    n_unknown_intent = int((~df["intent_type"].isin(F.INTENTS)).sum())
    df["intent_type"] = df["intent_type"].where(df["intent_type"].isin(F.INTENTS), "other")
    all_rows = set(load_d1_features_v2()["row_idx"].astype(int))
    missing = sorted(all_rows - set(df["row_idx"]))
    if missing:
        df = pd.concat([df, pd.DataFrame([{"row_idx": r, **coerce({})} for r in missing])], ignore_index=True)
    fields = ["intent_type"] + NUMERIC_FIELDS + BOOL_FIELDS
    return df[["row_idx"] + fields], {"path": str(path), "n_rows": len(all_rows),
                                      "n_default_filled": len(missing), "n_unknown_intent": n_unknown_intent}


def run_variant(v: dict, cfg: dict, paths: dict, out_dir: Path) -> dict:
    repo_root = Path(paths["repo_root"])
    run_dir = source_run_dir(paths, cfg["source_run_id"])
    vdir = out_dir / v["name"]
    vdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    probs = None
    if "coreader" in v["groups"]:
        p = out_dir / "coreader_probs_grouped.parquet"
        assert p.exists(), f"missing {p}; run the co-reader step first"
        probs = pd.read_parquet(p)

    d1_override, d1_info = None, None
    if v.get("d1_file"):
        d1_override, d1_info = load_d1_file(repo_root / v["d1_file"])
        write_json(vdir / "d1_file_info.json", d1_info)

    d1_extra = None
    if v.get("d1_extra"):
        d1_extra = F.load_d1_extra(v["d1_extra"], repo_root)
        write_json(vdir / "d1_extra_info.json",
                   {name: {"n_rows": len(enc), "n_features": len(cols)}
                    for name, (enc, cols) in d1_extra.items()})

    feats, y, meta = F.build(cfg, repo_root, v["d2d3"], set(v["groups"]), probs,
                             d1_override=d1_override, d1_extra=d1_extra)
    X, yy, g, Xt = F.pool(feats, y, meta)
    cols = list(feats["train"].columns)
    if v["protocol"] == "grouped_clean":
        assert check_group_disjoint(make_folds(yy, g, "grouped_clean", 42, cfg["n_folds"]), g) == 0

    if v.get("reuse_v1_params"):
        v1 = json.loads((run_dir / cfg["v1_artefacts"]["v1_config"]).read_text())
        params = {"xgb": v1["best_xgb_params"], "lgbm": v1["best_lgbm_params"], "source": "v1.0 LMEIC_v9_d1v2"}
    elif v.get("params_from"):
        params = json.loads((out_dir / v["params_from"] / "params.json").read_text())
        params["source"] = v["params_from"]
    else:
        params = M.tune(X, yy, g, v["protocol"], cfg["n_folds"], cfg["optuna_trials"],
                        cfg["inner_es_fraction"], tune_lgbm=v["arch"] == "stacking")
    write_json(vdir / "params.json", params)

    if v["arch"] == "stacking":
        res = M.stacking(X, yy, g, Xt, params, v["protocol"], cfg["seeds"], cfg["n_folds"],
                         cfg["inner_es_fraction"])
    elif v["arch"] == "xgb":
        res = M.single_xgb(X, yy, Xt, params, cfg["seeds"])
    else:
        raise ValueError(v["arch"])

    yt, roles = y["test"], meta["test"]["role"].values
    seed_metrics = [compute_metrics(yt, res[s]["pred"], roles) for s in cfg["seeds"]]
    agg = aggregate_seed_metrics(seed_metrics)
    agg["n_test"] = int(len(yt))

    rows = []
    for s in cfg["seeds"]:
        for i, mid in enumerate(meta["test"]["message_id"].values):
            pr = res[s]["proba"][i]
            rows.append({"seed": s, "message_id": mid, "role": roles[i], "gold": LABELS[yt[i]],
                         "pred": LABELS[int(res[s]["pred"][i])],
                         "p_high": float(pr[0]), "p_med": float(pr[1]), "p_low": float(pr[2])})
    pd.DataFrame(rows).to_parquet(vdir / "test_predictions_all_seeds.parquet", index=False)

    info = {"variant": v, "d1_file_info": d1_info, "n_features": len(cols), "feature_columns": cols,
            "n_dev": int(len(yy)), "n_test": int(len(yt)), "seeds": cfg["seeds"],
            "oof_meta_f1": {s: res[s].get("oof_f1") for s in cfg["seeds"]},
            "elapsed_s": round(time.time() - t0, 1)}
    write_json(vdir / "config.json", info)
    write_json(vdir / "metrics.json", agg)
    mf = agg["overall"]["macro_f1"]
    logger.info("[%s] macro-F1 %.4f +/- %.4f (nfeat=%d, %.0fs)", v["name"], mf["mean"], mf["std"],
                len(cols), info["elapsed_s"])
    return {"variant": v["name"], "d2d3": v["d2d3"], "protocol": v["protocol"], "arch": v["arch"],
            "groups": "+".join(v["groups"]), "n_features": len(cols),
            "macro_f1_mean": mf["mean"], "macro_f1_std": mf["std"],
            "Mgr": agg["per_role"]["Mgr"]["macro_f1"]["mean"],
            "CS": agg["per_role"]["CS"]["macro_f1"]["mean"],
            "Sch": agg["per_role"]["Sch"]["macro_f1"]["mean"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--only", default=None, help="comma-separated variant name prefixes")
    args = ap.parse_args()
    paths, cfg = load_config(CONFIG_NAME)
    out_dir = Path(args.out_dir) if args.out_dir else new_output_dir(paths, cfg["output_prefix"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"run_meta_{time.strftime('%Y%m%d_%H%M%S')}.json",
               {"config": cfg, "git_commit": git_commit(Path(paths["repo_root"])), "argv": sys.argv})
    logger.info("Output: %s", out_dir)

    variants = cfg["variants"]
    if args.only:
        pref = args.only.split(",")
        variants = [v for v in variants if any(v["name"].startswith(p) for p in pref)]

    summ_path = out_dir / "summary.csv"
    summary = []
    for v in variants:
        row = run_variant(v, cfg, paths, out_dir)
        summary = pd.read_csv(summ_path).to_dict("records") if summ_path.exists() else []
        summary = [r for r in summary if r["variant"] != row["variant"]] + [row]
        pd.DataFrame(summary).to_csv(summ_path, index=False)
    logger.info("Summary:\n%s", pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
