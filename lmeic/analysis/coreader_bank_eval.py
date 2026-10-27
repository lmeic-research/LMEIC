"""
Evaluation of the co-reader bank (revision 1, E10).
Version: v1 | Date: 2026-10-22

For every completed candidate-seed in <run dir>/coreader_bank (plus the five-seed
average of a candidate when several seeds exist):
  - stand-alone test macro-F1 of the co-reader;
  - fusion: the full reference variant (explicit features + co-reader probabilities) is
    refitted with that co-reader's scores and the reference variant's tuned parameters,
    five stacking seeds; paired thread-cluster bootstrap against the explicit variant;
  - complementarity with the explicit variant: Yule's Q of per-instance correctness,
    disagreement rate and error correlation (phi), averaged over explicit seeds.

Usage:
  python -m lmeic.analysis.coreader_bank_eval --run-dir <rev1_reference run dir> \
      [--full-variant R3_v3_grouped_full] [--explicit-variant R2_v3_grouped_explicit]
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

from lmeic.analysis.d1_perturbation import cluster_boot, paired  # noqa: E402
from lmeic.reference import features as F  # noqa: E402
from lmeic.reference import models as M  # noqa: E402
from lmeic.reference.analyse_reference import LABELS, macro  # noqa: E402
from lmeic.reference.coreader import PROB_COLS  # noqa: E402
from lmeic.common.rev1_common import git_commit, load_config, write_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
REF_CONFIG = "rev1_reference.yaml"


def yule_q(a: np.ndarray, b: np.ndarray) -> float:
    """Yule's Q for two boolean correctness vectors."""
    n11 = float(np.sum(a & b)); n00 = float(np.sum(~a & ~b))
    n10 = float(np.sum(a & ~b)); n01 = float(np.sum(~a & b))
    den = n11 * n00 + n01 * n10
    return (n11 * n00 - n01 * n10) / den if den > 0 else float("nan")


def phi(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.astype(float), b.astype(float))[0, 1])


def load_bank(bank_dir: Path) -> dict[str, pd.DataFrame]:
    bank = {}
    for d in sorted(bank_dir.glob("*_s*")):
        if (d / "probs.parquet").exists():
            bank[d.name] = pd.read_parquet(d / "probs.parquet")
    by_cand: dict[str, list[str]] = {}
    for k in bank:
        by_cand.setdefault(k.rsplit("_s", 1)[0], []).append(k)
    for cand, keys in by_cand.items():
        if len(keys) > 1:
            base = bank[keys[0]][["message_id", "split"]].copy()
            for k in keys[1:]:
                assert (bank[k]["message_id"].values == base["message_id"].values).all()
            base[PROB_COLS] = np.mean([bank[k][PROB_COLS].values for k in keys], axis=0)
            bank[f"{cand}_avg{len(keys)}"] = base
    return bank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--full-variant", default="R3_v3_grouped_full")
    ap.add_argument("--explicit-variant", default="R2_v3_grouped_explicit")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--only", default=None, help="comma-separated bank entries")
    ap.add_argument("--ensemble", default=None,
                    help="comma-separated bank entries averaged into one text channel, e.g. "
                         "xlmr_large_s42,bge_m3_s42,qwen25_7b_lora_text_s42")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    paths, cfg = load_config(REF_CONFIG)
    repo_root = Path(paths["repo_root"])
    fdir, edir = run_dir / args.full_variant, run_dir / args.explicit_variant
    vcfg = json.loads((fdir / "config.json").read_text())["variant"]
    params = json.loads((fdir / "params.json").read_text())
    assert "coreader" in vcfg["groups"] and vcfg["arch"] == "stacking"
    out_dir = run_dir / f"coreader_bank_eval_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir()

    bank = load_bank(run_dir / "coreader_bank")
    if args.only:
        bank = {k: v for k, v in bank.items() if k in args.only.split(",")}
    if args.ensemble:
        members = args.ensemble.split(",")
        missing = [m for m in members if m not in bank]
        assert not missing, f"not in bank: {missing}"
        base = bank[members[0]][["message_id", "split"]].copy()
        for m in members[1:]:
            assert (bank[m]["message_id"].values == base["message_id"].values).all()
        base[PROB_COLS] = np.mean([bank[m][PROB_COLS].values for m in members], axis=0)
        bank = {f"ENSEMBLE[{'+'.join(members)}]": base}
    logger.info("bank entries: %s", list(bank))

    lab = {l: i for i, l in enumerate(LABELS)}
    ex = pd.read_parquet(edir / "test_predictions_all_seeds.parquet")
    rows, preds = [], []
    boots = None
    for name, probs in bank.items():
        feats, y, meta = F.build(cfg, repo_root, vcfg["d2d3"], set(vcfg["groups"]), probs)
        X, yy, g, Xt = F.pool(feats, y, meta)
        yt, threads, mids = y["test"], meta["test"]["thread_id"].values, meta["test"]["message_id"].values
        if boots is None:
            boots = cluster_boot(yt, threads, args.n_boot, 20261022)
            P_ex = np.vstack([ex[ex["seed"] == s].set_index("message_id").loc[mids, "pred"].map(lab).values
                              for s in cfg["seeds"]])
        res = M.stacking(X, yy, g, Xt, params, vcfg["protocol"], cfg["seeds"], cfg["n_folds"],
                         cfg["inner_es_fraction"])
        P = np.vstack([res[s]["pred"] for s in cfg["seeds"]])
        cr = probs[probs["split"] == "test"].set_index("message_id").loc[mids, PROB_COLS].values.argmax(1)
        ok_cr = cr == yt
        qs = [yule_q(ok_cr, P_ex[i] == yt) for i in range(len(P_ex))]
        ph = [phi(~ok_cr, P_ex[i] != yt) for i in range(len(P_ex))]
        dis = [float((cr != P_ex[i]).mean()) for i in range(len(P_ex))]
        r = {"coreader": name, "coreader_test_macro_f1": macro(yt, cr[None, :]),
             "fused_macro_f1": macro(yt, P), "explicit_macro_f1": macro(yt, P_ex),
             **{f"fused_vs_explicit_{k}": v for k, v in paired(yt, P, P_ex, boots).items()},
             "yule_q": float(np.mean(qs)), "error_phi": float(np.mean(ph)), "disagreement": float(np.mean(dis))}
        rows.append(r)
        logger.info("%s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        for si, s in enumerate(cfg["seeds"]):
            preds.append(pd.DataFrame({"coreader": name, "seed": s, "message_id": mids, "gold": yt, "pred": P[si]}))
        pd.DataFrame(rows).to_csv(out_dir / "coreader_bank_results.csv", index=False)

    pd.concat(preds).to_parquet(out_dir / "fused_predictions.parquet", index=False)
    write_json(out_dir / "meta.json", {"full_variant": args.full_variant, "explicit_variant": args.explicit_variant,
                                       "params": "full variant tuned parameters (not re-tuned per co-reader)",
                                       "bootstrap": {"unit": "thread", "n_boot": args.n_boot, "seed": 20261022},
                                       "git_commit": git_commit(repo_root)})
    logger.info("results:\n%s", pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
