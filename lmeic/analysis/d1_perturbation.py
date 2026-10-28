"""
D1 perturbation study on the reference model: prompt sensitivity, extraction failure and
graceful degradation (revision 1, E5 and E8).
Version: v1 | Date: 2026-10-22

The reference variant (tuned parameters and protocol from a rev1_reference run) is fitted
once on the development pool with v1.0 descriptors. The test split is then scored under
several test-time D1 versions; only D1 columns and the products derived from them change.
  v1_d1               v1.0 descriptors (reference)
  snapshot_original   gpt-4o-2024-11-20, original prompt (snapshot drift relative to v1_d1)
  prompt:original     same snapshot and prompt, re-run on the endpoint used for the variants;
                      reference for the variants (its difference from snapshot_original
                      measures repeat-call and endpoint variation)
  prompt:<variant>    gpt-4o-2024-11-20, prompt / output-structure variant
  fail_<p>_r<k>       fraction p of test rows replaced by the extractor's default output
                      (what the pipeline imputes after a failed call), replicate k
Graceful degradation compares, for each failure rate, default imputation under the full
model with routing the failed rows to a fallback model fitted without D1.

Usage:
  python -m lmeic.analysis.d1_perturbation --run-dir <rev1_reference run dir> \
      --variant R2_v3_grouped_explicit --prompt-dir <rev1_d1_prompt_variants dir>
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

from lmeic.analysis.drift_d1 import descriptor_agreement  # noqa: E402
from lmeic.reference import features as F  # noqa: E402
from lmeic.reference import models as M  # noqa: E402
from lmeic.reference.analyse_reference import macro  # noqa: E402
from lmeic.common.rev1_common import git_commit, load_config, read_jsonl, write_json  # noqa: E402
from lmeic.common.data_loader import load_d1_features_v2  # noqa: E402
from lmeic.common.lmeic_features import build_interactions  # noqa: E402
from lmeic.features.d1_prompt_variants import VARIANTS  # noqa: E402
from lmeic.features.extract_d1 import BOOL_FIELDS, NUMERIC_FIELDS, coerce  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
REF_CONFIG = "rev1_reference.yaml"
RAW_FIELDS = ["intent_type"] + NUMERIC_FIELDS + BOOL_FIELDS
FAIL_RATES = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
N_REPLICATES = 5


def read_d1_file(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(read_jsonl(path))
    df["row_idx"] = df["id"].astype(int)
    df = df.drop_duplicates("row_idx", keep="last")
    df["intent_type"] = df["intent_type"].where(df["intent_type"].isin(F.INTENTS), "other")
    return df


class TestMatrixFactory:
    """Rebuild test-split D1 columns and D1-derived products for a given raw D1 table."""

    def __init__(self, feats_test: pd.DataFrame, meta_test: pd.DataFrame):
        self.base = feats_test.reset_index(drop=True)
        self.meta = meta_test.reset_index(drop=True)
        self.d1_raw = load_d1_features_v2().drop_duplicates("row_idx").set_index("row_idx")
        self.mid_to_row = {m: r for r, m in zip(self.d1_raw.index, self.d1_raw["message_id"])}
        self.test_rows = np.array([self.mid_to_row[m] for m in self.meta["message_id"]])
        _, self.d1_cols = F.load_d1()
        self.d1_cols = [c for c in self.d1_cols if c in self.base.columns]

    def matrix(self, override: pd.DataFrame | None) -> np.ndarray:
        enc, _ = F.load_d1(override)
        enc = enc.set_index("message_id").loc[self.meta["message_id"], self.d1_cols]
        df = self.base.copy()
        df[self.d1_cols] = enc.values.astype(float)
        df = build_interactions(df, self.meta[["role"]], "v2")
        assert list(df.columns) == list(self.base.columns)
        X = df.values.astype(np.float32)
        np.nan_to_num(X, copy=False)
        return X

    def default_rows(self, rows: np.ndarray) -> pd.DataFrame:
        d = coerce({})
        return pd.DataFrame([{"row_idx": int(r), **d} for r in rows])


def cluster_boot(y, threads, n_boot, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(threads)
    members = {t: np.where(threads == t)[0] for t in uniq}
    return [np.concatenate([members[t] for t in rng.choice(uniq, len(uniq), replace=True)]) for _ in range(n_boot)]


def paired(y, Pa, Pb, boots) -> dict:
    d = np.array([macro(y, Pa, i) - macro(y, Pb, i) for i in boots])
    return {"delta": macro(y, Pa) - macro(y, Pb), "ci95_low": float(np.percentile(d, 2.5)),
            "ci95_high": float(np.percentile(d, 97.5)),
            "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def summarise_agreement(tab: pd.DataFrame) -> dict:
    b = tab[tab["type"] == "bool"]
    n = tab[tab["type"] == "0-10"]
    return {"intent_kappa": float(tab.loc[tab["field"] == "intent_type", "kappa"].iloc[0]),
            "bool_kappa_mean": float(b["kappa"].mean()), "bool_kappa_min": float(b["kappa"].min()),
            "numeric_icc_mean": float(n["icc_a1"].mean()), "numeric_icc_min": float(n["icc_a1"].min())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", default="R2_v3_grouped_explicit")
    ap.add_argument("--prompt-dir", default=None, help="omit to run only the failure / degradation part")
    ap.add_argument("--prompt-set", default="test_prompt_subset")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seeds", default=None, help="comma-separated subset of the configured seeds")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    prompt_dir = Path(args.prompt_dir) if args.prompt_dir else Path()
    paths, cfg = load_config(REF_CONFIG)
    if args.seeds:
        cfg["seeds"] = [int(s) for s in args.seeds.split(",")]
    repo_root = Path(paths["repo_root"])
    vdir = run_dir / args.variant
    vcfg = json.loads((vdir / "config.json").read_text())["variant"]
    params = json.loads((vdir / "params.json").read_text())
    assert vcfg["arch"] == "stacking" and "coreader" not in vcfg["groups"], "explicit stacking variants only"
    groups = set(vcfg["groups"])
    out_dir = vdir / f"d1_perturbation_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir()

    feats, y, meta = F.build(cfg, repo_root, vcfg["d2d3"], groups)
    X, yy, g, Xt0 = F.pool(feats, y, meta)
    fac = TestMatrixFactory(feats["test"], meta["test"])
    assert np.allclose(fac.matrix(None), Xt0), "factory does not reproduce the reference test matrix"
    yt, threads, n = y["test"], meta["test"]["thread_id"].values, len(y["test"])

    mats, info = {"v1_d1": Xt0}, {}
    snap = read_d1_file(repo_root / cfg["external_sets"]["d1_dir"] / "d1_test_drift.jsonl")
    assert set(snap["row_idx"]) == set(fac.test_rows), "snapshot extraction does not cover the test split"
    mats["snapshot_original"] = fac.matrix(snap[["row_idx"] + RAW_FIELDS])
    v1_test = fac.d1_raw.loc[fac.test_rows].reset_index()
    agree_rows = [{"version": "snapshot_original", "reference": "v1_d1",
                   **summarise_agreement(descriptor_agreement(v1_test, snap)[1])}]
    pfiles = {v: prompt_dir / (f"d1_{args.prompt_set}__{v}.jsonl" if v != "original" else
                               f"d1_{args.prompt_set}.jsonl") for v in VARIANTS}
    pdfs = {v: read_d1_file(p) for v, p in pfiles.items() if p.exists()} if args.prompt_dir else {}
    if pdfs:
        assert "original" in pdfs, "prompt variants need the original prompt run on the same endpoint"
    subset = set.intersection(*[set(d["row_idx"]) for d in pdfs.values()]) if pdfs else set()
    idx_s = np.where(np.isin(fac.test_rows, sorted(subset)))[0]
    logger.info("prompt subset: %d test rows common to %d prompt versions", len(idx_s), len(pdfs))
    prompt_ref = "prompt:original" if pdfs else None
    ref_df = pdfs["original"][pdfs["original"]["row_idx"].isin(subset)] if pdfs else None
    for v, dv in pdfs.items():
        info[v] = {"n_rows": int(len(dv)), "n_in_subset": int(dv["row_idx"].isin(subset).sum()),
                   "share_retried": float((dv.get("attempts", pd.Series([1])) > 1).mean())}
        dv = dv[dv["row_idx"].isin(subset)]
        full = pd.concat([dv, snap[~snap["row_idx"].isin(subset)]], ignore_index=True)
        mats[f"prompt:{v}"] = fac.matrix(full[["row_idx"] + RAW_FIELDS])
        ref_name, ref_tab = (("snapshot_original", snap[snap["row_idx"].isin(subset)]) if v == "original"
                             else (prompt_ref, ref_df))
        agree_rows.append({"version": f"prompt:{v}", "reference": ref_name, "n": int(len(dv)),
                           **summarise_agreement(descriptor_agreement(ref_tab, dv)[1])})

    rng = np.random.default_rng(20261022)
    fail_masks = {}
    for p in FAIL_RATES:
        for k in range(N_REPLICATES if p < 1 else 1):
            pos = np.sort(rng.choice(n, int(round(p * n)), replace=False))
            name = f"fail_{p:.2f}_r{k}"
            fail_masks[name] = pos
            mats[name] = fac.matrix(fac.default_rows(fac.test_rows[pos]))

    names = list(mats)
    logger.info("scoring %d test versions with one fit of %s", len(names), args.variant)
    res = M.stacking(X, yy, g, np.vstack([mats[k] for k in names]), params, vcfg["protocol"],
                     cfg["seeds"], cfg["n_folds"], cfg["inner_es_fraction"])
    P = {k: np.vstack([res[s]["pred"][i * n:(i + 1) * n] for s in cfg["seeds"]]) for i, k in enumerate(names)}

    fb_groups = groups - {"d1", "interactions"}
    ffb, yfb, mfb = F.build(cfg, repo_root, vcfg["d2d3"], fb_groups)
    Xf, yf, gf, Xtf = F.pool(ffb, yfb, mfb)
    assert (mfb["test"]["message_id"].values == meta["test"]["message_id"].values).all()
    rfb = M.stacking(Xf, yf, gf, Xtf, params, vcfg["protocol"], cfg["seeds"], cfg["n_folds"],
                     cfg["inner_es_fraction"])
    P_fb = np.vstack([rfb[s]["pred"] for s in cfg["seeds"]])

    boots = cluster_boot(yt, threads, args.n_boot, 20261022)
    boots_s = cluster_boot(yt[idx_s], threads[idx_s], args.n_boot, 20261022) if len(idx_s) else []
    rows = []
    for k in names:
        ref = (("snapshot_original" if k == "prompt:original" else prompt_ref) if k.startswith("prompt:")
               else "v1_d1")
        r = {"version": k, "eval_rows": "full_test", "n_eval": n, "macro_f1": macro(yt, P[k]), "reference": ref,
             "flip_rate_vs_ref": float((P[k] != P[ref]).mean())}
        if k == "snapshot_original":
            r.update(paired(yt, P[k], P[ref], boots))
        if k.startswith("prompt:"):
            ys, pk, pr = yt[idx_s], P[k][:, idx_s], P[ref][:, idx_s]
            r.update({"eval_rows": "prompt_subset", "n_eval": len(idx_s), "macro_f1": macro(ys, pk),
                      "flip_rate_vs_ref": float((pk != pr).mean()), **paired(ys, pk, pr, boots_s)})
        rows.append(r)
    for k in (["v1_d1", "snapshot_original"] if len(idx_s) else []):
        rows.append({"version": k, "eval_rows": "prompt_subset", "n_eval": len(idx_s),
                     "macro_f1": macro(yt[idx_s], P[k][:, idx_s]), "reference": None})
    tab = pd.DataFrame(rows)
    pv = tab[tab["version"].str.startswith("prompt:") & (tab["version"] != "prompt:original")]["macro_f1"]
    prompt_summary = {"n_variants": int(len(pv)), "macro_f1_mean": float(pv.mean()) if len(pv) else None,
                      "macro_f1_sd": float(pv.std(ddof=1)) if len(pv) > 1 else None,
                      "macro_f1_range": [float(pv.min()), float(pv.max())] if len(pv) else None}

    deg = []
    for p in FAIL_RATES:
        reps = [k for k in fail_masks if k.startswith(f"fail_{p:.2f}_")]
        f_default, f_route = [], []
        for k in reps:
            routed = P[k].copy()
            routed[:, fail_masks[k]] = P_fb[:, fail_masks[k]]
            f_default.append(macro(yt, P[k]))
            f_route.append(macro(yt, routed))
        deg.append({"fail_rate": p, "n_replicates": len(reps),
                    "macro_f1_default_fill": float(np.mean(f_default)),
                    "macro_f1_default_fill_sd": float(np.std(f_default, ddof=1)) if len(reps) > 1 else 0.0,
                    "macro_f1_fallback_route": float(np.mean(f_route)),
                    "macro_f1_fallback_route_sd": float(np.std(f_route, ddof=1)) if len(reps) > 1 else 0.0})
    deg = pd.DataFrame(deg)
    deg.insert(1, "macro_f1_no_failure", macro(yt, P["v1_d1"]))
    deg["macro_f1_fallback_model_alone"] = macro(yt, P_fb)

    tab.to_csv(out_dir / "test_versions.csv", index=False)
    pd.DataFrame(agree_rows).to_csv(out_dir / "descriptor_agreement.csv", index=False)
    deg.to_csv(out_dir / "degradation.csv", index=False)
    pred_rows = []
    for k in ["v1_d1", "snapshot_original"] + [x for x in names if x.startswith("prompt:")]:
        for si, s in enumerate(cfg["seeds"]):
            pred_rows.append(pd.DataFrame({"version": k, "seed": s, "message_id": meta["test"]["message_id"],
                                           "gold": yt, "pred": P[k][si]}))
    for si, s in enumerate(cfg["seeds"]):
        pred_rows.append(pd.DataFrame({"version": "fallback_no_d1", "seed": s,
                                       "message_id": meta["test"]["message_id"], "gold": yt, "pred": P_fb[si]}))
    pd.concat(pred_rows).to_parquet(out_dir / "predictions.parquet", index=False)
    write_json(out_dir / "summary.json", {
        "variant": args.variant, "protocol": vcfg["protocol"], "groups": sorted(groups),
        "fallback_groups": sorted(fb_groups), "fallback_params": "reference variant parameters (not re-tuned)",
        "prompt_dir": str(prompt_dir), "prompt_set": args.prompt_set, "n_prompt_subset": int(len(idx_s)),
        "prompt_reference": prompt_ref, "prompt_variant_info": info, "prompt_summary": prompt_summary,
        "fail_rates": FAIL_RATES, "n_replicates": N_REPLICATES, "default_output": coerce({}),
        "bootstrap": {"unit": "thread", "n_boot": args.n_boot, "seed": 20261022},
        "git_commit": git_commit(repo_root)})
    logger.info("test versions:\n%s", tab.round(4).to_string(index=False))
    logger.info("descriptor agreement:\n%s", pd.DataFrame(agree_rows).round(3).to_string(index=False))
    logger.info("degradation:\n%s", deg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
