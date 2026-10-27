#!/usr/bin/env python3
"""
E13: per-email latency, throughput and cost of the LMEIC inference path.
Version: v1.1 | Date: 2026-10-23 (v1.1: masked extraction listed separately, relative source paths)

Timed on the EMAIL-3R test split (run on an otherwise idle GPU):
  d2d3       pre-t D2/D3 features for one email from the history index (CPU)
  text       TF-IDF + SVD transform of one email (CPU)
  coreader   XLM-R forward pass, batch 1 and batch 64 (GPU); the distilled D1 encoder has the
             same backbone and sequence length, so its cost equals this row
  stacking   XGB + LGBM base learners and the logistic meta-learner on one feature row (batch 1)
             and on the whole split
Extraction rows come from recorded runs: local vLLM throughput from the E4 extraction stats and
GPT-4o tokens per email from the revision-1 extraction stats. Cost per million emails uses the
prices and GPU-hour rate in rev1_e13_deploy.yaml (latency.prices), each with its source.

Usage:
  python -m lmeic.deploy.latency --run-dir <rev1_reference run dir> \
      [--variant R2_v3_grouped_explicit] [--n 500]
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.reference import features as F  # noqa: E402
from lmeic.reference import models as M  # noqa: E402
from lmeic.common.rev1_common import git_commit, load_config, source_run_dir, write_json  # noqa: E402
from lmeic.common.data_loader import load_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def summarise(name: str, secs: list[float], unit_n: int = 1, hw: str = "") -> dict:
    a = np.array(secs) * 1000
    return {"component": name, "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "throughput_per_s": float(unit_n * len(a) / (a.sum() / 1000)),
            "n_timed": len(a), "hardware": hw}


def time_d2d3(n: int) -> dict:
    from lmeic.features.d2d3_v3 import FeatureSpec, HistoryIndex, compute_features, read_jsonl_gz
    paths, fcfg = load_config("rev1_features_d2d3_v3.yaml")
    run_dir = source_run_dir(paths, fcfg["source_run_id"])
    t0 = time.time()
    idx = HistoryIndex.from_file(run_dir / fcfg["history_relpath"])
    load_s = time.time() - t0
    spec = FeatureSpec(fcfg, Path(paths["repo_root"]))
    recs = list(read_jsonl_gz(run_dir / fcfg["sampled_relpath"]))[:n]
    secs = []
    for r in recs:
        t = time.perf_counter(); compute_features([r], idx, spec); secs.append(time.perf_counter() - t)
    out = summarise("d2d3_features", secs, hw="CPU, 1 thread")
    out["index_load_s"] = round(load_s, 1)
    return out


def time_text(n: int) -> dict:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    tr, te = load_split("train"), load_split("test")
    # vectoriser settings must match run_tide_v9.build_text_features
    wv = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=3, max_df=0.9)
    ws = TruncatedSVD(40, random_state=42).fit(wv.fit_transform(tr["latest_text"].fillna("")))
    cv = TfidfVectorizer(max_features=30000, ngram_range=(2, 4), sublinear_tf=True, min_df=5, max_df=0.9, analyzer="char_wb")
    cs = TruncatedSVD(20, random_state=42).fit(cv.fit_transform(tr["latest_text"].fillna("")))
    secs = []
    for x in te["latest_text"].fillna("").values[:n]:
        t = time.perf_counter(); ws.transform(wv.transform([x])); cs.transform(cv.transform([x])); secs.append(time.perf_counter() - t)
    return summarise("text_tfidf_svd", secs, hw="CPU, 1 thread")


def time_coreader(n: int) -> list[dict]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _, rcfg = load_config("rev1_reference.yaml")
    cc = rcfg["coreader"]
    tok = AutoTokenizer.from_pretrained(cc["model_name"])
    m = AutoModelForSequenceClassification.from_pretrained(cc["model_name"], num_labels=3).cuda().eval()
    texts = load_split("test")["latest_text"].fillna("").tolist()
    gpu = torch.cuda.get_device_name(0)
    out = []
    with torch.no_grad():
        for _ in range(10):
            m(**tok(texts[:1], truncation=True, max_length=cc["max_len"], return_tensors="pt").to("cuda"))
        secs = []
        for x in texts[:n]:
            torch.cuda.synchronize(); t = time.perf_counter()
            m(**tok([x], truncation=True, max_length=cc["max_len"], return_tensors="pt").to("cuda"))
            torch.cuda.synchronize(); secs.append(time.perf_counter() - t)
        out.append(summarise("coreader_xlmr_batch1", secs, hw=gpu))
        secs = []
        for i in range(0, len(texts), 64):
            torch.cuda.synchronize(); t = time.perf_counter()
            m(**tok(texts[i:i + 64], truncation=True, max_length=cc["max_len"], padding=True, return_tensors="pt").to("cuda"))
            torch.cuda.synchronize(); secs.append(time.perf_counter() - t)
        r = summarise("coreader_xlmr_batch64", secs, unit_n=64, hw=gpu)
        r["throughput_per_s"] = float(len(texts) / sum(secs))
        out.append(r)
    return out


def time_stacking(run_dir: Path, variant: str, n: int) -> list[dict]:
    paths, rcfg = load_config("rev1_reference.yaml")
    v = next(x for x in rcfg["variants"] if x["name"] == variant)
    params = json.loads((run_dir / variant / "params.json").read_text())
    feats, y, meta = F.build(rcfg, Path(paths["repo_root"]), v["d2d3"], set(v["groups"]))
    X, yy, g, Xt = F.pool(feats, y, meta)
    xm = M._xgb(params["xgb"], 42, prob=True).fit(X, yy, sample_weight=M.sample_weights(yy))
    lm = M._lgbm(params["lgbm"], 42).fit(X, yy, sample_weight=M.sample_weights(yy))
    from sklearn.linear_model import LogisticRegression
    meta_lr = LogisticRegression(max_iter=1000).fit(np.hstack([xm.predict_proba(X), lm.predict_proba(X)]), yy)
    xm.set_params(device="cpu")
    pred = lambda Z: meta_lr.predict(np.hstack([xm.predict_proba(Z), lm.predict_proba(Z)]))  # noqa: E731
    for _ in range(5):
        pred(Xt[:1])
    secs = []
    for i in range(n):
        t = time.perf_counter(); pred(Xt[i:i + 1]); secs.append(time.perf_counter() - t)
    out = [summarise("stacking_batch1", secs, hw="CPU")]
    t = time.perf_counter(); pred(Xt); el = time.perf_counter() - t
    out.append({"component": "stacking_full_split", "throughput_per_s": float(len(Xt) / el), "n_timed": len(Xt),
                "hardware": "CPU", "note": "one fitted XGB+LGBM+LR (seed 42); five-seed inference scales linearly"})
    return out


def extraction_rows(paths: dict) -> list[dict]:
    base = Path(paths["output"]["runs_base_dir"])
    rows = []
    for f in sorted(glob.glob(str(base / "rev1_d1_local_*" / "stats_*.json"))):
        d = json.loads(Path(f).read_text())
        if d["ok"] >= 10000:
            suffix = "_masked" if "masked" in Path(f).parent.name else ""
            rows.append({"component": f"d1_local_{d['model']}{suffix}", "throughput_per_s": d["throughput_per_s"],
                         "n_timed": d["ok"], "hardware": "vLLM, one RTX 5090 (shared with other jobs)",
                         "source": str(Path(f).relative_to(base))})
    pt = ct = n = 0
    for f in glob.glob(str(base / "rev1_d1_gpt4o_20241120_*" / "d1_*_stats_*.json")):
        d = json.loads(Path(f).read_text())
        if d.get("variant", "original") == "original" and d["ok"]:
            pt += d["prompt_tokens"]; ct += d["completion_tokens"]; n += d["ok"]
    rows.append({"component": "d1_gpt4o_api", "prompt_tokens_per_email": pt / max(n, 1),
                 "completion_tokens_per_email": ct / max(n, 1), "n_timed": n, "hardware": "API"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", default="R2_v3_grouped_explicit")
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    paths, cfg = load_config("rev1_e13_deploy.yaml")
    run_dir = Path(args.run_dir)
    out = run_dir / f"e13_latency_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir()
    rows = [time_d2d3(args.n), time_text(args.n), *time_coreader(args.n), *time_stacking(run_dir, args.variant, args.n),
            *extraction_rows(paths)]
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "latency.csv", index=False)

    pr = cfg["latency"]["prices"]
    api = next(r for r in rows if r["component"] == "d1_gpt4o_api")
    per_m = 1e6
    cost = {"gpt4o_api_usd_per_million": per_m * (api["prompt_tokens_per_email"] * pr["gpt4o_input_usd_per_mtok"]
                                                   + api["completion_tokens_per_email"] * pr["gpt4o_output_usd_per_mtok"]) / 1e6}
    for r in rows:
        if r["component"].startswith("d1_local_"):
            hrs = per_m / r["throughput_per_s"] / 3600
            cost[f"{r['component']}_gpu_hours_per_million"] = hrs
            if pr.get("gpu_usd_per_hour"):
                cost[f"{r['component']}_usd_per_million"] = hrs * pr["gpu_usd_per_hour"]
    write_json(out / "summary.json", {"rows": rows, "cost_per_million_emails": cost, "prices": pr,
                                      "variant": args.variant, "git_commit": git_commit(Path(paths["repo_root"]))})
    logger.info("latency:\n%s", tab.round(3).to_string(index=False))
    logger.info("cost: %s", cost)


if __name__ == "__main__":
    main()
