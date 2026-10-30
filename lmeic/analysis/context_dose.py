"""
E29a: gain of the context domains (D2 + D3) stratified by the amount of history (revision 1).
Version: v1.2 | Date: 2026-10-29 (v1.1: --pair explicit runs the same bins for LMEIC-explicit against
  E9_explicit_minus_d2d3, 5 decision-layer seeds each; added after the v1 result, reported as post hoc;
  v1.2: also writes the sender-history distribution of sender-seen and sender-unseen test instances,
  sender-unseen as in analyse_reference: the sender never appears in the development pool)

LMEIC-full (E21_full, 5 co-reader seeds x 5 decision-layer seeds) against the same system without
D2 and D3 (E21_full_minus_d2d3, 25 runs), on the test split, by bins fixed before looking at results:
  d2_sender_freq_90d   : 0 / 1-4 / 5-19 / >=20
  d2_thread_prior_count: 0 / 1 / >=2
Per bin: n, macro-F1 of both systems (mean over runs), difference with a thread-clustered paired
bootstrap 95% interval (5,000 replicates, seed 20261022) and Holm within each binning variable.
Exploratory analysis.

Usage:
  python -m lmeic.analysis.context_dose --run-dir <rev1_reference run dir> [--pair full|explicit]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.reference import features as F  # noqa: E402
from lmeic.reference.analyse_reference import LABELS, macro, pooled_systems  # noqa: E402
from lmeic.common.rev1_common import (  # noqa: E402
    git_commit, load_config, norm_mid, read_jsonl, source_run_dir, write_json,
)
from lmeic.common.data_loader import load_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PAIRS = {"full": ("E21_full_5cr", "E21_full_minus_d2d3_5cr", 25),
         "explicit": ("R2_v3_grouped_explicit", "E9_explicit_minus_d2d3", 5)}
BINS = {"d2_sender_freq_90d": ([0, 1, 5, 20, np.inf], ["0", "1-4", "5-19", ">=20"]),
        "d2_thread_prior_count": ([0, 1, 2, np.inf], ["0", "1", ">=2"])}


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pair", choices=list(PAIRS), default="full")
    args = ap.parse_args()
    A, B, n_runs = PAIRS[args.pair]
    run_dir = Path(args.run_dir)
    paths, cfg = load_config("rev1_reference.yaml")
    cfg["_repo_root"] = paths["repo_root"]
    ac = cfg["analysis"]

    te = load_split("test")
    test_mids = [norm_mid(m) for m in te["message_id"]]
    y = te["label_final"].map({l: i for i, l in enumerate(LABELS)}).values
    threads = te["thread_id"].values

    keep = {k: ac["pooled_systems"].get(k, {"kind": "variant_glob", "glob": k, "min_groups": 1}) for k in (A, B)}
    pooled, groups = pooled_systems(run_dir, {**cfg, "analysis": {**ac, "pooled_systems": keep}}, test_mids)
    assert set(pooled) == {A, B} and all(sum(groups[k]) == n_runs for k in (A, B)), groups

    d2 = F.load_d2d3("v3", cfg, Path(paths["repo_root"]))
    d2 = d2.assign(message_id=d2["message_id"].map(norm_mid)).drop_duplicates("message_id").set_index("message_id")
    feat = d2.loc[test_mids, list(BINS)]
    assert feat.notna().all().all()

    src_dir = source_run_dir(paths, cfg["source_run_id"])
    sender = {norm_mid(r["message_id"]): (r.get("from_addr") or "").strip().lower()
              for r in read_jsonl(src_dir / ac["sampled_relpath"])}
    dev = pd.concat([load_split("train"), load_split("val")])
    dev_senders = {sender[norm_mid(m)] for m in dev["message_id"]}
    unseen = np.array([sender[m] not in dev_senders for m in test_mids])
    hist = d2.loc[test_mids, ["d2_sender_freq_30d", "d2_sender_freq_90d", "d2_sender_reply_rate"]]
    sender_hist = {grp: {"n": int(m.sum()), **{c: {"zero_share": float((hist[c].values[m] == 0).mean()),
                                                 "median": float(np.median(hist[c].values[m]))} for c in hist}}
                   for grp, m in (("sender_seen", ~unseen), ("sender_unseen", unseen))}
    logger.info("sender history by dev-pool sender status: %s", sender_hist)

    rng = np.random.default_rng(ac["boot_seed"])
    uniq = np.unique(threads)
    members = {t: np.where(threads == t)[0] for t in uniq}
    boots = [np.concatenate([members[t] for t in rng.choice(uniq, len(uniq), replace=True)])
             for _ in range(ac["n_boot"])]

    rows = []
    overall = {"variable": "all", "bin": "all", "n": len(y)}
    for var, (edges, names) in BINS.items():
        b = pd.cut(feat[var].values, edges, right=False, labels=names)
        assert b.notna().all()
        block = []
        for name in names:
            idx = np.where(b == name)[0]
            d_obs = macro(y, pooled[A], idx) - macro(y, pooled[B], idx)
            d = []
            for bi in boots:
                j = bi[np.isin(bi, idx)]
                d.append(macro(y, pooled[A], j) - macro(y, pooled[B], j))
            d = np.array(d)
            block.append({"variable": var, "bin": name, "n": len(idx),
                          "n_threads": int(len(np.unique(threads[idx]))),
                          "with_d2d3": macro(y, pooled[A], idx), "without_d2d3": macro(y, pooled[B], idx),
                          "delta_pp": 100 * d_obs, "ci95_low_pp": 100 * float(np.percentile(d, 2.5)),
                          "ci95_high_pp": 100 * float(np.percentile(d, 97.5)),
                          "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))})
        for r, p in zip(block, holm([r["p_two_sided"] for r in block])):
            r["p_holm"] = p
        assert sum(r["n"] for r in block) == len(y)
        rows += block
    overall.update({"with_d2d3": macro(y, pooled[A]), "without_d2d3": macro(y, pooled[B]),
                    "delta_pp": 100 * (macro(y, pooled[A]) - macro(y, pooled[B]))})
    tab = pd.DataFrame([overall] + rows)

    out = run_dir / f"e29_context_dose_{args.pair}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir()
    tab.to_csv(out / "context_dose.csv", index=False)
    write_json(out / "meta.json", {"systems": [A, B], "pooled_groups": groups, "sender_history": sender_hist,
                                   "bins": {k: [str(e) for e in v[0]] for k, v in BINS.items()},
                                   "bootstrap": {"unit": "thread", "n_boot": ac["n_boot"], "seed": ac["boot_seed"]},
                                   "git_commit": git_commit(Path(paths["repo_root"]))})
    logger.info("\n%s", tab.round(4).to_string(index=False))
    logger.info("Output: %s", out)


if __name__ == "__main__":
    main()
