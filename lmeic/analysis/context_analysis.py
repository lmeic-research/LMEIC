"""
E9: contribution of the context domains (D2/D3) by test-set situation.
Version: v1 | Date: 2026-10-22

For each configured pair (with context, without context) of reference variants, the
contribution is delta = macro-F1(with) - macro-F1(without), averaged over the five
seeds, on the full test split and on situational subsets: short emails (bottom tercile
of latest_chars), forwarded emails, language, replies inside a thread vs thread
starters, sender-unseen emails, each role, and emails on which the content-only model
is least confident (bottom tercile of its seed-averaged maximum class probability).
Intervals come from a thread-clustered paired bootstrap within each subset; p-values
are Holm-adjusted across subsets within each pair.

Usage:
  python -m lmeic.analysis.context_analysis --run-dir <rev1_reference run dir>
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.reference.analyse_reference import LABELS, load_systems, macro  # noqa: E402
from lmeic.common.rev1_common import (  # noqa: E402
    git_commit, load_config, norm_mid, read_jsonl, source_run_dir, write_json,
)
from lmeic.common.data_loader import load_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
CONFIG_NAME = "rev1_reference.yaml"

PAIRS = [
    ("D2+D3", "R2_v3_grouped_explicit", "E9_explicit_minus_d2d3"),
    ("D2", "R2_v3_grouped_explicit", "E9_explicit_minus_d2"),
    ("D3", "R2_v3_grouped_explicit", "E9_explicit_minus_d3"),
    ("D2+D3+role_cond", "R2_v3_grouped_explicit", "E9_explicit_no_context"),
    ("full: D2+D3", "R3_v3_grouped_full", "E9_full_minus_d2d3"),
    ("full: D2", "R3_v3_grouped_full", "E9_full_minus_d2"),
    ("full: D3", "R3_v3_grouped_full", "E9_full_minus_d3"),
    ("full: D2+D3+role_cond", "R3_v3_grouped_full", "E9_full_no_context"),
    ("co-reader", "R3_v3_grouped_full", "R2_v3_grouped_explicit"),
]
CONTENT_ONLY = "E9_explicit_no_context"
FORWARD_RE = re.compile(r"^\s*(re\s*[:\uff1a]\s*)*(fw|fwd)\s*[:\uff1a]|^\s*(re\s*[:\uff1a]\s*)*\u8f6c\u53d1", re.I)


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        adj[i] = running
    return adj


def subsets(meta: pd.DataFrame, unseen: np.ndarray, low_conf: np.ndarray | None) -> dict[str, np.ndarray]:
    chars = meta["latest_chars"].values
    out = {
        "all": np.ones(len(meta), bool),
        "short (bottom tercile of length)": chars <= np.quantile(chars, 1 / 3),
        "long (top tercile of length)": chars >= np.quantile(chars, 2 / 3),
        "forwarded": meta["subject"].fillna("").map(lambda s: bool(FORWARD_RE.search(s))).values,
        "thread reply (depth > 0)": meta["thread_depth"].values > 0,
        "thread starter (depth = 0)": meta["thread_depth"].values == 0,
        "sender unseen": unseen,
        "sender seen": ~unseen,
    }
    for lang in ["en", "zh", "mixed"]:
        out[f"language {lang}"] = meta["language"].values == lang
    for role in ["Mgr", "CS", "Sch"]:
        out[f"role {role}"] = meta["role"].values == role
    if low_conf is not None:
        out["content-only model low confidence"] = low_conf
        out["content-only model high confidence"] = ~low_conf
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    paths, cfg = load_config(CONFIG_NAME)
    cfg["_repo_root"] = paths["repo_root"]   # load_systems resolves extra_predictions against it
    src_dir = source_run_dir(paths, cfg["source_run_id"])

    te = load_split("test")
    dev = pd.concat([load_split("train"), load_split("val")])
    test_mids = [norm_mid(m) for m in te["message_id"]]
    y = te["label_final"].map({l: i for i, l in enumerate(LABELS)}).values
    threads = te["thread_id"].values

    ids = set(test_mids)
    rows = {}
    dev_mids = {norm_mid(m) for m in dev["message_id"]}
    dev_senders = set()
    for r in read_jsonl(src_dir / cfg["analysis"]["sampled_relpath"]):
        m = norm_mid(r["message_id"])
        sender = (r.get("from_addr") or "").strip().lower()
        if m in ids:
            rows[m] = {"subject": r.get("subject"), "latest_chars": r.get("latest_chars"),
                       "language": r.get("language"), "thread_depth": r.get("thread_depth"),
                       "sender": sender}
        if m in dev_mids:
            dev_senders.add(sender)
    meta = pd.DataFrame([rows[m] for m in test_mids])
    meta["role"] = te["role"].values
    unseen = ~meta["sender"].isin(dev_senders).values

    systems = load_systems(run_dir, src_dir, cfg, test_mids)
    low_conf = None
    p = run_dir / CONTENT_ONLY / "test_predictions_all_seeds.parquet"
    if p.exists():
        pr = pd.read_parquet(p).assign(message_id=lambda d: d["message_id"].map(norm_mid))
        pmax = pr.groupby("message_id")[["p_high", "p_med", "p_low"]].mean().max(axis=1).loc[test_mids].values
        low_conf = pmax <= np.quantile(pmax, 1 / 3)
    subs = subsets(meta, unseen, low_conf)

    rng = np.random.default_rng(args.seed)
    boot_idx = {}
    for name, mask in subs.items():
        idx = np.where(mask)[0]
        th = threads[idx]
        uniq = np.unique(th)
        members = {t: idx[th == t] for t in uniq}
        boot_idx[name] = (idx, [np.concatenate([members[t] for t in rng.choice(uniq, len(uniq), replace=True)])
                                for _ in range(args.n_boot)], len(uniq))

    out_rows = []
    for label, a, b in PAIRS:
        if a not in systems or b not in systems:
            logger.info("skip %s (%s or %s missing)", label, a, b)
            continue
        part = []
        for name, (idx, boots, n_thr) in boot_idx.items():
            fa, fb = macro(y, systems[a], idx), macro(y, systems[b], idx)
            d = np.array([macro(y, systems[a], i) - macro(y, systems[b], i) for i in boots])
            part.append({"contribution": label, "with": a, "without": b, "subset": name,
                         "n": int(len(idx)), "n_threads": n_thr, "f1_with": fa, "f1_without": fb,
                         "delta": fa - fb, "ci95_low": float(np.percentile(d, 2.5)),
                         "ci95_high": float(np.percentile(d, 97.5)),
                         "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))})
        ph = holm(np.array([r["p_two_sided"] for r in part]))
        for r, q in zip(part, ph):
            r["p_holm"] = float(q)
        out_rows += part
    res = pd.DataFrame(out_rows)

    out = run_dir / f"e9_context_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir()
    res.to_csv(out / "context_deltas.csv", index=False)
    write_json(out / "meta.json", {
        "pairs": PAIRS, "content_only_model": CONTENT_ONLY if low_conf is not None else None,
        "subset_sizes": {k: int(v.sum()) for k, v in subs.items()},
        "bootstrap": {"unit": "thread", "n_boot": args.n_boot, "seed": args.seed},
        "git_commit": git_commit(Path(paths["repo_root"]))})
    if len(res):
        logger.info("\n%s", res[["contribution", "subset", "n", "delta", "ci95_low", "ci95_high", "p_holm"]]
                    .round(4).to_string(index=False))
    logger.info("saved %s", out)


if __name__ == "__main__":
    main()
