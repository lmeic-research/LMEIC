"""
Analysis of the reference rebuild: per-variant metrics, sender-disjoint subsets and
thread-clustered paired bootstrap comparisons.
Version: v1.1 | Date: 2026-10-23 (v1.1: single-seed external systems via analysis.extra_predictions)

Sender-disjoint subset: test emails whose sender address never occurs in the
development pool (train+val). Paired comparisons resample test threads with
replacement (cluster bootstrap) and average macro-F1 over the five seeds of each
system, so the interval reflects test-set sampling at the thread level.

Usage:
  python -m lmeic.reference.analyse_reference --run-dir <rev1_reference run dir>
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

from lmeic.common.rev1_common import (  # noqa: E402
    git_commit, load_config, norm_mid, read_jsonl, source_run_dir, write_json,
)
from lmeic.common.data_loader import load_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
CONFIG_NAME = "rev1_reference.yaml"
LABELS = ["High", "Medium", "Low"]


def load_systems(run_dir: Path, src_dir: Path, cfg: dict, test_mids: list[str]) -> dict[str, np.ndarray]:
    """Return {system: int array (n_seeds, n_test)} aligned to test_mids."""
    lab = {l: i for i, l in enumerate(LABELS)}
    out = {}
    frames = {v["name"]: pd.read_parquet(run_dir / v["name"] / "test_predictions_all_seeds.parquet")
              for v in cfg["variants"] if (run_dir / v["name"] / "test_predictions_all_seeds.parquet").exists()}
    for name, rel in cfg["analysis"]["published_predictions"].items():
        frames[name] = pd.DataFrame(read_jsonl(src_dir / rel))
    for name, rel in cfg["analysis"].get("extra_predictions", {}).items():
        path = Path(cfg["_repo_root"]) / rel
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            df = df[df["split"] == "test"] if "split" in df else df
            pred = df[["plm_p_high", "plm_p_med", "plm_p_low"]].values.argmax(1)
            frames[name] = pd.DataFrame({"seed": 42, "message_id": df["message_id"].values,
                                         "pred": [LABELS[i] for i in pred]})
        else:
            df = pd.DataFrame(read_jsonl(path))
            frames[name] = pd.DataFrame({"seed": 42, "message_id": df["id"].values, "pred": df["pred"].values})
    for name, df in frames.items():
        df = df.assign(message_id=df["message_id"].map(norm_mid))
        mats = []
        for s in sorted(df["seed"].unique()):
            m = df[df["seed"] == s].drop_duplicates("message_id").set_index("message_id")
            missing = set(test_mids) - set(m.index)
            assert not missing, f"{name} seed {s}: {len(missing)} test ids missing"
            mats.append(m.loc[test_mids, "pred"].map(lab).values)
        out[name] = np.vstack(mats)
    return out


def macro(y, P, idx=None) -> float:
    """Seed-averaged macro-F1 over the three classes (zero when a class has no support or predictions)."""
    if idx is not None:
        y, P = y[idx], P[:, idx]
    cm = np.stack([np.bincount(y * 3 + p, minlength=9).reshape(3, 3) for p in P]).astype(float)
    tp = np.diagonal(cm, axis1=1, axis2=2)
    denom = cm.sum(2) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    paths, cfg = load_config(CONFIG_NAME)
    cfg["_repo_root"] = paths["repo_root"]
    src_dir = source_run_dir(paths, cfg["source_run_id"])
    ac = cfg["analysis"]

    te = load_split("test")
    dev = pd.concat([load_split("train"), load_split("val")])
    test_mids = [norm_mid(m) for m in te["message_id"]]
    y = te["label_final"].map({l: i for i, l in enumerate(LABELS)}).values
    roles = te["role"].values
    threads = te["thread_id"].values

    sender = {norm_mid(r["message_id"]): (r.get("from_addr") or "").strip().lower()
              for r in read_jsonl(src_dir / ac["sampled_relpath"])}
    dev_senders = {sender[norm_mid(m)] for m in dev["message_id"]}
    unseen = np.array([sender[m] not in dev_senders for m in test_mids])
    logger.info("sender-unseen test emails: %d / %d", unseen.sum(), len(unseen))

    systems = load_systems(run_dir, src_dir, cfg, test_mids)
    rows = []
    for name, P in systems.items():
        r = {"system": name, "n_seeds": P.shape[0], "macro_f1": macro(y, P),
             "sender_seen": macro(y, P, np.where(~unseen)[0]),
             "sender_unseen": macro(y, P, np.where(unseen)[0])}
        for role in ["Mgr", "CS", "Sch"]:
            r[role] = macro(y, P, np.where(roles == role)[0])
        rows.append(r)
    tab = pd.DataFrame(rows)

    rng = np.random.default_rng(ac["boot_seed"])
    uniq = np.unique(threads)
    members = {t: np.where(threads == t)[0] for t in uniq}
    boots = [np.concatenate([members[t] for t in rng.choice(uniq, len(uniq), replace=True)])
             for _ in range(ac["n_boot"])]
    comps = []
    for a, b in ac["comparisons"]:
        if a not in systems or b not in systems:
            logger.info("skip %s vs %s (not available)", a, b)
            continue
        d_obs = macro(y, systems[a]) - macro(y, systems[b])
        d = np.array([macro(y, systems[a], i) - macro(y, systems[b], i) for i in boots])
        comps.append({"a": a, "b": b, "delta": d_obs, "ci95_low": float(np.percentile(d, 2.5)),
                      "ci95_high": float(np.percentile(d, 97.5)),
                      "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
                      "delta_sender_unseen": macro(y, systems[a], np.where(unseen)[0]) -
                      macro(y, systems[b], np.where(unseen)[0])})
    comp = pd.DataFrame(comps)

    out = run_dir / f"analysis_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir()
    tab.to_csv(out / "system_metrics.csv", index=False)
    comp.to_csv(out / "paired_comparisons.csv", index=False)
    write_json(out / "analysis_meta.json", {
        "n_test": int(len(y)), "n_test_threads": int(len(uniq)), "n_sender_unseen": int(unseen.sum()),
        "bootstrap": {"unit": "thread", "n_boot": ac["n_boot"], "seed": ac["boot_seed"]},
        "git_commit": git_commit(Path(paths["repo_root"]))})
    logger.info("systems:\n%s", tab.round(4).to_string(index=False))
    if len(comp):
        logger.info("comparisons:\n%s", comp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
