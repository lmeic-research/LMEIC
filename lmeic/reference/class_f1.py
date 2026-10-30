"""E28: per-class F1, run SD and between-co-reader SD of the pooled systems in the main table (revision 1).

Version: v1 | Date: 2026-10-30
Writes <analysis dir>/class_f1.csv for E21_full_5cr, E22_lora_5seed and the E28 systems: macro-F1 and
per-class F1 averaged over runs, population SD over runs, SD of the co-reader (group) means.

Usage:
  python -m lmeic.reference.class_f1 --run-dir <rev1_reference run dir> --analysis-dir <analysis_*>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.reference.analyse_reference import LABELS, pooled_systems  # noqa: E402
from lmeic.common.rev1_common import load_config, norm_mid  # noqa: E402
from lmeic.common.data_loader import load_split  # noqa: E402

SYSTEMS = ["E21_full_5cr", "E22_lora_5seed", "E28_full_xlmrl_5cr", "E28_full_loratext_5cr",
           "E28_xlmrl_5seed", "E28_loratext_5seed"]


def class_f1(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3).astype(float)
    tp = np.diag(cm); den = cm.sum(0) + cm.sum(1)
    return np.divide(2 * tp, den, out=np.zeros(3), where=den > 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--analysis-dir", required=True)
    args = ap.parse_args()
    paths, cfg = load_config("rev1_reference.yaml")
    cfg["_repo_root"] = paths["repo_root"]
    te = load_split("test")
    mids = [norm_mid(m) for m in te["message_id"]]
    y = te["label_final"].map({l: i for i, l in enumerate(LABELS)}).values
    keep = {k: cfg["analysis"]["pooled_systems"][k] for k in SYSTEMS}
    P, G = pooled_systems(Path(args.run_dir), {**cfg, "analysis": {**cfg["analysis"], "pooled_systems": keep}}, mids)
    rows = []
    for k in SYSTEMS:
        per = np.array([class_f1(y, p) for p in P[k]])
        mac = per.mean(1)
        grp = [g.mean() for g in np.split(mac, np.cumsum(G[k])[:-1])]
        rows.append({"system": k, "n_runs": len(mac), "macro_f1": mac.mean(), "sd_runs": mac.std(),
                     "sd_between_groups": float(np.std(grp)), "f1_high": per[:, 0].mean(),
                     "f1_medium": per[:, 1].mean(), "f1_low": per[:, 2].mean(),
                     "group_means": ";".join(f"{g:.4f}" for g in grp)})
    out = pd.DataFrame(rows)
    out.to_csv(Path(args.analysis_dir) / "class_f1.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
