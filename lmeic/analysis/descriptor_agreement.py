"""
Descriptor agreement between a local open-weight extractor and the v1.0 GPT-4o descriptors (E4).
Version: v1.1 | Date: 2026-10-23 (v1.1: optional local reference file, used by E6)

Per-field agreement on all 15K instances and per split: Cohen's kappa for the categorical
and boolean fields; exact / within-1 agreement, quadratic-weighted kappa, Spearman rho and
ICC(A,1) for the 0-10 fields; plus extraction statistics (coverage, token counts).

Usage:
  python -m lmeic.analysis.descriptor_agreement --d1-file <local run>/d1_all.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.analysis.d1_perturbation import summarise_agreement  # noqa: E402
from lmeic.analysis.drift_d1 import descriptor_agreement  # noqa: E402
from lmeic.common.rev1_common import (  # noqa: E402
    git_commit, load_config, norm_mid, read_jsonl, write_json,
)
from lmeic.common.data_loader import load_d1_features_v2, load_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1-file", required=True)
    ap.add_argument("--ref-file", default=None, help="reference descriptor jsonl keyed by row_idx (default: v1.0 GPT-4o)")
    ap.add_argument("--ref-name", default="v1")
    args = ap.parse_args()
    paths, _ = load_config("rev1_reference.yaml")
    f = Path(args.d1_file)
    new = pd.DataFrame(read_jsonl(f)).drop_duplicates("row_idx", keep="last")
    if args.ref_file:
        ref = Path(paths["repo_root"]) / args.ref_file
        v1 = pd.DataFrame(read_jsonl(ref)).drop_duplicates("row_idx", keep="last")
    else:
        v1 = load_d1_features_v2().drop_duplicates("row_idx")
    out = f.parent / f"agreement_vs_{args.ref_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir()

    split_of = {}
    for s in ["train", "val", "test"]:
        for m in load_split(s)["message_id"]:
            split_of[norm_mid(m)] = s
    v1["split"] = v1["message_id"].map(norm_mid).map(split_of).fillna("unassigned")

    summary, tab = descriptor_agreement(v1, new)
    tab.to_csv(out / "agreement_all.csv", index=False)
    rows = [{"subset": "all", "n": summary["n_aligned"], **summarise_agreement(tab)}]
    for s in ["train", "val", "test"]:
        sub = v1[(v1["split"] == s) & v1["row_idx"].isin(new["row_idx"])]
        if len(sub) < 2:
            continue
        _, t = descriptor_agreement(sub, new)
        t.to_csv(out / f"agreement_{s}.csv", index=False)
        rows.append({"subset": s, "n": int(len(sub)), **summarise_agreement(t)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "agreement_summary.csv", index=False)
    write_json(out / "summary.json", {
        "d1_file": str(f), "reference": args.ref_file or "v1.0 GPT-4o", "n_rows": int(len(new)),
        "n_v1_rows": int(len(v1)), **summary,
        "completion_tokens_mean": float(new["completion_tokens"].mean()) if "completion_tokens" in new else None,
        "prompt_tokens_mean": float(new["prompt_tokens"].mean()) if "prompt_tokens" in new else None,
        "git_commit": git_commit(Path(paths["repo_root"]))})
    logger.info("per-field (all):\n%s", tab.round(3).to_string(index=False))
    logger.info("summary:\n%s", res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
