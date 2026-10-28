"""E1d: human-agent confusion on all instances with both labels, and a census of the analysis masks.

Version: v1 | Date: 2026-10-28

The E1b confusion matrix used the mask "agent label and three round-zero votes present"
(sel = ok & maj >= 0), which drops two fresh hold-out items whose debate-fallback vote was
stored as a fourth round-zero vote; its text count (305) summed only four disagreement cells.
This script recomputes the consensus confusion on every instance with a human majority and an
agent label, and lists the instance counts of both masks per subset, so that the manuscript's
denominators can be stated explicitly.

Masks:
  agreement mask : agent label present and three round-zero votes parsed (alt-test, Table 4 human-agent rows)
  rescoring mask : human majority exists (model re-scoring, Table 8, SI S5)

Usage: python -m lmeic.human_study.confusion_census --e1-dir <E1 run dir>

Requires the confidential EMAIL-3R human-study files (items_with_labels.jsonl and the
annotation database); it is included to document the analysis and does not run without them.

"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.common.rev1_common import git_commit, write_json  # noqa: E402

LABELS = ["High", "Medium", "Low"]


def three_votes(v) -> bool:
    return isinstance(v, list) and len(v) == 3 and all(x in LABELS for x in v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-dir", required=True)
    args = ap.parse_args()
    e1 = Path(args.e1_dir)
    items = pd.read_json(e1 / "items_with_labels.jsonl", lines=True)

    has_agent = items["agent_label"].isin(LABELS)
    has_major = items["human_majority"].isin(LABELS)
    has_votes = items["agent_votes_r0"].apply(three_votes)
    in_pool = items["eval_tier"].eq("thread_in_training_pool") if "eval_tier" in items else False
    masks = {"agreement_mask": has_agent & has_votes, "rescoring_mask": has_agent & has_major}

    census = {}
    for name, m in masks.items():
        census[name] = {"all": int(m.sum()),
                        "excluding_training_pool_threads": int((m & ~in_pool).sum()),
                        **{src: int((m & items["source"].eq(src)).sum()) for src in sorted(items["source"].unique())}}
    census["dropped_by_agreement_mask"] = items.loc[masks["rescoring_mask"] & ~masks["agreement_mask"], "item_id"].tolist()
    census["dropped_by_rescoring_mask"] = items.loc[masks["agreement_mask"] & ~masks["rescoring_mask"], "item_id"].tolist()

    d = items[masks["rescoring_mask"]]
    conf = {}
    for src, g in [("all", d)] + list(d.groupby("source")):
        conf[src] = pd.crosstab(g["human_majority"], g["agent_label"]).reindex(
            index=LABELS, columns=LABELS, fill_value=0).to_dict("index")
    disagree = d[d["human_majority"] != d["agent_label"]]
    out = e1 / f"e1d_confusion_census_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir()
    write_json(out / "summary.json", {
        "e1_dir": str(e1.relative_to(_REPO_ROOT)) if e1.is_absolute() else str(e1),
        "git_commit": git_commit(_REPO_ROOT),
        "n_both_labels": int(len(d)), "n_disagree": int(len(disagree)),
        "disagree_human_medium": int((disagree["human_majority"] == "Medium").sum()),
        "disagree_human_medium_agent_low": int(((disagree["human_majority"] == "Medium") & (disagree["agent_label"] == "Low")).sum()),
        "disagree_human_medium_agent_high": int(((disagree["human_majority"] == "Medium") & (disagree["agent_label"] == "High")).sum()),
        "disagree_high_low_swaps": int(disagree["human_majority"].isin(["High", "Low"]).mul(disagree["agent_label"].isin(["High", "Low"])).sum()),
        "confusion_human_rows_agent_cols": conf, "mask_census": census})
    print(json.dumps({"n_both": len(d), "n_disagree": len(disagree), "out": str(out)}))


if __name__ == "__main__":
    main()
