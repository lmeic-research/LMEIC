"""E1c: alt-test of the agent consensus by evaluation-set source (audit finding D065).

The pooled alt-test in e1b_human_supplement.py passes at epsilon 0.20 but not at the
pre-registered 0.15; this script reports omega per source (core_test / cross_role /
fresh_holdout) and on the subset without thread_in_training_pool items, at 0.10 / 0.15 / 0.20.
Writes e1b_supplement/alt_test_by_source.json.

Usage: python -m lmeic.human_study.alt_test_by_source

Requires the confidential EMAIL-3R human-study files (items_with_labels.jsonl and the
annotation database); it is included to document the analysis and does not run without them.

"""
import json
import sys

import numpy as np
import pandas as pd

from lmeic.human_study.annotation_store.store import Store
from lmeic.human_study.agreement import LAB
from lmeic.human_study.supplement import alt_test

RUN = "private/runs/rev1_e1_human_r2_20261025_130349"
DB = "private/runs/rev1_eval_sets_20261022_161427/human_annotation/web_annotations_r2.sqlite"

it = pd.read_json(f"{RUN}/items_with_labels.jsonl", lines=True)
ann = pd.DataFrame([r for r in Store(DB, read_only=True).all_rows() if r["phase"] == "main"])
w = ann.pivot_table(index="item_id", columns="annotator", values="label", aggfunc="first")
it = it[it.agent_label.notna() & it.agent_votes_r0.notna()].reset_index(drop=True)
H = np.stack([w.loc[it.item_id, a].map(LAB).to_numpy(int) for a in ["A1", "A2", "A3"]], 1)
f = it.agent_label.map(LAB).to_numpy()
subsets = {"all": np.ones(len(it), bool), "core_test": (it.source == "core_test").to_numpy(),
           "cross_role": (it.source == "cross_role").to_numpy(),
           "fresh_holdout": (it.source == "fresh_holdout").to_numpy(),
           "no_overlap": (it.eval_tier != "thread_in_training_pool").to_numpy()}
res = {}
for name, m in subsets.items():
    res[name] = {"n": int(m.sum()), **{str(e): alt_test(H[m], f[m], e) for e in (0.10, 0.15, 0.20)}}
    print(name, res[name]["n"], {e: res[name][str(e)]["omega"] for e in (0.10, 0.15, 0.20)})
json.dump(res, open(f"{RUN}/e1b_supplement/alt_test_by_source.json", "w"), indent=1)
