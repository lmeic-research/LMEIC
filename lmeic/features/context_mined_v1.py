#!/usr/bin/env python3
"""
E7 richer context features, part 3: role-discriminative n-grams mined from training labels.

Version: v1 | Date: 2026-10-22

Replaces the hand-written role keyword lists with n-grams learned per role. For each
role, smoothed log-odds of High vs not-High and of Low vs not-Low are estimated from
binary n-gram occurrence in that role's labelled emails; the top-k n-grams by absolute
log-odds (document frequency >= min_df) are kept, and an email's score is the sum of the
log-odds of the kept n-grams it contains divided by sqrt(1 + number present):
  d3x_mined_high, d3x_mined_low, d3x_mined_hits
Development-pool rows are scored out of fold (thread-grouped folds, fixed fold seed);
test rows and external instances are scored with n-grams mined on the whole development
pool. Tokens are word unigrams/bigrams with CJK characters as single tokens.

Usage:
  python -m lmeic.features.context_mined_v1 --out-dir <rev1_context_rich dir>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedGroupKFold

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.common.rev1_common import git_commit, load_config, norm_mid, write_json  # noqa: E402
from lmeic.features.context_rich_v1 import CONFIG_NAME, LABELS, ROLES, load_targets  # noqa: E402
from lmeic.common.data_loader import load_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
TOKEN_PATTERN = r"(?u)\b[^\W\d_\u4e00-\u9fff]{2,}\b|[\u4e00-\u9fff]"


class RoleMiner:
    def __init__(self, mcfg: dict):
        self.c = mcfg
        self.models = {}

    def fit(self, texts: np.ndarray, roles: np.ndarray, y: np.ndarray) -> "RoleMiner":
        for role in ROLES:
            m = roles == role
            vec = CountVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=tuple(self.c["ngram_range"]),
                                  min_df=self.c["min_df"], binary=True, lowercase=True)
            X = vec.fit_transform(texts[m])
            yr = y[m]
            w = {}
            for name, pos in (("high", yr == 0), ("low", yr == 2)):
                a = self.c["alpha"]
                p1 = (np.asarray(X[pos].sum(0)).ravel() + a) / (pos.sum() + 2 * a)
                p0 = (np.asarray(X[~pos].sum(0)).ravel() + a) / ((~pos).sum() + 2 * a)
                lo = np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0))
                keep = np.argsort(-np.abs(lo))[: self.c["top_k"]]
                vec_w = np.zeros(X.shape[1])
                vec_w[keep] = lo[keep]
                w[name] = vec_w
            mask = (w["high"] != 0) | (w["low"] != 0)
            self.models[role] = (vec, w, mask)
        return self

    def score(self, texts: np.ndarray, roles: np.ndarray) -> np.ndarray:
        out = np.zeros((len(texts), 3))
        for role, (vec, w, mask) in self.models.items():
            m = roles == role
            if not m.any():
                continue
            X = vec.transform(texts[m])
            hits = np.asarray(X[:, mask].sum(1)).ravel()
            den = np.sqrt(1 + hits)
            out[m, 0] = (X @ w["high"]) / den
            out[m, 1] = (X @ w["low"]) / den
            out[m, 2] = hits
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    paths, cfg = load_config(CONFIG_NAME)
    mc = cfg["mined_ngrams"]

    targets = load_targets(cfg, paths)
    targets["text"] = (targets["subject"].fillna("") + " " + targets["latest_text"].fillna("")).str.strip()
    dev = pd.concat([load_split("train"), load_split("val")])
    lab = {l: i for i, l in enumerate(LABELS)}
    smp = targets[targets["set"] == "sampled"].set_index("key")
    dev_mids = dev["message_id"].map(norm_mid).values
    texts = smp.loc[dev_mids, "text"].values
    roles = dev["role"].values
    y = dev["label_final"].map(lab).values
    groups = dev["thread_id"].values

    oof = np.zeros((len(dev), 3))
    for k, (tr, va) in enumerate(StratifiedGroupKFold(mc["n_folds"], shuffle=True, random_state=mc["fold_seed"])
                                 .split(texts, y, groups)):
        oof[va] = RoleMiner(mc).fit(texts[tr], roles[tr], y[tr]).score(texts[va], roles[va])
        logger.info("fold %d scored", k + 1)
    full = RoleMiner(mc).fit(texts, roles, y)

    cols = ["d3x_mined_high", "d3x_mined_low", "d3x_mined_hits"]
    scores = full.score(targets["text"].values, targets["role"].values)
    feats = pd.DataFrame(scores, columns=cols)
    feats.insert(0, "key", targets["key"].values)
    feats.insert(1, "set", targets["set"].values)
    smp_rows = feats.index[feats["set"] == "sampled"]
    key_to_row = dict(zip(feats.loc[smp_rows, "key"], smp_rows))
    feats.loc[[key_to_row[m] for m in dev_mids], cols] = oof
    for name, g in feats.groupby("set"):
        g.drop(columns="set").to_parquet(out_dir / f"mined_{name}.parquet", index=False)

    top = {}
    for role, (vec, w, _) in full.models.items():
        vocab = np.array(vec.get_feature_names_out())
        top[role] = {"n_vocab": int(len(vocab)),
                     "n_high_kept": int((w["high"] != 0).sum()), "n_low_kept": int((w["low"] != 0).sum())}
    write_json(out_dir / "mined_stats.json", {
        "config": mc, "git_commit": git_commit(Path(paths["repo_root"])), "n_dev_oof": int(len(dev)),
        "per_role": top, "describe": feats.drop(columns=["key", "set"]).describe().round(4).to_dict()})
    logger.info("saved mined n-gram features")


if __name__ == "__main__":
    main()
