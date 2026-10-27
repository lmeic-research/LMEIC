#!/usr/bin/env python3
"""
E7 richer context features, part 2: semantic thread history and role-description similarity.

Version: v1 | Date: 2026-10-22

Dense embeddings (CLS pooling, L2-normalised) from a local encoder on the GPU:
  d2x_thread_hist_sim   cosine between the target email and the mean embedding of up to N
                        earlier messages of the same thread (dated strictly before t)
  d2x_thread_hist_n     number of earlier thread messages used
  d3x_desc_sim_own      cosine between the target email and the core-responsibility statement
                        of its recipient role
  d3x_desc_sim_margin   own-role similarity minus the highest similarity to another role
Role statements are the "ROLE ... Core responsibilities ..." lines of the agent role
criteria; the High/Medium/Low criteria and calibration anchors are not used.

Usage:
  python -m lmeic.features.context_semantic_v1 --out-dir <rev1_context_rich dir>
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.labelling.system_prompts import ROLE_CRITERIA  # noqa: E402
from lmeic.common.rev1_common import git_commit, load_config, norm_mid, source_run_dir, write_json  # noqa: E402
from lmeic.features.context_rich_v1 import CONFIG_NAME, ROLES, load_targets  # noqa: E402
from lmeic.features.d2d3_v3 import parse_date, read_jsonl_gz  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def role_statements() -> dict[str, str]:
    out = {}
    for role in ROLES:
        text = ROLE_CRITERIA[role]
        out[role] = " ".join(text.split("Classification criteria")[0].split())
    return out


@torch.no_grad()
def embed(texts: list[str], model, tok, max_len: int, bs: int) -> np.ndarray:
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(**enc).last_hidden_state[:, 0]
        out[idx] = torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()
        if (i // bs) % 200 == 0:
            logger.info("embedded %d / %d", i + len(idx), len(texts))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    paths, cfg = load_config(CONFIG_NAME)
    sc = cfg["semantic"]
    assert torch.cuda.is_available(), "CUDA required"

    targets = load_targets(cfg, paths)
    thread_msgs: dict[str, list] = defaultdict(list)
    hist_text: dict[str, str] = {}
    for r in read_jsonl_gz(source_run_dir(paths, cfg["source_run_id"]) / cfg["history_relpath"]):
        t, mid = parse_date(r.get("date_utc")), norm_mid(r.get("message_id") or "")
        if t is None or not mid or mid in hist_text:
            continue
        hist_text[mid] = ((r.get("subject") or "") + "\n" + (r.get("latest_text") or "")).strip()
        thread_msgs[r.get("thread_id") or ""].append((t, mid))
    for k in thread_msgs:
        thread_msgs[k].sort()

    prev_ids = []
    for r in targets.itertuples(index=False):
        if r.date is None or pd.isna(r.date) or not r.thread_id:
            prev_ids.append([])
            continue
        prev = [m for (u, m) in thread_msgs.get(r.thread_id, []) if u < r.date and m != r.message_id]
        prev_ids.append(prev[-sc["thread_history_max"]:])
    need = sorted({m for p in prev_ids for m in p})
    tgt_text = (targets["subject"].fillna("") + "\n" + targets["latest_text"].fillna("")).str.strip().tolist()
    stm = role_statements()

    tok = AutoTokenizer.from_pretrained(sc["model_name"])
    model = AutoModel.from_pretrained(sc["model_name"]).cuda().eval()
    t0 = time.time()
    E_t = embed(tgt_text, model, tok, sc["max_len"], sc["batch_size"])
    E_h = embed([hist_text[m] for m in need], model, tok, sc["max_len"], sc["batch_size"]) if need else None
    E_r = embed([stm[r] for r in ROLES], model, tok, sc["max_len"], sc["batch_size"])
    logger.info("embeddings done in %.0fs (%d targets, %d history messages)", time.time() - t0, len(E_t), len(need))
    pos = {m: i for i, m in enumerate(need)}

    sims = E_t @ E_r.T
    ridx = {r: i for i, r in enumerate(ROLES)}
    rows = []
    for i, (r, prev) in enumerate(zip(targets.itertuples(index=False), prev_ids)):
        f = {"d2x_thread_hist_n": len(prev), "d2x_thread_hist_sim": 0.0}
        if prev:
            v = E_h[[pos[m] for m in prev]].mean(0)
            n = np.linalg.norm(v)
            f["d2x_thread_hist_sim"] = float(E_t[i] @ v / n) if n > 0 else 0.0
        j = ridx.get(r.role)
        if j is None:
            f.update({"d3x_desc_sim_own": 0.0, "d3x_desc_sim_margin": 0.0})
        else:
            others = np.delete(sims[i], j)
            f.update({"d3x_desc_sim_own": float(sims[i, j]), "d3x_desc_sim_margin": float(sims[i, j] - others.max())})
        rows.append(f)
    feats = pd.DataFrame(rows)
    feats.insert(0, "key", targets["key"].values)
    feats.insert(1, "set", targets["set"].values)
    for name, g in feats.groupby("set"):
        g.drop(columns="set").to_parquet(out_dir / f"semantic_{name}.parquet", index=False)
    write_json(out_dir / "semantic_stats.json", {
        "config": sc, "git_commit": git_commit(Path(paths["repo_root"])),
        "role_statement_chars": {r: len(s) for r, s in stm.items()},
        "n_history_messages_embedded": len(need),
        "describe": feats.drop(columns=["key", "set"]).describe().round(4).to_dict()})
    logger.info("saved semantic features")


if __name__ == "__main__":
    main()
