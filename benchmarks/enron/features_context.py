#!/usr/bin/env python3
"""
E12 step 3: D2 (header metadata and pre-t mailbox history) and D3 (role statement similarity)
for the Enron instances.
Version: v1 | Date: 2026-10-23

Every history count uses only events of the recipient's mailbox (or, for sender activity, the
whole corpus) dated strictly before the target email. Mailbox folders are never used as features.
  D2-meta  d2_n_to, d2_n_cc, d2_owner_in_to, d2_sender_internal, d2_sender_rank, d2_sender_is_exec,
           d2_hour, d2_weekday, d2_is_reply, d2_is_forward, d2_has_attachment, d2_body_length_log
  D2-hist  d2_sender_freq_30d, d2_sender_freq_90d, d2_owner_sent_to_sender_90d, d2_sender_reply_rate,
           d2_days_since_sender, d2_thread_depth, d2_owner_in_thread_before, d2_sender_activity_90d,
           d2_owner_load_24h
  D3       role one-hot, d3_desc_sim_own, d3_desc_sim_margin (bge-m3, GPU)
Cross-fitted mined role n-grams (D3) need labels and are built by the model runner.

Usage:
  python -m benchmarks.enron.features_context --run-dir <build_instances dir>
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.enron.enron_common import CONFIG_NAME, load_positions  # noqa: E402
from lmeic.common.rev1_common import git_commit, load_config, write_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
DAY = pd.Timedelta(days=1)


def count_between(times: list, lo, hi) -> int:
    return bisect.bisect_left(times, hi) - bisect.bisect_left(times, lo)


def history_features(inst: pd.DataFrame, ev: pd.DataFrame, addr_rank: dict[str, int]) -> pd.DataFrame:
    ev = ev.sort_values("t")
    recv, sent_to, owner_recv, thread_ev = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for r in ev.itertuples():
        if r.direction == "received":
            recv[(r.owner, r.sender)].append(r.t)
            owner_recv[r.owner].append(r.t)
        else:
            for a in set(list(r.to) + list(r.cc)):
                sent_to[(r.owner, a)].append(r.t)
        thread_ev[(r.owner, r.thread_id)].append((r.t, r.direction))
    glob_send = defaultdict(list)
    for r in ev.drop_duplicates("email_key").itertuples():
        glob_send[r.sender].append(r.t)

    rows = []
    for r in inst.itertuples():
        t, o, s = pd.Timestamp(r.date_utc), r.owner, r.from_addr
        rv, st = recv.get((o, s), []), sent_to.get((o, s), [])
        n_recv_prior = bisect.bisect_left(rv, t)
        n_sent_prior = bisect.bisect_left(st, t)
        last = rv[n_recv_prior - 1] if n_recv_prior else None
        th = [d for (tt, d) in thread_ev.get((o, r.thread_id), []) if tt < t]
        subj = str(r.subject or "").strip().lower()
        rank = addr_rank.get(s, 0)
        rows.append({
            "instance_id": r.instance_id,
            "d2_n_to": r.to_addrs_count, "d2_n_cc": r.cc_addrs_count,
            "d2_owner_in_to": int(r.target_user_position == "to"),
            "d2_sender_internal": int(s.endswith("@enron.com")), "d2_sender_rank": rank,
            "d2_sender_is_exec": int(rank == 3), "d2_hour": t.hour, "d2_weekday": t.weekday(),
            "d2_is_reply": int(subj.startswith("re:")), "d2_is_forward": int(subj.startswith(("fw:", "fwd:"))),
            "d2_has_attachment": int(r.has_attachment), "d2_body_length_log": float(np.log1p(len(r.latest_text))),
            "d2_sender_freq_30d": count_between(rv, t - 30 * DAY, t),
            "d2_sender_freq_90d": count_between(rv, t - 90 * DAY, t),
            "d2_owner_sent_to_sender_90d": count_between(st, t - 90 * DAY, t),
            "d2_sender_reply_rate": n_sent_prior / (n_recv_prior + 1),
            "d2_days_since_sender": min(365.0, (t - last) / DAY) if last is not None else 365.0,
            "d2_thread_depth": len(th), "d2_owner_in_thread_before": int(any(d == "sent" for d in th)),
            "d2_sender_activity_90d": count_between(glob_send.get(s, []), t - 90 * DAY, t),
            "d2_owner_load_24h": count_between(owner_recv.get(o, []), t - DAY, t),
        })
    return pd.DataFrame(rows)


def semantic_features(inst: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    import torch
    from transformers import AutoModel, AutoTokenizer
    from lmeic.features.context_semantic_v1 import embed
    assert torch.cuda.is_available(), "CUDA not available"
    sc = cfg["semantic"]
    tok = AutoTokenizer.from_pretrained(sc["model"])
    model = AutoModel.from_pretrained(sc["model"], torch_dtype=torch.float16).cuda().eval()
    roles = list(cfg["role_statements"])
    R = embed([cfg["role_statements"][r] for r in roles], model, tok, sc["max_len"], sc["batch_size"])
    texts = (inst["subject"].fillna("") + "\n" + inst["latest_text"].fillna("")).tolist()
    E = embed(texts, model, tok, sc["max_len"], sc["batch_size"])
    S = E @ R.T
    own = np.array([roles.index(r) for r in inst["role"]])
    sim_own = S[np.arange(len(S)), own]
    other = np.where(np.eye(len(roles), dtype=bool)[own], -np.inf, S).max(1)
    out = pd.DataFrame({"instance_id": inst["instance_id"].values, "d3_desc_sim_own": sim_own,
                        "d3_desc_sim_margin": sim_own - other})
    for r in roles:
        out[f"d3_role_{r}"] = (inst["role"].values == r).astype(int)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    paths, cfg = load_config(CONFIG_NAME)
    repo = Path(paths["repo_root"])
    t0 = time.time()
    inst = pd.read_json(run_dir / "instances.jsonl.gz", lines=True, dtype={"instance_id": str})
    ev = pd.read_parquet(run_dir / "events.parquet")
    pos = load_positions(repo / cfg["positions"]["csv"])
    rank_of = {s: int(k) for k, ss in cfg["sender_rank"].items() for s in ss}
    owner_addrs = json.loads((run_dir / "owner_addresses.json").read_text())
    status = pos.set_index("mailbox")["status"].to_dict()
    addr_rank = {}
    for mb, addrs in owner_addrs.items():
        for a in addrs:
            addr_rank[a] = rank_of.get(status.get(mb, "N/A"), 1)
    for r in pos.itertuples():
        for a in r.addrs:
            addr_rank[a] = rank_of.get(r.status, 1)

    hist = history_features(inst, ev, addr_rank)
    logger.info("history features: %s (%.0fs)", hist.shape, time.time() - t0)
    sem = semantic_features(inst, cfg)
    feats = hist.merge(sem, on="instance_id", how="left")
    assert len(feats) == len(inst) and feats["instance_id"].is_unique
    out = run_dir / f"context_features_{time.strftime('%Y%m%d_%H%M%S')}.parquet"
    feats.to_parquet(out, index=False)
    write_json(out.with_suffix(".json"), {
        "n": len(feats), "columns": list(feats.columns), "git_commit": git_commit(repo),
        "describe": feats.drop(columns=["instance_id"]).describe().round(4).to_dict()})
    logger.info("saved %s (%.0fs)", out, time.time() - t0)


if __name__ == "__main__":
    main()
