#!/usr/bin/env python3
"""
E7 richer context features, part 1: history, communication graph and label prior.

Version: v1 | Date: 2026-10-22

Every feature uses only records dated strictly before the target email (t):
  d2x_role_load_{h}h        emails received by the role mailbox in the h hours before t
  d2x_sender_gap_days       days since the sender's previous email to this role (capped)
  d2x_owner_to_sender_{d}d  messages from the role owner to the sender in the d days before t
  d2x_reciprocity           owner-to-sender messages / (sender-to-role messages + 1), all before t
  d2x_owner_reply_share     share of the sender's earlier emails to the role answered by the owner
  d2x_owner_reply_latency_h median owner reply latency to this sender (hours; -1 if none)
  d2x_thread_owner_msgs     earlier messages from the role owner in the same thread
  d2x_thread_participants   distinct senders in the same thread before t
  d2x_graph_degree          distinct correspondents of the sender in the communication graph
  d2x_graph_pagerank_log    log PageRank of the sender
  d2x_graph_owner_cos       cosine between sender and role-owner spectral embeddings of the
                            normalised adjacency, leading (degree) component removed
                            (graph snapshot = all records before the first day of t's month)
  d2x_prior_high/low        smoothed share of High / Low among earlier development-pool labels,
                            backing off sender+role -> sender -> sender domain -> role; labels of
                            the target's own thread are excluded
  d2x_prior_n_log           log(1 + earlier labelled emails at the sender+role level)
Role owners are the most frequent To address of each role mailbox; owner messages are
recognised by sender address in any mailbox, so owner activity is only partly observed
for roles without a sent folder. Mailbox filing folders are never read.

Usage:
  python -m lmeic.features.context_rich_v1 [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import bisect
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.common.rev1_common import (  # noqa: E402
    git_commit, load_config, new_output_dir, norm_mid, read_jsonl, source_run_dir, write_json,
)
from lmeic.features.d2d3_v3 import parse_date, read_jsonl_gz  # noqa: E402
from lmeic.common.data_loader import load_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
CONFIG_NAME = "rev1_context_rich.yaml"
LABELS = ["High", "Medium", "Low"]
ROLES = ["Mgr", "CS", "Sch"]


def _addr(a) -> str:
    return (a or "").strip().lower()


def _addr_list(v) -> list[str]:
    if not v:
        return []
    if isinstance(v, str):
        v = [v]
    return [_addr(a) for a in v if a]


def load_targets(cfg: dict, paths: dict) -> pd.DataFrame:
    """One row per target instance: EMAIL-3R sampled rows and the external evaluation sets."""
    repo = Path(paths["repo_root"])
    src = source_run_dir(paths, cfg["source_run_id"])
    cols = ["key", "set", "message_id", "role", "from_addr", "date", "thread_id", "subject", "latest_text"]
    rows = []
    for r in read_jsonl_gz(src / cfg["sampled_relpath"]):
        mid = norm_mid(r["message_id"])
        rows.append([mid, "sampled", mid, r.get("role"), _addr(r.get("from_addr")), parse_date(r.get("date_utc")),
                     r.get("thread_id") or "", r.get("subject") or "", r.get("latest_text") or ""])
    for name, rel in cfg["external_sets"].items():
        for r in read_jsonl(repo / rel):
            rows.append([str(r["instance_id"]), name, norm_mid(r["message_id"]), r.get("role"),
                         _addr(r.get("from_addr")), parse_date(r.get("date_utc")), r.get("thread_id") or "",
                         r.get("subject") or "", r.get("latest_text") or ""])
    df = pd.DataFrame(rows, columns=cols)
    assert not df[df["set"] == "sampled"]["key"].duplicated().any(), "duplicate sampled message_id"
    return df


class HistoryExt:
    def __init__(self, history: list[dict]):
        to_counts: dict[str, Counter] = defaultdict(Counter)
        for r in history:
            for a in _addr_list(r["to_addrs"]):
                to_counts[r["role"]][a] += 1
        self.owner = {role: to_counts[role].most_common(1)[0][0] for role in ROLES}
        owners = set(self.owner.values())

        self.role_dates: dict[str, list] = defaultdict(list)
        self.sender_role: dict[tuple, list] = defaultdict(list)
        self.owner_to: dict[tuple, list] = defaultdict(list)
        self.thread: dict[str, list] = defaultdict(list)
        msg = {}
        seen_thread, seen_owner = set(), set()
        for r in history:
            t = r["date"]
            self.role_dates[r["role"]].append(t)
            self.sender_role[(r["from"], r["role"])].append(t)
            uid = r["mid"] or id(r)
            if (r["thread_id"], uid) not in seen_thread:
                seen_thread.add((r["thread_id"], uid))
                self.thread[r["thread_id"]].append((t, r["from"]))
            if r["mid"]:
                msg.setdefault(r["mid"], (r["from"], t, r["role"]))
            if r["from"] in owners and uid not in seen_owner:
                seen_owner.add(uid)
                for a in set(_addr_list(r["to_addrs"]) + _addr_list(r["cc_addrs"])):
                    self.owner_to[(r["from"], a)].append(t)
        self.replies: dict[tuple, list] = defaultdict(list)    # (owner, sender) -> [(parent_t, reply_t)]
        seen = set()
        for r in history:
            if r["from"] in owners and r["irt"] and r["irt"] in msg and (r["mid"], r["irt"]) not in seen:
                seen.add((r["mid"], r["irt"]))
                p_from, p_t, _ = msg[r["irt"]]
                if r["date"] > p_t:
                    self.replies[(r["from"], p_from)].append((p_t, r["date"]))
        for d in (self.role_dates, self.sender_role, self.owner_to):
            for k in d:
                d[k].sort()
        for k in self.thread:
            self.thread[k].sort(key=lambda x: x[0])
        for k in self.replies:
            self.replies[k].sort()

    @staticmethod
    def _count(dates: list, lo, hi) -> int:
        return bisect.bisect_left(dates, hi) - bisect.bisect_left(dates, lo)

    def features(self, role: str, sender: str, tid: str, t: datetime, hcfg: dict) -> dict:
        out = {}
        rd = self.role_dates.get(role, [])
        for h in hcfg["load_windows_hours"]:
            out[f"d2x_role_load_{h}h"] = self._count(rd, t - timedelta(hours=h), t)
        sr = self.sender_role.get((sender, role), [])
        k = bisect.bisect_left(sr, t)
        cap = hcfg["gap_cap_days"]
        out["d2x_sender_gap_days"] = min(cap, (t - sr[k - 1]).total_seconds() / 86400) if k else cap
        owner = self.owner.get(role, "")
        ot = self.owner_to.get((owner, sender), [])
        d = hcfg["owner_window_days"]
        out[f"d2x_owner_to_sender_{d}d"] = self._count(ot, t - timedelta(days=d), t)
        n_ot = bisect.bisect_left(ot, t)
        out["d2x_reciprocity"] = n_ot / (k + 1)
        rep = [(p, q) for p, q in self.replies.get((owner, sender), []) if q < t]
        out["d2x_owner_reply_share"] = min(1.0, len(rep) / k) if k else 0.0
        out["d2x_owner_reply_latency_h"] = (float(np.median([(q - p).total_seconds() / 3600 for p, q in rep]))
                                            if rep else -1.0)
        prev = [s for (u, s) in self.thread.get(tid, []) if u < t] if tid else []
        out["d2x_thread_owner_msgs"] = sum(1 for s in prev if s == owner)
        out["d2x_thread_participants"] = len(set(prev))
        return out


class GraphSnapshots:
    """Monthly communication-graph snapshots built from all records before each month start."""

    def __init__(self, history: list[dict], months: list[datetime], gcfg: dict, owners: dict[str, str]):
        nodes = {}
        edges = []
        for r in history:
            u = nodes.setdefault(r["from"], len(nodes))
            for a in set(_addr_list(r["to_addrs"]) + _addr_list(r["cc_addrs"])):
                if a and a != r["from"]:
                    edges.append((r["date"], u, nodes.setdefault(a, len(nodes))))
        edges.sort(key=lambda e: e[0])
        self.nodes, self.owners, self.snap = nodes, owners, {}
        times = [e[0] for e in edges]
        n = len(nodes)
        for m in months:
            j = bisect.bisect_left(times, m)
            if j == 0:
                self.snap[m] = None
                continue
            u = np.array([e[1] for e in edges[:j]])
            v = np.array([e[2] for e in edges[:j]])
            A = sparse.coo_matrix((np.ones(j), (u, v)), shape=(n, n)).tocsr()
            A = A + A.T
            A.data = np.log1p(A.data)
            deg_w = np.asarray(A.sum(1)).ravel()
            degree = np.diff((A > 0).astype(np.int8).tocsr().indptr)
            inv = np.where(deg_w > 0, 1 / np.sqrt(np.maximum(deg_w, 1e-12)), 0)
            S = sparse.diags(inv) @ A @ sparse.diags(inv)
            U, s, _ = svds(S.astype(np.float64), k=gcfg["n_components"] + 1)
            keep = np.argsort(s)[::-1][1:]
            emb = U[:, keep] * s[keep]
            P = sparse.diags(np.where(deg_w > 0, 1 / np.maximum(deg_w, 1e-12), 0)) @ A
            pr = np.full(n, 1 / n)
            dmp = gcfg["pagerank_damping"]
            for _ in range(gcfg["pagerank_iters"]):
                pr = (1 - dmp) / n + dmp * (P.T @ pr) + dmp * pr[deg_w == 0].sum() / n
            self.snap[m] = (degree, emb, pr)
            logger.info("graph snapshot %s: %d edges", m.date(), j)

    def features(self, sender: str, role: str, t: datetime) -> dict:
        m = datetime(t.year, t.month, 1, tzinfo=t.tzinfo)
        snap = self.snap.get(m)
        i = self.nodes.get(sender)
        if snap is None or i is None:
            return {"d2x_graph_degree": 0, "d2x_graph_pagerank_log": float(np.log(1e-9)), "d2x_graph_owner_cos": 0.0}
        degree, emb, pr = snap
        o = self.nodes.get(self.owners.get(role, ""))
        cos = 0.0
        if o is not None:
            a, b = emb[i], emb[o]
            den = np.linalg.norm(a) * np.linalg.norm(b)
            cos = float(a @ b / den) if den > 0 else 0.0
        return {"d2x_graph_degree": int(degree[i]), "d2x_graph_pagerank_log": float(np.log(max(pr[i], 1e-9))),
                "d2x_graph_owner_cos": cos}


class LabelPrior:
    """Smoothed class shares from earlier development-pool labels, excluding the target thread."""

    def __init__(self, dev: pd.DataFrame, pcfg: dict):
        self.m = pcfg["smoothing"]
        self.levels = pcfg["levels"]
        self.idx: dict[tuple, tuple] = {}
        self.thr: dict[tuple, tuple] = {}
        self.base = {role: np.bincount(g["y"], minlength=3) / len(g) for role, g in dev.groupby("role")}
        for lvl in self.levels:
            for key, g in dev.groupby(self._key_cols(lvl)):
                g = g.sort_values("date")
                key = key if isinstance(key, tuple) else (key,)
                oh = np.eye(3)[g["y"].values]
                self.idx[(lvl,) + key] = (list(g["date"]), np.vstack([np.zeros(3), np.cumsum(oh, 0)]))
                for tid, h in g.groupby("thread_id"):
                    ohh = np.eye(3)[h["y"].values]
                    self.thr[(lvl,) + key + (tid,)] = (list(h["date"]), np.vstack([np.zeros(3), np.cumsum(ohh, 0)]))

    @staticmethod
    def _key_cols(lvl: str) -> list[str]:
        return {"sender_role": ["sender", "role"], "sender": ["sender"], "domain": ["domain"], "role": ["role"]}[lvl]

    def _counts(self, lvl: str, key: tuple, tid: str, t) -> np.ndarray:
        e = self.idx.get((lvl,) + key)
        if e is None:
            return np.zeros(3)
        c = e[1][bisect.bisect_left(e[0], t)].copy()
        h = self.thr.get((lvl,) + key + (tid,))
        if h is not None:
            c -= h[1][bisect.bisect_left(h[0], t)]
        return c

    def features(self, sender: str, role: str, tid: str, t) -> dict:
        domain = sender.split("@")[-1] if "@" in sender else ""
        keys = {"sender_role": (sender, role), "sender": (sender,), "domain": (domain,), "role": (role,)}
        p = self.base.get(role, np.full(3, 1 / 3))
        n_top = 0.0
        for lvl in reversed(self.levels):
            c = self._counts(lvl, keys[lvl], tid, t)
            p = (c + self.m * p) / (c.sum() + self.m)
            if lvl == self.levels[0]:
                n_top = c.sum()
        return {"d2x_prior_high": float(p[0]), "d2x_prior_low": float(p[2]), "d2x_prior_n_log": float(np.log1p(n_top))}


def load_history(path: Path) -> list[dict]:
    out = []
    for r in read_jsonl_gz(path):
        t = parse_date(r.get("date_utc"))
        if t is None:
            continue
        out.append({"date": t, "role": r.get("role"), "from": _addr(r.get("from_addr")),
                    "to_addrs": r.get("to_addrs"), "cc_addrs": r.get("cc_addrs"),
                    "thread_id": r.get("thread_id") or "", "mid": norm_mid(r.get("message_id") or ""),
                    "irt": norm_mid(r.get("in_reply_to") or "")})
    logger.info("history: %d records", len(out))
    return out


def dev_labels(targets: pd.DataFrame) -> pd.DataFrame:
    dev = pd.concat([load_split("train"), load_split("val")])
    lab = {l: i for i, l in enumerate(LABELS)}
    s = targets[targets["set"] == "sampled"].set_index("key")
    mids = dev["message_id"].map(norm_mid)
    out = pd.DataFrame({"sender": s.loc[mids, "from_addr"].values, "role": dev["role"].values,
                        "date": s.loc[mids, "date"].tolist(), "thread_id": dev["thread_id"].values,
                        "y": dev["label_final"].map(lab).values})
    out["domain"] = out["sender"].map(lambda a: a.split("@")[-1] if "@" in a else "")
    return out.dropna(subset=["date"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    paths, cfg = load_config(CONFIG_NAME)
    repo = Path(paths["repo_root"])
    out_dir = Path(args.out_dir) if args.out_dir else new_output_dir(paths, cfg["output_prefix"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output: %s", out_dir)

    targets = load_targets(cfg, paths)
    history = load_history(source_run_dir(paths, cfg["source_run_id"]) / cfg["history_relpath"])
    hx = HistoryExt(history)
    t_valid = targets["date"].dropna()
    months = sorted({datetime(t.year, t.month, 1, tzinfo=t.tzinfo) for t in t_valid})
    graph = GraphSnapshots(history, months, cfg["graph"], hx.owner)
    prior = LabelPrior(dev_labels(targets), cfg["label_prior"])

    rows = []
    for r in targets.itertuples(index=False):
        if r.date is None or pd.isna(r.date):
            rows.append({})
            continue
        f = hx.features(r.role, r.from_addr, r.thread_id, r.date, cfg["history"])
        f.update(graph.features(r.from_addr, r.role, r.date))
        f.update(prior.features(r.from_addr, r.role, r.thread_id, r.date))
        rows.append(f)
    feats = pd.DataFrame(rows).fillna(0.0)
    feats.insert(0, "key", targets["key"].values)
    feats.insert(1, "set", targets["set"].values)
    for name, g in feats.groupby("set"):
        g.drop(columns="set").to_parquet(out_dir / f"history_{name}.parquet", index=False)
    write_json(out_dir / "history_stats.json", {
        "config": cfg, "git_commit": git_commit(repo), "n_targets": targets["set"].value_counts().to_dict(),
        "n_missing_date": int(targets["date"].isna().sum()),
        "owner_messages_observed": {r: int(sum(len(v) for (o, _), v in hx.owner_to.items() if o == hx.owner[r]))
                                    for r in ROLES},
        "describe": feats.drop(columns=["key", "set"]).describe().round(4).to_dict()})
    logger.info("saved history features for %s", feats["set"].value_counts().to_dict())


if __name__ == "__main__":
    main()
