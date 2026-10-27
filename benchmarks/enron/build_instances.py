#!/usr/bin/env python3
"""
E12 step 1: build Enron recipient-aware instances, the mailbox event history and splits.
Version: v1 | Date: 2026-10-23

Outputs (new timestamped directory under the runs base):
  events.parquet         one row per (owner mailbox, unique email) with direction sent/received,
                         time, sender, recipients, normalised subject; read by the D2 builder
  instances.jsonl.gz     labelled-to-be instances (email x recipient owner) with the fields the
                         agent panel and the D1 extractor read
  stats.json

Unique emails are keyed by sender, timestamp, normalised subject and the start of the body, so
copies of one email in several folders or mailboxes share a key. Cross-role emails (received by
owners of two or more roles) form the role-sensitivity test set; every other instance is split
thread-disjointly into development and test.

Usage:
  python -m benchmarks.enron.build_instances
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.enron.enron_common import (  # noqa: E402
    CONFIG_NAME, ascii_ratio, has_attachment, load_positions, norm_addr, norm_subject, split_latest,
)
from lmeic.common.rev1_common import git_commit, load_config, new_output_dir, write_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_corpus(cfg: dict, repo: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(repo / cfg["corpus"]["parquet_glob"])))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    parts = df["file_name"].astype(str).str.split("/")
    df["mailbox"], df["folder"] = parts.str[0], parts.str[1].str.lower()
    df["from_addr"] = df["from"].map(norm_addr)
    for c in ["to", "cc", "bcc"]:
        df[c] = df[c].map(lambda xs: [norm_addr(x) for x in (xs if xs is not None else []) if x])
    df["nsubj"] = df["subject"].map(norm_subject)
    lo, hi = (pd.Timestamp(x, tz="UTC") for x in cfg["history_window"])
    df = df[(df["date"] >= lo) & (df["date"] <= hi)].copy()
    head = df["body"].fillna("").str.slice(0, 200)
    key_src = df["from_addr"] + "|" + df["date"].astype(str) + "|" + df["nsubj"] + "|" + head
    df["email_key"] = key_src.map(lambda s: hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:16])
    return df


def owner_addresses(df: pd.DataFrame, pos: pd.DataFrame, sent_folders: list[str]) -> dict[str, set[str]]:
    out = {r.mailbox: set(r.addrs) for r in pos.itertuples()}
    sent = df[df["folder"].isin(sent_folders)]
    for mb, g in sent.groupby("mailbox"):
        top = g["from_addr"].value_counts()
        if len(top):
            out.setdefault(mb, set()).add(top.index[0])
    return out


def build_events(df: pd.DataFrame, addrs: dict[str, set[str]], sent_folders: list[str]) -> pd.DataFrame:
    own = df["mailbox"].map(lambda m: addrs.get(m, set()))
    is_self = [f in a for f, a in zip(df["from_addr"], own)]
    direction = np.where(df["folder"].isin(sent_folders) | np.array(is_self), "sent", "received")
    ev = pd.DataFrame({"owner": df["mailbox"].values, "direction": direction, "t": pd.DatetimeIndex(df["date"]),
                       "sender": df["from_addr"].values, "to": df["to"].values, "cc": df["cc"].values,
                       "nsubj": df["nsubj"].values, "email_key": df["email_key"].values,
                       "folder": df["folder"].values, "row": df.index.values})
    ev = ev.sort_values(["owner", "t", "email_key"]).drop_duplicates(["owner", "email_key"]).reset_index(drop=True)
    ev["owner_in_to"] = [o_addr & set(to) != set() for o_addr, to in zip(ev["owner"].map(lambda m: addrs.get(m, set())), ev["to"])]
    th = ev["nsubj"].where(ev["nsubj"] != "", "nosubj:" + ev["email_key"])
    ev["thread_id"] = th
    ev["thread_depth"] = ev.groupby(["owner", "thread_id"]).cumcount()
    return ev


def candidate_targets(df: pd.DataFrame, ev: pd.DataFrame, owner_role: dict[str, str], cfg: dict) -> pd.DataFrame:
    ic = cfg["instances"]
    lo, hi = (pd.Timestamp(x, tz="UTC") for x in cfg["target_window"])
    c = ev[(ev["direction"] == "received") & ev["owner"].isin(owner_role) & (ev["t"] >= lo) & (ev["t"] <= hi)
           & (ev["sender"] != "")].copy()
    c["role"] = c["owner"].map(owner_role)
    bodies = df.loc[c["row"], "body"].fillna("").values
    split = [split_latest(b) for b in bodies]
    c["latest_text"] = [s[0] if len(s[0]) >= ic["min_body_chars"] else b.strip() for s, b in zip(split, bodies)]
    c["is_forward_shell"] = [len(s[0]) < ic["min_body_chars"] and s[1] for s in split]
    c["has_attachment"] = [has_attachment(b) for b in bodies]
    c["latest_text"] = c["latest_text"].str.slice(0, ic["max_body_chars"])
    c["subject"] = df.loc[c["row"], "subject"].fillna("").values
    ok = (c["latest_text"].str.len() >= ic["min_body_chars"]) & (c["latest_text"].map(ascii_ratio) >= ic["ascii_ratio"])
    return c[ok].copy()


def sample_role(c: pd.DataFrame, n: int, min_owner: int, rng: np.random.Generator) -> pd.DataFrame:
    sizes = c["owner"].value_counts()
    owners = sorted(sizes[sizes >= min_owner].index)
    quota = {o: 0 for o in owners}
    left = n
    while left > 0 and owners:
        share = max(1, math.floor(left / len(owners)))
        for o in list(owners):
            take = min(share, sizes[o] - quota[o], left)
            quota[o] += take
            left -= take
            if quota[o] >= sizes[o]:
                owners.remove(o)
            if left == 0:
                break
    parts = [g.iloc[rng.permutation(len(g))[:quota[o]]] for o, g in c.groupby("owner") if quota.get(o, 0) > 0]
    return pd.concat(parts) if parts else c.iloc[:0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    paths, cfg = load_config(CONFIG_NAME)
    repo = Path(paths["repo_root"])
    out_dir = Path(args.out_dir) if args.out_dir else new_output_dir(paths, cfg["output_prefix"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ic, rng = cfg["instances"], np.random.default_rng(cfg["instances"]["seed"])

    pos = load_positions(repo / cfg["positions"]["csv"])
    df = load_corpus(cfg, repo)
    logger.info("corpus rows in history window: %d", len(df))
    addrs = owner_addresses(df, pos, cfg["sent_folders"])
    ev = build_events(df, addrs, cfg["sent_folders"])
    logger.info("owner events: %d (%s)", len(ev), ev["direction"].value_counts().to_dict())

    status_to_role = {s: r for r, ss in cfg["roles"].items() for s in ss}
    owner_role = {r.mailbox: status_to_role[r.status] for r in pos.itertuples() if r.status in status_to_role}
    cand = candidate_targets(df, ev, owner_role, cfg)
    logger.info("candidate targets: %d", len(cand))

    per_key_roles = cand.groupby("email_key")["role"].nunique()
    multi = per_key_roles[per_key_roles >= 2].index.values
    n_rcpt = df.loc[cand.drop_duplicates("email_key").set_index("email_key").loc[multi, "row"]]
    n_rcpt = (n_rcpt["to"].map(len) + n_rcpt["cc"].map(len)).values
    small, large = multi[n_rcpt <= 15], multi[n_rcpt > 15]
    half = ic["cross_role_emails"] // 2
    pick_small = rng.choice(small, min(len(small), half), replace=False)
    pick_large = rng.choice(large, min(len(large), ic["cross_role_emails"] - len(pick_small)), replace=False)
    cross_keys = set(pick_small) | set(pick_large)
    cr = cand[cand["email_key"].isin(cross_keys)]
    cr = cr.iloc[rng.permutation(len(cr))].drop_duplicates(["email_key", "role"])
    cr = cr.assign(cross_role=1)

    rest = cand[~cand["email_key"].isin(cross_keys)].drop_duplicates(["email_key", "role"])
    main_parts = [sample_role(rest[rest["role"] == r], ic["per_role"], ic["min_owner_received"], rng)
                  for r in cfg["roles"]]
    inst = pd.concat(main_parts + [cr], ignore_index=True)
    inst["cross_role"] = inst["cross_role"].fillna(0).astype(int)

    sc = cfg["splits"]
    rng_s = np.random.default_rng(sc["seed"])
    cross_threads = set(inst.loc[inst["cross_role"] == 1, "thread_id"])
    other = sorted(set(inst["thread_id"]) - cross_threads)
    n_test = int(round(sc["test_share"] * len(other)))
    test_threads = cross_threads | set(rng_s.choice(other, n_test, replace=False))
    inst["split"] = np.where(inst["thread_id"].isin(test_threads), "test", "dev")
    assert not (set(inst.loc[inst["split"] == "dev", "thread_id"]) & test_threads)

    inst = inst.sort_values(["split", "role", "owner", "t"]).reset_index(drop=True)
    inst["instance_id"] = [f"e12_{i:05d}" for i in range(len(inst))]
    pos_by_mb = pos.set_index("mailbox")["status"].to_dict()
    recs = []
    for r in inst.itertuples():
        recs.append({
            "instance_id": r.instance_id, "email_key": r.email_key, "owner": r.owner, "role": r.role,
            "owner_status": pos_by_mb.get(r.owner, ""), "from_addr": r.sender, "from_display": r.sender,
            "date_utc": pd.Timestamp(r.t).isoformat(), "subject": r.subject, "latest_text": r.latest_text,
            "to_addrs_count": len(r.to), "cc_addrs_count": len(r.cc),
            "target_user_position": "to" if r.owner_in_to else "cc", "has_attachment": bool(r.has_attachment),
            "is_forward_shell": bool(r.is_forward_shell), "thread_id": r.thread_id,
            "thread_depth": int(r.thread_depth), "language": "en", "cross_role": int(r.cross_role),
            "split": r.split})
    import gzip
    with gzip.open(out_dir / "instances.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ev.drop(columns=["row"]).to_parquet(out_dir / "events.parquet", index=False)
    json.dump({mb: sorted(a) for mb, a in addrs.items()}, open(out_dir / "owner_addresses.json", "w"))

    tab = inst.groupby(["split", "role", "cross_role"]).size().rename("n").reset_index()
    write_json(out_dir / "stats.json", {
        "config": cfg, "git_commit": git_commit(repo), "n_corpus_rows": int(len(df)), "n_events": int(len(ev)),
        "n_candidates": int(len(cand)), "n_multi_role_emails": int(len(multi)),
        "n_cross_role_emails": len(cross_keys), "n_instances": int(len(inst)),
        "owners_per_role": inst.groupby("role")["owner"].nunique().to_dict(),
        "counts": tab.to_dict(orient="records"),
        "n_threads": {"dev": int(inst.loc[inst["split"] == "dev", "thread_id"].nunique()),
                      "test": int(inst.loc[inst["split"] == "test", "thread_id"].nunique())}})
    logger.info("instances %d -> %s\n%s", len(inst), out_dir, tab.to_string(index=False))


if __name__ == "__main__":
    main()
