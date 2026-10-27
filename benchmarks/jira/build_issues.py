#!/usr/bin/env python3
"""
E18: Apache Jira public benchmark — build the issue table (creation-time priority).
Version: v1.1 | Date: 2026-10-26
v1.1 (audit 2026-10-26): every creation-time field is recovered from the change log by taking the
  EARLIEST event of that field (v1 replayed edits in stream order and kept the value before the last
  edit, and kept text that was added later when the creation value was empty). Also recovered:
  issue type, components (undo adds/removes after creation), assignee and reporter at creation;
  has_attachment counts only attachments added within MERGE_MINUTES of creation.

Two passes over the MSR 2023 Apache Jira dump (Diamantopoulos et al., CC BY 4.0):
  pass 1  events.bson.gz  -> for every priority change, (issue_id, timestamp, from, to); keep the
                             earliest change per issue, which gives the value at creation
  pass 2  issues.bson.gz  -> one row per issue: created, project, components, type, status,
                             resolution, reporter, assignee-at-creation (if no reassignment event),
                             summary/description at creation (recovered from the events when the
                             text was edited later), current priority and label provenance

Priority at creation follows Li 2024 §3.4.2 R2/R3/R4: the first priority value in the change log
is the creation value; changes that happen within MERGE_MINUTES of creation are treated as part of
the initial triage (Li's example: Major set on creation, corrected to Critical one minute later is
recorded as Critical at creation). Selection of projects follows Li's criteria that the dump
supports (five priority levels, >5 years of activity, >150 priority-changed bugs); C3 (commits) is
not in this dump and is reported as not applied.

Output: issues.parquet, events_priority.parquet, build_stats.json
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.jira.bson_stream import iter_docs  # noqa: E402
from lmeic.common.rev1_common import git_commit, write_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

LEVELS = {"Blocker": 1, "Critical": 2, "Major": 3, "Minor": 4, "Trivial": 5}
MERGE_MINUTES = 5
TEXT_FIELDS = ("summary", "description", "Component", "components", "environment")


def ts(s):
    if not s:
        return None
    if isinstance(s, dt.datetime):
        return s if s.tzinfo else s.replace(tzinfo=dt.timezone.utc)
    return dt.datetime.strptime(str(s)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)


def pass_events(path: Path) -> tuple[dict[str, list], dict[str, list], dict[str, list], set, int]:
    """First/last priority change, text edits and reassignment events per issue."""
    first_pri: dict[str, tuple] = {}
    last_pri: dict[str, tuple] = {}
    text_edits: dict[str, list] = collections.defaultdict(list)
    reassigned: set[str] = set()
    n_events = 0
    for e in iter_docs(str(path)):
        n_events += 1
        iid = str(e.get("issue"))
        t = ts(e.get("created"))
        for it in e.get("items") or []:
            f = (it.get("field") or "").lower()
            if f == "priority":
                rec = (t, it.get("fromString"), it.get("toString"))
                if iid not in first_pri or t < first_pri[iid][0]:
                    first_pri[iid] = rec
                if iid not in last_pri or t > last_pri[iid][0]:
                    last_pri[iid] = rec
            elif f in ("summary", "description", "issuetype", "assignee", "reporter"):
                # store the value BEFORE the change; the earliest one is the value at creation
                val = it.get("from") if f in ("assignee", "reporter") else it.get("fromString")
                text_edits[iid].append((t, f, val, None))
                if f == "assignee":
                    reassigned.add(iid)
            elif f == "component":
                # add: from None, to name; remove: from name, to None
                text_edits[iid].append((t, "component", it.get("fromString"), it.get("toString")))
    return first_pri, last_pri, text_edits, reassigned, n_events


def creation_priority(first: tuple | None, current: str | None, created: dt.datetime) -> tuple[str | None, str]:
    """Return (label at creation, provenance)."""
    if first is None:
        return current, "no_change_use_current"
    t, frm, to = first
    if frm is None:                     # priority was empty before the first change
        return to, "first_change_from_null"
    if (t - created).total_seconds() <= MERGE_MINUTES * 60:
        return to, "merged_initial_triage"   # Li R3: within 5 minutes counts as creation value
    return frm, "earliest_change_from"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="datasets/public/apache_jira/raw")
    ap.add_argument("--out", default="datasets/public/apache_jira/processed")
    ap.add_argument("--merge-minutes", type=int, default=MERGE_MINUTES)
    args = ap.parse_args()
    raw = Path(args.raw_dir)
    t0 = time.time()

    logger.info("pass 1: events")
    first_pri, last_pri, text_edits, reassigned, n_events = pass_events(raw / "events.bson.gz")
    logger.info("events %d | issues with at least one priority change %d", n_events, len(first_pri))

    logger.info("pass 2: issues")
    rows = []
    prov = collections.Counter()
    for d in iter_docs(raw / "issues.bson.gz"):
        iid = str(d.get("_id"))
        created = ts(d.get("created"))
        cur = (d.get("priority") or {}).get("name")
        label, p = creation_priority(first_pri.get(iid), cur, created)
        prov[p] += 1
        pr = d.get("project") or {}
        cur_vals = {"summary": d.get("summary"), "description": d.get("description"),
                    "issuetype": (d.get("issuetype") or {}).get("name"), "assignee": d.get("assignee"),
                    "reporter": d.get("reporter")}
        evs = sorted(text_edits.get(iid, []), key=lambda e: e[0])
        first = {}
        for t, f, frm, to in evs:
            if f != "component" and f not in first:
                first[f] = frm                        # value before the first change = value at creation
        at = {f: (first[f] if f in first else cur_vals[f]) for f in cur_vals}
        comp_names = [c.get("name", "") for c in (d.get("components") or [])]
        comps_now = set(comp_names)
        for t, f, frm, to in reversed(evs):           # undo component changes, latest first
            if f != "component":
                continue
            if to is not None and frm is None:
                comps_now.discard(to)
            elif frm is not None and to is None:
                comps_now.add(frm)
        comp_changed = comps_now != set(comp_names)
        summary, desc = at["summary"], at["description"]
        edited = ("summary" in first) or ("description" in first)
        rows.append({
            "issue_id": iid, "issue_key": d.get("key"), "project": pr.get("key") or d.get("projectname"),
            "created": created, "updated": ts(d.get("updated")),
            "priority_current": cur, "priority_creation": label, "priority_provenance": p,
            "priority_changed": iid in first_pri,
            "changed_to": (last_pri[iid][2] if iid in last_pri else None),
            "type": at["issuetype"], "type_changed_later": "issuetype" in first,
            "status": (d.get("status") or {}).get("name"),
            "resolution": (d.get("resolution") or {}).get("name"),
            "reporter": at["reporter"], "assignee_at_creation": at["assignee"],
            "reassigned_later": iid in reassigned,
            "components": "|".join(sorted(comps_now)), "n_components": len(comps_now),
            "components_changed_later": comp_changed,
            "summary": summary, "description": desc, "text_edited_later": edited,
            "has_attachment": any((ts(str(a.get("created", ""))[:19].replace("T", " ")) or created) <= created
                                  + dt.timedelta(minutes=MERGE_MINUTES) for a in (d.get("attachment") or [])),
            "labels": "|".join(d.get("labels") or []),
            "votes": (d.get("votes") or {}).get("votes"), "watches": (d.get("watches") or {}).get("watchCount"),
        })
    df = pd.DataFrame(rows)
    logger.info("issues %d", len(df))

    # Li 2024 project criteria that this dump supports
    pr = df.groupby("project").agg(n=("issue_id", "size"), n_changed=("priority_changed", "sum"),
                                   tmin=("created", "min"), tmax=("created", "max"),
                                   n_levels=("priority_creation", lambda s: s.dropna().nunique()))
    pr["years"] = (pr["tmax"] - pr["tmin"]).dt.days / 365.25
    keep5 = pr["n_levels"] == 5
    keep_years = pr["years"] > 5
    keep_changed = pr["n_changed"] > 150
    pr["selected"] = keep5 & keep_years & keep_changed
    df["project_selected"] = df["project"].map(pr["selected"]).fillna(False)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "issues.parquet", index=False)
    pr.to_csv(out / "projects.csv")
    lv = df["priority_creation"].value_counts(dropna=False).to_dict()
    stats = {
        "source": str(raw), "n_issues": int(len(df)), "n_projects": int(df["project"].nunique()),
        "n_events": int(n_events), "n_priority_changed": int(df["priority_changed"].sum()),
        "pct_priority_changed": round(100 * df["priority_changed"].mean(), 2),
        "provenance": dict(prov),
        "priority_creation": {str(k): int(v) for k, v in lv.items()},
        "share_major": round(float((df["priority_creation"] == "Major").mean()), 4),
        "date_range": [str(df["created"].min()), str(df["created"].max())],
        "text_edited_later": int(df["text_edited_later"].sum()),
        "projects_selected": int(pr["selected"].sum()),
        "selected_issues": int(df["project_selected"].sum()),
        "criteria": {"five_levels": int(keep5.sum()), "over_5_years": int(keep_years.sum()),
                     "over_150_changed": int(keep_changed.sum()),
                     "note": "Li 2024 C3 (>3000 commits) is not available in this dump and is not applied"},
        "merge_minutes": args.merge_minutes, "elapsed_s": round(time.time() - t0, 1),
        "git_commit": git_commit(_REPO_ROOT)}
    write_json(out / "build_stats.json", stats)
    logger.info("%s", json.dumps({k: stats[k] for k in ("n_issues", "n_projects", "pct_priority_changed",
                                                        "share_major", "projects_selected", "selected_issues")}, ensure_ascii=False))
    logger.info("saved %s", out)


if __name__ == "__main__":
    main()
