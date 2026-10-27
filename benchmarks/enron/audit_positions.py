#!/usr/bin/env python3
"""
E12 position audit: compares the owner positions used for the Enron instances with the
original ISI employee status list and writes the subset whose role class agrees.
Version: v1 | Date: 2026-10-23

Owners are matched to ISI names by normalised full-name similarity; an owner is consistent
when the matched ISI status maps to the same role class (rev1_e12_enron.yaml: roles).

Usage:
  python -m benchmarks.enron.audit_positions --run-dir <build_instances dir>
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import time
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.enron.enron_common import CONFIG_NAME, load_positions  # noqa: E402
from lmeic.common.rev1_common import git_commit, load_config, write_json  # noqa: E402


def _norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())


def role_class(status: str, roles: dict[str, list[str]]) -> str:
    return next((r for r, sts in roles.items() if status in sts), "none")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    paths, ecfg = load_config(CONFIG_NAME)
    repo = Path(paths["repo_root"])
    pcfg = ecfg["positions"]
    out = run_dir / f"position_audit_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir()

    pos = load_positions(repo / pcfg["csv"]).set_index("mailbox")
    isi = pd.read_csv(repo / pcfg["isi_reference_csv"], dtype=str, keep_default_na=False)
    isi_keys = [_norm(n) for n in isi["name"]]
    inst = pd.read_json(run_dir / "instances.jsonl.gz", lines=True)

    rows = []
    for owner, g in inst.groupby("owner"):
        p = pos.loc[owner]
        name = f"{p['firstName']} {p['lastName']}"
        ratios = [difflib.SequenceMatcher(None, k, _norm(name)).ratio() for k in isi_keys]
        j = max(range(len(ratios)), key=ratios.__getitem__)
        matched = ratios[j] >= pcfg["isi_match_min_ratio"]
        isi_status = isi.loc[j, "status"] if matched else ""
        isi_role = role_class(isi_status, ecfg["roles"]) if matched else "absent"
        rows.append({"owner": owner, "name": name, "status_used": p["status"], "role_used": g["role"].iloc[0],
                     "isi_name": isi.loc[j, "name"] if matched else "", "isi_status": isi_status,
                     "match_ratio": round(ratios[j], 3), "isi_role": isi_role,
                     "consistent": isi_role == g["role"].iloc[0],
                     "n_instances": len(g), "n_test": int((g["split"] == "test").sum())})
    owners = pd.DataFrame(rows)
    owners.to_csv(out / "owners.csv", index=False)
    keep = inst[inst["owner"].isin(owners.loc[owners["consistent"], "owner"])]
    keep["instance_id"].to_csv(out / "isi_consistent_instances.txt", index=False, header=False)

    def counts(mask):
        sub = owners[mask]
        return {"owners": int(len(sub)), "instances": int(sub["n_instances"].sum()), "test": int(sub["n_test"].sum())}

    write_json(out / "summary.json", {
        "positions_used": pcfg["csv"], "isi_reference": pcfg["isi_reference_csv"],
        "match_min_ratio": pcfg["isi_match_min_ratio"],
        "consistent": counts(owners["consistent"]),
        "role_changed": counts(~owners["consistent"] & (owners["isi_role"] != "absent")),
        "absent_from_isi": counts(owners["isi_role"] == "absent"),
        "kept_by_role": keep["role"].value_counts().to_dict(),
        "kept_cross_role": int(keep["cross_role"].sum()),
        "git_commit": git_commit(repo)})
    print(owners[~owners["consistent"]].to_string(index=False))
    print(f"kept {len(keep)} of {len(inst)} instances -> {out}")


if __name__ == "__main__":
    main()
