#!/usr/bin/env python3
"""
E12 step 2: agent-panel labels and D1 descriptors for the Enron instances.
Version: v1.1 | Date: 2026-10-23 (v1.1: --profile for d1)

Runs the revision-1 labelling and D1 extraction code unchanged except for the domain:
  labels  registers the Enron role cards (rev1_e12_enron.yaml: role_criteria) and replaces the
          shipping-specific examples in the agent prompts (agent_prompt_replacements)
  d1      registers the Enron role descriptions (role_desc) and replaces the company domain in
          the D1 system prompt (d1_domain)
  d1_local  the same D1 prompt changes for the local vLLM extractor (run in the vLLM environment);
          output rows are keyed by the instance file's row position
The instance file is the set "e12_enron" in rev1_agent_labelling.yaml and rev1_d1_extraction.yaml.

Usage:
  python -m benchmarks.enron.run_api labels --out-dir DIR [--limit N]
  python -m benchmarks.enron.run_api d1 --out-dir DIR [--limit N]
  python -m benchmarks.enron.run_api d1_local --out-dir DIR [--model qwen3_8b]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.enron.enron_common import CONFIG_NAME  # noqa: E402
from lmeic.common.rev1_common import load_config  # noqa: E402

SET_NAME = "e12_enron"


def patch_labelling(ecfg: dict) -> None:
    import prompts.system_prompts as sp
    sp.ROLE_CRITERIA.update(ecfg["role_criteria"])
    for old, new in ecfg["agent_prompt_replacements"]:
        for k in sp.AGENT_PROMPTS:
            sp.AGENT_PROMPTS[k] = sp.AGENT_PROMPTS[k].replace(old, new)
        sp.DECISION_BOUNDARY_RULES = sp.DECISION_BOUNDARY_RULES.replace(old, new)
        assert all(old not in p for p in [*sp.AGENT_PROMPTS.values(), sp.DECISION_BOUNDARY_RULES])


def patch_d1(ecfg: dict) -> None:
    import lmeic.features.d1_prompt_variants as pv
    import lmeic.features.extract_d1 as x
    old, new = ecfg["d1_domain"]
    assert old in pv.SYSTEM_PROMPT
    pv.SYSTEM_PROMPT = pv.SYSTEM_PROMPT.replace(old, new)
    x.ROLE_DESC.update(ecfg["role_desc"])


def run_local_d1(ecfg: dict, paths: dict, out_dir: Path, model: str, limit: int | None) -> None:
    import lmeic.features.d1_extract_local as dl
    import lmeic.features.extract_d1 as x
    old, new = ecfg["d1_domain"]
    assert old in dl.SYSTEM_PROMPT
    dl.SYSTEM_PROMPT = dl.SYSTEM_PROMPT.replace(old, new)
    x.ROLE_DESC.update(ecfg["role_desc"])
    _, acfg = load_config("rev1_agent_labelling.yaml")
    sys.argv = ["d1_extract_local", "--model", model, "--out-dir", str(out_dir),
                "--input", acfg["sets"][SET_NAME]["path"]] + (["--limit", str(limit)] if limit else [])
    dl.main()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["labels", "d1", "d1_local"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="qwen3_8b", help="d1_local: model key in rev1_d1_local.yaml")
    ap.add_argument("--profile", default=None, help="d1: api profile in rev1_d1_extraction.yaml")
    args = ap.parse_args()
    paths, ecfg = load_config(CONFIG_NAME)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.task == "d1_local":
        run_local_d1(ecfg, paths, out_dir, args.model, args.limit)
    elif args.task == "labels":
        from lmeic.labelling import agent_label as al
        patch_labelling(ecfg)
        asyncio.run(al.run(SET_NAME, out_dir, args.limit))
    else:
        from lmeic.features import d1_extract_rev1 as dx
        patch_d1(ecfg)
        asyncio.run(dx.run(SET_NAME, out_dir, args.limit, profile=args.profile))


if __name__ == "__main__":
    main()
