#!/usr/bin/env python3
"""
E18d: Apache Jira — zero-shot priority classification with a local open-weight LLM (vLLM).
Version: v1 | Date: 2026-10-25

Scores the E18 test rows (sample_issue_ids.txt, valid D1 record, created >= 2023-01-01; the
same rows as run_d1_fusion / run_id_ablation). Decoding is constrained to the label names:
  S1  High / Medium / Low on all test rows (Blocker+Critical=High, Major=Medium, Minor+Trivial=Low)
  S2  High / Low on test rows whose creation priority is not Major
The prompt states the mapping in Jira's own terms; the model never sees the priority field.
Runs in the vLLM environment (no scikit-learn), so macro-F1 is computed here.

Usage (GPU, vLLM environment):
  python -m benchmarks.jira.zero_shot --model qwen25_7b
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.jira.sample_ids import DEFAULT_SAMPLE_IDS, read_sample_ids  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CONFIG_PATH = _REPO_ROOT / "configs" / "rev1_d1_local.yaml"
MAP5 = {"Blocker": "High", "Critical": "High", "Major": "Medium", "Minor": "Low", "Trivial": "Low"}

PROMPT = """You are triaging a new issue in a software project's issue tracker for the {component} component.
Classify the priority the issue should receive when it is filed.

Definitions:
- High: blocks development or testing work, causes crashes, data loss or a severe loss of function (Jira Blocker or Critical).
- Medium: a significant problem or request that should be handled in normal planning (Jira Major).
- Low: a minor loss of function, cosmetic problem or nice-to-have improvement (Jira Minor or Trivial).
{s2_note}
Issue type: {itype}
Summary: {summary}
Description:
{description}

Respond with exactly one word: {choices}."""


def macro_f1(y: np.ndarray, p: np.ndarray, k: int) -> tuple[float, list[float]]:
    f = []
    for c in range(k):
        tp = ((y == c) & (p == c)).sum()
        d = (y == c).sum() + (p == c).sum()
        f.append(float(2 * tp / d) if d else 0.0)
    return float(np.mean(f)), f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen25_7b")
    ap.add_argument("--processed", default="datasets/public/apache_jira/processed")
    ap.add_argument("--sample-ids", default=DEFAULT_SAMPLE_IDS)
    ap.add_argument("--d1", default="datasets/public/apache_jira/d1/d1_jira_qwen25_7b.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import yaml
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    mc = yaml.safe_load(CONFIG_PATH.read_text())["models"][args.model]
    proc = Path(args.processed)
    df = pd.read_parquet(proc / "issues.parquet")
    df["issue_id"] = df["issue_id"].astype(str)
    ids = read_sample_ids(proc, args.sample_ids)
    d1_ids = {json.loads(l)["issue_id"] for l in Path(args.d1).read_text().splitlines() if l.strip()}
    df = df.set_index("issue_id").loc[ids].reset_index()
    df = df[df["issue_id"].isin(d1_ids)]
    df["created"] = pd.to_datetime(df["created"], utc=True, errors="coerce")
    te = df[df["created"] >= "2023-01-01"].copy()
    te["label3"] = te["priority_creation"].map(MAP5)
    assert te["label3"].notna().all()
    logger.info("test rows %d (S2 %d)", len(te), int((te["priority_creation"] != "Major").sum()))

    llm = LLM(model=mc["hf_id"], dtype=mc["dtype"], max_model_len=mc["max_model_len"],
              gpu_memory_utilization=mc["gpu_memory_utilization"], seed=42)
    chat_kw = {"chat_template_kwargs": mc["chat_template_kwargs"]} if mc.get("chat_template_kwargs") else {}

    def txt(v) -> str:
        return v if isinstance(v, str) else ""          # parquet gives NaN for missing text

    def prompt(r, choices, s2_note=""):
        comp = txt(r["components"]).strip() or "unspecified"
        return PROMPT.format(component=comp, itype=txt(r["type"]) or "unspecified", summary=txt(r["summary"]),
                             description=txt(r["description"])[:3500], choices=", ".join(choices[:-1]) + " or " + choices[-1],
                             s2_note=s2_note)

    res, out_rows = {"model": args.model, "hf_id": mc["hf_id"]}, []
    for setting, labels, rows, note in (
            ("S1", ["High", "Medium", "Low"], te, ""),
            ("S2", ["High", "Low"], te[te["priority_creation"] != "Major"],
             "This issue is known not to have the default priority; decide between High and Low.\n")):
        sp = SamplingParams(temperature=0, max_tokens=4, seed=42, structured_outputs=StructuredOutputsParams(choice=labels))
        t0 = time.time()
        outs = llm.chat([[{"role": "user", "content": prompt(r, labels, note)}] for _, r in rows.iterrows()], sp,
                        use_tqdm=False, **chat_kw)
        pred = [o.outputs[0].text.strip() for o in outs]
        idx = {l: i for i, l in enumerate(labels)}
        y = rows["label3"].map(idx).to_numpy()
        p = np.array([idx.get(x, -1) for x in pred])
        mf, per = macro_f1(y, p, len(labels))
        res[setting] = {"n": len(rows), "macro_f1": mf, "per_class_f1": dict(zip(labels, per)),
                        "invalid": int((p < 0).sum()), "pred_share": {l: float((p == i).mean()) for l, i in idx.items()},
                        "elapsed_s": round(time.time() - t0, 1)}
        logger.info("%s macro-F1 %.4f per-class %s pred share %s", setting, mf, np.round(per, 3), res[setting]["pred_share"])
        out_rows += [{"setting": setting, "issue_id": i, "gold": g, "pred": x}
                     for i, g, x in zip(rows["issue_id"], rows["label3"], pred)]
    out_dir = Path(args.out) if args.out else proc / f"zero_shot_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    logger.info("saved %s", out_dir)


if __name__ == "__main__":
    main()
