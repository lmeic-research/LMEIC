"""
Three-agent labelling with debate and adjudication for revision-1 evaluation sets.
Version: v1 | Date: 2026-10-22

Flow per instance (identical to the v1.0 pipeline): three agents label independently;
a 2/3 majority is accepted; a 1-1-1 split triggers one debate round; a remaining split
is adjudicated. Prompt builders, JSON recovery and the consensus rule are imported from
the v1.0 annotation code. Every vote records the requested model, the returned model
string and, where available, the serving provider. Resumable.

Usage:
  python -m lmeic.labelling.agent_label --set core_test [--out-dir DIR] [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.common.rev1_common import (  # noqa: E402
    git_commit, load_config, new_output_dir, norm_mid, read_jsonl, source_run_dir, write_json,
)
from lmeic.labelling.system_prompts import (  # noqa: E402
    build_adjudicator_prompt, build_agent_system_prompt, build_debate_prompt, build_email_input,
)

from lmeic.labelling.consensus import check_consensus, extract_json_from_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
CONFIG_NAME = "rev1_agent_labelling.yaml"
VALID = {"high": "High", "medium": "Medium", "low": "Low"}


def load_set(name: str, cfg: dict, paths: dict) -> list[tuple[str, dict]]:
    sc = cfg["sets"][name]
    repo = Path(paths["repo_root"])
    if sc["kind"] == "jsonl":
        rows = list(read_jsonl(repo / sc["path"]))
        return [(str(r[sc["id_field"]]), r) for r in rows]
    if sc["kind"] == "sampled_by_key_source":
        want = [norm_mid(r["message_id"]) for r in read_jsonl(repo / cfg["human_package_key"])
                if r["source"] == sc["key_source"]]
        run_dir = source_run_dir(paths, cfg["source_run_id"])
        by_mid = {norm_mid(r["message_id"]): r for r in read_jsonl(run_dir / sc["relpath"])}
        assert all(m in by_mid for m in want), "core_test ids missing from sampled_all"
        return [(m, by_mid[m]) for m in dict.fromkeys(want)]
    raise ValueError(sc["kind"])


class Caller:
    def __init__(self, cfg: dict):
        from openai import OpenAI
        self.req = cfg["request"]
        self.clients = {k: OpenAI(api_key=os.environ[v["key_env"]], base_url=os.environ[v["base_url_env"]],
                                  timeout=self.req["request_timeout_s"], max_retries=0)
                        for k, v in cfg["endpoints"].items()}
        per = self.req.get("max_concurrent_per_model", {})
        self.sems: dict[str, asyncio.Semaphore] = {}
        self.per = per

    def _sem(self, model: str) -> asyncio.Semaphore:
        if model not in self.sems:
            self.sems[model] = asyncio.Semaphore(self.per.get(model, self.req["max_concurrent"]))
        return self.sems[model]

    def _kwargs(self, spec: dict, messages: list[dict]) -> dict:
        kw = {"model": spec["model"], "messages": messages, "temperature": self.req["temperature"],
              "max_tokens": self.req["max_tokens"], "timeout": self.req["request_timeout_s"]}
        if spec.get("json_mode"):
            kw["response_format"] = {"type": "json_object"}
        if spec.get("provider_order"):
            kw["extra_body"] = {"provider": {"order": spec["provider_order"], "allow_fallbacks": False}}
        return kw

    async def __call__(self, spec: dict, messages: list[dict]) -> dict | None:
        n = self.req["max_retries"]
        for attempt in range(n):
            use = spec["fallback"] if spec.get("fallback") and attempt >= n // 2 else spec
            kw = self._kwargs({**spec, **use}, messages)
            client = self.clients[use["endpoint"]]
            try:
                async with self._sem(use["model"]):
                    resp = await asyncio.wait_for(asyncio.to_thread(client.chat.completions.create, **kw),
                                                  timeout=self.req["request_timeout_s"] + 15)
                parsed = extract_json_from_text(resp.choices[0].message.content or "")
                label = VALID.get(str((parsed or {}).get("label", "")).strip().lower())
                if label is None:
                    logger.warning("unparseable output from %s (attempt %d)", spec["model"], attempt + 1)
                    continue
                parsed["label"] = label
                parsed["_model_returned"] = getattr(resp, "model", None)
                parsed["_provider"] = getattr(resp, "provider", None) or use["endpoint"]
                return parsed
            except Exception as e:  # network / API errors are retried
                rate = "RateLimit" in type(e).__name__
                wait = min(60.0, 4.0 * 2 ** attempt) if rate else 2.0 * (attempt + 1)
                logger.warning("%s from %s@%s (attempt %d)", type(e).__name__, use["model"], use["endpoint"],
                               attempt + 1)
                await asyncio.sleep(wait + random.uniform(0, 1))
        return None


def _vote(aid: str, spec: dict, parsed: dict, rnd: int) -> dict:
    try:
        conf = int(parsed.get("confidence", 3))
    except (TypeError, ValueError):
        conf = 3
    v = {"agent": aid, "model": spec["model"], "model_returned": parsed.get("_model_returned"),
         "provider": parsed.get("_provider"), "label": parsed["label"], "confidence": conf,
         "rationale": parsed.get("rationale", ""), "round": rnd}
    if rnd > 0:
        v["changed"] = parsed.get("changed", False)
    return v


async def annotate(call: Caller, cfg: dict, iid: str, rec: dict) -> dict:
    t0 = time.time()
    role = rec.get("role", "Mgr")
    email = build_email_input(rec)
    agents = cfg["agents"]

    async def r0(aid):
        p = await call(agents[aid], [{"role": "system", "content": build_agent_system_prompt(aid, role)},
                                     {"role": "user", "content": email}])
        return _vote(aid, agents[aid], p, 0) if p else None

    votes0 = [v for v in await asyncio.gather(*[r0(a) for a in agents]) if v]
    if len(votes0) < len(agents):
        return {"id": iid, "message_id": rec.get("message_id"), "role": role, "label_final": None,
                "status": "incomplete_panel", "n_votes_round0": len(votes0)}
    final, status = check_consensus(votes0)
    all_votes, debate_rounds, adj = list(votes0), 0, None

    if final is None:
        debate_rounds = 1

        async def r1(aid):
            mine = next(v for v in votes0 if v["agent"] == aid)
            others = [v for v in votes0 if v["agent"] != aid]
            msgs = [{"role": "system", "content": build_agent_system_prompt(aid, role)},
                    {"role": "user", "content": email},
                    {"role": "assistant", "content": json.dumps({k: mine[k] for k in ("label", "confidence", "rationale")})},
                    {"role": "user", "content": build_debate_prompt(mine["label"], mine["rationale"], others)}]
            p = await call(agents[aid], msgs)
            return _vote(aid, agents[aid], p, 1) if p else mine

        votes1 = list(await asyncio.gather(*[r1(a) for a in agents]))
        all_votes += votes1
        final, status = check_consensus(votes1)
        if final is None:
            spec = cfg["adjudicator"]
            p = await call(spec, [{"role": "system", "content": build_adjudicator_prompt(role, votes1)},
                                  {"role": "user", "content": email}])
            if p:
                final, adj, status = p["label"], {"rationale": p.get("rationale", ""),
                                                  "model_returned": p.get("_model_returned")}, "adjudicated"
            else:
                final, status = Counter(v["label"] for v in votes1).most_common(1)[0][0], "fallback_majority"

    return {"id": iid, "message_id": rec.get("message_id"), "role": role, "label_final": final,
            "status": status, "n_votes_round0": len(votes0), "agent_votes": all_votes,
            "debate_rounds": debate_rounds, "adjudication": adj,
            "timestamp": datetime.now(timezone.utc).isoformat(), "elapsed_s": round(time.time() - t0, 2)}


async def run(set_name: str, out_dir: Path, limit: int | None) -> None:
    paths, cfg = load_config(CONFIG_NAME)
    load_dotenv(Path(paths["repo_root"]) / cfg["env_file"])
    items = load_set(set_name, cfg, paths)
    if limit:
        items = items[:limit]
    out_path = out_dir / f"agent_labels_{set_name}.jsonl"
    n_agents = len(cfg["agents"])
    done = set()
    if out_path.exists():
        # an instance counts as done only with a full round-0 panel; later records supersede earlier ones
        latest = {}
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            latest[r["id"]] = r
        done = {i for i, r in latest.items() if r.get("n_votes_round0", 0) == n_agents}
    todo = [(i, r) for i, r in items if i not in done]
    logger.info("set=%s total=%d done=%d todo=%d -> %s", set_name, len(items), len(done), len(todo), out_path)

    call = Caller(cfg)
    t0, n, status = time.time(), 0, Counter()
    tasks = [asyncio.create_task(annotate(call, cfg, i, r)) for i, r in todo]
    with open(out_path, "a", encoding="utf-8") as fout:
        for fut in asyncio.as_completed(tasks):
            res = await fut
            n += 1
            status[res["status"]] += 1
            if res["label_final"] is not None:
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()
            if n % 100 == 0:
                logger.info("progress %d/%d %s (%.0fs)", n, len(todo), dict(status), time.time() - t0)

    write_json(out_dir / f"agent_labels_{set_name}_stats_{time.strftime('%Y%m%d_%H%M%S')}.json", {
        "set": set_name, "n_items": len(items), "previously_done": len(done), "this_run": n,
        "status": dict(status), "agents": cfg["agents"], "adjudicator": cfg["adjudicator"],
        "request": cfg["request"], "elapsed_s": round(time.time() - t0, 1),
        "git_commit": git_commit(Path(paths["repo_root"]))})
    logger.info("done %s", dict(status))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    paths, cfg = load_config(CONFIG_NAME)
    out_dir = Path(args.out_dir) if args.out_dir else new_output_dir(paths, cfg["output_prefix"])
    out_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args.set, out_dir, args.limit))


if __name__ == "__main__":
    main()
