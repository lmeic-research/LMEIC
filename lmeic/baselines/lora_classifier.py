#!/usr/bin/env python3
"""
E11 / E10: LoRA-tuned LLM sequence classifier on EMAIL-3R (GPU).
Version: v1.2 | Date: 2026-10-28 (v1.1: --external; v1.2: --seed overrides lora.seed, for E22)

  --input role   "Recipient role: <description>\\nSubject: ...\\n\\n<body>" (E11 direct classifier
                 baseline; role descriptions of the v1.0 direct-LLM baselines)
  --input text   latest_text only, as the XLM-R co-reader (E10 co-reader candidate)
  --mode full    one model on the full development pool scores the test split
  --mode oof     additionally thread-grouped out-of-fold scores for the development pool
                 (fold seed and folds as in coreader_bank), so the output can enter the stack
  --external     (full mode) the full-pool model also scores the revision-1 external sets
                 (rev1_reference.yaml: external_sets) -> external_predictions.parquet in the
                 eval_external layout (seed, set, id, pred, p_high, p_med, p_low)
  --set e12      (full mode) train on the E12 Enron dev split and score its test split, with the
                 Enron role descriptions (rev1_e12_enron.yaml: role_desc); ids are instance_id
Output: <out-root>/<name>_s<seed>/probs.parquet (message_id, split trainval/test, plm_p_*) and
stats.json, the co-reader bank layout read by coreader_bank_eval. Folds are resumable.

Usage:
  python -m lmeic.baselines.lora_classifier --out-root DIR \
      --input role --mode full
  python -m lmeic.baselines.lora_classifier --out-root DIR \
      --input role --set e12 --e12-run-dir <rev1_e12_enron run> --e12-labels <agent_labels_e12_enron.jsonl>
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.baselines.zero_shot_local import v1_prompt  # noqa: E402
from lmeic.reference.coreader import PROB_COLS  # noqa: E402
from lmeic.reference.folds import check_group_disjoint, make_folds  # noqa: E402
from lmeic.common.rev1_common import git_commit, load_config, write_json  # noqa: E402
from lmeic.common.data_loader import LABEL_MAP, load_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def inputs(df: pd.DataFrame, mode: str, role_desc: dict | None = None) -> list[str]:
    body = df["latest_text"].fillna("").str.slice(0, 4000)
    if mode == "text":
        return body.tolist()
    if role_desc is None:
        _, role_desc = v1_prompt()
    return [f"Recipient role: {role_desc.get(r, 'an office worker')}\nSubject: {s}\n\n{b}"
            for r, s, b in zip(df["role"], df["subject"].fillna(""), body)]


def build_model(lc: dict, tok):
    from peft import LoraConfig, TaskType, get_peft_model
    m = AutoModelForSequenceClassification.from_pretrained(lc["model_name"], num_labels=3, dtype=torch.bfloat16)
    m.config.pad_token_id = tok.pad_token_id
    if lc["gradient_checkpointing"]:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.enable_input_require_grads()
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=lc["lora_r"], lora_alpha=lc["lora_alpha"],
                     lora_dropout=lc["lora_dropout"], target_modules=lc["target_modules"])
    return get_peft_model(m, cfg).cuda()


def batches(enc: list[list[int]], idx: np.ndarray, bs: int, pad_id: int):
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        L = max(len(enc[j]) for j in b)
        ids = torch.full((len(b), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(b), L), dtype=torch.long)
        for k, j in enumerate(b):
            ids[k, :len(enc[j])] = torch.tensor(enc[j]); att[k, :len(enc[j])] = 1
        yield b, ids.cuda(), att.cuda()


def train(lc: dict, tok, enc, y: np.ndarray, seed: int):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(lc, tok)
    cls, cnt = np.unique(y, return_counts=True)
    w = torch.tensor([len(y) / (3 * cnt[list(cls).index(c)]) for c in range(3)], dtype=torch.float32, device="cuda")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lc["lr"], weight_decay=lc["weight_decay"])
    steps = math.ceil(len(y) / (lc["batch_size"] * lc["grad_accum"])) * lc["epochs"]
    sch = get_linear_schedule_with_warmup(opt, int(lc["warmup_ratio"] * steps), steps)
    yt = torch.tensor(y, dtype=torch.long, device="cuda")
    rng = np.random.default_rng(seed)
    model.train()
    t0, step = time.time(), 0
    for ep in range(lc["epochs"]):
        order = rng.permutation(len(y))
        for i, (b, ids, att) in enumerate(batches(enc, order, lc["batch_size"], tok.pad_token_id)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=att).logits.float()
            loss = F.cross_entropy(logits, yt[b], weight=w) / lc["grad_accum"]
            loss.backward()
            if (i + 1) % lc["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sch.step(); opt.zero_grad(); step += 1
                if step % 100 == 0:
                    logger.info("epoch %d step %d/%d loss %.4f (%.0fs)", ep + 1, step, steps,
                                loss.item() * lc["grad_accum"], time.time() - t0)
    return model


@torch.no_grad()
def predict(model, tok, enc, bs: int = 16) -> np.ndarray:
    model.eval()
    order = np.argsort([len(e) for e in enc])
    out = np.zeros((len(enc), 3), dtype=np.float32)
    for b, ids, att in batches(enc, order, bs, tok.pad_token_id):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[b] = F.softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1).cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--input", choices=["role", "text"], required=True)
    ap.add_argument("--mode", choices=["full", "oof"], default="full")
    ap.add_argument("--model-name", default=None, help="override lora.model_name (smoke tests)")
    ap.add_argument("--limit", type=int, default=None, help="subsample dev and test (smoke tests)")
    ap.add_argument("--external", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="override lora.seed")
    ap.add_argument("--set", default="email3r", choices=["email3r", "e12"])
    ap.add_argument("--e12-run-dir", default=None)
    ap.add_argument("--e12-labels", default=None)
    args = ap.parse_args()
    assert not (args.external and args.mode != "full"), "--external requires --mode full"
    assert args.set == "email3r" or (args.mode == "full" and not args.external
                                     and args.e12_run_dir and args.e12_labels), \
        "--set e12 needs --mode full, no --external, --e12-run-dir and --e12-labels"
    paths, cfg = load_config("rev1_e11_llm_baselines.yaml")
    _, rcfg = load_config("rev1_reference.yaml")
    lc = cfg["lora"]
    if args.model_name:
        lc["model_name"] = args.model_name
    if args.seed is not None:
        lc["seed"] = args.seed
    assert torch.cuda.is_available(), "CUDA required"
    name = f"qwen25_7b_lora_{args.input}" if not args.model_name else f"smoke_lora_{args.input}"
    if args.set == "e12":
        name += "_e12"
    cdir = Path(args.out_root) / f"{name}_s{lc['seed']}"
    cdir.mkdir(parents=True, exist_ok=True)

    role_desc, roles = None, ["Mgr", "CS", "Sch"]
    if args.set == "e12":
        from benchmarks.enron import data as E12
        _, ecfg = load_config("rev1_e12_enron.yaml")
        role_desc, roles = ecfg["role_desc"], E12.ROLES
        df = E12.load_labelled(Path(args.e12_run_dir), Path(args.e12_labels))
        df = df.rename(columns={"instance_id": "message_id"})   # id column name of the bank layout
        dev = df[df["split"] == "dev"].reset_index(drop=True)
        te = df[df["split"] == "test"].reset_index(drop=True)
    else:
        dev = pd.concat([load_split("train"), load_split("val")]).reset_index(drop=True)
        te = load_split("test")
    if args.limit:
        dev, te = dev.sample(args.limit, random_state=0).reset_index(drop=True), te.iloc[: args.limit // 4]
    tok = AutoTokenizer.from_pretrained(lc["model_name"])
    tok.padding_side = "right"
    enc_d = tok(inputs(dev, args.input, role_desc), truncation=True, max_length=lc["max_len"])["input_ids"]
    enc_t = tok(inputs(te, args.input, role_desc), truncation=True, max_length=lc["max_len"])["input_ids"]
    y, y_te = dev["label_final"].map(LABEL_MAP).values, te["label_final"].map(LABEL_MAP).values
    t0 = time.time()

    full_path = cdir / "test_probs_full.npy"
    ext_path = cdir / "external_predictions.parquet"
    if full_path.exists() and (ext_path.exists() or not args.external):
        p_te = np.load(full_path)
    else:
        m = train(lc, tok, enc_d, y, lc["seed"] + 100)
        p_te = predict(m, tok, enc_t)
        np.save(full_path, p_te)
        if args.external:
            frames = [pd.DataFrame({"set": "test", "id": te["message_id"].values})]
            probs = [p_te]
            for set_name, rel in rcfg["external_sets"]["sets"].items():
                ext = pd.read_json(Path(paths["repo_root"]) / rel, lines=True)
                enc_x = tok(inputs(ext, args.input), truncation=True, max_length=lc["max_len"])["input_ids"]
                probs.append(predict(m, tok, enc_x))
                frames.append(pd.DataFrame({"set": set_name, "id": ext["instance_id"].astype(str).values}))
                logger.info("scored external set %s (%d)", set_name, len(ext))
            ex = pd.concat(frames, ignore_index=True)
            ex[["p_high", "p_med", "p_low"]] = np.vstack(probs)
            ex.insert(0, "seed", lc["seed"])
            ex["pred"] = np.array(["High", "Medium", "Low"])[ex[["p_high", "p_med", "p_low"]].values.argmax(1)]
            ex[["seed", "set", "id", "pred", "p_high", "p_med", "p_low"]].to_parquet(ext_path, index=False)
        del m; torch.cuda.empty_cache()
    test_f1 = float(f1_score(y_te, p_te.argmax(1), average="macro"))
    logger.info("%s full-pool test macro-F1 %.4f (%.0fs)", name, test_f1, time.time() - t0)
    stats = {"candidate": name, "input": args.input, "mode": args.mode, "set": args.set, "seed": lc["seed"],
             "lora": lc, "n_dev": len(dev), "n_test": len(te),
             "e12_run_dir": args.e12_run_dir, "e12_labels": args.e12_labels,
             "test_macro_f1": test_f1, "test_macro_f1_by_role": {
                 r: float(f1_score(y_te[(te["role"] == r).values], p_te[(te["role"] == r).values].argmax(1),
                                   average="macro")) for r in roles if (te["role"] == r).any()},
             "gpu": torch.cuda.get_device_name(0), "hf_home": os.environ.get("HF_HOME"),
             "git_commit": git_commit(Path(paths["repo_root"]))}
    trows = pd.DataFrame(p_te, columns=PROB_COLS)
    trows.insert(0, "split", "test")
    trows.insert(0, "message_id", te["message_id"].values)

    if args.mode == "oof":
        g = dev["thread_id"].values
        folds = make_folds(y, g, "grouped_clean", lc["fold_seed"], rcfg["n_folds"])
        assert check_group_disjoint(folds, g) == 0
        oof = np.zeros((len(dev), 3), dtype=np.float32)
        fold_f1 = []
        for k, (a, b) in enumerate(folds):
            fp = cdir / f"oof_fold{k}.npy"
            if fp.exists():
                oof[b] = np.load(fp)
            else:
                m = train(lc, tok, [enc_d[i] for i in a], y[a], lc["seed"] + k)
                oof[b] = predict(m, tok, [enc_d[i] for i in b])
                np.save(fp, oof[b])
                del m; torch.cuda.empty_cache()
            fold_f1.append(float(f1_score(y[b], oof[b].argmax(1), average="macro")))
            logger.info("%s fold %d OOF macro-F1 %.4f (%.0fs)", name, k + 1, fold_f1[-1], time.time() - t0)
        rows = pd.DataFrame(oof, columns=PROB_COLS)
        rows.insert(0, "split", "trainval")
        rows.insert(0, "message_id", dev["message_id"].values)
        stats.update({"fold_seed": lc["fold_seed"], "fold_oof_macro_f1": fold_f1,
                      "oof_macro_f1": float(f1_score(y, oof.argmax(1), average="macro"))})
        pd.concat([rows, trows], ignore_index=True).to_parquet(cdir / "probs.parquet", index=False)
    else:
        trows.to_parquet(cdir / "test_probs.parquet", index=False)
    stats["elapsed_s"] = round(time.time() - t0, 1)
    write_json(cdir / "stats.json", stats)
    logger.info("done %s", stats)


if __name__ == "__main__":
    main()
