"""E25: human-label re-scoring and cross-role sensitivity with the multi-run systems (E21, E22).

Version: v1 | Date: 2026-10-28

Replaces the single-label scoring of E1/E1c (seed-averaged probabilities, one co-reader draw,
LoRA run B only) by the estimand of the main table: every run is scored separately and macro-F1
is averaged over runs. Systems: LMEIC-explicit (five decision-layer seeds), LMEIC-full (five
co-reader seeds x five decision-layer seeds, E21), the LoRA classifier (five seeds, E22) and, on
the core test subset only, the five fp32 co-readers as fine-tuned XLM-R, D1-only XGBoost and
D2/D3-only XGBoost (for the rank correlation).

Outputs (under the E1 run directory, e25_multiseed_<timestamp>/):
  rescoring.csv    macro-F1 against the human majority and the agent consensus per subset
  paired.csv       thread-clustered paired bootstrap (2,000) on human labels, Holm within subset
  role_pairs.csv   every pair of role instances of the same email (cross-role emails, including the
                   31 whose other instance belongs to the core test part)
  summary.json     rank correlation, cross-role shares with email-clustered intervals, settings

Usage: python -m lmeic.human_study.crossrole_multiseed --e1-dir <E1 run dir> --lora-root <E22 output root>

Requires the confidential EMAIL-3R human-study files (items_with_labels.jsonl and the
annotation database); it is included to document the analysis and does not run without them.

"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.common.rev1_common import git_commit, norm_mid, write_json  # noqa: E402

LABELS = ["High", "Medium", "Low"]
LAB = {l: i for i, l in enumerate(LABELS)}
REF = _REPO_ROOT / "private/runs/rev1_reference_20261022_191130"
N_BOOT, BOOT_SEED = 2000, 20261028


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3).astype(float)
    tp = np.diag(cm); den = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.divide(2 * tp, den, out=np.zeros(3), where=den > 0)))


def runs_from_frame(df: pd.DataFrame, id_col: str, set_col: str | None) -> list[dict]:
    """One lookup {(set, id): label} per seed; labels from 'pred' or argmax of probability columns."""
    df = df.copy()
    df["_id"] = df[id_col].map(lambda x: norm_mid(str(x)))
    df["_set"] = df[set_col] if set_col else "test"
    if "pred" in df:
        df["_y"] = df["pred"].map(LAB)
    else:
        cols = [c for c in ("p_high", "p_med", "p_low") if c in df] or ["plm_p_high", "plm_p_med", "plm_p_low"]
        df["_y"] = df[cols].to_numpy().argmax(1)
    seeds = sorted(df["seed"].unique()) if "seed" in df else [None]
    out = []
    for s in seeds:
        g = df if s is None else df[df["seed"] == s]
        out.append(dict(zip(zip(g["_set"], g["_id"]), g["_y"].astype(int))))
    return out


def load_systems(lora_root: Path) -> dict[str, list[dict]]:
    sysruns: dict[str, list[dict]] = {}
    ext = sorted((REF / "R2_v3_grouped_explicit").glob("external_*/external_predictions_all_seeds.parquet"))
    sysruns["LMEIC_explicit"] = runs_from_frame(pd.read_parquet(ext[0]), "id", "set")
    full = []
    for d in sorted(REF.glob("E21_full_cr*")):
        f = sorted(d.glob("external_*/external_predictions_all_seeds.parquet"))
        assert f, f"no external predictions in {d}"
        full += runs_from_frame(pd.read_parquet(f[-1]), "id", "set")
    assert len(full) == 25, f"LMEIC-full runs: {len(full)}"
    sysruns["LMEIC_full"] = full
    lora = []
    for d in sorted(lora_root.glob("qwen25_7b_lora_role_s*")):
        test = runs_from_frame(pd.read_parquet(d / "test_probs.parquet").assign(seed=0), "message_id", None)[0]
        ex = runs_from_frame(pd.read_parquet(d / "external_predictions.parquet"), "id", "set")[0]
        lora.append({**test, **ex})
    assert len(lora) == 5, f"LoRA runs: {len(lora)}"
    sysruns["LoRA"] = lora
    xl = [runs_from_frame(pd.read_parquet(f).query("split == 'test'").assign(seed=0), "message_id", None)[0]
          for f in sorted((REF / "coreader_bank").glob("xlmr_base_fp32_s*/probs.parquet"))]
    assert len(xl) == 5
    sysruns["XLM-R_finetuned"] = xl
    for name, var in [("D1_only_XGB", "R5_B12_grouped"), ("D2D3_only_XGB", "R4_B11_v3_grouped")]:
        sysruns[name] = runs_from_frame(pd.read_parquet(REF / var / "test_predictions_all_seeds.parquet"), "message_id", None)
    return sysruns


def score(runs: list[dict], keys: list, y: np.ndarray) -> float | None:
    if not all(k in runs[0] for k in keys):
        return None
    return float(np.mean([macro_f1(y, np.array([r[k] for k in keys])) for r in runs]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-dir", required=True)
    ap.add_argument("--lora-root", required=True, help="E22 output root (five-seed LoRA runs)")
    args = ap.parse_args()
    e1 = Path(args.e1_dir)
    lora_root = Path(args.lora_root)
    it = pd.read_json(e1 / "items_with_labels.jsonl", lines=True)
    it["pred_id"] = it["pred_id"].map(lambda x: norm_mid(str(x)))
    S = load_systems(lora_root)

    d = it[it["human_majority"].isin(LABELS) & it["agent_label"].isin(LABELS)].reset_index(drop=True)
    yh, ya = d["human_majority"].map(LAB).to_numpy(), d["agent_label"].map(LAB).to_numpy()
    keys = list(zip(d["pred_set"], d["pred_id"]))
    subsets = {"all": np.ones(len(d), bool),
               "excluding_training_pool_threads": (d["eval_tier"] != "thread_in_training_pool").to_numpy(),
               **{s: (d["source"] == s).to_numpy() for s in ("core_test", "cross_role", "fresh_holdout")}}
    rows, paired = [], []
    rng = np.random.default_rng(BOOT_SEED)
    for sub, m in subsets.items():
        idx = np.where(m)[0]; k = [keys[i] for i in idx]
        for name, runs in S.items():
            h = score(runs, k, yh[idx])
            if h is not None:
                rows.append({"subset": sub, "system": name, "n": int(len(idx)), "n_runs": len(runs),
                             "macro_f1_vs_human": h, "macro_f1_vs_agent": score(runs, k, ya[idx])})
        th = d["thread_id"].to_numpy()[idx]; u = np.unique(th); mem = {t: np.where(th == t)[0] for t in u}
        boots = [np.concatenate([mem[t] for t in rng.choice(u, len(u))]) for _ in range(N_BOOT)]
        mats = {n: np.array([[r[kk] for kk in k] for r in S[n]]) for n in ("LMEIC_full", "LMEIC_explicit", "LoRA")}
        fam = []
        for a, b in [("LMEIC_full", "LoRA"), ("LMEIC_full", "LMEIC_explicit"), ("LoRA", "LMEIC_explicit")]:
            f = lambda P, bi: float(np.mean([macro_f1(yh[idx][bi], p[bi]) for p in P]))  # noqa: E731
            allidx = np.arange(len(idx))
            obs = f(mats[a], allidx) - f(mats[b], allidx)
            dd = np.array([f(mats[a], bi) - f(mats[b], bi) for bi in boots])
            fam.append({"subset": sub, "a": a, "b": b, "n": int(len(idx)), "delta": obs,
                        "ci95_low": float(np.percentile(dd, 2.5)), "ci95_high": float(np.percentile(dd, 97.5)),
                        "p_two_sided": float(min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean())))})
        p = np.array([r["p_two_sided"] for r in fam]); order = np.argsort(p); run = 0.0
        for j, i in enumerate(order):
            run = max(run, (len(p) - j) * p[i]); fam[i]["p_holm"] = min(1.0, run)
        paired += fam
    sc = pd.DataFrame(rows)
    core = sc[sc["subset"] == "core_test"]
    t = kendalltau(core["macro_f1_vs_human"], core["macro_f1_vs_agent"])
    tau = {"tau": float(t.statistic), "p": float(t.pvalue), "systems": core["system"].tolist()}

    # cross-role pairs: every pair of role instances of the same email among cross-role emails
    cr_mids = set(it.loc[it["source"] == "cross_role", "mid"])
    pool = it[it["mid"].isin(cr_mids) & it["human_majority"].isin(LABELS)]
    pairs = []
    for mid, g in pool.groupby("mid"):
        for (_, ra), (_, rb) in itertools.combinations(g.iterrows(), 2):
            ka, kb = (ra.pred_set, ra.pred_id), (rb.pred_set, rb.pred_id)
            row = {"mid": mid, "roles": f"{ra.role}|{rb.role}", "sources": f"{ra.source}|{rb.source}",
                   "tier": rb.eval_tier if ra.source == "core_test" else ra.eval_tier,
                   "human_differ": int(ra.human_majority != rb.human_majority),
                   "agent_differ": int(ra.agent_label != rb.agent_label) if ra.agent_label in LABELS and rb.agent_label in LABELS else np.nan}
            for name in ("LMEIC_explicit", "LMEIC_full", "LoRA"):
                runs = S[name]
                if ka in runs[0] and kb in runs[0]:
                    row[f"{name}_differ"] = float(np.mean([r[ka] != r[kb] for r in runs]))
                    row[f"{name}_both_correct"] = float(np.mean([(r[ka] == LAB[ra.human_majority]) and
                                                                  (r[kb] == LAB[rb.human_majority]) for r in runs]))
            pairs.append(row)
    pr = pd.DataFrame(pairs)

    def share(df, col):
        v = df[col].dropna().to_numpy()
        if not len(v):
            return None
        b = [rng.choice(v, len(v)).mean() for _ in range(N_BOOT)]
        return {"n": int(len(v)), "mean": float(v.mean()), "ci95": np.percentile(b, [2.5, 97.5]).round(4).tolist()}

    cols = ["human_differ", "agent_differ"] + [f"{n}_differ" for n in ("LMEIC_explicit", "LMEIC_full", "LoRA")]
    hd = pr[pr["human_differ"] == 1]
    cross = {"n_pairs": int(len(pr)), "n_emails": int(pr["mid"].nunique()),
             "all_pairs": {c: share(pr, c) for c in cols},
             "pairs_humans_differ": {c: share(hd, c) for c in cols[1:] + [f"{n}_both_correct" for n in ("LMEIC_explicit", "LMEIC_full", "LoRA")]},
             "note": "pairs are mostly one per email, so the bootstrap resamples pairs"}

    out = e1 / f"e25_multiseed_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir()
    sc.to_csv(out / "rescoring.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)
    pr.to_csv(out / "role_pairs.csv", index=False)
    write_json(out / "summary.json", {"e1_dir": str(e1), "lora_root": str(lora_root), "git_commit": git_commit(_REPO_ROOT),
                                      "estimand": "macro-F1 per run, averaged over runs",
                                      "n_runs": {k: len(v) for k, v in S.items()}, "kendall_tau_core": tau,
                                      "cross_role": cross, "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED}})
    print(sc.round(4).to_string(index=False)); print(pd.DataFrame(paired).round(4).to_string(index=False))
    print(json.dumps({"tau": tau, "cross_role": cross}, indent=1))


if __name__ == "__main__":
    main()
