"""
Multi-perspective training on cross-role emails (E17).
Version: v1.1 | Date: 2026-10-23
v1.1 (after a one-seed smoke test, on structural grounds only): only 4 eligible emails
fall in the clean tier, because most clean threads lie in the test split and are
excluded; and 387 eligible emails have one role instance already in the development
pool, so the model has seen that instance. The primary pair subset is therefore
"unseen": both instances of the pair absent from the development pool. "all" is
kept as secondary; subsets with fewer than 10 reference-differing pairs are not
reported. Bootstrap filtering uses boolean masks.

EMAIL-3R v1.0 pairs every email with a single role, so no model ever sees the same
email labelled for two roles. On the revision-1 cross-role set, the reference system
and the role-conditioned LoRA classifier change their prediction for only about one
in ten role pairs whose reference labels differ. This experiment tests whether that
is a training-data limitation: real cross-role instances labelled by the revision-1
agent panel are added to the development pool in an email-grouped cross-fitting
design, and role sensitivity is measured on held-out emails.

Arms (LMEIC-explicit, R2 tuned parameters, five seeds):
  control  development pool only (stored R2 predictions, same features and seeds)
  aug_all  + every real role instance of the training-fold emails (multiple views)
  aug_one  + one randomly chosen real instance per training-fold email that has no
           instance in the development pool (more data, single view); separates
           "more training rows" from "several views of the same email"

Instances already in the development pool (same message and role) are never added
again. Emails whose message or thread occurs in the test split are excluded, so the
test split stays untouched. Outer folds are assigned by thread.

Usage:
  python -m lmeic.analysis.e17_multiview_training [--smoke]
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lmeic.reference import features as F  # noqa: E402
from lmeic.reference import models as M  # noqa: E402
from lmeic.common.rev1_common import (  # noqa: E402
    git_commit, load_config, new_output_dir, norm_mid, read_jsonl, write_json,
)
from lmeic.common.data_loader import LABEL_MAP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("e17")

P3 = ["p_high", "p_med", "p_low"]
IMPORTANCE = np.array([2.0, 1.0, 0.0])   # High, Medium, Low (LABEL_MAP order)
REF = "private/runs/rev1_reference_20261022_191130"
VARIANT = "R2_v3_grouped_explicit"
CONTROL_EXT = f"{REF}/{VARIANT}/external_20261022_211532/external_predictions_all_seeds.parquet"
N_OUTER = 5
FOLD_SEED = 20261023
N_BOOT = 2000
BOOT_SEED = 20261023
ARMS = ("aug_all", "aug_one")

CARD = {
    "experiment": "E17 multi-perspective training",
    "question": "Does adding several role views of the same email to training make the model "
                "change its prediction when the recipient role changes?",
    "answers": ["R1.2", "R3.5", "R3.3"],
    "model": f"LMEIC-explicit, {VARIANT} tuned parameters, stacking (XGB + LGBM + LR)",
    "arms": {"control": "development pool only",
             "aug_all": "+ all real role instances of training-fold cross-role emails",
             "aug_one": "+ one real instance per training-fold email not already in the pool"},
    "labels_for_added_rows": "revision-1 agent panel, latest record with three round-0 votes",
    "eligibility": "real (not counterfactual) cross-role instances with an agent label; message and "
                   "thread not in the test split; email has >= 2 such instances",
    "outer_folds": f"{N_OUTER}, by thread, seed {FOLD_SEED}",
    "primary": "role sensitivity on held-out emails, pairs whose two instances are both absent from "
               "the development pool ('unseen'): share of pairs with differing reference labels "
               "whose predictions differ; expected-importance order agreement on those pairs",
    "secondary": ["false sensitivity (pairs with equal reference labels)",
                  "macro-F1 on held-out cross-role instances", "test-split macro-F1 (no regression)",
                  "clean tier separately"],
    "statistics": f"email-clustered bootstrap for pair metrics, thread-clustered for F1, "
                  f"{N_BOOT} resamples, seed {BOOT_SEED}; differences aug_all-control, "
                  f"aug_one-control, aug_all-aug_one",
    "known_confounds": ["added rows use revision-1 GPT-4o descriptors and the revision-1 agent "
                        "panel; the development pool uses v1.0 descriptors and labels",
                        "aug_one adds fewer rows than aug_all"],
    "acceptance": "descriptive; arms and metrics fixed before running, nothing tuned on test",
}


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3).astype(float)
    tp = np.diag(cm)
    prec, rec = tp / np.maximum(cm.sum(0), 1), tp / np.maximum(cm.sum(1), 1)
    return float(np.mean(2 * prec * rec / np.maximum(prec + rec, 1e-12)))


def pair_values(pc: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> dict[str, np.ndarray]:
    pred = pc.argmax(1)
    score = pc @ IMPORTANCE
    return {"differ": (pred[ia] != pred[ib]).astype(float), "score_gap": score[ia] - score[ib]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="one seed, one fold, aug_all only; output to tmp/")
    args = ap.parse_args()
    assert LABEL_MAP == {"High": 0, "Medium": 1, "Low": 2}, LABEL_MAP
    paths, cfg = load_config("rev1_reference.yaml")
    repo = Path(paths["repo_root"])
    seeds = cfg["seeds"][:1] if args.smoke else list(cfg["seeds"])
    arms = ARMS[:1] if args.smoke else ARMS
    folds = [0] if args.smoke else list(range(N_OUTER))
    if args.smoke:
        out = repo / "tmp" / f"e17_smoke_{time.strftime('%Y%m%d_%H%M%S')}"
        out.mkdir(parents=True)
    else:
        out = new_output_dir(paths, "rev1_e17_multiview")
    write_json(out / "experiment_card.json", {**CARD, "smoke_test": args.smoke, "seeds": seeds})

    # development pool and test split, exactly as the reference run
    v = next(x for x in cfg["variants"] if x["name"] == VARIANT)
    groups = set(v["groups"])
    params = json.loads((repo / REF / VARIANT / "params.json").read_text())
    feats, y, meta = F.build(cfg, repo, v["d2d3"], groups, None)
    X, yy, g, Xt = F.pool(feats, y, meta)
    cols = list(feats["train"].columns)
    dev_meta = pd.concat([meta["train"], meta["val"]])
    dev_keys = set(zip(dev_meta["message_id"].map(norm_mid), dev_meta["role"]))
    test_mids = meta["test"]["message_id"].map(norm_mid).to_numpy()
    y_te = np.asarray(y["test"], dtype=int)

    # cross-role instances with the same feature pipeline as score_external
    es = cfg["external_sets"]
    ext = pd.DataFrame(read_jsonl(repo / es["sets"]["cross_role"]))
    d1 = pd.DataFrame(read_jsonl(repo / es["d1_dir"] / "d1_cross_role.jsonl"))
    dd = pd.read_parquet(repo / es["d2d3_dir"] / "d2d3_v3_cross_role.parquet")
    Xc = np.nan_to_num(F.build_external(cfg, repo, groups, cols, ext, d1, dd, None).values.astype(np.float32))
    ext["mid"] = ext["message_id"].map(norm_mid)
    ext["iid"] = ext["instance_id"].astype(str)
    latest = {}
    for r in read_jsonl(repo / es["agent_dir"] / "agent_labels_cross_role.jsonl"):
        latest[str(r["id"])] = r
    lab = {i: LABEL_MAP[r["label_final"]] for i, r in latest.items() if r.get("n_votes_round0") == 3}
    ext["y"] = ext["iid"].map(lab)
    ext["in_dev"] = [(m, r) in dev_keys for m, r in zip(ext["mid"], ext["role"])]

    real = ~ext["is_counterfactual"].astype(bool)
    keep = (real & ext["y"].notna() & ~ext["mid"].isin(set(test_mids))
            & ~ext["thread_id"].isin(set(meta["test"]["thread_id"])))
    E = ext[keep].copy()
    E = E[E["mid"].map(E.groupby("mid").size()) >= 2]
    E["y"] = E["y"].astype(int)
    threads = np.array(sorted(E["thread_id"].unique()))
    np.random.default_rng(FOLD_SEED).shuffle(threads)
    E["fold"] = E["thread_id"].map({t: i % N_OUTER for i, t in enumerate(threads)})
    rng = np.random.default_rng(FOLD_SEED + 1)
    one_view = set()
    for _, grp in E.groupby("mid"):
        if not grp["in_dev"].any():
            one_view.add(int(grp.index[rng.integers(len(grp))]))
    counts = {"cross_role_instances": int(len(ext)), "real": int(real.sum()),
              "eligible_instances": int(len(E)), "eligible_emails": int(E["mid"].nunique()),
              "eligible_already_in_dev": int(E["in_dev"].sum()),
              "by_tier": E.groupby("eval_tier")["mid"].nunique().to_dict(),
              "one_view_candidates": len(one_view)}
    logger.info("counts: %s", counts)

    # train arms
    preds, fold_info = [], []
    for arm in arms:
        for k in folds:
            pool = E[(E["fold"] != k) & ~E["in_dev"]]
            if arm == "aug_one":
                pool = pool[pool.index.isin(one_view)]
            add = pool.index.to_numpy()
            ev = E[E["fold"] == k]
            X_tr = np.vstack([X, Xc[add]])
            y_tr = np.concatenate([yy, ext.loc[add, "y"].to_numpy(int)])
            g_tr = np.concatenate([g, ext.loc[add, "thread_id"].to_numpy()])
            X_ev = np.vstack([Xt, Xc[ev.index.to_numpy()]])
            t0 = time.time()
            res = M.stacking(X_tr, y_tr, g_tr, X_ev, params, v["protocol"], seeds, cfg["n_folds"],
                             cfg["inner_es_fraction"])
            ids = list(test_mids) + ev["iid"].tolist()
            sets = ["test"] * len(test_mids) + ["cross_role"] * len(ev)
            for s in seeds:
                pr = res[s]["proba"]
                preds.append(pd.DataFrame({"arm": arm, "fold": k, "seed": s, "set": sets, "id": ids,
                                           "p_high": pr[:, 0], "p_med": pr[:, 1], "p_low": pr[:, 2]}))
            fold_info.append({"arm": arm, "fold": k, "n_added": int(len(add)),
                              "n_added_emails": int(pool["mid"].nunique()), "n_eval": int(len(ev)),
                              "elapsed_s": round(time.time() - t0, 1)})
            logger.info("%s fold %d: +%d rows, eval %d, %.0fs", arm, k, len(add), len(ev), time.time() - t0)
            pd.concat(preds).to_parquet(out / "predictions.parquet", index=False)
    P = pd.concat(preds)

    # aligned probabilities per arm and seed
    E_eval = E[E["fold"].isin(folds)].reset_index(drop=True)
    eval_ids = E_eval["iid"].tolist()
    C = pd.read_parquet(repo / CONTROL_EXT)
    C = C[C["set"] == "cross_role"].assign(id=lambda d: d["id"].astype(str))
    stored = pd.read_parquet(repo / REF / VARIANT / "test_predictions_all_seeds.parquet")
    stored["id"] = stored["message_id"].map(norm_mid)
    probs: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for arm in ("control",) + tuple(arms):
        probs[arm] = {}
        for s in seeds:
            if arm == "control":
                pc = C[C["seed"] == s].drop_duplicates("id").set_index("id").loc[eval_ids, P3].to_numpy()
                pt = stored[stored["seed"] == s].drop_duplicates("id").set_index("id").loc[test_mids, P3].to_numpy()
            else:
                d = P[(P["arm"] == arm) & (P["seed"] == s)]
                pc = d[d["set"] == "cross_role"].drop_duplicates("id").set_index("id").loc[eval_ids, P3].to_numpy()
                pt = d[d["set"] == "test"].groupby("id")[P3].mean().loc[test_mids].to_numpy()
            probs[arm][s] = (pc, pt)

    # role pairs on held-out emails
    ia, ib, email_of = [], [], []
    for mid, grp in E_eval.groupby("mid"):
        for a, b in itertools.combinations(grp.index.to_numpy(), 2):
            ia.append(a); ib.append(b); email_of.append(mid)
    ia, ib, email_of = np.array(ia), np.array(ib), np.array(email_of)
    ya, yb = E_eval["y"].to_numpy()[ia], E_eval["y"].to_numpy()[ib]
    ref_differ = ya != yb
    ref_gap = IMPORTANCE[ya] - IMPORTANCE[yb]
    clean = E_eval["eval_tier"].to_numpy()[ia] == "clean"
    in_dev_ev = E_eval["in_dev"].to_numpy(bool)
    unseen = ~in_dev_ev[ia] & ~in_dev_ev[ib]
    y_ev = E_eval["y"].to_numpy()
    th_ev = E_eval["thread_id"].to_numpy()

    pv = {arm: {s: pair_values(probs[arm][s][0], ia, ib) for s in seeds} for arm in probs}

    def pair_stat(arm: str, idx: np.ndarray, metric: str) -> float:
        idx = np.asarray(idx, dtype=int)
        vals = []
        for s in seeds:
            d = pv[arm][s]
            if metric == "sensitivity":
                m = idx[ref_differ[idx]]
                vals.append(d["differ"][m].mean() if len(m) else np.nan)
            elif metric == "order_agreement":
                m = idx[ref_differ[idx]]
                vals.append((np.sign(d["score_gap"][m]) == np.sign(ref_gap[m])).mean() if len(m) else np.nan)
            else:  # false_sensitivity
                m = idx[~ref_differ[idx]]
                vals.append(d["differ"][m].mean() if len(m) else np.nan)
        return float(np.nanmean(vals))

    def f1_stat(arm: str, idx: np.ndarray) -> float:
        return float(np.mean([macro_f1(y_ev[idx], probs[arm][s][0][idx].argmax(1)) for s in seeds]))

    def test_f1(arm: str, idx: np.ndarray) -> float:
        return float(np.mean([macro_f1(y_te[idx], probs[arm][s][1][idx].argmax(1)) for s in seeds]))

    rng_b = np.random.default_rng(BOOT_SEED)
    emails = np.unique(email_of)
    by_email = {e: np.where(email_of == e)[0] for e in emails}
    pair_boots = [np.concatenate([by_email[e] for e in rng_b.choice(emails, len(emails))]) for _ in range(N_BOOT)]
    ths = np.unique(th_ev)
    by_th = {t: np.where(th_ev == t)[0] for t in ths}
    inst_boots = [np.concatenate([by_th[t] for t in rng_b.choice(ths, len(ths))]) for _ in range(N_BOOT)]
    te_threads = meta["test"]["thread_id"].to_numpy()
    tth = np.unique(te_threads)
    by_tt = {t: np.where(te_threads == t)[0] for t in tth}
    test_boots = [np.concatenate([by_tt[t] for t in rng_b.choice(tth, len(tth))]) for _ in range(N_BOOT)]

    all_pairs = np.arange(len(ia))
    subsets = {name: idx for name, idx in (("unseen", all_pairs[unseen]), ("all", all_pairs),
                                           ("clean", all_pairs[clean]))
               if ref_differ[idx].sum() >= 10}
    logger.info("pair subsets: %s", {k: (int(len(v)), int(ref_differ[v].sum())) for k, v in subsets.items()})
    arm_rows = []
    for arm in probs:
        r = {"arm": arm,
             "cross_role_macro_f1": f1_stat(arm, np.arange(len(y_ev))),
             "test_macro_f1": test_f1(arm, np.arange(len(y_te))),
             "test_macro_f1_by_seed": [macro_f1(y_te, probs[arm][s][1].argmax(1)) for s in seeds]}
        for name, idx in subsets.items():
            for metric in ("sensitivity", "order_agreement", "false_sensitivity"):
                r[f"{metric}_{name}"] = pair_stat(arm, idx, metric)
        arm_rows.append(r)
        logger.info("%s: %s", arm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    comps = []
    pairs_to_compare = [(a, b) for a, b in (("aug_all", "control"), ("aug_one", "control"), ("aug_all", "aug_one"))
                        if a in probs and b in probs]
    for a, b in pairs_to_compare:
        for name, idx in subsets.items():
            for metric in ("sensitivity", "order_agreement", "false_sensitivity"):
                obs = pair_stat(a, idx, metric) - pair_stat(b, idx, metric)
                mask = np.zeros(len(ia), bool)
                mask[idx] = True
                d = np.array([pair_stat(a, bt[mask[bt]], metric) - pair_stat(b, bt[mask[bt]], metric)
                              for bt in pair_boots])
                comps.append({"a": a, "b": b, "metric": f"{metric}_{name}", "delta": obs,
                              "ci95_low": float(np.nanpercentile(d, 2.5)), "ci95_high": float(np.nanpercentile(d, 97.5)),
                              "p_two_sided": float(min(1.0, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0))))})
        for metric, fn, boots, n in (("cross_role_macro_f1", f1_stat, inst_boots, len(y_ev)),
                                     ("test_macro_f1", test_f1, test_boots, len(y_te))):
            obs = fn(a, np.arange(n)) - fn(b, np.arange(n))
            d = np.array([fn(a, bt) - fn(b, bt) for bt in boots])
            comps.append({"a": a, "b": b, "metric": metric, "delta": obs,
                          "ci95_low": float(np.percentile(d, 2.5)), "ci95_high": float(np.percentile(d, 97.5)),
                          "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))})

    pd.DataFrame(arm_rows).to_csv(out / "arm_metrics.csv", index=False)
    pd.DataFrame(comps).to_csv(out / "comparisons.csv", index=False)
    write_json(out / "summary.json", {
        "counts": counts, "folds_run": folds, "fold_info": fold_info,
        "n_eval_instances": int(len(E_eval)), "n_pairs": int(len(ia)),
        "n_pairs_ref_differ": int(ref_differ.sum()), "n_pairs_clean": int(clean.sum()),
        "n_pairs_clean_ref_differ": int((clean & ref_differ).sum()),
        "n_pairs_unseen": int(unseen.sum()), "n_pairs_unseen_ref_differ": int((unseen & ref_differ).sum()),
        "subsets_reported": {k: [int(len(v)), int(ref_differ[v].sum())] for k, v in subsets.items()},
        "arms": arm_rows, "git_commit": git_commit(repo)})
    logger.info("output: %s", out)
    print(pd.DataFrame(arm_rows).drop(columns=["test_macro_f1_by_seed"]).round(4).to_string(index=False))
    print(pd.DataFrame(comps).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
