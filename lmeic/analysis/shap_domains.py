"""Domain-level SHAP attribution recomputed on the revision-1 explicit model (R2).

v1.0 (experiments/analysis/shap_analysis.py) used the v2 D2 features and assigned every column
starting with "role_" to the role-conditional group, which put the D1 descriptor role_relevance
there (the same prefix issue as D050). This rerun:
  - uses the R2 feature set (D2/D3 v3, pre-timestamp) and the stored R2 XGBoost parameters
  - fits XGBoost on train+val (the XGB base learner of the R2 stack, seed 42)
  - assigns role_relevance to D1; role-conditional = role_Mgr_/role_CS_/role_Sch_ prefixes
TreeSHAP on the 2,489 test emails; mean |SHAP| over classes, aggregated by domain and role.

Usage: python -m lmeic.analysis.shap_domains
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from lmeic.reference import features as F
from lmeic.reference import models as M
from lmeic.common.rev1_common import git_commit, load_config, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("shap_v3")
_REPO_ROOT = Path(__file__).resolve().parents[2]
REF = _REPO_ROOT / "private/runs/rev1_reference_20261022_191130"


def domain_of(c: str) -> str:
    if c.startswith("inter_"):
        return "interactions"
    if c.startswith(F.ROLE_COND_PREFIXES):
        return "role_cond"
    if c.startswith(("word_svd", "char_svd")):
        return "text"
    if c.startswith("d2_"):
        return "D2"
    if c.startswith("d3_"):
        return "D3"
    return "D1"


def main() -> None:
    paths, rcfg = load_config("rev1_reference.yaml")
    v = next(x for x in rcfg["variants"] if x["name"] == "R2_v3_grouped_explicit")
    params = json.loads((REF / "R2_v3_grouped_explicit" / "params.json").read_text())
    feats, y, meta = F.build(rcfg, Path(paths["repo_root"]), v["d2d3"], set(v["groups"]))
    X, yy, g, Xt = F.pool(feats, y, meta)
    cols = list(feats["train"].columns)
    model = M._xgb(params["xgb"], 42, prob=True).fit(X, yy, sample_weight=M.sample_weights(yy))
    t0 = time.time()
    sv = shap.TreeExplainer(model).shap_values(Xt)
    absm = np.mean([np.abs(s) for s in sv], axis=0) if isinstance(sv, list) else np.abs(sv).mean(axis=2)
    logger.info("SHAP done (%.0fs)", time.time() - t0)
    imp = pd.DataFrame({"feature": cols, "mean_abs_shap": absm.mean(0), "domain": [domain_of(c) for c in cols]})
    dom = imp.groupby("domain")["mean_abs_shap"].sum()
    roles = meta["test"]["role"].to_numpy()
    per_role = {}
    for r in ("Mgr", "CS", "Sch"):
        d = pd.Series(absm[roles == r].mean(0), index=cols).groupby([domain_of(c) for c in cols]).sum()
        per_role[r] = (d / d.sum() * 100).round(2).to_dict()
    out = REF / f"shap_v3_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir()
    imp.sort_values("mean_abs_shap", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    res = {"model": "XGB base learner of R2 (stored params, seed 42, train+val)", "n_test": int(len(Xt)),
           "n_features": len(cols), "domain_attribution_pct": (dom / dom.sum() * 100).round(2).to_dict(),
           "n_features_by_domain": imp["domain"].value_counts().to_dict(), "per_role_domain_attribution_pct": per_role,
           "top20": imp.sort_values("mean_abs_shap", ascending=False).head(20)[["feature", "domain", "mean_abs_shap"]]
           .round(4).to_dict("records"), "v1_reference": "v1_experiments/analysis/shap_20260530_093412",
           "git_commit": git_commit(_REPO_ROOT)}
    write_json(out / "summary.json", res)
    print(json.dumps({k: res[k] for k in ("domain_attribution_pct", "n_features_by_domain", "per_role_domain_attribution_pct")}, indent=1))
    print(pd.DataFrame(res["top20"]).to_string(index=False))
    print("saved", out)


if __name__ == "__main__":
    main()
