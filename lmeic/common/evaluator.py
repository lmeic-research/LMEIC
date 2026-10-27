"""
Unified evaluator for Phase 4 experiments.
Version: 1.0 | 2026-05-28

Computes all metrics defined in the baseline protocol and outputs
standardised metrics.json.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)

LABEL_NAMES = ["High", "Medium", "Low"]
LABEL_MAP = {"High": 0, "Medium": 1, "Low": 2}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    roles: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels (0=High, 1=Medium, 2=Low)
        y_pred: Predicted labels
        roles: Optional role array for per-role breakdown

    Returns:
        Dictionary matching metrics.json schema
    """
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    acc = float(accuracy_score(y_true, y_pred))

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    per_class = {}
    for i, name in enumerate(LABEL_NAMES):
        per_class[name] = {
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f[i]), 4),
            "support": int(s[i]),
        }

    per_role = {}
    if roles is not None:
        for role in ["Mgr", "CS", "Sch"]:
            mask = roles == role
            if mask.sum() > 0:
                per_role[role] = {
                    "macro_f1": round(
                        float(f1_score(y_true[mask], y_pred[mask], average="macro")),
                        4,
                    ),
                    "support": int(mask.sum()),
                }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    report_text = classification_report(
        y_true, y_pred, target_names=LABEL_NAMES, digits=4, zero_division=0
    )

    return {
        "overall": {
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "accuracy": round(acc, 4),
        },
        "per_class": per_class,
        "per_role": per_role,
        "confusion_matrix": cm,
        "classification_report_text": report_text,
    }


def aggregate_seed_metrics(
    seed_metrics: List[Dict],
) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple seeds.
    Returns mean +/- std for numeric fields.
    """
    if len(seed_metrics) == 1:
        m = seed_metrics[0]
        result = {
            "overall": {
                k: {"mean": v, "std": None, "seeds": [v]}
                for k, v in m["overall"].items()
            },
            "per_class": m["per_class"],
            "per_role": m["per_role"],
            "confusion_matrix": m["confusion_matrix"],
            "classification_report_text": m["classification_report_text"],
        }
        return result

    overall_keys = ["macro_f1", "weighted_f1", "accuracy"]
    overall_agg = {}
    for k in overall_keys:
        vals = [m["overall"][k] for m in seed_metrics]
        overall_agg[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "seeds": [round(v, 4) for v in vals],
        }

    per_role_agg = {}
    for role in ["Mgr", "CS", "Sch"]:
        vals = [m["per_role"].get(role, {}).get("macro_f1", 0) for m in seed_metrics]
        per_role_agg[role] = {
            "macro_f1": {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)},
            "support": seed_metrics[0]["per_role"].get(role, {}).get("support", 0),
        }

    return {
        "overall": overall_agg,
        "per_class": seed_metrics[0]["per_class"],
        "per_role": per_role_agg,
        "confusion_matrix": seed_metrics[0]["confusion_matrix"],
        "classification_report_text": seed_metrics[0]["classification_report_text"],
    }


def save_results(
    output_dir: Path,
    baseline_id: str,
    baseline_name: str,
    config: Dict,
    metrics: Dict,
    predictions: Optional[List[Dict]] = None,
):
    """Save complete results to standardised directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"baseline_id": baseline_id, "baseline_name": baseline_name, **metrics}, f, indent=2)

    if predictions:
        with open(output_dir / "predictions.jsonl", "w") as f:
            for p in predictions:
                f.write(json.dumps(p, default=str) + "\n")

    logger.info(f"Results saved to {output_dir}")
