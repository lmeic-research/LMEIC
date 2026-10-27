"""
Fold construction for the reference rebuild.
Version: v1 | Date: 2026-10-22

Two protocols:
  v1_original   : StratifiedKFold over messages (as in EMAIL-3R v1.0 experiments).
  grouped_clean : StratifiedGroupKFold over threads, so that no thread contributes
                  messages to both the fitting and the scored fold.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold

PROTOCOLS = ("v1_original", "grouped_clean")


def make_folds(y: np.ndarray, groups: np.ndarray, protocol: str, seed: int,
               n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if protocol == "v1_original":
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(len(y)), y))
    if protocol == "grouped_clean":
        sgkf = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
        return list(sgkf.split(np.zeros(len(y)), y, groups))
    raise ValueError(protocol)


def inner_es_split(tr: np.ndarray, groups: np.ndarray, frac: float,
                   seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a training fold into fit and early-stopping parts without sharing threads."""
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    fit_rel, es_rel = next(gss.split(tr, groups=groups[tr]))
    return tr[fit_rel], tr[es_rel]


def check_group_disjoint(folds: list[tuple[np.ndarray, np.ndarray]], groups: np.ndarray) -> int:
    """Return the number of groups that appear on both sides of any fold."""
    n = 0
    for tr, va in folds:
        n += len(set(groups[tr]) & set(groups[va]))
    return n
