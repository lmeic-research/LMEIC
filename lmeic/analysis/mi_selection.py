"""
Information-theoretic feature ranking for E8: mRMR and JMI on discretised features.
Version: v1 | Date: 2026-10-22

Features with more than n_bins distinct values are discretised into equal-frequency
bins; mutual information is the plug-in estimate (nats) from contingency counts.
mRMR (Peng et al. 2005): argmax I(X_k;Y) - mean_{s in S} I(X_k;X_s).
JMI (Brown et al. 2012): argmax sum_{s in S} I(X_k,X_s;Y); the first pick maximises I(X_k;Y).
"""

from __future__ import annotations

import numpy as np


def discretise(X: np.ndarray, n_bins: int = 10) -> np.ndarray:
    out = np.zeros(X.shape, dtype=np.int64)
    for j in range(X.shape[1]):
        col = X[:, j]
        uniq = np.unique(col)
        if len(uniq) <= n_bins:
            out[:, j] = np.searchsorted(uniq, col)
        else:
            edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)[1:-1]))
            out[:, j] = np.searchsorted(edges, col, side="right")
    return out


def _mi_codes(a: np.ndarray, b: np.ndarray) -> float:
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    joint = np.bincount(ia * len(ub) + ib, minlength=len(ua) * len(ub)).reshape(len(ua), len(ub))
    p = joint / joint.sum()
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum())


def mutual_info(a: np.ndarray, b: np.ndarray) -> float:
    return _mi_codes(a, b)


def joint_mi(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
    """I(A,B;Y) with the pair (A,B) encoded as one variable."""
    return _mi_codes(a * (b.max() + 1) + b, y)


def rank_mrmr(D: np.ndarray, y: np.ndarray, k: int) -> list[int]:
    n = D.shape[1]
    rel = np.array([mutual_info(D[:, j], y) for j in range(n)])
    sel = [int(rel.argmax())]
    red_sum = np.zeros(n)
    while len(sel) < min(k, n):
        last = sel[-1]
        red_sum += np.array([mutual_info(D[:, j], D[:, last]) if j not in sel else 0.0 for j in range(n)])
        score = rel - red_sum / len(sel)
        score[sel] = -np.inf
        sel.append(int(score.argmax()))
    return sel


def rank_jmi(D: np.ndarray, y: np.ndarray, k: int) -> list[int]:
    n = D.shape[1]
    rel = np.array([mutual_info(D[:, j], y) for j in range(n)])
    sel = [int(rel.argmax())]
    acc = np.zeros(n)
    while len(sel) < min(k, n):
        last = sel[-1]
        acc += np.array([joint_mi(D[:, j], D[:, last], y) if j not in sel else 0.0 for j in range(n)])
        score = acc.copy()
        score[sel] = -np.inf
        sel.append(int(score.argmax()))
    return sel
