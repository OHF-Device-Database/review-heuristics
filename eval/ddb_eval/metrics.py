"""Threshold-free and threshold metrics for binary labels and scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_Z95 = 1.959964
# The Jigsaw Unintended Bias benchmark's power-mean exponent and weight of the overall AUC.
_BIAS_POWER = -5
_BIAS_OVERALL_WEIGHT = 0.25


@dataclass(frozen=True, slots=True)
class Bias:
    """Per-subgroup AUCs and the combined Jigsaw bias score."""

    subgroups: dict[str, dict[str, float | int | None]]
    overall_auc: float | None
    final: float | None


def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Return the area under the ROC curve; None unless both classes are present."""
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    _, inverse, counts = np.unique(scores[order], return_inverse=True, return_counts=True)
    # Tied scores share the average of the ranks they span.
    average_rank = np.cumsum(counts) - (counts - 1) / 2
    ranks = np.empty(len(scores))
    ranks[order] = average_rank[inverse]
    rank_sum = ranks[labels].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def auroc_interval(auc: float, positives: int, negatives: int) -> tuple[float, float]:
    """Return a 95% confidence interval for an AUC (Hanley and McNeil, 1982)."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    variance = (
        auc * (1 - auc) + (positives - 1) * (q1 - auc**2) + (negatives - 1) * (q2 - auc**2)
    ) / (positives * negatives)
    margin = _Z95 * float(np.sqrt(max(variance, 0.0)))
    return max(auc - margin, 0.0), min(auc + margin, 1.0)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Return the area under the precision-recall curve; None when there are no positives."""
    if not labels.any():
        return None
    order = np.argsort(-scores, kind="mergesort")
    ranked_scores, ranked_labels = scores[order], labels[order]
    true_positives = np.cumsum(ranked_labels)
    false_positives = np.cumsum(~ranked_labels)
    # One operating point per distinct score, so tied items enter together.
    last_of_tie = np.r_[np.diff(ranked_scores) != 0, True]
    true_positives, false_positives = true_positives[last_of_tie], false_positives[last_of_tie]
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / true_positives[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def wilson(successes: int, trials: int) -> tuple[float, float]:
    """Return the 95% Wilson score interval for a proportion."""
    if trials == 0:
        return 0.0, 1.0
    rate = successes / trials
    denominator = 1 + _Z95**2 / trials
    centre = (rate + _Z95**2 / (2 * trials)) / denominator
    margin = _Z95 * np.sqrt(rate * (1 - rate) / trials + _Z95**2 / (4 * trials**2)) / denominator
    return max(centre - margin, 0.0), min(centre + margin, 1.0)


def bias(labels: np.ndarray, scores: np.ndarray, subgroups: dict[str, np.ndarray]) -> Bias:
    """
    Return the Jigsaw Unintended Bias metrics for one head.

    Per subgroup: the AUC within the subgroup, BPSN (subgroup negatives against background
    positives) and BNSP (subgroup positives against background negatives). ``final`` is
    undefined (None) when any of those is.
    """
    rows: dict[str, dict[str, float | int | None]] = {}
    for name, members in subgroups.items():
        background = ~members
        bpsn = (members & ~labels) | (background & labels)
        bnsp = (members & labels) | (background & ~labels)
        rows[name] = {
            "size": int(members.sum()),
            "positives": int((members & labels).sum()),
            "subgroup_auc": auroc(labels[members], scores[members]),
            "bpsn_auc": auroc(labels[bpsn], scores[bpsn]),
            "bnsp_auc": auroc(labels[bnsp], scores[bnsp]),
        }

    overall = auroc(labels, scores)
    columns = ("subgroup_auc", "bpsn_auc", "bnsp_auc")
    values = [[row[column] for row in rows.values()] for column in columns]
    if overall is None or any(v is None for column in values for v in column):
        return Bias(rows, overall, None)
    means = [_power_mean(np.array(column, dtype=float)) for column in values]
    final = _BIAS_OVERALL_WEIGHT * overall + (1 - _BIAS_OVERALL_WEIGHT) * float(np.mean(means))
    return Bias(rows, overall, final)


def _power_mean(values: np.ndarray) -> float:
    # A zero AUC makes the negative power infinite, and the mean correctly comes out as 0.
    with np.errstate(divide="ignore"):
        return float(np.mean(values**_BIAS_POWER) ** (1 / _BIAS_POWER))
