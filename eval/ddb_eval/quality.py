"""Gate 2 for toxicity: Civil Comments accuracy and bias, and false flags on device text."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import metrics
from .civil_comments import CivilComments
from .datasets import TextItem

DECISION_THRESHOLD = 0.5
TOXIC_LABEL = "toxic"
# Detoxify `original` names its heads after the 2018 Jigsaw challenge; these are the
# Civil Comments attributes they predict.
_ATTRIBUTE_FOR_HEAD = {
    "toxic": "toxicity",
    "severe_toxic": "severe_toxicity",
    "identity_hate": "identity_attack",
}
_SHOWN_ERRORS = 50


def attribute(head: str) -> str:
    """Return the Civil Comments attribute a model head predicts."""
    return _ATTRIBUTE_FOR_HEAD.get(head, head)


def civil_comments_metrics(
    data: CivilComments, heads: list[str], scores: np.ndarray
) -> dict[str, Any]:
    """
    Return per-head accuracy on Civil Comments, and bias metrics for the toxicity head.

    ``scores`` has one column per entry in ``heads``.
    """
    per_head = {}
    for column, head in enumerate(heads):
        labels, head_scores = data.labels[attribute(head)], scores[:, column]
        positives = int(labels.sum())
        auc = metrics.auroc(labels, head_scores)
        flagged = head_scores >= DECISION_THRESHOLD
        per_head[head] = {
            "attribute": attribute(head),
            "positives": positives,
            "auroc": auc,
            "auroc_95": metrics.auroc_interval(auc, positives, len(labels) - positives)
            if auc is not None
            else None,
            "average_precision": metrics.average_precision(labels, head_scores),
            "recall_at_threshold": _rate(flagged & labels, labels),
            "false_positive_rate_at_threshold": _rate(flagged & ~labels, ~labels),
        }

    toxicity_head = next(head for head in heads if attribute(head) == "toxicity")
    result = metrics.bias(
        data.labels["toxicity"], scores[:, heads.index(toxicity_head)], data.identities
    )
    return {
        "texts": len(data.items),
        "threshold": DECISION_THRESHOLD,
        "heads": per_head,
        "bias": {
            "head": toxicity_head,
            "overall_auc": result.overall_auc,
            "final": result.final,
            "subgroups": result.subgroups,
        },
    }


def device_metrics(items: list[TextItem], heads: list[str], scores: np.ndarray) -> dict[str, Any]:
    """
    Return how often each category of device text is flagged by any head.

    For benign categories that is the false-flag rate; for toxic ones, recall. ``errors``
    lists the benign texts flagged and the toxic texts missed, highest score first.
    """
    flagged = (scores >= DECISION_THRESHOLD).any(axis=1)
    top = scores.argmax(axis=1)
    categories: dict[str, dict[str, Any]] = {}
    for category in dict.fromkeys(item.category for item in items):
        rows = [i for i, item in enumerate(items) if item.category == category]
        hits = int(flagged[rows].sum())
        categories[str(category)] = {
            "label": items[rows[0]].label,
            "texts": len(rows),
            "flagged": hits,
            "flagged_rate": hits / len(rows),
            "flagged_95": metrics.wilson(hits, len(rows)),
            "flagged_by_head": {
                head: int((scores[rows, column] >= DECISION_THRESHOLD).sum())
                for column, head in enumerate(heads)
            },
        }

    toxic = np.array([item.label == TOXIC_LABEL for item in items])
    wrong = np.flatnonzero(flagged != toxic)
    wrong = wrong[np.argsort(-scores[wrong].max(axis=1))]
    return {
        "texts": len(items),
        "threshold": DECISION_THRESHOLD,
        "benign_flagged": _summary(flagged[~toxic]),
        "toxic_caught": _summary(flagged[toxic]),
        "categories": categories,
        "errors": [
            {
                "id": items[i].id,
                "text": items[i].text,
                "label": items[i].label,
                "head": heads[top[i]],
                "score": float(scores[i, top[i]]),
            }
            for i in wrong[:_SHOWN_ERRORS]
        ],
    }


def _rate(hits: np.ndarray, base: np.ndarray) -> float | None:
    total = int(base.sum())
    return int(hits.sum()) / total if total else None


def _summary(hits: np.ndarray) -> dict[str, Any]:
    count = int(hits.sum())
    return {
        "count": count,
        "of": len(hits),
        "rate": count / len(hits) if len(hits) else None,
        "rate_95": metrics.wilson(count, len(hits)),
    }
