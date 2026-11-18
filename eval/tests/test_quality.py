"""Quality metrics, the toxicity summaries built on them, and served scoring order."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from ddb_eval import metrics, parity, quality
from ddb_eval.civil_comments import CivilComments
from ddb_eval.datasets import TextItem


def test_auroc_counts_ties_as_half() -> None:
    labels = np.array([False, False, True, True])

    assert metrics.auroc(labels, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert metrics.auroc(labels, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert metrics.auroc(np.array([True, True]), np.array([0.1, 0.9])) is None


def test_auroc_interval_narrows_with_more_data() -> None:
    small = metrics.auroc_interval(0.9, 10, 10)
    large = metrics.auroc_interval(0.9, 1000, 1000)

    assert small[0] < large[0] < 0.9 < large[1] < small[1]


def test_average_precision() -> None:
    labels = np.array([True, False, True, False])

    # Ranked: T, T, F, F -> perfect.
    assert metrics.average_precision(labels, np.array([0.9, 0.2, 0.8, 0.1])) == pytest.approx(1)
    # Ranked: F, T, F, T -> precision 1/2 at recall 0.5, 2/4 at recall 1.
    assert metrics.average_precision(labels, np.array([0.8, 0.9, 0.1, 0.7])) == pytest.approx(0.5)
    assert metrics.average_precision(np.array([False]), np.array([0.3])) is None


def test_wilson_interval_brackets_the_rate() -> None:
    low, high = metrics.wilson(0, 100)

    assert low == 0
    assert high == pytest.approx(0.037, abs=1e-3)
    assert metrics.wilson(5, 10)[0] < 0.5 < metrics.wilson(5, 10)[1]


def test_bias_penalises_scores_that_track_identity() -> None:
    labels = np.array([False, True] * 4)
    members = np.array([True, True, True, True, False, False, False, False])
    fair = np.array([0.1, 0.9] * 4)
    # Benign subgroup members score as high as toxic background comments.
    skewed = np.array([0.8, 0.9, 0.8, 0.9, 0.1, 0.7, 0.1, 0.7])

    fair_bias = metrics.bias(labels, fair, {"group": members})
    skewed_bias = metrics.bias(labels, skewed, {"group": members})

    assert fair_bias.final == pytest.approx(1)
    assert skewed_bias.subgroups["group"]["bpsn_auc"] < 0.5
    assert skewed_bias.final is not None
    assert skewed_bias.final < fair_bias.final


def test_original_heads_map_to_civil_comments_attributes() -> None:
    assert quality.attribute("toxic") == "toxicity"
    assert quality.attribute("identity_hate") == "identity_attack"
    assert quality.attribute("insult") == "insult"


def test_civil_comments_metrics_use_the_mapped_attribute() -> None:
    toxic = np.array([True, False, True, False])
    data = CivilComments(
        version="test",
        items=[TextItem(str(i), "text") for i in range(4)],
        labels={"toxicity": toxic, "insult": np.zeros(4, dtype=bool)},
        identities={"group": np.array([True, True, False, False])},
    )
    scores = np.array([[0.9, 0.1], [0.2, 0.6], [0.7, 0.2], [0.1, 0.1]])

    result = quality.civil_comments_metrics(data, ["toxic", "insult"], scores)

    assert result["heads"]["toxic"]["auroc"] == pytest.approx(1)
    assert result["heads"]["toxic"]["recall_at_threshold"] == 1
    assert result["heads"]["insult"]["auroc"] is None
    assert result["heads"]["insult"]["false_positive_rate_at_threshold"] == pytest.approx(0.25)
    assert result["bias"]["head"] == "toxic"


def test_device_metrics_report_false_flags_and_misses() -> None:
    items = [
        TextItem("a", "kill switch", "benign", "technical-terms"),
        TextItem("b", "dead zone", "benign", "technical-terms"),
        TextItem("c", "idiot", "toxic", "insult"),
        TextItem("d", "moron", "toxic", "insult"),
    ]
    scores = np.array([[0.7, 0.1], [0.2, 0.1], [0.3, 0.9], [0.1, 0.2]])

    result = quality.device_metrics(items, ["toxicity", "insult"], scores)

    assert result["benign_flagged"]["count"] == 1
    assert result["toxic_caught"]["count"] == 1
    assert result["categories"]["technical-terms"]["flagged_by_head"] == {
        "toxicity": 1,
        "insult": 0,
    }
    assert [error["id"] for error in result["errors"]] == ["a", "d"]


def test_served_scores_restore_dataset_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["texts"]
        return httpx.Response(
            200,
            json={"results": [{"id": t["id"], "scores": {"x": len(t["text"])}} for t in texts]},
        )

    items = [TextItem(str(i), "t" * length) for i, length in enumerate([5, 1, 300, 3])]
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://serve") as client:
        scores = parity.served_scores(client, "toxicity/x@fp32", ["x"], items)

    assert scores[:, 0].tolist() == [5, 1, 300, 3]
