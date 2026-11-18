"""The harness's pure logic: parity comparison, capacity, ids and manifests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ddb_eval import datasets, images, parity
from ddb_eval.cost import LoadStep, capacity
from ddb_eval.parity import compare, passes
from ddb_eval.registry import VariantId, split_source


def test_identical_scores_pass_fp32_parity() -> None:
    scores = np.array([[0.1, 0.9], [0.6, 0.2]])

    comparison = compare(scores, scores.copy())

    assert comparison.max_abs_diff == 0
    assert comparison.decision_agreement == 1
    assert passes("fp32", comparison) is True


def test_a_flipped_decision_fails_fp32_parity_and_is_listed_first() -> None:
    reference = np.array([[0.1, 0.49], [0.6, 0.2]])
    served = np.array([[0.1, 0.51], [0.6, 0.2]])

    comparison = compare(reference, served)

    assert comparison.decision_agreement == pytest.approx(0.75)
    assert comparison.worst[0][:2] == (0, 1)
    assert passes("fp32", comparison) is False
    assert passes("int8", comparison) is None


def _step(rate: float, p95: float, rejected: int = 0) -> LoadStep:
    return LoadStep(rate, 10, 10 - rejected, rejected, 0, p95 / 2, p95, p95, 1.0, 0.1)


def test_capacity_is_the_highest_clean_rate_under_the_ceiling() -> None:
    steps = [_step(1, 20), _step(5, 40), _step(10, 900, rejected=2), _step(20, 3000)]

    assert capacity(steps, ceiling_ms=1000) == 5
    assert capacity([_step(1, 3000)], ceiling_ms=1000) is None


def test_variant_ids_round_trip() -> None:
    parsed = VariantId.parse("toxicity/detoxify-unbiased@int8")

    assert str(parsed) == "toxicity/detoxify-unbiased@int8"
    assert parsed.with_precision("fp32").directory == Path("toxicity/detoxify-unbiased/fp32")
    with pytest.raises(ValueError, match="precision"):
        VariantId.parse("toxicity/detoxify-unbiased")


def test_sources_must_pin_a_revision() -> None:
    assert split_source("org/model@abc123") == ("org/model", "abc123")
    with pytest.raises(ValueError, match="pin a revision"):
        split_source("org/model")


def test_manifest_version_follows_its_content(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(json.dumps({"id": "a", "text": "Hello."}) + "\n")
    first = datasets.load_text(path)
    path.write_text(json.dumps({"id": "a", "text": "Hello!"}) + "\n")

    assert first.items == [datasets.TextItem("a", "Hello.")]
    assert datasets.load_text(path).version != first.version
    assert datasets.load_text(path, with_long_text=True).items[-1].id.startswith("synthetic/")


def test_synthetic_images_are_fixed_and_decode_upright_on_white() -> None:
    first, second = images.synthetic(), images.synthetic()
    decoded = {image.id: images.reference_decode(image.data) for image in first}

    assert [image.data for image in first] == [image.data for image in second]
    assert all(image.mode == "RGB" for image in decoded.values())
    assert decoded["exif-rotated-jpeg"].size == (300, 600)
    assert decoded["rgba-png-transparent"].getpixel((0, 0)) == (255, 255, 255)
    assert decoded["palette-gif-transparent"].getpixel((0, 0)) == (255, 255, 255)


def test_min_cosine_ignores_scale() -> None:
    reference = np.array([[1.0, 0.0], [0.0, 2.0]])

    assert parity.min_cosine(reference, reference * 3) == pytest.approx(1)
    assert parity.min_cosine(reference, np.array([[1.0, 1.0], [0.0, 1.0]])) == pytest.approx(
        2**-0.5
    )
