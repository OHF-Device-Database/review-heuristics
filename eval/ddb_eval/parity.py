"""Gate 1: check that the served variant answers like the original PyTorch model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
import torch
from timm.data import create_transform, resolve_data_config
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .datasets import NliPair, TextItem
from .images import EncodedImage, reference_decode
from .registry import split_source

# fp32 pass criteria from the benchmark plan; quantized variants are reported, not gated.
FP32_MAX_ABS_DIFF = 1e-3
FP32_MIN_AGREEMENT = 0.999
FP32_MIN_COSINE = 0.999
DECISION_THRESHOLD = 0.5
# The most texts serve accepts in one request.
_REQUEST_BATCH = 128
_PROGRESS_EVERY = 5_000
_REFERENCE_BATCH = 16
_IMAGE_BATCH = 8
_WORST_SHOWN = 5
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Comparison:
    """How far served scores are from the reference scores."""

    max_abs_diff: float
    mean_abs_diff: float
    decision_agreement: float
    worst: list[tuple[int, int, float, float]]


def compare(reference: np.ndarray, served: np.ndarray) -> Comparison:
    """
    Compare two score matrices of the same shape (texts by labels).

    ``worst`` lists the largest differences as (row, column, reference, served).
    """
    diff = np.abs(reference - served)
    agreement = (reference >= DECISION_THRESHOLD) == (served >= DECISION_THRESHOLD)
    flat = np.argsort(diff, axis=None)[::-1][:_WORST_SHOWN]
    rows, columns = np.unravel_index(flat, diff.shape)
    return Comparison(
        max_abs_diff=float(diff.max()),
        mean_abs_diff=float(diff.mean()),
        decision_agreement=float(agreement.mean()),
        worst=[
            (int(r), int(c), float(reference[r, c]), float(served[r, c]))
            for r, c in zip(rows, columns, strict=True)
        ],
    )


def passes(precision: str, comparison: Comparison) -> bool | None:
    """Return whether an fp32 variant passes parity; None for precisions that aren't gated."""
    if precision != "fp32":
        return None
    return (
        comparison.max_abs_diff <= FP32_MAX_ABS_DIFF
        and comparison.decision_agreement >= FP32_MIN_AGREEMENT
    )


def served_scores(
    client: httpx.Client, variant_id: str, labels: list[str], items: list[TextItem]
) -> np.ndarray:
    """Return the served variant's scores for ``items``, columns in ``labels`` order."""
    # Texts of similar length share a request, so the server pads each batch little.
    order = sorted(range(len(items)), key=lambda i: len(items[i].text))
    scores = np.empty((len(items), len(labels)))
    for start in range(0, len(order), _REQUEST_BATCH):
        chunk = order[start : start + _REQUEST_BATCH]
        response = client.post(
            "/v1/toxicity",
            json={
                "model": variant_id,
                "texts": [{"id": items[i].id, "text": items[i].text} for i in chunk],
            },
        )
        response.raise_for_status()
        for i, result in zip(chunk, response.json()["results"], strict=True):
            scores[i] = [result["scores"][label] for label in labels]
        done = start + len(chunk)
        if done % _PROGRESS_EVERY < _REQUEST_BATCH and done >= _PROGRESS_EVERY:
            _LOGGER.info("Scored %d of %d texts", done, len(items))
    return scores


def reference_scores(source: str, labels: list[str], items: list[TextItem]) -> np.ndarray:
    """Return the original PyTorch model's scores for ``items``, columns in ``labels`` order."""
    repo, revision = split_source(source)
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(repo, revision=revision).eval()
    config: Any = model.config
    columns = [config.label2id[label] for label in labels]

    rows = []
    with torch.no_grad():
        for start in range(0, len(items), _REFERENCE_BATCH):
            chunk = [item.text for item in items[start : start + _REFERENCE_BATCH]]
            encoded = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
            logits = model(**encoded).logits
            if config.problem_type == "multi_label_classification":
                probabilities = torch.sigmoid(logits)
            else:
                probabilities = torch.softmax(logits, dim=-1)
            rows.append(probabilities[:, columns].numpy())
    return np.concatenate(rows)


def min_cosine(reference: np.ndarray, served: np.ndarray) -> float:
    """Return the lowest cosine similarity between matching rows of two embedding matrices."""
    reference = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    served = served / np.linalg.norm(served, axis=1, keepdims=True)
    return float((reference * served).sum(axis=1).min())


def served_nli(
    client: httpx.Client, variant_id: str, labels: list[str], pairs: list[NliPair]
) -> np.ndarray:
    """Return the served variant's probabilities per pair, columns in ``labels`` order."""
    rows = []
    for start in range(0, len(pairs), _REQUEST_BATCH):
        chunk = pairs[start : start + _REQUEST_BATCH]
        response = client.post(
            "/v1/texts/nli",
            json={
                "model": variant_id,
                "pairs": [
                    {"id": p.id, "premise": p.premise, "hypothesis": p.hypothesis} for p in chunk
                ],
            },
        )
        response.raise_for_status()
        rows.extend([r["scores"][label] for label in labels] for r in response.json()["results"])
    return np.array(rows)


def reference_nli(source: str, labels: list[str], pairs: list[NliPair]) -> np.ndarray:
    """Return the original PyTorch model's probabilities per pair, columns in ``labels`` order."""
    repo, revision = split_source(source)
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(repo, revision=revision).eval()
    config: Any = model.config
    columns = [config.label2id[label] for label in labels]
    rows = []
    with torch.no_grad():
        for start in range(0, len(pairs), _REFERENCE_BATCH):
            chunk = pairs[start : start + _REFERENCE_BATCH]
            encoded = tokenizer(
                [p.premise for p in chunk],
                [p.hypothesis for p in chunk],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            rows.append(torch.softmax(model(**encoded).logits, dim=-1)[:, columns].numpy())
    return np.concatenate(rows)


def served_obscenity(
    client: httpx.Client, variant_id: str, labels: list[str], images: list[EncodedImage]
) -> np.ndarray:
    """Return the served variant's probabilities per image, columns in ``labels`` order."""
    results = _post_images(client, "/v1/obscenity", {"model": variant_id}, images)
    return np.array([[r["scores"][label] for label in labels] for r in results])


def reference_obscenity(source: str, labels: list[str], images: list[EncodedImage]) -> np.ndarray:
    """Return the original model's probabilities per image, preprocessed as its card does."""
    repo, revision = split_source(source)
    model = AutoModelForImageClassification.from_pretrained(
        repo, revision=revision, dtype=torch.float32
    ).eval()
    transform = create_transform(**resolve_data_config({}, model=model.timm_model))
    config: Any = model.config
    columns = [config.label2id[label] for label in labels]
    with torch.no_grad():
        pixels = torch.stack([transform(reference_decode(image.data)) for image in images])
        return torch.softmax(model(pixel_values=pixels).logits, dim=-1)[:, columns].numpy()


def served_embeddings(
    client: httpx.Client, variant_id: str, images: list[EncodedImage]
) -> np.ndarray:
    """Return the served variant's image embeddings."""
    results = _post_images(client, "/v1/images/embeddings", {"model": variant_id}, images)
    return np.array([r["embedding"] for r in results])


def served_zero_shot(
    client: httpx.Client, variant_id: str, prompts: list[str], images: list[EncodedImage]
) -> np.ndarray:
    """Return the served zero-shot softmax per image, one class per prompt."""
    body = {"model": variant_id, "classes": {prompt: [prompt] for prompt in prompts}}
    results = _post_images(client, "/v1/images/classify", body, images)
    return np.array([[r["scores"][prompt] for prompt in prompts] for r in results])


def reference_siglip(
    source: str, prompts: list[str], images: list[EncodedImage]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return SigLIP's image embeddings and zero-shot softmax from the original model.

    Images go through Transformers' Pillow-based image processor, which serve reproduces.
    Transformers' default torchvision processor resizes slightly differently: its embeddings
    are at cosine 0.99997 to the Pillow ones. Prompts are lowercased and padded to 64 tokens,
    as the model was trained, and the zero-shot logits are the model's own
    ``logits_per_image``.
    """
    repo, revision = split_source(source)
    model = AutoModel.from_pretrained(repo, revision=revision).eval()
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    processor = AutoImageProcessor.from_pretrained(repo, revision=revision, use_fast=False)
    decoded = [reference_decode(image.data) for image in images]
    with torch.no_grad():
        text = tokenizer(
            [prompt.lower() for prompt in prompts],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        pixels = processor(images=decoded, return_tensors="pt")
        outputs = model(input_ids=text["input_ids"], pixel_values=pixels["pixel_values"])
    return outputs.image_embeds.numpy(), torch.softmax(outputs.logits_per_image, dim=-1).numpy()


def _post_images(
    client: httpx.Client, route: str, body: dict[str, Any], images: list[EncodedImage]
) -> list[dict[str, Any]]:
    results = []
    for start in range(0, len(images), _IMAGE_BATCH):
        chunk = images[start : start + _IMAGE_BATCH]
        response = client.post(
            route, json={**body, "images": [{"id": i.id, "data": i.base64} for i in chunk]}
        )
        response.raise_for_status()
        results.extend(response.json()["results"])
    return results
