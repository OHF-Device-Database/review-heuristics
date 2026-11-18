"""``POST /v1/images/embeddings`` and ``/v1/images/classify``: SigLIP 2 image features."""

from __future__ import annotations

from http import HTTPStatus
from time import perf_counter

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .runtime import ModelPool
from .schemas import ImageItem, ImageTimingsMs, check_images, image_timings
from .text import Timings
from .vision import SiglipEncoder, decode_all

MAX_CLASSES = 32
MAX_PROMPTS = 256
_TASK = "image-embed"

router = APIRouter(tags=["images"])


class EmbeddingsRequest(BaseModel):
    """A batch of images to embed."""

    model: str | None = Field(default=None, description="Variant id; optional with one loaded")
    images: list[ImageItem] = Field(min_length=1)


class Embedding(BaseModel):
    """One image's L2-normalised embedding."""

    id: str
    embedding: list[float]


class EmbeddingsResponse(BaseModel):
    """Embeddings for every image in the request, and the exact variant that produced them."""

    model: str
    model_sha256: str
    results: list[Embedding]
    timings_ms: ImageTimingsMs


class ClassifyRequest(BaseModel):
    """A batch of images, and the classes to choose between, each described by text prompts."""

    model: str | None = Field(default=None, description="Variant id; optional with one loaded")
    classes: dict[str, list[str]] = Field(
        min_length=2,
        description="Class name to one or more prompts, e.g. {'packshot': ['a product photo "
        "on a plain background']}",
    )
    images: list[ImageItem] = Field(min_length=1)


class Classification(BaseModel):
    """
    One image's zero-shot scores.

    ``scores`` picks one class among those given, summing to 1. ``match`` is SigLIP's own
    probability that each class's prompts describe the image, independent of the others.
    """

    id: str
    scores: dict[str, float]
    match: dict[str, float]


class ClassifyResponse(BaseModel):
    """Scores for every image in the request, and the exact variant that produced them."""

    model: str
    model_sha256: str
    results: list[Classification]
    timings_ms: ImageTimingsMs


@router.post("/v1/images/embeddings", response_model=EmbeddingsResponse)
async def embed_images(body: EmbeddingsRequest, request: Request) -> EmbeddingsResponse:
    """Return an L2-normalised embedding for each image."""
    began = perf_counter()
    check_images(body.images)
    pool: ModelPool = request.app.state.pool
    loaded = pool.resolve(_TASK, body.model)
    model: SiglipEncoder = loaded.model

    def run() -> tuple[np.ndarray, Timings]:
        timings = Timings()
        images = decode_all([(item.id, item.data) for item in body.images], timings)
        return model.embed_images(images, timings), timings

    embeddings, timings = await loaded.submit(run)
    return EmbeddingsResponse(
        model=loaded.variant.id,
        model_sha256=loaded.meta.sha256,
        results=[
            Embedding(id=item.id, embedding=row.tolist())
            for item, row in zip(body.images, embeddings, strict=True)
        ],
        timings_ms=image_timings(timings, began),
    )


@router.post("/v1/images/classify", response_model=ClassifyResponse)
async def classify_images(body: ClassifyRequest, request: Request) -> ClassifyResponse:
    """Return zero-shot class scores for each image, using the prompts in the request."""
    began = perf_counter()
    check_images(body.images)
    prompts = [prompt for class_prompts in body.classes.values() for prompt in class_prompts]
    if len(body.classes) > MAX_CLASSES or len(prompts) > MAX_PROMPTS:
        raise HTTPException(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            f"at most {MAX_CLASSES} classes and {MAX_PROMPTS} prompts per request",
        )
    if empty := [name for name, class_prompts in body.classes.items() if not class_prompts]:
        raise HTTPException(
            HTTPStatus.UNPROCESSABLE_ENTITY, f"classes without prompts: {', '.join(empty)}"
        )
    pool: ModelPool = request.app.state.pool
    loaded = pool.resolve(_TASK, body.model)
    model: SiglipEncoder = loaded.model

    def run() -> tuple[np.ndarray, np.ndarray, Timings]:
        timings = Timings()
        images = decode_all([(item.id, item.data) for item in body.images], timings)
        scores, match = model.classify(images, body.classes, timings)
        return scores, match, timings

    scores, match, timings = await loaded.submit(run)
    names = list(body.classes)
    return ClassifyResponse(
        model=loaded.variant.id,
        model_sha256=loaded.meta.sha256,
        results=[
            Classification(
                id=item.id,
                scores=dict(zip(names, score_row.tolist(), strict=True)),
                match=dict(zip(names, match_row.tolist(), strict=True)),
            )
            for item, score_row, match_row in zip(body.images, scores, match, strict=True)
        ],
        timings_ms=image_timings(timings, began),
    )
