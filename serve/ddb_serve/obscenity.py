"""``POST /v1/obscenity``: how explicit each image is."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from .runtime import ModelPool
from .schemas import ImageItem, ImageTimingsMs, check_images, image_timings
from .text import Timings
from .vision import ImageClassifier, decode_all

router = APIRouter(tags=["obscenity"])


class ObscenityRequest(BaseModel):
    """A batch of images to score."""

    model: str | None = Field(default=None, description="Variant id; optional with one loaded")
    images: list[ImageItem] = Field(min_length=1)


class ObscenityResult(BaseModel):
    """
    One image's scores.

    ``scores`` has one probability per model label. For graded models, ``at_least`` gives the
    probability of each level or worse (``at_least.medium`` = P(medium) + P(high)). ``nsfw``
    is 1 - P(safe label), comparable across models.
    """

    id: str
    scores: dict[str, float]
    at_least: dict[str, float] | None
    nsfw: float


class ObscenityResponse(BaseModel):
    """Scores for every image in the request, and the exact variant that produced them."""

    model: str
    model_sha256: str
    results: list[ObscenityResult]
    timings_ms: ImageTimingsMs


@router.post("/v1/obscenity", response_model=ObscenityResponse)
async def score_obscenity(body: ObscenityRequest, request: Request) -> ObscenityResponse:
    """Return explicitness scores for each image."""
    began = perf_counter()
    check_images(body.images)
    pool: ModelPool = request.app.state.pool
    loaded = pool.resolve("obscenity", body.model)
    model: ImageClassifier = loaded.model

    def run() -> tuple[np.ndarray, Timings]:
        timings = Timings()
        images = decode_all([(item.id, item.data) for item in body.images], timings)
        return model.predict(images, timings), timings

    probabilities, timings = await loaded.submit(run)
    results = []
    for item, row in zip(body.images, probabilities, strict=True):
        signals = model.signals(row)
        results.append(
            ObscenityResult(
                id=item.id,
                scores={label: signals[label] for label in model.labels},
                at_least={
                    key.removeprefix("at_least_"): value
                    for key, value in signals.items()
                    if key.startswith("at_least_")
                }
                or None,
                nsfw=signals["nsfw"],
            )
        )
    return ObscenityResponse(
        model=loaded.variant.id,
        model_sha256=loaded.meta.sha256,
        results=results,
        timings_ms=image_timings(timings, began),
    )
