"""``POST /v1/toxicity``: per-head toxicity scores for a batch of texts."""

from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from .runtime import ModelPool
from .schemas import TextItem, TimingsMs, check_texts, text_timings

router = APIRouter(tags=["toxicity"])


class ToxicityRequest(BaseModel):
    """A batch of texts, and optionally the variant to score them with."""

    model: str | None = Field(default=None, description="Variant id; optional with one loaded")
    texts: list[TextItem] = Field(min_length=1)


class ToxicityResult(BaseModel):
    """Scores for one text, between 0 and 1 per head."""

    id: str
    scores: dict[str, float]


class ToxicityResponse(BaseModel):
    """Scores for every text in the request, and the exact variant that produced them."""

    model: str
    model_sha256: str
    results: list[ToxicityResult]
    timings_ms: TimingsMs


@router.post("/v1/toxicity", response_model=ToxicityResponse)
async def score_toxicity(body: ToxicityRequest, request: Request) -> ToxicityResponse:
    """Return per-head toxicity scores for each text."""
    began = perf_counter()
    check_texts(body.texts)
    pool: ModelPool = request.app.state.pool
    loaded = pool.resolve("toxicity", body.model)
    probabilities, timings = await loaded.submit(
        loaded.model.predict, [item.text for item in body.texts]
    )
    labels = loaded.model.labels
    return ToxicityResponse(
        model=loaded.variant.id,
        model_sha256=loaded.meta.sha256,
        results=[
            ToxicityResult(id=item.id, scores=dict(zip(labels, row.tolist(), strict=True)))
            for item, row in zip(body.texts, probabilities, strict=True)
        ],
        timings_ms=text_timings(timings, began),
    )
