"""``POST /v1/texts/language``: which language each text is in, and how likely English."""

from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from .language_id import LanguageIdentifier
from .runtime import ModelPool
from .schemas import TextItem, TimingsMs, check_texts, text_timings
from .text import Timings

MAX_VOCABULARY = 1000

router = APIRouter(tags=["language"])


class LanguageRequest(BaseModel):
    """A batch of texts, and how to score them."""

    model: str | None = Field(default=None, description="Variant id; optional with one loaded")
    texts: list[TextItem] = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=20, description="How many languages to return")
    mask: bool = Field(
        default=True,
        description="Remove names, codes and capitalised words before scoring",
    )
    vocabulary: list[str] = Field(
        default_factory=list,
        max_length=MAX_VOCABULARY,
        description="Extra words to remove before scoring, such as brand and protocol names",
    )


class LanguageScore(BaseModel):
    """One candidate language, as an ISO 639-3 code with its script, and its probability."""

    language: str
    probability: float


class LanguageResult(BaseModel):
    """
    The languages one text scored as.

    ``english`` is null, and ``top`` empty, when masking left no words to score.
    """

    id: str
    english: float | None
    top: list[LanguageScore]
    letters: int = Field(description="Letters left after masking")
    scored_text: str


class LanguageResponse(BaseModel):
    """Results for every text in the request, and the exact variant that produced them."""

    model: str
    model_sha256: str
    results: list[LanguageResult]
    timings_ms: TimingsMs


@router.post("/v1/texts/language", response_model=LanguageResponse)
async def identify_language(body: LanguageRequest, request: Request) -> LanguageResponse:
    """Return the most likely languages for each text, and its probability of being English."""
    began = perf_counter()
    check_texts(body.texts)
    pool: ModelPool = request.app.state.pool
    loaded = pool.resolve("language", body.model)
    model: LanguageIdentifier = loaded.model
    vocabulary = frozenset(word.lower() for word in body.vocabulary)

    def run() -> tuple[list, Timings]:
        started = perf_counter()
        found = model.identify(
            [item.text for item in body.texts],
            top_k=body.top_k,
            masked=body.mask,
            vocabulary=vocabulary,
        )
        return found, Timings(inference=(perf_counter() - started) * 1000)

    found, timings = await loaded.submit(run)
    return LanguageResponse(
        model=loaded.variant.id,
        model_sha256=loaded.meta.sha256,
        results=[
            LanguageResult(
                id=item.id,
                english=result.english,
                top=[LanguageScore(language=code, probability=p) for code, p in result.top],
                letters=result.letters,
                scored_text=result.scored_text,
            )
            for item, result in zip(body.texts, found, strict=True)
        ],
        timings_ms=text_timings(timings, began),
    )
