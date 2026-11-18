"""``POST /v1/contributions/check``: every configured rule over one contribution."""

from __future__ import annotations

import base64
import binascii
from functools import partial
from http import HTTPStatus
from time import perf_counter
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from PIL import Image as PilImage
from pydantic import BaseModel, Field

from ddb_checks import engine
from ddb_checks.config import Config
from ddb_checks.contribution import Contribution, Image
from ddb_checks.scorer import LanguageScore

from .runtime import ModelPool
from .schemas import MAX_IMAGES, MAX_TEXT_CHARS, ImageItem
from .text import Timings
from .vision import ImageError, decode

_LANGUAGE_TOP_K = 5

router = APIRouter(tags=["contributions"])


class Fields(BaseModel):
    """The contribution's text; steps and quirks as a list, or one per line."""

    name: str | None = None
    summary: str | None = None
    getting_going: str | list[str] | None = None
    quirks: str | list[str] | None = None


class CheckRequest(BaseModel):
    """One contribution: structured device data, text fields and images."""

    config_version: str | None = Field(
        default=None,
        description="Refuse to check unless the server's configuration has this version",
    )
    device: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data, e.g. brand, category, connectivity, integration, "
        "identifiers, entities",
    )
    fields: Fields = Field(default_factory=Fields)
    images: list[ImageItem] = Field(default_factory=list, max_length=MAX_IMAGES)


class EvidenceOut(BaseModel):
    """A score behind a hit, and the threshold it crossed."""

    model: str
    signal: str
    score: float
    threshold: float | None
    detail: str | None


class HitOut(BaseModel):
    """
    One rule firing on one unit of text, or one image.

    ``unit`` counts the field's units from 0; ``item`` is the list position for fields sent as
    a list; ``span`` is the character range in that text. For images, ``unit`` is the image's
    position and ``text`` its id.
    """

    rule: str
    field: str
    unit: int
    item: int | None
    text: str | None
    span: tuple[int, int] | None
    action: Literal["warn", "reject"]
    message: str
    guideline: str
    evidence: list[EvidenceOut]


class ConfigOut(BaseModel):
    """The configuration the contribution was checked with."""

    version: str
    sha256: str


class CheckResponse(BaseModel):
    """The verdict: ``reject`` if any hit rejects, ``warn`` if any warns, else ``accept``."""

    outcome: Literal["accept", "warn", "reject"]
    config: ConfigOut
    rules: list[str]
    hits: list[HitOut]
    timings_ms: dict[str, float]


@router.post("/v1/contributions/check", response_model=CheckResponse)
async def check_contribution(body: CheckRequest, request: Request) -> CheckResponse:
    """Return the outcome of every configured rule, with the evidence for each hit."""
    began = perf_counter()
    config: Config = request.app.state.checks
    if body.config_version is not None and body.config_version != config.version:
        raise HTTPException(
            HTTPStatus.CONFLICT,
            f"this server runs config {config.version}, not {body.config_version}",
        )
    texts = [value for value in body.fields.model_dump().values() if value]
    flat = [t for value in texts for t in (value if isinstance(value, list) else [value])]
    if any(len(text) > MAX_TEXT_CHARS for text in flat):
        raise HTTPException(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"fields over {MAX_TEXT_CHARS} characters"
        )
    images = []
    for item in body.images:
        try:
            images.append(Image(item.id, base64.b64decode(item.data, validate=True)))
        except binascii.Error as err:
            raise ImageError(f"image {item.id}: data is not valid base64") from err

    contribution = Contribution(
        device=body.device, fields=body.fields.model_dump(), images=tuple(images)
    )
    result = await engine.check(contribution, config, PoolScorer(request.app.state.pool))
    return CheckResponse(
        outcome=result.outcome,
        config=ConfigOut(version=result.config_version, sha256=result.config_sha256),
        rules=result.rules_run,
        hits=[HitOut.model_validate(hit, from_attributes=True) for hit in result.hits],
        timings_ms={"total": round((perf_counter() - began) * 1000, 3)},
    )


class PoolScorer:
    """Scores for the checks from the server's own loaded models."""

    def __init__(self, pool: ModelPool) -> None:
        """Score with ``pool``; images are decoded once per request."""
        self._pool = pool
        self._decoded: dict[Image, PilImage.Image] = {}

    async def toxicity(self, model: str, texts: list[str]) -> list[dict[str, float]]:
        """Return each text's score per toxicity head."""
        loaded = self._pool.resolve("toxicity", model)
        probabilities, _ = await loaded.submit(loaded.model.predict, texts)
        return [dict(zip(loaded.model.labels, row.tolist(), strict=True)) for row in probabilities]

    async def language(
        self, model: str, texts: list[str], vocabulary: frozenset[str]
    ) -> list[LanguageScore]:
        """Return each text's language scores, after masking."""
        loaded = self._pool.resolve("language", model)
        found = await loaded.submit(
            partial(
                loaded.model.identify,
                texts,
                top_k=_LANGUAGE_TOP_K,
                masked=True,
                vocabulary=vocabulary,
            )
        )
        return [LanguageScore(r.english, tuple(r.top), r.letters) for r in found]

    async def nli(self, model: str, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """Return each pair's probability per NLI label."""
        loaded = self._pool.resolve("nli", model)
        probabilities, _ = await loaded.submit(loaded.model.predict, pairs)
        return [dict(zip(loaded.model.labels, row.tolist(), strict=True)) for row in probabilities]

    async def image_classes(
        self, model: str, images: list[Image], classes: dict[str, tuple[str, ...]]
    ) -> list[dict[str, float]]:
        """Return each image's zero-shot probability per class."""
        loaded = self._pool.resolve("image-embed", model)

        def run() -> list[dict[str, float]]:
            scores, _ = loaded.model.classify(
                self._decode(images), {name: list(p) for name, p in classes.items()}, Timings()
            )
            return [dict(zip(classes, row.tolist(), strict=True)) for row in scores]

        return await loaded.submit(run)

    async def obscenity(self, model: str, images: list[Image]) -> list[dict[str, float]]:
        """Return each image's explicitness signals."""
        loaded = self._pool.resolve("obscenity", model)

        def run() -> list[dict[str, float]]:
            probabilities = loaded.model.predict(self._decode(images), Timings())
            return [loaded.model.signals(row) for row in probabilities]

        return await loaded.submit(run)

    def _decode(self, images: list[Image]) -> list[PilImage.Image]:
        for image in images:
            if image not in self._decoded:
                try:
                    self._decoded[image] = decode(image.data)
                except ImageError as err:
                    raise type(err)(f"image {image.id}: {err}") from err
        return [self._decoded[image] for image in images]
