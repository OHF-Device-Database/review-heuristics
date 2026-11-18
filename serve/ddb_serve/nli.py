"""``POST /v1/texts/nli``: entailment between text pairs, and zero-shot classification."""

from __future__ import annotations

from http import HTTPStatus
from time import perf_counter
from typing import Literal, Self, cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from .runtime import ModelPool
from .schemas import MAX_TEXT_CHARS, MAX_TEXTS, TextItem, TimingsMs, text_timings
from .text import SequenceClassifier, softmax

MAX_LABELS = 32
_ENTAILMENT = "entailment"
# Three-way models contrast entailment with contradiction; two-way ones with not_entailment.
_AGAINST_ENTAILMENT = ("contradiction", "not_entailment")

router = APIRouter(tags=["nli"])


class Pair(BaseModel):
    """A premise and a hypothesis; the model judges whether the first implies the second."""

    id: str
    premise: str
    hypothesis: str


class ZeroShot(BaseModel):
    """Candidate labels, each turned into a hypothesis by the template."""

    labels: list[str] = Field(min_length=1, max_length=MAX_LABELS)
    hypothesis_template: str = Field(
        default="This example is {}.", description="Must contain {} where the label goes"
    )
    multi_label: bool = Field(
        default=False,
        description="Score each label on its own instead of picking one among them",
    )


class NliRequest(BaseModel):
    """Either ``pairs``, or ``texts`` with ``zero_shot`` labels."""

    model: str | None = Field(default=None, description="Variant id; optional with one loaded")
    pairs: list[Pair] | None = Field(default=None, min_length=1)
    texts: list[TextItem] | None = Field(default=None, min_length=1)
    zero_shot: ZeroShot | None = None

    @model_validator(mode="after")
    def _one_mode(self) -> Self:
        if (self.pairs is None) == (self.texts is None):
            raise ValueError("send either pairs, or texts with zero_shot")
        if (self.texts is None) != (self.zero_shot is None):
            raise ValueError("texts and zero_shot go together")
        if self.zero_shot is not None and "{}" not in self.zero_shot.hypothesis_template:
            raise ValueError("hypothesis_template must contain {}")
        return self


class NliResult(BaseModel):
    """
    Scores for one pair or text.

    For pairs, one probability per model label (entailment, neutral, contradiction). For
    zero-shot, one probability per candidate label: across labels summing to 1, or with
    ``multi_label`` each label's own entailment probability.
    """

    id: str
    scores: dict[str, float]


class NliResponse(BaseModel):
    """Scores for every item in the request, and the exact variant that produced them."""

    model: str
    model_sha256: str
    mode: Literal["pairs", "zero_shot"]
    results: list[NliResult]
    timings_ms: TimingsMs


@router.post("/v1/texts/nli", response_model=NliResponse)
async def score_nli(body: NliRequest, request: Request) -> NliResponse:
    """Return entailment scores for each pair, or zero-shot label scores for each text."""
    began = perf_counter()
    pool: ModelPool = request.app.state.pool
    loaded = pool.resolve("nli", body.model)
    model: SequenceClassifier = loaded.model

    if body.pairs is not None:
        _check_size([(pair.id, pair.premise + pair.hypothesis) for pair in body.pairs], 1)
        probabilities, timings = await loaded.submit(
            model.predict, [(pair.premise, pair.hypothesis) for pair in body.pairs]
        )
        results = [
            NliResult(id=pair.id, scores=dict(zip(model.labels, row.tolist(), strict=True)))
            for pair, row in zip(body.pairs, probabilities, strict=True)
        ]
        mode: Literal["pairs", "zero_shot"] = "pairs"
    else:
        # The validator guarantees both are set in zero-shot mode.
        texts, zero_shot = cast(list[TextItem], body.texts), cast(ZeroShot, body.zero_shot)
        _check_size([(item.id, item.text) for item in texts], len(zero_shot.labels))
        against, entailment = _entailment_columns(model)
        hypotheses = [zero_shot.hypothesis_template.format(label) for label in zero_shot.labels]
        logits, timings = await loaded.submit(
            model.logits, [(item.text, hypothesis) for item in texts for hypothesis in hypotheses]
        )
        logits = logits.reshape(len(texts), len(hypotheses), -1)
        if zero_shot.multi_label:
            scores = softmax(logits[..., [against, entailment]])[..., 1]
        else:
            scores = softmax(logits[..., entailment], axis=1)
        results = [
            NliResult(id=item.id, scores=dict(zip(zero_shot.labels, row.tolist(), strict=True)))
            for item, row in zip(texts, scores, strict=True)
        ]
        mode = "zero_shot"

    return NliResponse(
        model=loaded.variant.id,
        model_sha256=loaded.meta.sha256,
        mode=mode,
        results=results,
        timings_ms=text_timings(timings, began),
    )


def _check_size(items: list[tuple[str, str]], pairs_per_item: int) -> None:
    if len(items) * pairs_per_item > MAX_TEXTS:
        raise HTTPException(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            f"at most {MAX_TEXTS} pairs per request (texts times labels in zero-shot mode)",
        )
    if too_long := [item_id for item_id, text in items if len(text) > MAX_TEXT_CHARS]:
        raise HTTPException(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            f"items over {MAX_TEXT_CHARS} characters: {', '.join(too_long)}",
        )


def _entailment_columns(model: SequenceClassifier) -> tuple[int, int]:
    """Return the logit columns for (against entailment, entailment)."""
    names = [label.lower() for label in model.model_labels]
    against = next((names.index(name) for name in _AGAINST_ENTAILMENT if name in names), None)
    if _ENTAILMENT not in names or against is None:
        raise HTTPException(
            HTTPStatus.UNPROCESSABLE_ENTITY, "this model has no entailment output for zero-shot"
        )
    return against, names.index(_ENTAILMENT)
