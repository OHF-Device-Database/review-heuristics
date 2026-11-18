"""Apply the configured rules to a contribution, using scores from a Scorer."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import Config, Rule
from .contribution import Contribution, Unit, split_units
from .scorer import Scorer

ACTIONS = ("pass", "warn", "reject")
_OUTCOMES = {"pass": "accept", "warn": "warn", "reject": "reject"}


@dataclass(frozen=True, slots=True)
class Evidence:
    """A score that crossed a threshold, or explains why the rule fired."""

    model: str
    signal: str
    score: float
    threshold: float | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class Hit:
    """
    One rule firing on one unit or image.

    ``unit`` indexes the field's units; for images, the image's position. ``span`` is the
    unit's character range (see Unit), and None for images.
    """

    rule: str
    field: str
    unit: int
    text: str | None
    span: tuple[int, int] | None
    item: int | None
    action: str
    message: str
    guideline: str
    evidence: list[Evidence]


@dataclass(frozen=True, slots=True)
class Result:
    """The checks' verdict on a contribution: ``reject`` if any hit rejects, else ``warn``."""

    outcome: str
    hits: list[Hit]
    config_version: str
    config_sha256: str
    rules_run: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return the result as plain JSON-ready data."""
        return asdict(self)


async def check(contribution: Contribution, config: Config, scorer: Scorer) -> Result:
    """
    Return the outcome of every configured rule on ``contribution``.

    Rules run concurrently; each asks the scorer for one batch of scores.
    """
    units = split_units(contribution)
    batches = await asyncio.gather(
        *(
            _EVALUATORS[rule.type](rule, contribution, units, config, scorer)
            for rule in config.rules
        )
    )
    hits = [hit for batch in batches for hit in batch]
    worst = max((ACTIONS.index(hit.action) for hit in hits), default=0)
    return Result(
        outcome=_OUTCOMES[ACTIONS[worst]],
        hits=hits,
        config_version=config.version,
        config_sha256=config.sha256,
        rules_run=[rule.id for rule in config.rules],
    )


def render_facts(device: dict[str, Any], config: Config) -> list[str]:
    """Return the device's structured data as short English sentences, per ``facts.json``."""
    sentences = []
    for key, spec in config.facts.items():
        value = device.get(key)
        if value in (None, "", []):
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if "values" in spec:
                if (sentence := spec["values"].get(str(item))) is not None:
                    sentences.append(sentence)
            else:
                sentences.append(spec["template"].format(value=item))
    return sentences


def _against(thresholds: dict[str, dict[str, float]], scores: dict[str, float], model: str):
    """Return the action and evidence for scores where higher is worse."""
    action, evidence = "pass", []
    for signal, limits in thresholds.items():
        score = scores.get(signal)
        if score is None:
            continue
        for level in ("reject", "warn"):
            limit = limits.get(level)
            if limit is not None and score >= limit:
                evidence.append(Evidence(model, signal, score, limit))
                if ACTIONS.index(level) > ACTIONS.index(action):
                    action = level
                break
    return action, evidence


def _hit(rule: Rule, unit: Unit, action: str, evidence: list[Evidence]) -> Hit:
    return Hit(
        rule.id,
        unit.field,
        unit.index,
        unit.text,
        unit.span,
        unit.item,
        action,
        rule.message,
        rule.guideline,
        evidence,
    )


def _in_fields(rule: Rule, units: list[Unit]) -> list[Unit]:
    return [unit for unit in units if unit.field in rule.fields]


async def _toxicity(
    rule: Rule, contribution: Contribution, units: list[Unit], config: Config, scorer: Scorer
) -> list[Hit]:
    selected = _in_fields(rule, units)
    if not selected:
        return []
    scores = await scorer.toxicity(rule.model, [unit.text for unit in selected])
    hits = []
    for unit, unit_scores in zip(selected, scores, strict=True):
        action, evidence = _against(rule.settings["thresholds"], unit_scores, rule.model)
        if action != "pass":
            hits.append(_hit(rule, unit, action, evidence))
    return hits


async def _language(
    rule: Rule, contribution: Contribution, units: list[Unit], config: Config, scorer: Scorer
) -> list[Hit]:
    selected = _in_fields(rule, units)
    if not selected:
        return []
    settings = rule.settings
    # The device's own names and codes say nothing about the language around them.
    own = [contribution.device.get(key) for key in ("brand", "integration")]
    own += list(contribution.device.get("identifiers", []))
    vocabulary = config.vocabulary | {str(word).lower() for word in own if word}
    scores = await scorer.language(rule.model, [unit.text for unit in selected], vocabulary)
    hits = []
    for unit, score in zip(selected, scores, strict=True):
        if score.english is None or score.letters < settings["min_letters"]:
            continue
        other, other_probability = next(
            ((code, p) for code, p in score.top if code != settings["language"]), ("", 0.0)
        )
        evidence = [
            Evidence(rule.model, settings["language"], score.english, settings["warn_below"]),
            Evidence(rule.model, other or "other", other_probability, detail="top other language"),
        ]
        if (
            score.english < settings["reject_below"]
            and other_probability >= settings["reject_needs_other_at_least"]
        ):
            hits.append(_hit(rule, unit, "reject", evidence))
        elif score.english < settings["warn_below"]:
            hits.append(_hit(rule, unit, "warn", evidence))
    return hits


async def _nli(
    rule: Rule, contribution: Contribution, units: list[Unit], config: Config, scorer: Scorer
) -> list[Hit]:
    selected = _in_fields(rule, units)
    facts = render_facts(contribution.device, config)
    if not selected or not facts:
        return []
    # One fact per premise: a long fact sheet would overflow the model's input, and the
    # strongest single fact is the clearest evidence.
    pairs = [(fact, unit.text) for unit in selected for fact in facts]
    scores = await scorer.nli(rule.model, pairs)
    signal = "contradiction" if rule.type == "nli_contradiction" else "entailment"
    hits = []
    for position, unit in enumerate(selected):
        unit_scores = scores[position * len(facts) : (position + 1) * len(facts)]
        strongest = max(range(len(facts)), key=lambda i: unit_scores[i][signal])
        action, evidence = _against(rule.settings["thresholds"], unit_scores[strongest], rule.model)
        if action != "pass":
            evidence = [
                Evidence(e.model, e.signal, e.score, e.threshold, detail=facts[strongest])
                for e in evidence
            ]
            hits.append(_hit(rule, unit, action, evidence))
    return hits


async def _images(
    rule: Rule, contribution: Contribution, units: list[Unit], config: Config, scorer: Scorer
) -> list[Hit]:
    images = list(contribution.images)
    if not images:
        return []
    if rule.type == "image_class":
        scores = await scorer.image_classes(rule.model, images, config.image_classes)
    else:
        scores = await scorer.obscenity(rule.model, images)
    hits = []
    for index, (image, image_scores) in enumerate(zip(images, scores, strict=True)):
        action, evidence = _against(rule.settings["thresholds"], image_scores, rule.model)
        if action != "pass":
            hits.append(
                Hit(
                    rule.id,
                    "images",
                    index,
                    image.id,
                    None,
                    None,
                    action,
                    rule.message,
                    rule.guideline,
                    evidence,
                )
            )
    return hits


_EVALUATORS: dict[
    str, Callable[[Rule, Contribution, list[Unit], Config, Scorer], Awaitable[list[Hit]]]
] = {
    "toxicity": _toxicity,
    "language": _language,
    "nli_contradiction": _nli,
    "nli_entailment": _nli,
    "image_class": _images,
    "obscenity": _images,
}
