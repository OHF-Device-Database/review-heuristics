"""What the checks need from the models: one async call per component kind."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .contribution import Image


@dataclass(frozen=True, slots=True)
class LanguageScore:
    """
    A text's language scores after masking.

    ``english`` is None, and ``top`` empty, when masking left nothing to score.
    """

    english: float | None
    top: tuple[tuple[str, float], ...]
    letters: int


class Scorer(Protocol):
    """
    Produces model scores for the checks.

    The model server implements it with its loaded models; the benchmark with a score cache, so
    both run the same rules over the same numbers. Each method returns one result per input, in
    order.
    """

    async def toxicity(self, model: str, texts: list[str]) -> list[dict[str, float]]:
        """Return each text's score per toxicity head."""

    async def language(
        self, model: str, texts: list[str], vocabulary: frozenset[str]
    ) -> list[LanguageScore]:
        """Return each text's language scores, after masking ``vocabulary`` and names."""

    async def nli(self, model: str, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """Return each (premise, hypothesis) pair's probability per NLI label."""

    async def image_classes(
        self, model: str, images: list[Image], classes: dict[str, tuple[str, ...]]
    ) -> list[dict[str, float]]:
        """Return each image's zero-shot probability per class, summing to 1."""

    async def obscenity(self, model: str, images: list[Image]) -> list[dict[str, float]]:
        """
        Return each image's explicitness signals.

        The model's own labels, ``at_least_<level>`` for each graded level, and ``nsfw``.
        """
