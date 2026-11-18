"""Language identification with a fastText model, after masking brand-like tokens."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import fasttext

from .registry import ModelMeta, Variant

_LABEL_PREFIX = "__label__"
_SENTENCE_END = (".", "!", "?")
_SURROUNDING_PUNCTUATION = (
    "\"'()[]{}<>,;:.!?«»“”\N{LEFT SINGLE QUOTATION MARK}\N{RIGHT SINGLE QUOTATION MARK}"
)
# Tokens with this many capitals read as codes or brands (SNSW, ZHA), not words.
_CODE_CAPITALS = 2


class FastTextModel(Protocol):
    """The part of a loaded fastText model this module uses."""

    def predict(self, text: str, k: int = 1, threshold: float = 0.0) -> tuple[Any, Any]:
        """Return the top ``k`` labels and their probabilities."""


@dataclass(frozen=True, slots=True)
class Identification:
    """The languages a text scored as, after masking."""

    scored_text: str
    letters: int
    top: list[tuple[str, float]]
    english: float | None


class LanguageIdentifier:
    """Scores texts with a fastText language-ID model."""

    labels: tuple[str, ...] = ()

    def __init__(self, model: FastTextModel, english: str) -> None:
        """Wrap a loaded model; ``english`` is its label for English, without the prefix."""
        self._model = model
        self._english = english

    @classmethod
    def load(cls, directory: Path, meta: ModelMeta, variant: Variant) -> LanguageIdentifier:
        """Return the identifier for the exported variant in ``directory``."""
        model = fasttext.load_model(str(directory / meta.require("file")))
        return cls(model, meta.require("english"))

    def warm_up(self) -> None:
        """Run one prediction so the first real request pays no one-time costs."""
        self.identify(["warm-up"], top_k=1, masked=False, vocabulary=frozenset())

    def identify(
        self, texts: list[str], *, top_k: int, masked: bool, vocabulary: frozenset[str]
    ) -> list[Identification]:
        """
        Return the top ``top_k`` languages for each text, and its probability of being English.

        With ``masked``, tokens that read as names, codes or terms in ``vocabulary`` are removed
        first, since they say little about the language around them. A text with no words left
        gets no prediction.
        """
        results = []
        for text in texts:
            scored = mask(text, vocabulary) if masked else " ".join(text.split())
            if not scored:
                results.append(Identification(scored, 0, [], None))
                continue
            labels, probabilities = self._model.predict(scored, k=-1, threshold=0.0)
            ranked = [
                (label.removeprefix(_LABEL_PREFIX), float(probability))
                for label, probability in zip(labels, probabilities, strict=True)
            ]
            results.append(
                Identification(
                    scored_text=scored,
                    letters=sum(character.isalpha() for character in scored),
                    top=ranked[:top_k],
                    english=next((p for label, p in ranked if label == self._english), 0.0),
                )
            )
        return results


def mask(text: str, vocabulary: frozenset[str]) -> str:
    """
    Return ``text`` without the tokens that say little about its language.

    Removed: tokens containing a digit, tokens with two or more capitals (SNSW, ZHA, iRobot is
    kept), capitalised words that don't start a sentence, and words in ``vocabulary``
    (compared case-insensitively, so pass them lowercased).
    """
    kept = []
    starts_sentence = True
    for token in text.split():
        word = token.strip(_SURROUNDING_PUNCTUATION)
        drop = (
            any(character.isdigit() for character in word)
            or sum(character.isupper() for character in word) >= _CODE_CAPITALS
            or (len(word) > 1 and word[0].isupper() and not starts_sentence)
            or word.lower() in vocabulary
        )
        if not drop:
            kept.append(token)
        starts_sentence = token.endswith(_SENTENCE_END)
    return " ".join(kept)
