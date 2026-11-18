"""A contribution, and the units its text is checked in."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

FIELDS = ("name", "summary", "getting_going", "quirks")

# A sentence ends at . ! or ?, optionally followed by closing quotes or brackets, then space and
# a capital, a digit or an opening quote.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
_ABBREVIATIONS = frozenset({"e.g.", "i.e.", "etc.", "approx.", "vs.", "no.", "fig.", "incl."})
_CODE_FENCE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


@dataclass(frozen=True, slots=True)
class Image:
    """An uploaded image's bytes, with the id the result refers to it by."""

    id: str
    data: bytes


@dataclass(frozen=True, slots=True)
class Contribution:
    """
    One submission: the device's structured data, its text fields and its images.

    ``getting_going`` and ``quirks`` may be a list of steps or bullets, or one string with one
    per line.
    """

    device: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, str | list[str] | None] = field(default_factory=dict)
    images: tuple[Image, ...] = ()


@dataclass(frozen=True, slots=True)
class Unit:
    """
    One piece of text that is checked on its own.

    ``span`` is the character range in the field's text; for a field given as a list, in that
    list item, whose position is ``item``.
    """

    field: str
    index: int
    text: str
    span: tuple[int, int]
    item: int | None = None


def split_units(contribution: Contribution) -> list[Unit]:
    """
    Return the units of every text field, in field order.

    The name is one unit; the summary is split into sentences; each getting-going step and
    each quirk is a unit. Code blocks are kept whole.
    """
    units: list[Unit] = []
    for name in FIELDS:
        value = contribution.fields.get(name)
        if not value:
            continue
        if name == "name":
            pieces = [(value.strip(), _strip_span(value), None)] if isinstance(value, str) else []
        elif name == "summary" and isinstance(value, str):
            pieces = [(text, span, None) for text, span in sentences(value)]
        elif isinstance(value, list):
            pieces = [(item.strip(), _strip_span(item), i) for i, item in enumerate(value)]
        else:
            pieces = [(text, span, None) for text, span in _lines(value)]
        units.extend(
            Unit(name, index, text, span, item)
            for index, (text, span, item) in enumerate(p for p in pieces if p[0])
        )
    return units


def sentences(text: str) -> list[tuple[str, tuple[int, int]]]:
    """Return each sentence in ``text`` with its character span; code blocks stay whole."""
    found = []
    position = 0
    for block in _CODE_FENCE.finditer(text):
        found.extend(_prose_sentences(text, position, block.start()))
        found.append((block.group().strip(), (block.start(), block.end())))
        position = block.end()
    found.extend(_prose_sentences(text, position, len(text)))
    return [(sentence, span) for sentence, span in found if sentence]


def _prose_sentences(text: str, start: int, end: int) -> list[tuple[str, tuple[int, int]]]:
    found = []
    sentence_start = start
    for boundary in _SENTENCE_BOUNDARY.finditer(text, start, end):
        before = text[sentence_start : boundary.start()].split()
        last_word = before[-1].lower() if before else ""
        # "e.g." and single initials ("A.") end with a period but rarely end a sentence.
        if last_word in _ABBREVIATIONS or re.fullmatch(r"[a-z]\.", last_word):
            continue
        found.append(_trimmed(text, sentence_start, boundary.start()))
        sentence_start = boundary.end()
    found.append(_trimmed(text, sentence_start, end))
    return found


def _lines(text: str) -> list[tuple[str, tuple[int, int]]]:
    found = []
    offset = 0
    for line in text.splitlines(keepends=True):
        marker = _LIST_MARKER.match(line)
        start = offset + (marker.end() if marker else 0)
        found.append(_trimmed(text, start, offset + len(line)))
        offset += len(line)
    return found


def _trimmed(text: str, start: int, end: int) -> tuple[str, tuple[int, int]]:
    piece = text[start:end]
    leading = len(piece) - len(piece.lstrip())
    stripped = piece.strip()
    return stripped, (start + leading, start + leading + len(stripped))


def _strip_span(text: str) -> tuple[int, int]:
    return _trimmed(text, 0, len(text))[1]
