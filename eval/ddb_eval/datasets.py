"""Dataset manifests: JSON Lines files whose hash is the dataset's version."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# Long enough to exceed every text model's token limit, so truncation is exercised.
_LONG_TEXT = "The relay reports power, voltage and current every ten seconds. " * 120


@dataclass(frozen=True, slots=True)
class TextItem:
    """One text from a manifest, with its label and category when the manifest has them."""

    id: str
    text: str
    label: str | None = None
    category: str | None = None


@dataclass(frozen=True, slots=True)
class NliPair:
    """A premise and hypothesis from a manifest."""

    id: str
    premise: str
    hypothesis: str


@dataclass(frozen=True, slots=True)
class TextDataset:
    """A loaded manifest and its version."""

    name: str
    version: str
    items: list[TextItem]


def load_text(path: Path, *, with_long_text: bool = False) -> TextDataset:
    """
    Return the texts in a manifest.

    ``with_long_text`` appends one text over every model's token limit, for parity checks.
    """
    raw = path.read_bytes()
    items = [
        TextItem(record["id"], record["text"], record.get("label"), record.get("category"))
        for record in (json.loads(line) for line in raw.decode().splitlines() if line.strip())
    ]
    if with_long_text:
        items.append(TextItem("synthetic/over-token-limit", _LONG_TEXT))
    return TextDataset(path.name, hashlib.sha256(raw).hexdigest()[:12], items)


def load_pairs(path: Path) -> tuple[str, list[NliPair]]:
    """Return a pair manifest's version and its premise and hypothesis pairs."""
    raw = path.read_bytes()
    pairs = [
        NliPair(record["id"], record["premise"], record["hypothesis"])
        for record in (json.loads(line) for line in raw.decode().splitlines() if line.strip())
    ]
    return hashlib.sha256(raw).hexdigest()[:12], pairs
