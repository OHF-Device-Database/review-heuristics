"""Registry lookups the harness needs: a variant's source, and where its files go."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ID_PATTERN = re.compile(r"^(?P<task>[a-z0-9-]+)/(?P<name>[a-z0-9.-]+)@(?P<precision>[a-z0-9]+)$")


@dataclass(frozen=True, slots=True)
class VariantId:
    """A parsed ``<task>/<name>@<precision>`` id."""

    task: str
    name: str
    precision: str

    @classmethod
    def parse(cls, variant_id: str) -> VariantId:
        """Return the parsed id; raises ValueError when it is malformed."""
        match = _ID_PATTERN.match(variant_id)
        if match is None:
            raise ValueError(f"variant id {variant_id!r} is not <task>/<name>@<precision>")
        return cls(match["task"], match["name"], match["precision"])

    def __str__(self) -> str:
        """Return the id in registry form."""
        return f"{self.task}/{self.name}@{self.precision}"

    @property
    def directory(self) -> Path:
        """Return where this variant's files live, relative to the models root."""
        return Path(self.task, self.name, self.precision)

    def with_precision(self, precision: str) -> VariantId:
        """Return the same model at another precision."""
        return VariantId(self.task, self.name, precision)


def find(registry_path: Path, variant_id: str) -> dict[str, Any]:
    """Return the registry entry for ``variant_id``; raises KeyError when it isn't listed."""
    for entry in json.loads(registry_path.read_text())["models"]:
        if entry.get("id") == variant_id:
            return entry
    raise KeyError(f"{variant_id} is not in {registry_path}")


def split_source(source: str) -> tuple[str, str]:
    """
    Return the Hugging Face repo and revision from ``repo@revision``.

    Raises ValueError when the revision is missing, since an unpinned source can change
    under a benchmark.
    """
    repo, _, revision = source.partition("@")
    if not revision:
        raise ValueError(f"source {source!r} must pin a revision as repo@revision")
    return repo, revision
