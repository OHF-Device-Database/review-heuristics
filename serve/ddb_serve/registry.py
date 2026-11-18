"""The model registry, and the metadata export writes next to each variant's files."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ID_PATTERN = re.compile(r"^(?P<task>[a-z0-9-]+)/(?P<name>[a-z0-9.-]+)@(?P<precision>[a-z0-9]+)$")
_HASH_CHUNK = 1 << 20


class RegistryError(Exception):
    """The registry or a variant's files are missing, malformed, or fail verification."""


@dataclass(frozen=True, slots=True)
class Runtime:
    """Execution settings for one variant."""

    intra_op_threads: int = 1
    max_concurrency: int = 1
    max_queue: int = 32
    batch_size: int = 32
    # ONNX Runtime keeps the memory a run allocated for reuse unless this is off. Off trades
    # some speed for memory that returns to the baseline after each request.
    memory_arena: bool = True


@dataclass(frozen=True, slots=True)
class Variant:
    """One model at one precision, as listed in ``models.json``."""

    id: str
    task: str
    name: str
    precision: str
    source: str
    license: str
    labels: tuple[str, ...] | None
    runtime: Runtime

    @property
    def directory(self) -> Path:
        """Return where this variant's files live, relative to the models root."""
        return Path(self.task, self.name, self.precision)


@dataclass(frozen=True, slots=True)
class ModelMeta:
    """What export recorded about a variant's files."""

    sha256: str
    files: dict[str, str]
    data: dict[str, Any]

    def require(self, key: str) -> Any:
        """Return a recorded value; raises RegistryError when export did not record it."""
        try:
            return self.data[key]
        except KeyError as err:
            raise RegistryError(f"meta.json has no {key!r}; re-export this variant") from err


def load_registry(path: Path) -> list[Variant]:
    """
    Return every variant listed in ``models.json``.

    Raises RegistryError when the file is malformed or lists an id twice.
    """
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as err:
        raise RegistryError(f"cannot read registry {path}: {err}") from err
    entries = document.get("models") if isinstance(document, dict) else None
    if not isinstance(entries, list):
        raise RegistryError(f"registry {path} must have a 'models' list")

    variants = [_parse_variant(entry) for entry in entries]
    ids = [variant.id for variant in variants]
    if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
        raise RegistryError(f"registry lists these ids more than once: {', '.join(duplicates)}")
    return variants


def select(variants: list[Variant], patterns: str) -> list[Variant]:
    """
    Return the variants matching any of the comma-separated glob patterns.

    Raises RegistryError when a pattern matches nothing, since that is almost always a typo.
    """
    globs = [glob.strip() for glob in patterns.split(",") if glob.strip()]
    if not globs:
        raise RegistryError("no model patterns given")
    for glob in globs:
        if not any(fnmatch.fnmatchcase(variant.id, glob) for variant in variants):
            raise RegistryError(f"pattern {glob!r} matches no variant in the registry")
    return [v for v in variants if any(fnmatch.fnmatchcase(v.id, glob) for glob in globs)]


def load_meta(directory: Path) -> ModelMeta:
    """
    Return the metadata export wrote for the variant in ``directory``.

    ``sha256`` identifies the variant's weights as a whole; ``files`` maps each file to its
    own SHA-256. Raises RegistryError when the variant has not been exported or the metadata
    is incomplete.
    """
    path = directory / "meta.json"
    try:
        raw = json.loads(path.read_text())
        return ModelMeta(sha256=raw.pop("sha256"), files=dict(raw.pop("files")), data=raw)
    except FileNotFoundError as err:
        raise RegistryError(f"{path} not found; export this variant first") from err
    except (OSError, ValueError, KeyError, TypeError) as err:
        raise RegistryError(f"cannot read {path}: {err}") from err


def verify(directory: Path, meta: ModelMeta) -> None:
    """Raise RegistryError unless every file's SHA-256 matches the one export recorded."""
    for name, expected in meta.files.items():
        digest = hashlib.sha256()
        try:
            with (directory / name).open("rb") as file:
                while chunk := file.read(_HASH_CHUNK):
                    digest.update(chunk)
        except OSError as err:
            raise RegistryError(f"cannot read {directory / name}: {err}") from err
        if digest.hexdigest() != expected:
            raise RegistryError(
                f"{directory / name} has SHA-256 {digest.hexdigest()}, but export recorded "
                f"{expected}"
            )


def _parse_variant(entry: Any) -> Variant:
    if not isinstance(entry, dict) or "id" not in entry:
        raise RegistryError(f"registry entry has no id: {entry!r}")
    match = _ID_PATTERN.match(str(entry["id"]))
    if match is None:
        raise RegistryError(f"variant id {entry['id']!r} is not <task>/<name>@<precision>")
    try:
        runtime = Runtime(**entry.get("runtime", {}))
        labels = entry.get("labels")
        return Variant(
            id=entry["id"],
            task=match["task"],
            name=match["name"],
            precision=match["precision"],
            source=entry["source"],
            license=entry["license"],
            labels=tuple(labels) if labels is not None else None,
            runtime=runtime,
        )
    except (KeyError, TypeError) as err:
        raise RegistryError(f"variant {entry['id']} is incomplete: {err}") from err
