"""The configuration in ``config/``: loaded, validated against its schemas, and cross-checked."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

FILES = ("models.json", "checks.json", "image-classes.json", "language.json", "facts.json")
SCHEMA_DIR = "schemas"
# The signals each rule type can set thresholds on, beyond a model's own labels.
_OBSCENITY_SIGNALS = frozenset({"nsfw", "at_least_low", "at_least_medium", "at_least_high"})
_NLI_SIGNALS = {"nli_contradiction": "contradiction", "nli_entailment": "entailment"}


class ConfigError(ValueError):
    """The configuration is missing, malformed, or inconsistent across files."""


@dataclass(frozen=True, slots=True)
class Rule:
    """
    One rule from ``checks.json``.

    ``settings`` holds the type-specific keys (thresholds and the like) as written in the file.
    """

    id: str
    type: str
    model: str
    fields: tuple[str, ...]
    message: str
    guideline: str
    settings: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Config:
    """
    Everything that decides how contributions are checked.

    ``sha256`` covers every file, so a result can always be traced to the exact configuration.
    """

    version: str
    sha256: str
    models: dict[str, dict[str, Any]]
    rules: tuple[Rule, ...]
    image_classes: dict[str, tuple[str, ...]]
    vocabulary: frozenset[str]
    facts: dict[str, dict[str, Any]]


def load(directory: Path) -> Config:
    """
    Return the configuration in ``directory``.

    Raises ConfigError when a file is missing, fails its schema, or refers to something another
    file doesn't define, such as a rule naming a model that isn't in ``models.json``.
    """
    raw = {name: _read(directory, name) for name in FILES}
    for name, document in raw.items():
        _validate(directory / SCHEMA_DIR / name.replace(".json", ".schema.json"), name, document)

    models = {entry["id"]: entry for entry in raw["models.json"]["models"]}
    if len(models) != len(raw["models.json"]["models"]):
        raise ConfigError("models.json lists a model id more than once")
    image_classes = {
        name: tuple(entry["prompts"])
        for name, entry in raw["image-classes.json"]["classes"].items()
    }
    rules = tuple(
        Rule(
            id=entry["id"],
            type=entry["type"],
            model=entry["model"],
            fields=tuple(entry.get("fields", ())),
            message=entry["message"],
            guideline=entry["guideline"],
            settings=entry,
        )
        for entry in raw["checks.json"]["rules"]
    )
    config = Config(
        version=raw["checks.json"]["version"],
        sha256=_hash(directory),
        models=models,
        rules=rules,
        image_classes=image_classes,
        vocabulary=frozenset(word.lower() for word in raw["language.json"]["mask_words"]),
        facts=raw["facts.json"]["facts"],
    )
    _cross_check(config)
    return config


def _cross_check(config: Config) -> None:
    ids = [rule.id for rule in config.rules]
    if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
        raise ConfigError(f"checks.json defines these rules more than once: {duplicates}")
    for rule in config.rules:
        model = config.models.get(rule.model)
        if model is None:
            raise ConfigError(f"rule {rule.id} uses {rule.model}, which models.json doesn't list")
        task = rule.model.split("/", 1)[0]
        signals = set(rule.settings.get("thresholds", {}))
        if rule.type == "toxicity":
            known = set(model.get("labels", ()))
        elif rule.type == "obscenity":
            known = _OBSCENITY_SIGNALS | set(model.get("labels", ()))
        elif rule.type == "image_class":
            known = set(config.image_classes)
        elif rule.type in _NLI_SIGNALS:
            known = {_NLI_SIGNALS[rule.type]}
        else:
            known = signals
        if unknown := sorted(signals - known):
            raise ConfigError(f"rule {rule.id} sets thresholds on unknown signals: {unknown}")
        expected_task = {
            "toxicity": "toxicity",
            "language": "language",
            "nli_contradiction": "nli",
            "nli_entailment": "nli",
            "image_class": "image-embed",
            "obscenity": "obscenity",
        }[rule.type]
        if task != expected_task:
            raise ConfigError(f"rule {rule.id} is a {rule.type} rule but uses a {task} model")


def _read(directory: Path, name: str) -> dict[str, Any]:
    try:
        return json.loads((directory / name).read_text())
    except FileNotFoundError as err:
        raise ConfigError(f"{directory / name} is missing") from err
    except json.JSONDecodeError as err:
        raise ConfigError(f"{name} is not valid JSON: {err}") from err


def _validate(schema_path: Path, name: str, document: dict[str, Any]) -> None:
    try:
        schema = json.loads(schema_path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        raise ConfigError(f"cannot read the schema for {name}: {err}") from err
    errors = sorted(Draft202012Validator(schema).iter_errors(document), key=lambda e: e.path)
    if errors:
        details = "; ".join(
            f"{'/'.join(str(part) for part in error.path) or '(top level)'}: {error.message}"
            for error in errors[:5]
        )
        raise ConfigError(f"{name} doesn't match its schema: {details}")


def _hash(directory: Path) -> str:
    digest = hashlib.sha256()
    paths = [directory / name for name in FILES] + sorted((directory / SCHEMA_DIR).glob("*.json"))
    for path in paths:
        digest.update(f"{path.relative_to(directory)}\n".encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
