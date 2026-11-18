"""
Known-good and known-bad contributions, run through the checks with cached model scores.

Model scores are cached per input, so a change to thresholds, actions or rule fields is compared
without a model server. A change that alters model inputs (models, prompts, mask words or fact
templates) needs fresh scores from a server first.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .config import Config
from .contribution import Contribution, Image
from .engine import ACTIONS, check
from .scorer import LanguageScore, Scorer

_TEXT_BATCH = 128
_IMAGE_BATCH = 8
_LANGUAGE_TOP_K = 5


class MissingScoresError(LookupError):
    """The cache lacks scores the configuration needs, and no model server was given."""


@dataclass(frozen=True, slots=True)
class Case:
    """
    One contribution with the outcome each rule should have on it.

    ``expect`` maps rule ids to ``pass``, ``warn`` or ``reject``; rules not listed aren't judged
    on this case.
    """

    id: str
    contribution: Contribution
    expect: dict[str, str]
    why: str = ""


@dataclass(slots=True)
class Tally:
    """How one configuration did on the judged (case, rule) pairs."""

    good: int = 0
    good_rejected: int = 0
    good_warned: int = 0
    bad: int = 0
    bad_caught: int = 0

    def add(self, expected: str, got: str) -> None:
        """Count one judged pair."""
        if expected == "pass":
            self.good += 1
            self.good_rejected += got == "reject"
            self.good_warned += got == "warn"
        else:
            self.bad += 1
            self.bad_caught += got != "pass"


@dataclass(slots=True)
class Run:
    """Each case's action per rule under one configuration, with tallies."""

    config_version: str
    actions: dict[str, dict[str, str]]
    total: Tally = field(default_factory=Tally)
    per_rule: dict[str, Tally] = field(default_factory=dict)


def load_cases(paths: list[Path], root: Path) -> list[Case]:
    """
    Return the cases in every JSON Lines file in ``paths``.

    Image paths in a case are relative to ``root``. Raises ValueError on a duplicate case id.
    """
    cases = []
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            body = raw["contribution"]
            images = tuple(
                Image(image["id"], (root / image["path"]).read_bytes())
                for image in body.get("images", [])
            )
            unknown = set(raw["expect"].values()) - set(ACTIONS)
            if unknown:
                raise ValueError(f"{path}:{number}: unknown expected actions {sorted(unknown)}")
            cases.append(
                Case(
                    raw["id"],
                    Contribution(body.get("device", {}), body.get("fields", {}), images),
                    raw["expect"],
                    raw.get("why", ""),
                )
            )
    ids = [case.id for case in cases]
    if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"duplicate case ids: {duplicates}")
    return cases


async def run(cases: list[Case], config: Config, scorer: Scorer) -> Run:
    """Return every case's action per configured rule, tallied against the expectations."""
    results = await asyncio.gather(*(check(case.contribution, config, scorer) for case in cases))
    configured = [rule.id for rule in config.rules]
    outcome = Run(config.version, {})
    for case, result in zip(cases, results, strict=True):
        actions = dict.fromkeys(configured, "pass")
        for hit in result.hits:
            if ACTIONS.index(hit.action) > ACTIONS.index(actions[hit.rule]):
                actions[hit.rule] = hit.action
        outcome.actions[case.id] = actions
        for rule, expected in case.expect.items():
            if rule in actions:
                outcome.total.add(expected, actions[rule])
                outcome.per_rule.setdefault(rule, Tally()).add(expected, actions[rule])
    return outcome


def compare(cases: list[Case], base: Run, head: Run) -> tuple[str, bool]:
    """
    Return a Markdown comparison of two runs, and whether ``head`` regressed.

    A regression is more good (case, rule) pairs rejected than before: the change contributors
    feel most. Everything else is shown for reviewers to judge.
    """

    def row(label: str, before: Tally | None, after: Tally | None) -> str:
        cells = []
        for metric, total in (
            ("good_rejected", "good"),
            ("good_warned", "good"),
            ("bad_caught", "bad"),
        ):
            values = [
                f"{getattr(t, metric)}/{getattr(t, total)}" if t else "n/a" for t in (before, after)
            ]
            cells.append(f"{values[0]} → {values[1]}" if values[0] != values[1] else values[0])
        return f"| {label} | " + " | ".join(cells) + " |"

    rules = sorted(set(base.per_rule) | set(head.per_rule))
    lines = [
        f"Config {base.config_version} → {head.config_version}, {len(cases)} cases.",
        "",
        "| Rule | Good, wrongly rejected | Good, wrongly warned | Bad, caught |",
        "| --- | --- | --- | --- |",
        row("**All rules**", base.total, head.total),
        *(row(rule, base.per_rule.get(rule), head.per_rule.get(rule)) for rule in rules),
    ]
    flips = []
    for case in cases:
        before, after = base.actions.get(case.id, {}), head.actions.get(case.id, {})
        for rule in sorted(set(before) | set(after)):
            old, new = before.get(rule, "n/a"), after.get(rule, "n/a")
            if old != new:
                expected = case.expect.get(rule, "not judged")
                flips.append(f"| {case.id} | {rule} | {old} → {new} | {expected} |")
    lines += ["", f"**Changed decisions: {len(flips)}**", ""]
    if flips:
        lines += ["| Case | Rule | Decision | Expected |", "| --- | --- | --- | --- |", *flips]
    regressed = head.total.good_rejected > base.total.good_rejected
    if regressed:
        lines += ["", "**More good contributions are rejected than before.**"]
    return "\n".join(lines) + "\n", regressed


def summary(head: Run) -> str:
    """Return a Markdown table of one run's results per rule."""

    def cell(t: Tally) -> str:
        return f"| {t.good_rejected}/{t.good} | {t.good_warned}/{t.good} | {t.bad_caught}/{t.bad} |"

    return (
        "\n".join(
            [
                f"Config {head.config_version}.",
                "",
                "| Rule | Good, wrongly rejected | Good, wrongly warned | Bad, caught |",
                "| --- | --- | --- | --- |",
                f"| **All rules** {cell(head.total)}",
                *(f"| {rule} {cell(t)}" for rule, t in sorted(head.per_rule.items())),
            ]
        )
        + "\n"
    )


class ScoreCache:
    """Model scores per input, stored as one JSON Lines file per model."""

    def __init__(self, entries: dict[str, dict[str, Any]] | None = None) -> None:
        """Start with ``entries``: model id to {input key: result}."""
        self.entries: dict[str, dict[str, Any]] = entries or {}
        self.used: dict[str, set[str]] = {}

    @classmethod
    def read(cls, directory: Path) -> ScoreCache:
        """Return the cache stored in ``directory``; empty when there is none."""
        cache = cls()
        cache.merge({path.stem: path.read_text() for path in sorted(directory.glob("*.jsonl"))})
        return cache

    def merge(self, files: dict[str, str]) -> None:
        """Add entries from file contents, keyed by file stem; existing entries win."""
        for stem, text in files.items():
            model = _model_id(stem)
            target = self.entries.setdefault(model, {})
            for line in text.splitlines():
                if line.strip():
                    record = json.loads(line)
                    target.setdefault(record["key"], record["result"])

    def write(self, directory: Path) -> None:
        """
        Write the entries the last runs used, one file per model, sorted for stable diffs.

        Entries no run needed are dropped, so the cache only holds what the configuration uses.
        """
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob("*.jsonl"):
            stale.unlink()
        for model, keys in sorted(self.used.items()):
            lines = [
                json.dumps({"key": key, "result": self.entries[model][key]}, sort_keys=True)
                for key in sorted(keys)
            ]
            (directory / f"{_file_stem(model)}.jsonl").write_text("\n".join(lines) + "\n")


class CachingScorer:
    """
    A Scorer that answers from a ScoreCache, asking ``live`` only for what's missing.

    Without ``live``, a missing score raises MissingScoresError naming what to re-score.
    """

    def __init__(self, cache: ScoreCache, live: Scorer | None = None) -> None:
        """Answer from ``cache``, filling gaps from ``live`` when given."""
        self._cache = cache
        self._live = live
        self._lock = asyncio.Lock()

    async def toxicity(self, model: str, texts: list[str]) -> list[dict[str, float]]:
        """Return each text's score per toxicity head."""
        return await self._cached(model, "toxicity", texts, lambda s, m, todo: s.toxicity(m, todo))

    async def language(
        self, model: str, texts: list[str], vocabulary: frozenset[str]
    ) -> list[LanguageScore]:
        """Return each text's language scores, after masking."""
        words = sorted(vocabulary)
        found = await self._cached(
            model,
            ["language", words],
            texts,
            lambda s, m, todo: _language_records(s, m, todo, vocabulary),
        )
        return [
            LanguageScore(r["english"], tuple(map(tuple, r["top"])), r["letters"]) for r in found
        ]

    async def nli(self, model: str, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """Return each pair's probability per NLI label."""
        return await self._cached(
            model,
            "nli",
            [list(pair) for pair in pairs],
            lambda s, m, todo: s.nli(m, [tuple(pair) for pair in todo]),
        )

    async def image_classes(
        self, model: str, images: list[Image], classes: dict[str, tuple[str, ...]]
    ) -> list[dict[str, float]]:
        """Return each image's zero-shot probability per class."""
        by_hash = {_sha256(image.data): image for image in images}
        prompts = {name: list(p) for name, p in classes.items()}
        return await self._cached(
            model,
            ["image_classes", prompts],
            list(by_hash),
            lambda s, m, todo: s.image_classes(m, [by_hash[h] for h in todo], classes),
            inputs=[_sha256(image.data) for image in images],
        )

    async def obscenity(self, model: str, images: list[Image]) -> list[dict[str, float]]:
        """Return each image's explicitness signals."""
        by_hash = {_sha256(image.data): image for image in images}
        return await self._cached(
            model,
            "obscenity",
            list(by_hash),
            lambda s, m, todo: s.obscenity(m, [by_hash[h] for h in todo]),
            inputs=[_sha256(image.data) for image in images],
        )

    async def _cached(
        self,
        model: str,
        kind: Any,
        unique: list[Any],
        fetch: Callable[[Scorer, str, list[Any]], Any],
        inputs: list[Any] | None = None,
    ) -> list[Any]:
        inputs = unique if inputs is None else inputs
        keys = [_key(kind, item) for item in inputs]
        async with self._lock:
            known = self._cache.entries.setdefault(model, {})
            missing = list(
                {
                    key: item for key, item in zip(keys, inputs, strict=True) if key not in known
                }.items()
            )
            if missing:
                if self._live is None:
                    raise MissingScoresError(
                        f"{len(missing)} {model} scores are not cached; run `make scores` "
                        "against a model server"
                    )
                results = await fetch(self._live, model, [item for _, item in missing])
                for (key, _), result in zip(missing, results, strict=True):
                    known[key] = result
            self._cache.used.setdefault(model, set()).update(keys)
            return [known[key] for key in keys]


class HttpScorer:
    """A Scorer that calls a running model server's component endpoints."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        """Send requests with ``client``, whose base URL is the server."""
        self._client = client

    async def toxicity(self, model: str, texts: list[str]) -> list[dict[str, float]]:
        """Return each text's score per toxicity head."""
        results = await self._texts("/v1/toxicity", {"model": model}, texts)
        return [r["scores"] for r in results]

    async def language(
        self, model: str, texts: list[str], vocabulary: frozenset[str]
    ) -> list[LanguageScore]:
        """Return each text's language scores, after masking."""
        body = {
            "model": model,
            "top_k": _LANGUAGE_TOP_K,
            "mask": True,
            "vocabulary": sorted(vocabulary),
        }
        results = await self._texts("/v1/texts/language", body, texts)
        return [
            LanguageScore(
                r["english"],
                tuple((t["language"], t["probability"]) for t in r["top"]),
                r["letters"],
            )
            for r in results
        ]

    async def nli(self, model: str, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """Return each pair's probability per NLI label."""
        results = []
        for start in range(0, len(pairs), _TEXT_BATCH):
            chunk = pairs[start : start + _TEXT_BATCH]
            body = {
                "model": model,
                "pairs": [
                    {"id": str(i), "premise": p, "hypothesis": h} for i, (p, h) in enumerate(chunk)
                ],
            }
            results.extend(await self._post("/v1/texts/nli", body))
        return [r["scores"] for r in results]

    async def image_classes(
        self, model: str, images: list[Image], classes: dict[str, tuple[str, ...]]
    ) -> list[dict[str, float]]:
        """Return each image's zero-shot probability per class."""
        body = {"model": model, "classes": {name: list(p) for name, p in classes.items()}}
        return [r["scores"] for r in await self._images("/v1/images/classify", body, images)]

    async def obscenity(self, model: str, images: list[Image]) -> list[dict[str, float]]:
        """Return each image's explicitness signals."""
        results = await self._images("/v1/obscenity", {"model": model}, images)
        return [
            {
                **r["scores"],
                **{f"at_least_{level}": p for level, p in (r["at_least"] or {}).items()},
                "nsfw": r["nsfw"],
            }
            for r in results
        ]

    async def _texts(self, route: str, body: dict[str, Any], texts: list[str]) -> list[Any]:
        results = []
        for start in range(0, len(texts), _TEXT_BATCH):
            chunk = texts[start : start + _TEXT_BATCH]
            items = [{"id": str(i), "text": text} for i, text in enumerate(chunk)]
            results.extend(await self._post(route, {**body, "texts": items}))
        return results

    async def _images(self, route: str, body: dict[str, Any], images: list[Image]) -> list[Any]:
        results = []
        for start in range(0, len(images), _IMAGE_BATCH):
            chunk = images[start : start + _IMAGE_BATCH]
            items = [
                {"id": str(i), "data": base64.b64encode(image.data).decode()}
                for i, image in enumerate(chunk)
            ]
            results.extend(await self._post(route, {**body, "images": items}))
        return results

    async def _post(self, route: str, body: dict[str, Any]) -> list[Any]:
        response = await self._client.post(route, json=body)
        response.raise_for_status()
        return response.json()["results"]


async def _language_records(
    scorer: Scorer, model: str, texts: list[str], vocabulary: frozenset[str]
) -> list[dict[str, Any]]:
    scores = await scorer.language(model, texts, vocabulary)
    return [
        {"english": s.english, "top": [list(t) for t in s.top], "letters": s.letters}
        for s in scores
    ]


def _key(kind: Any, item: Any) -> str:
    return _sha256(json.dumps([kind, item], sort_keys=True, ensure_ascii=False).encode())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_stem(model: str) -> str:
    return model.replace("/", "--").replace("@", "__")


def _model_id(stem: str) -> str:
    return stem.replace("--", "/", 1).replace("__", "@", 1)
