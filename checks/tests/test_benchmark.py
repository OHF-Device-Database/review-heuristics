"""The benchmark: cached scores, case loading and comparison."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ddb_checks import benchmark
from ddb_checks import config as config_module
from ddb_checks.benchmark import CachingScorer, Case, MissingScoresError, ScoreCache
from ddb_checks.contribution import Contribution

from .test_checks import CONFIG

MODEL = "toxicity/detoxify-unbiased@int8"


class CountingScorer:
    """Toxicity scores from a fixed table, counting how many texts it was asked for."""

    def __init__(self, scores: dict[str, float]) -> None:
        """Answer with ``scores`` per text."""
        self.scores = scores
        self.asked: list[str] = []

    async def toxicity(self, model: str, texts: list[str]) -> list[dict[str, float]]:
        self.asked.extend(texts)
        return [{"toxicity": self.scores[t], "insult": 0.0} for t in texts]


def _case(case_id: str, text: str, expected: str) -> Case:
    return Case(case_id, Contribution(fields={"summary": text}), {"abuse_spam": expected})


def _config_with_warn_at(tmp_path: Path, warn: float) -> config_module.Config:
    directory = tmp_path / f"config-{warn}"
    directory.mkdir()
    for path in CONFIG.rglob("*.json"):
        (directory / path.relative_to(CONFIG)).parent.mkdir(parents=True, exist_ok=True)
        (directory / path.relative_to(CONFIG)).write_text(path.read_text())
    checks = json.loads((directory / "checks.json").read_text())
    rule = next(r for r in checks["rules"] if r["id"] == "abuse_spam")
    rule["thresholds"] = {"toxicity": {"warn": warn}}
    checks["rules"] = [rule]
    (directory / "checks.json").write_text(json.dumps(checks))
    return config_module.load(directory)


def test_scores_are_fetched_once_then_served_from_the_cache(tmp_path: Path) -> None:
    live = CountingScorer({"fine": 0.1, "rude": 0.95})
    cache = ScoreCache()

    first = asyncio.run(CachingScorer(cache, live).toxicity(MODEL, ["fine", "rude", "fine"]))
    cache.write(tmp_path)
    again = asyncio.run(CachingScorer(ScoreCache.read(tmp_path)).toxicity(MODEL, ["rude", "fine"]))

    assert live.asked == ["fine", "rude"]
    assert [s["toxicity"] for s in first] == [0.1, 0.95, 0.1]
    assert [s["toxicity"] for s in again] == [0.95, 0.1]
    assert [p.name for p in tmp_path.iterdir()] == ["toxicity--detoxify-unbiased__int8.jsonl"]


def test_missing_scores_without_a_server_say_what_to_do() -> None:
    with pytest.raises(MissingScoresError, match="make scores"):
        asyncio.run(CachingScorer(ScoreCache()).toxicity(MODEL, ["new text"]))


def test_compare_flags_more_good_cases_rejected_as_a_regression(tmp_path: Path) -> None:
    cases = [_case("good", "fine", "pass"), _case("bad", "rude", "warn")]
    scorer = CachingScorer(ScoreCache(), CountingScorer({"fine": 0.6, "rude": 0.95}))

    strict = asyncio.run(benchmark.run(cases, _config_with_warn_at(tmp_path, 0.5), scorer))
    lenient = asyncio.run(benchmark.run(cases, _config_with_warn_at(tmp_path, 0.9), scorer))
    text, regressed = benchmark.compare(cases, strict, lenient)

    assert (strict.total.good_warned, lenient.total.good_warned) == (1, 0)
    assert lenient.total.bad_caught == 1
    assert not regressed
    assert "| good | abuse_spam | warn → pass | pass |" in text


def test_cases_load_images_relative_to_the_root(tmp_path: Path) -> None:
    (tmp_path / "img.png").write_bytes(b"png bytes")
    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text(
        json.dumps(
            {
                "id": "one",
                "contribution": {"images": [{"id": "photo", "path": "img.png"}]},
                "expect": {"obscenity": "pass"},
            }
        )
        + "\n"
    )

    (case,) = benchmark.load_cases([cases_file], tmp_path)

    assert case.contribution.images[0].data == b"png bytes"
    with pytest.raises(ValueError, match="duplicate"):
        benchmark.load_cases([cases_file, cases_file], tmp_path)
