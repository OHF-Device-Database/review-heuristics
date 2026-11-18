"""Unit splitting, configuration validation, and the rules against a fake scorer."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from ddb_checks import config as config_module
from ddb_checks.config import ConfigError
from ddb_checks.contribution import Contribution, Image, sentences, split_units
from ddb_checks.engine import check, render_facts
from ddb_checks.scorer import LanguageScore

CONFIG = Path(__file__).parents[2] / "config"


class FakeScorer:
    """Scores from simple keyword rules, so each test can predict the outcome."""

    def __init__(self) -> None:
        """Start with no calls recorded."""
        self.calls: list[str] = []

    async def toxicity(self, model: str, texts: list[str]) -> list[dict[str, float]]:
        self.calls.append("toxicity")
        return [{"toxicity": 0.99 if "idiot" in t else 0.01, "insult": 0.0} for t in texts]

    async def language(self, model, texts, vocabulary):
        self.calls.append("language")
        return [
            LanguageScore(0.01, (("deu_Latn", 0.98), ("eng_Latn", 0.01)), 30)
            if "der" in t.lower().split()
            else LanguageScore(0.97, (("eng_Latn", 0.97),), 30)
            for t in texts
        ]

    async def nli(self, model, pairs):
        self.calls.append("nli")
        return [
            {
                "contradiction": 0.99 if "Zigbee" in h and "no Zigbee" in p else 0.0,
                "entailment": 0.0,
                "neutral": 1.0,
            }
            for p, h in pairs
        ]

    async def image_classes(self, model, images, classes):
        self.calls.append("image_classes")
        return [{name: (1.0 if name == "unrelated" else 0.0) for name in classes} for _ in images]

    async def obscenity(self, model, images):
        self.calls.append("obscenity")
        return [{"at_least_low": 0.0, "at_least_high": 0.0, "nsfw": 0.0} for _ in images]


def test_summary_splits_into_sentences_keeping_abbreviations_and_code() -> None:
    text = (
        "Works with ZHA, e.g. via SkyConnect. Firmware v1.2 fixed it! ```yaml\na: 1. B\n``` Done."
    )

    assert [s for s, _ in sentences(text)] == [
        "Works with ZHA, e.g. via SkyConnect.",
        "Firmware v1.2 fixed it!",
        "```yaml\na: 1. B\n```",
        "Done.",
    ]


def test_units_carry_spans_into_their_field() -> None:
    contribution = Contribution(
        fields={"name": " Plug ", "summary": "One. Two.", "quirks": "- First\n- Second"}
    )

    units = split_units(contribution)

    assert [(u.field, u.index, u.text) for u in units] == [
        ("name", 0, "Plug"),
        ("summary", 0, "One."),
        ("summary", 1, "Two."),
        ("quirks", 0, "First"),
        ("quirks", 1, "Second"),
    ]
    summary = contribution.fields["summary"]
    assert summary[units[2].span[0] : units[2].span[1]] == "Two."


def test_the_shipped_config_loads() -> None:
    loaded = config_module.load(CONFIG)

    assert loaded.rules
    assert len(loaded.sha256) == 64


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c["rules"][0].update(model="toxicity/missing@fp32"), "doesn't list"),
        (lambda c: c["rules"][0]["thresholds"].update(spam={"warn": 0.5}), "unknown signals"),
        (lambda c: c["rules"][0].update(type="obscenity"), "uses a toxicity model"),
        (lambda c: c["rules"][0]["thresholds"]["toxicity"].update(warn=2), "schema"),
        (lambda c: c["rules"].append(dict(c["rules"][0])), "more than once"),
    ],
)
def test_inconsistent_config_is_rejected(tmp_path: Path, change, message: str) -> None:
    shutil.copytree(CONFIG, tmp_path / "config")
    checks = tmp_path / "config/checks.json"
    document = json.loads(checks.read_text())
    change(document)
    checks.write_text(json.dumps(document))

    with pytest.raises(ConfigError, match=message):
        config_module.load(tmp_path / "config")


def test_facts_render_from_templates_and_fixed_values() -> None:
    loaded = config_module.load(CONFIG)
    device = {"brand": "Nest", "connectivity": "local", "identifiers": ["A1", "B2"], "other": 1}

    assert render_facts(device, loaded) == [
        "This device is made by Nest.",
        "This device works locally, without an internet connection.",
        "The device's model number is A1.",
        "The device's model number is B2.",
    ]


def test_check_reports_each_rule_with_its_evidence() -> None:
    loaded = config_module.load(CONFIG)
    contribution = Contribution(
        device={"brand": "Nest", "category": "smart display"},
        fields={
            "summary": "Smart display. Der Sensor meldet Temperatur.",
            "quirks": ["Whoever wrote this is an idiot.", "Pairs over Zigbee."],
        },
        images=(Image("parrot", b"\x89PNG"),),
    )
    scorer = FakeScorer()
    loaded.facts["brand"] = {"template": "It has no Zigbee radio, says {value}."}

    result = asyncio.run(check(contribution, loaded, scorer))

    found = {(hit.rule, hit.field, hit.unit, hit.action) for hit in result.hits}
    assert found == {
        ("english", "summary", 1, "reject"),
        ("abuse_spam", "quirks", 0, "warn"),
        ("consistency", "quirks", 1, "warn"),
        ("image_category", "images", 0, "warn"),
    }
    assert result.outcome == "reject"
    assert result.config_sha256 == loaded.sha256
    consistency = next(hit for hit in result.hits if hit.rule == "consistency")
    assert consistency.evidence[0].detail == "It has no Zigbee radio, says Nest."


def test_an_empty_contribution_is_accepted_without_scoring() -> None:
    scorer = FakeScorer()

    result = asyncio.run(check(Contribution(), config_module.load(CONFIG), scorer))

    assert result.outcome == "accept"
    assert scorer.calls == []
