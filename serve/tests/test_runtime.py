"""Registry parsing and the bounded per-variant executor."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ddb_serve.registry import RegistryError, load_meta, load_registry, select
from ddb_serve.runtime import LoadedVariant, QueueFullError, Settings
from ddb_serve.text import SequenceClassifier, create_session

from .conftest import registry_entry


def test_select_matches_globs(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(["toxicity/tiny@fp32", "toxicity/tiny@int8", "toxicity/other@fp32"])
    variants = load_registry(settings.registry_path)

    chosen = select(variants, "toxicity/tiny@*, toxicity/other@fp32")
    assert [v.id for v in chosen] == [v.id for v in variants]
    assert [v.id for v in select(variants, "*@int8")] == ["toxicity/tiny@int8"]


def test_select_rejects_patterns_matching_nothing(make_settings: Callable[..., Settings]) -> None:
    variants = load_registry(make_settings(["toxicity/tiny@fp32"]).registry_path)

    with pytest.raises(RegistryError, match="matches no variant"):
        select(variants, "toxicity/tiny@fp16")


@pytest.mark.parametrize(
    ("registry", "message"),
    [
        ({"models": [registry_entry("Toxicity/Tiny")]}, "is not <task>/<name>@<precision>"),
        ({"models": [registry_entry("toxicity/tiny@fp32")] * 2}, "more than once"),
        (registry_entry("toxicity/tiny@fp32"), "must have a 'models' list"),
    ],
)
def test_malformed_registries_are_rejected(
    tmp_path: Path, registry: dict[str, Any], message: str
) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(registry))

    with pytest.raises(RegistryError, match=message):
        load_registry(path)


def test_unexported_variant_is_reported(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="export this variant first"):
        load_meta(tmp_path / "toxicity/tiny/fp32")


def test_exposing_an_unknown_label_fails(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(["toxicity/tiny@fp32"])
    _update_entry(settings, labels=["toxicity", "spam"])
    variant = load_registry(settings.registry_path)[0]
    directory = settings.models_dir / variant.directory

    with pytest.raises(RegistryError, match="spam"):
        SequenceClassifier.load(directory, load_meta(directory), variant)


def test_full_queue_rejects_instead_of_waiting(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"models": [registry_entry("toxicity/tiny@fp32", max_queue=1)]}))
    variant = load_registry(path)[0]
    loaded = LoadedVariant(variant, meta=None, model=None, rss_delta_mb=0.0)
    release = threading.Event()

    async def scenario() -> None:
        running = asyncio.create_task(loaded.submit(release.wait))
        queued = asyncio.create_task(loaded.submit(release.wait))
        await asyncio.sleep(0.05)
        with pytest.raises(QueueFullError):
            await loaded.submit(release.wait)
        release.set()
        assert await asyncio.gather(running, queued) == [True, True]

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        loaded.close()


@pytest.mark.parametrize("arena", [True, False])
def test_sessions_follow_the_memory_arena_setting(
    make_settings: Callable[..., Settings], arena: bool
) -> None:
    settings = make_settings(["toxicity/tiny@fp32"])
    _update_entry(settings, runtime={**registry_entry("x")["runtime"], "memory_arena": arena})
    variant = load_registry(settings.registry_path)[0]

    session = create_session(settings.models_dir / variant.directory / "model.onnx", variant)

    assert session.get_session_options().enable_cpu_mem_arena is arena


def _update_entry(settings: Settings, **changes: Any) -> None:
    document = json.loads(settings.registry_path.read_text())
    document["models"][0].update(changes)
    settings.registry_path.write_text(json.dumps(document))
