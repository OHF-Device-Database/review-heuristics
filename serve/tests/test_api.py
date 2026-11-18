"""The HTTP API, against the tiny classifier."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from ddb_checks.config import ConfigError
from ddb_serve.main import create_app
from ddb_serve.registry import RegistryError
from ddb_serve.runtime import Settings

from .conftest import MAX_LENGTH, expected_scores

TINY = "toxicity/tiny@fp32"
PROBLEM = "application/problem+json"


def test_scores_each_text_with_the_exposed_labels(make_settings: Callable[..., Settings]) -> None:
    texts = ["bad bad", "good", "unknown words here", "bad " * (MAX_LENGTH * 3)]
    with TestClient(create_app(make_settings([TINY]))) as client:
        response = client.post(
            "/v1/toxicity",
            json={"texts": [{"id": str(i), "text": text} for i, text in enumerate(texts)]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == TINY
    assert set(body["timings_ms"]) == {"preprocess", "inference", "total"}
    for result, text in zip(body["results"], texts, strict=True):
        assert result["scores"] == pytest.approx(expected_scores(text), abs=1e-6)


def test_lists_loaded_variants(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings([TINY])
    meta = json.loads((settings.models_dir / "toxicity/tiny/fp32/meta.json").read_text())
    with TestClient(create_app(settings)) as client:
        assert client.get("/readyz").json() == {
            "status": "ready",
            "config": "0.0.0",
            "models": [TINY],
        }
        listed = client.get("/v1/models").json()["models"]

    assert [(m["id"], m["sha256"], m["labels"]) for m in listed] == [
        (TINY, meta["sha256"], ["toxicity", "insult"])
    ]


def test_requires_a_model_when_several_serve_the_task(
    make_settings: Callable[..., Settings],
) -> None:
    with TestClient(create_app(make_settings([TINY, "toxicity/tiny@int8"]))) as client:
        response = client.post("/v1/toxicity", json={"texts": [{"id": "a", "text": "bad"}]})

    assert response.status_code == 422
    assert response.headers["content-type"] == PROBLEM


def test_unknown_model_is_not_found(make_settings: Callable[..., Settings]) -> None:
    with TestClient(create_app(make_settings([TINY]))) as client:
        response = client.post(
            "/v1/toxicity",
            json={"model": "toxicity/missing@fp32", "texts": [{"id": "a", "text": "bad"}]},
        )

    assert response.status_code == 404
    assert response.headers["content-type"] == PROBLEM
    assert response.json()["status"] == 404


def test_rejects_oversized_batches(make_settings: Callable[..., Settings]) -> None:
    texts = [{"id": str(i), "text": "bad"} for i in range(129)]
    with TestClient(create_app(make_settings([TINY]))) as client:
        response = client.post("/v1/toxicity", json={"texts": texts})

    assert response.status_code == 413
    assert response.headers["content-type"] == PROBLEM


def test_invalid_requests_get_problem_details(make_settings: Callable[..., Settings]) -> None:
    with TestClient(create_app(make_settings([TINY]))) as client:
        response = client.post("/v1/toxicity", json={"texts": []})

    assert response.status_code == 422
    assert response.headers["content-type"] == PROBLEM
    assert "texts" in response.json()["detail"]


def test_startup_fails_when_the_model_file_changed(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings([TINY])
    with (settings.models_dir / "toxicity/tiny/fp32/model.onnx").open("ab") as file:
        file.write(b"tampered")

    with pytest.raises(RegistryError, match="SHA-256"), TestClient(create_app(settings)):
        pass


def test_serves_the_playground(make_settings: Callable[..., Settings]) -> None:
    with TestClient(create_app(make_settings([TINY]))) as client:
        response = client.get("/playground")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "/v1/models" in response.text


def test_cross_origin_calls_need_cors_origins(make_settings: Callable[..., Settings]) -> None:
    origin = {"Origin": "http://elsewhere.test"}
    closed = make_settings([TINY])
    with TestClient(create_app(closed)) as client:
        assert "access-control-allow-origin" not in client.get("/readyz", headers=origin).headers

    opened = Settings(closed.config_dir, closed.models_dir, "*", ("http://elsewhere.test",))
    with TestClient(create_app(opened)) as client:
        allowed = client.get("/readyz", headers=origin).headers
    assert allowed["access-control-allow-origin"] == "http://elsewhere.test"


def test_checks_a_contribution_with_the_configured_rules(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings([TINY])
    rule = {
        "id": "abuse_spam",
        "type": "toxicity",
        "model": TINY,
        "fields": ["quirks"],
        "thresholds": {"toxicity": {"warn": 0.58}},
        "message": "Keep it civil.",
        "guideline": "docs/guidelines.md#rules",
    }
    checks = json.loads((settings.config_dir / "checks.json").read_text())
    (settings.config_dir / "checks.json").write_text(json.dumps({**checks, "rules": [rule]}))
    request = {"fields": {"name": "Plug", "quirks": ["bad bad", "bad"]}}

    with TestClient(create_app(settings)) as client:
        body = client.post("/v1/contributions/check", json=request).json()
        stale = client.post("/v1/contributions/check", json={**request, "config_version": "9.9.9"})

    assert body["outcome"] == "warn"
    assert body["rules"] == ["abuse_spam"]
    assert [(h["field"], h["unit"], h["item"], h["text"]) for h in body["hits"]] == [
        ("quirks", 0, 0, "bad bad")
    ]
    evidence = body["hits"][0]["evidence"][0]
    assert evidence["signal"] == "toxicity"
    assert evidence["score"] == pytest.approx(expected_scores("bad bad")["toxicity"])
    assert stale.status_code == 409


def test_startup_fails_on_an_invalid_config(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings([TINY])
    (settings.config_dir / "checks.json").write_text('{"version": "1", "rules": []}')

    with pytest.raises(ConfigError, match="schema"), TestClient(create_app(settings)):
        pass
