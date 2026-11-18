"""NLI, language ID, image embeddings and zero-shot classification, and obscenity."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers

from ddb_serve import language_id
from ddb_serve.language_id import mask
from ddb_serve.main import create_app
from ddb_serve.runtime import Settings
from ddb_serve.vision import MAX_PIXELS, decode

from .conftest import build_variant, write_config, write_meta

NLI = "nli/tiny@fp32"
LANGUAGE = "language/tiny@ftz"
SIGLIP = "image-embed/tiny@fp32"
OBSCENITY = "obscenity/tiny@fp32"
IMAGE_SIZE = 8
TEXT_LENGTH = 6
PREPROCESS = {
    "size": [IMAGE_SIZE, IMAGE_SIZE],
    "resample": "bilinear",
    "mean": [0.5] * 3,
    "std": [0.5] * 3,
}
EMBEDDING = np.array([[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 1.0]])
GRADES = np.array([[2.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 1.0]])


class FakeFastText:
    """Calls a text English when it contains 'the', German otherwise."""

    def predict(self, text: str, k: int = 1, threshold: float = 0.0) -> tuple[Any, Any]:
        english = 0.9 if "the" in text.split() else 0.1
        labels = ("__label__eng_Latn", "__label__deu_Latn")
        probabilities = (english, 1 - english)
        order = sorted(range(2), key=lambda i: -probabilities[i])
        return tuple(labels[i] for i in order), tuple(probabilities[i] for i in order)


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Build one tiny variant per task, and a registry listing them."""
    models_dir = tmp_path / "models"
    build_variant(
        models_dir / "nli/tiny/fp32",
        labels=["contradiction", "entailment", "neutral"],
        output="softmax",
    )

    language = models_dir / "language/tiny/ftz"
    language.mkdir(parents=True)
    (language / "model.ftz").write_bytes(b"stand-in")
    write_meta(language, ["model.ftz"], file="model.ftz", english="eng_Latn")
    monkeypatch.setattr(language_id.fasttext, "load_model", lambda _: FakeFastText())

    siglip = models_dir / "image-embed/tiny/fp32"
    siglip.mkdir(parents=True)
    _image_model(siglip / "vision.onnx", EMBEDDING)
    _text_tower(siglip / "text.onnx")
    tokenizer = Tokenizer(
        models.WordLevel({"<pad>": 0, "<unk>": 1, "box": 2, "room": 3}, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(siglip / "tokenizer.json"))
    write_meta(
        siglip,
        ["vision.onnx", "text.onnx", "tokenizer.json"],
        preprocess=PREPROCESS,
        lowercase=True,
        logit_scale=float(np.log(10)),
        logit_bias=-5.0,
        text_length=TEXT_LENGTH,
        pad_id=0,
        pad_token="<pad>",
    )

    obscenity = models_dir / "obscenity/tiny/fp32"
    obscenity.mkdir(parents=True)
    _image_model(obscenity / "model.onnx", GRADES)
    write_meta(
        obscenity,
        ["model.onnx"],
        labels=["neutral", "low", "medium", "high"],
        safe_label="neutral",
        graded=True,
        preprocess=PREPROCESS,
    )

    entries = [
        {
            "id": variant_id,
            "source": "example/tiny@0000000",
            "license": "Apache-2.0",
            "runtime": {"batch_size": 2},
        }
        for variant_id in (NLI, LANGUAGE, SIGLIP, OBSCENITY)
    ]
    return Settings(write_config(tmp_path / "config", entries), models_dir, "*")


def test_nli_pairs_get_a_probability_per_model_label(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/texts/nli",
            json={"pairs": [{"id": "a", "premise": "good good", "hypothesis": "bad"}]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "pairs"
    scores = body["results"][0]["scores"]
    assert set(scores) == {"contradiction", "entailment", "neutral"}
    assert sum(scores.values()) == pytest.approx(1)


def test_nli_zero_shot_scores_each_label(settings: Settings) -> None:
    request = {
        "texts": [{"id": "a", "text": "good"}, {"id": "b", "text": "bad bad"}],
        "zero_shot": {"labels": ["good", "bad", "neither"], "hypothesis_template": "It is {}"},
    }
    with TestClient(create_app(settings)) as client:
        single = client.post("/v1/texts/nli", json=request).json()
        request["zero_shot"]["multi_label"] = True
        multi = client.post("/v1/texts/nli", json=request).json()

    assert single["mode"] == "zero_shot"
    for result in single["results"]:
        assert list(result["scores"]) == ["good", "bad", "neither"]
        assert sum(result["scores"].values()) == pytest.approx(1)
    for result in multi["results"]:
        assert all(0 < score < 1 for score in result["scores"].values())


@pytest.mark.parametrize(
    "body",
    [
        {
            "pairs": [{"id": "a", "premise": "p", "hypothesis": "h"}],
            "texts": [{"id": "b", "text": "t"}],
            "zero_shot": {"labels": ["x"]},
        },
        {"texts": [{"id": "b", "text": "t"}]},
        {
            "texts": [{"id": "b", "text": "t"}],
            "zero_shot": {"labels": ["x"], "hypothesis_template": "no slot"},
        },
    ],
)
def test_nli_rejects_mixed_or_incomplete_modes(settings: Settings, body: dict[str, Any]) -> None:
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/texts/nli", json=body).status_code == 422


def test_masking_removes_names_codes_and_vocabulary() -> None:
    text = "Shelly Plus 1PM pairs with the ZHA coordinator. It works with Zigbee and iRobot."

    assert (
        mask(text, frozenset({"zigbee"}))
        == "Shelly pairs with the coordinator. It works with and iRobot."
    )


def test_language_reports_english_after_masking(settings: Settings) -> None:
    texts = [
        {"id": "en", "text": "Press the button on the SNSW-001P16EU."},
        {"id": "de", "text": "Drücken Sie die Taste."},
        {"id": "codes", "text": "SNSW-001P16EU ZHA"},
    ]
    with TestClient(create_app(settings)) as client:
        results = client.post("/v1/texts/language", json={"texts": texts, "top_k": 1}).json()[
            "results"
        ]

    assert results[0]["english"] == pytest.approx(0.9)
    assert results[0]["scored_text"] == "Press the button on the"
    assert results[1]["top"] == [{"language": "deu_Latn", "probability": pytest.approx(0.9)}]
    assert results[2] == {
        "id": "codes",
        "english": None,
        "top": [],
        "letters": 0,
        "scored_text": "",
    }


def test_decoding_composites_transparency_onto_white_and_applies_exif() -> None:
    transparent = Image.new("RGBA", (4, 2), (255, 0, 0, 0))
    portrait = Image.new("RGB", (4, 2))
    exif = portrait.getexif()
    exif[0x0112] = 6  # Rotate 90° clockwise to display.

    assert decode(_png(transparent)).getpixel((0, 0)) == (255, 255, 255)
    assert decode(_jpeg(portrait, exif.tobytes())).size == (2, 4)


def test_image_embeddings_are_normalised(settings: Settings) -> None:
    images = [{"id": "red", "data": _b64(_png(Image.new("RGB", (20, 10), (255, 0, 0))))}]
    with TestClient(create_app(settings)) as client:
        body = client.post("/v1/images/embeddings", json={"images": images}).json()

    vector = np.array(body["results"][0]["embedding"])
    assert np.linalg.norm(vector) == pytest.approx(1)
    assert set(body["timings_ms"]) == {"decode", "preprocess", "inference", "total"}


def test_zero_shot_classification_scores_every_class(settings: Settings) -> None:
    request = {
        "classes": {"box": ["box", "a box"], "room": ["room"]},
        "images": [{"id": "x", "data": _b64(_png(Image.new("RGB", (9, 9), (0, 200, 50))))}],
    }
    with TestClient(create_app(settings)) as client:
        result = client.post("/v1/images/classify", json=request).json()["results"][0]

    assert list(result["scores"]) == ["box", "room"]
    assert sum(result["scores"].values()) == pytest.approx(1)
    assert all(0 < score < 1 for score in result["match"].values())


def test_obscenity_reports_cumulative_levels(settings: Settings) -> None:
    images = [{"id": "x", "data": _b64(_png(Image.new("RGB", (12, 12), (250, 10, 120))))}]
    with TestClient(create_app(settings)) as client:
        result = client.post("/v1/obscenity", json={"images": images}).json()["results"][0]

    scores = result["scores"]
    assert result["at_least"] == {
        "low": pytest.approx(scores["low"] + scores["medium"] + scores["high"]),
        "medium": pytest.approx(scores["medium"] + scores["high"]),
        "high": pytest.approx(scores["high"]),
    }
    assert result["nsfw"] == pytest.approx(1 - scores["neutral"])


@pytest.mark.parametrize(
    ("data", "status"),
    [
        ("not base64!", 422),
        (base64.b64encode(b"not an image").decode(), 422),
    ],
)
def test_bad_images_get_problem_details(settings: Settings, data: str, status: int) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/obscenity", json={"images": [{"id": "x", "data": data}]})

    assert response.status_code == status
    assert response.headers["content-type"] == "application/problem+json"
    assert "image x" in response.json()["detail"]
    assert "BytesIO" not in response.json()["detail"]


def test_images_over_the_pixel_limit_are_refused(settings: Settings) -> None:
    side = int(MAX_PIXELS**0.5) + 10
    huge = _png(Image.new("1", (side, side)))
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/obscenity", json={"images": [{"id": "x", "data": _b64(huge)}]})

    assert response.status_code == 413


def _image_model(path: Path, weights: np.ndarray) -> None:
    """Write a model mapping pixel_values (mean per channel) through ``weights``."""
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["pixel_values"], ["pooled"], axes=[2, 3], keepdims=0),
            helper.make_node("MatMul", ["pooled", "weights"], ["output"]),
        ],
        "tiny-image",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["b", 3, "h", "w"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["b", weights.shape[1]])],
        initializer=[
            numpy_helper.from_array(weights.astype(np.float32), "weights"),
        ],
    )
    _save(graph, path)


def _text_tower(path: Path) -> None:
    """Write a model mapping padded input_ids to a 4-wide embedding."""
    weights = np.arange(TEXT_LENGTH * 4, dtype=np.float32).reshape(TEXT_LENGTH, 4) / 10 + 0.1
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["input_ids"], ["ids"], to=TensorProto.FLOAT),
            helper.make_node("MatMul", ["ids", "weights"], ["output"]),
        ],
        "tiny-text",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", TEXT_LENGTH])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["b", 4])],
        initializer=[numpy_helper.from_array(weights, "weights")],
    )
    _save(graph, path)


def _save(graph: onnx.GraphProto, path: Path) -> None:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg(image: Image.Image, exif: bytes) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()
