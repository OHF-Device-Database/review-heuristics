"""A tiny real ONNX classifier and registry, so tests exercise ONNX Runtime end to end."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, models, pre_tokenizers

from ddb_serve.runtime import Settings

# logits = (sum of the token ids) * WEIGHTS, so expected scores are easy to compute.
WEIGHTS = [0.1, -0.1, 0.0]
LABELS = ["toxicity", "insult", "identity_marker"]
VOCAB = {"[PAD]": 0, "[UNK]": 1, "bad": 2, "good": 3}
MAX_LENGTH = 8
REPO_CONFIG = Path(__file__).parents[2] / "config"


def build_variant(directory: Path, labels: list[str] = LABELS, output: str = "sigmoid") -> None:
    """Write model.onnx, tokenizer.json and meta.json for the tiny text classifier."""
    directory.mkdir(parents=True)
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["input_ids"], ["ids"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["attention_mask"], ["mask"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["ids", "mask"], ["masked"]),
            helper.make_node("ReduceSum", ["masked", "axes"], ["total"], keepdims=1),
            helper.make_node("Mul", ["total", "weights"], ["logits"]),
        ],
        "tiny",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"]),
            helper.make_tensor_value_info(
                "attention_mask", TensorProto.INT64, ["batch", "sequence"]
            ),
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", len(WEIGHTS)])],
        initializer=[
            numpy_helper.from_array(np.array([1], dtype=np.int64), "axes"),
            numpy_helper.from_array(np.array([WEIGHTS], dtype=np.float32), "weights"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    # onnx writes its newest IR version by default, which ONNX Runtime may not read yet.
    model.ir_version = 8
    onnx.save(model, directory / "model.onnx")

    tokenizer = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))

    write_meta(
        directory,
        ["model.onnx"],
        labels=labels,
        output=output,
        max_length=MAX_LENGTH,
        pad_token="[PAD]",
        pad_id=0,
    )


def write_meta(directory: Path, files: list[str], **data: object) -> None:
    """Write meta.json as export does, hashing ``files`` in ``directory``."""
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files}
    sha256 = (
        hashes[files[0]]
        if len(files) == 1
        else hashlib.sha256(
            "".join(f"{name} {digest}\n" for name, digest in sorted(hashes.items())).encode()
        ).hexdigest()
    )
    (directory / "meta.json").write_text(json.dumps({"sha256": sha256, "files": hashes, **data}))


def expected_scores(text: str) -> dict[str, float]:
    """Return what the tiny classifier should score ``text``, for the exposed labels."""
    ids = [VOCAB.get(word, VOCAB["[UNK]"]) for word in text.split()][:MAX_LENGTH]
    logits = np.array(WEIGHTS) * sum(ids)
    scores = 1 / (1 + np.exp(-logits))
    return {"toxicity": float(scores[0]), "insult": float(scores[1])}


def registry_entry(variant_id: str, max_queue: int = 4, **runtime: object) -> dict[str, Any]:
    """Return a models.json entry exposing two of the tiny classifier's three labels."""
    return {
        "id": variant_id,
        "source": "example/tiny@0000000",
        "license": "Apache-2.0",
        "labels": ["toxicity", "insult"],
        "runtime": {
            "intra_op_threads": 1,
            "max_concurrency": 1,
            "max_queue": max_queue,
            "batch_size": 2,
            **runtime,
        },
    }


def write_config(
    directory: Path, entries: list[dict[str, Any]], rules: list[dict[str, Any]] | None = None
) -> Path:
    """Write a configuration directory: these models and rules, the rest from the repo's."""
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_CONFIG / "schemas", directory / "schemas", dirs_exist_ok=True)
    for name in ("image-classes.json", "language.json", "facts.json"):
        shutil.copyfile(REPO_CONFIG / name, directory / name)
    (directory / "models.json").write_text(json.dumps({"models": entries}))
    (directory / "checks.json").write_text(json.dumps({"version": "0.0.0", "rules": rules or []}))
    return directory


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[[list[str], str], Settings]:
    """Return a factory for settings over freshly built variants."""

    def factory(variant_ids: list[str], serve_models: str = "*") -> Settings:
        models_dir = tmp_path / "models"
        for variant_id in variant_ids:
            task, rest = variant_id.split("/")
            name, precision = rest.split("@")
            build_variant(models_dir / task / name / precision)
        config_dir = write_config(tmp_path / "config", [registry_entry(v) for v in variant_ids])
        return Settings(config_dir, models_dir, serve_models)

    return factory
