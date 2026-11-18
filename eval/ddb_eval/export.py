"""Export registry variants to the files and metadata ``serve`` loads."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .registry import VariantId, find, split_source

_LOGGER = logging.getLogger(__name__)
# Tokenizers report this when they have no length limit of their own.
_UNBOUNDED_LENGTH = 1_000_000
_HASH_CHUNK = 1 << 20
# SigLIP's text tower was trained on prompts padded to this many tokens, and lowercased.
_SIGLIP_TEXT_LENGTH = 64
_SIGLIP_IMAGE_BATCH = 2
_RESAMPLE_NAMES = {2: "bilinear", 3: "bicubic"}
_ENGLISH = "eng_Latn"


def export(variant_id: str, registry_path: Path, models_dir: Path) -> Path:
    """
    Export one variant and return the directory holding its files.

    An int8 variant is quantized from the fp32 export of the same model, which is exported
    first if it isn't there yet. Raises ValueError for tasks or precisions with no exporter.
    """
    parsed = VariantId.parse(variant_id)
    entry = find(registry_path, variant_id)
    exporter = _EXPORTERS.get(parsed.task)
    if exporter is None:
        raise ValueError(f"no exporter for task {parsed.task!r} yet")

    out = models_dir / parsed.directory
    if parsed.precision == "int8":
        fp32 = models_dir / parsed.with_precision("fp32").directory
        if not (fp32 / "meta.json").exists():
            exporter(entry, fp32)
        _quantize_dynamic(fp32, out)
    elif parsed.precision in {"fp32", "ftz"}:
        exporter(entry, out)
    else:
        raise ValueError(f"no exporter for precision {parsed.precision!r}")
    _LOGGER.info("Exported %s to %s", variant_id, out)
    return out


def sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _export_sequence_classifier(entry: dict[str, Any], out: Path) -> None:
    repo, revision = split_source(entry["source"])
    tokenizer = _fast_tokenizer(repo, revision)
    model = AutoModelForSequenceClassification.from_pretrained(repo, revision=revision).eval()

    # Two samples of different lengths, so both axes are traced as dynamic.
    sample = tokenizer(
        ["Export sample.", "A second, longer export sample so both axes vary."],
        padding=True,
        return_tensors="pt",
    )
    names = list(sample.keys())
    batch, sequence = torch.export.Dim("batch"), torch.export.Dim("sequence")
    _export_onnx(
        model,
        {name: sample[name] for name in names},
        {name: {0: batch, 1: sequence} for name in names},
        out / "model.onnx",
    )
    tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))
    _write_meta(
        out,
        ["model.onnx", "tokenizer.json"],
        {"source": entry["source"], **_classifier_meta(model.config, tokenizer)},
        "torch",
        "transformers",
        "onnx",
        "onnxscript",
    )


def _copy_onnx_classifier(entry: dict[str, Any], out: Path) -> None:
    """Take the repo's own ONNX export, with the tokenizer and labels from its config."""
    repo, revision = split_source(entry["source"])
    tokenizer = _fast_tokenizer(repo, revision)
    config = AutoConfig.from_pretrained(repo, revision=revision)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_download(repo, entry["artifact"], revision), out / "model.onnx")
    tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))
    _write_meta(
        out,
        ["model.onnx", "tokenizer.json"],
        {
            "source": entry["source"],
            "artifact": entry["artifact"],
            **_classifier_meta(config, tokenizer),
        },
        "transformers",
    )


def _copy_fasttext(entry: dict[str, Any], out: Path) -> None:
    repo, revision = split_source(entry["source"])
    out.mkdir(parents=True, exist_ok=True)
    name = Path(entry["artifact"]).name
    shutil.copyfile(_download(repo, entry["artifact"], revision), out / name)
    _write_meta(
        out,
        [name],
        {
            "source": entry["source"],
            "artifact": entry["artifact"],
            "file": name,
            "english": _ENGLISH,
        },
    )


def _export_siglip(entry: dict[str, Any], out: Path) -> None:
    """Export the image and text towers separately, so each request runs only the one it needs."""
    repo, revision = split_source(entry["source"])
    model = AutoModel.from_pretrained(repo, revision=revision).eval()
    tokenizer = _fast_tokenizer(repo, revision)
    processor = AutoImageProcessor.from_pretrained(repo, revision=revision)
    height, width = processor.size["height"], processor.size["width"]

    vision = _ImageTower(model.vision_model)
    text = _TextTower(model.text_model)
    batch = torch.export.Dim("batch")
    _export_onnx(
        vision,
        {"pixel_values": torch.randn(_SIGLIP_IMAGE_BATCH, 3, height, width)},
        {"pixel_values": {0: batch}},
        out / "vision.onnx",
    )
    ids = tokenizer(
        ["a photo of a smart plug", "an installed light switch"],
        padding="max_length",
        max_length=_SIGLIP_TEXT_LENGTH,
        truncation=True,
        return_tensors="pt",
    )["input_ids"]
    _export_onnx(text, {"input_ids": ids}, {"input_ids": {0: batch}}, out / "text.onnx")
    tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))
    _write_meta(
        out,
        ["vision.onnx", "text.onnx", "tokenizer.json"],
        {
            "source": entry["source"],
            "preprocess": {
                "size": [width, height],
                "resample": _resample_name(processor.resample),
                "mean": list(processor.image_mean),
                "std": list(processor.image_std),
            },
            "lowercase": True,
            "text_length": _SIGLIP_TEXT_LENGTH,
            "pad_token": tokenizer.pad_token,
            "pad_id": tokenizer.pad_token_id,
            "logit_scale": float(model.logit_scale.item()),
            "logit_bias": float(model.logit_bias.item()),
        },
        "torch",
        "transformers",
        "onnx",
        "onnxscript",
    )


def _export_timm_classifier(entry: dict[str, Any], out: Path) -> None:
    repo, revision = split_source(entry["source"])
    model = AutoModelForImageClassification.from_pretrained(
        repo, revision=revision, dtype=torch.float32
    ).eval()
    config = model.config
    pretrained = config.pretrained_cfg
    # serve resizes straight to the input size; any other crop would need its own code.
    if pretrained.get("crop_mode") != "squash" or pretrained.get("crop_pct") != 1.0:
        raise ValueError(f"{repo} needs a crop serve doesn't implement: {pretrained}")
    _, height, width = pretrained["input_size"]

    labels = [config.id2label[i] for i in range(config.num_labels)]
    if entry["safe_label"] not in labels:
        raise ValueError(f"{repo} has no label {entry['safe_label']!r}: {labels}")
    _export_onnx(
        _Logits(model),
        {"pixel_values": torch.randn(_SIGLIP_IMAGE_BATCH, 3, height, width)},
        {"pixel_values": {0: torch.export.Dim("batch")}},
        out / "model.onnx",
    )
    _write_meta(
        out,
        ["model.onnx"],
        {
            "source": entry["source"],
            "labels": labels,
            "safe_label": entry["safe_label"],
            "graded": bool(entry.get("graded", False)),
            "preprocess": {
                "size": [width, height],
                "resample": pretrained["interpolation"],
                "mean": list(pretrained["mean"]),
                "std": list(pretrained["std"]),
            },
        },
        "torch",
        "transformers",
        "timm",
        "onnx",
        "onnxscript",
    )


def _quantize_dynamic(fp32: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as scratch:
        prepared = Path(scratch) / "prepared.onnx"
        quant_pre_process(str(fp32 / "model.onnx"), str(prepared))
        quantize_dynamic(str(prepared), str(out / "model.onnx"), weight_type=QuantType.QInt8)
    meta = json.loads((fp32 / "meta.json").read_text())
    files = list(meta.pop("files"))
    for name in files:
        if name != "model.onnx":
            shutil.copyfile(fp32 / name, out / name)
    meta["quantization"] = {"method": "dynamic", "weight_type": "QInt8", "from": meta.pop("sha256")}
    exported_with = meta.pop("exported_with", {})
    meta["exported_with"] = {**exported_with, **_versions("onnxruntime")}
    _write_meta(out, files, meta)


class _ImageTower(torch.nn.Module):
    """SigLIP's image tower, returning its pooled embedding."""

    def __init__(self, tower: torch.nn.Module) -> None:
        super().__init__()
        self.tower = tower

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.tower(pixel_values=pixel_values).pooler_output


class _TextTower(torch.nn.Module):
    """SigLIP's text tower, returning its pooled embedding."""

    def __init__(self, tower: torch.nn.Module) -> None:
        super().__init__()
        self.tower = tower

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.tower(input_ids=input_ids).pooler_output


class _Logits(torch.nn.Module):
    """An image classifier, returning only its logits."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


def _export_onnx(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    dynamic_shapes: dict[str, dict[int, Any]],
    path: Path,
) -> None:
    names = list(inputs)
    program = torch.onnx.export(
        model,
        (),
        kwargs=inputs,
        input_names=names,
        output_names=["logits" if path.name == "model.onnx" else "embeddings"],
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
        external_data=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(path), external_data=False)


def _classifier_meta(config: Any, tokenizer: Any) -> dict[str, Any]:
    max_length = tokenizer.model_max_length
    if max_length >= _UNBOUNDED_LENGTH:
        max_length = config.max_position_embeddings
    return {
        "labels": [config.id2label[i] for i in range(config.num_labels)],
        "output": "sigmoid" if config.problem_type == "multi_label_classification" else "softmax",
        "max_length": max_length,
        "pad_token": tokenizer.pad_token,
        "pad_id": tokenizer.pad_token_id,
    }


def _fast_tokenizer(repo: str, revision: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    if not tokenizer.is_fast:
        raise ValueError(f"{repo} has no fast tokenizer, so serve can't load it")
    return tokenizer


def _download(repo: str, filename: str, revision: str) -> str:
    return hf_hub_download(repo, filename, revision=revision)


def _resample_name(code: int) -> str:
    if code not in _RESAMPLE_NAMES:
        raise ValueError(f"resampling filter {code} isn't supported by serve")
    return _RESAMPLE_NAMES[code]


def _write_meta(out: Path, files: list[str], meta: dict[str, Any], *packages: str) -> None:
    """
    Write ``meta.json`` with every file's SHA-256, and one hash for the variant as a whole.

    The variant hash is the file's own hash for a single-file variant, so it matches what
    ``sha256sum`` prints for it.
    """
    hashes = {name: sha256(out / name) for name in files}
    if len(hashes) == 1:
        variant_hash = next(iter(hashes.values()))
    else:
        listing = "".join(f"{name} {digest}\n" for name, digest in sorted(hashes.items()))
        variant_hash = hashlib.sha256(listing.encode()).hexdigest()
    if packages:
        meta = {**meta, "exported_with": _versions(*packages)}
    meta = {**meta, "files": hashes, "sha256": variant_hash}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _versions(*packages: str) -> dict[str, str]:
    return {package: version(package) for package in packages}


_EXPORTERS: dict[str, Callable[[dict[str, Any], Path], None]] = {
    "toxicity": _export_sequence_classifier,
    "nli": _copy_onnx_classifier,
    "language": _copy_fasttext,
    "image-embed": _export_siglip,
    "obscenity": _export_timm_classifier,
}
