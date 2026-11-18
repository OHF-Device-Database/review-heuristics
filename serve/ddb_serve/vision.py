"""Image decoding and the image models: a classifier and the SigLIP 2 dual encoder."""

from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError
from tokenizers import Tokenizer

from .registry import ModelMeta, RegistryError, Variant
from .text import Timings, create_session, softmax

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000
_RESAMPLE = {"bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC}
_TEXT_CACHE_SIZE = 4096

# Pillow refuses images over twice this size outright, before decoding any pixels.
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class ImageError(ValueError):
    """An image could not be decoded."""


class ImageTooLargeError(ImageError):
    """An image is over the size or pixel limit."""


@dataclass(frozen=True, slots=True)
class Preprocess:
    """How a model expects its input images: resized to ``size``, then normalised."""

    size: tuple[int, int]
    resample: Image.Resampling
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_meta(cls, meta: ModelMeta) -> Preprocess:
        """Return the preprocessing export recorded for the variant."""
        raw = meta.require("preprocess")
        if raw["resample"] not in _RESAMPLE:
            raise RegistryError(f"unsupported resampling filter {raw['resample']!r}")
        width, height = raw["size"]
        return cls(
            size=(width, height),
            resample=_RESAMPLE[raw["resample"]],
            mean=np.array(raw["mean"], dtype=np.float32).reshape(3, 1, 1),
            std=np.array(raw["std"], dtype=np.float32).reshape(3, 1, 1),
        )

    def apply(self, images: list[Image.Image]) -> np.ndarray:
        """Return a float32 batch shaped (images, 3, height, width)."""
        arrays = [
            np.asarray(image.resize(self.size, self.resample), dtype=np.float32).transpose(2, 0, 1)
            for image in images
        ]
        return ((np.stack(arrays) / 255 - self.mean) / self.std).astype(np.float32)


def decode_all(items: list[tuple[str, str]], timings: Timings) -> list[Image.Image]:
    """
    Return each ``(id, base64 data)`` item decoded to an upright RGB image.

    EXIF orientation is applied, and transparency is composited onto white. Raises
    ImageTooLargeError over the byte or pixel limit, and ImageError for anything undecodable;
    both name the item.
    """
    began = perf_counter()
    images = []
    for item_id, data in items:
        try:
            images.append(decode(base64.b64decode(data, validate=True)))
        except binascii.Error as err:
            raise ImageError(f"image {item_id}: data is not valid base64") from err
        except ImageError as err:
            raise type(err)(f"image {item_id}: {err}") from err
    timings.decode += (perf_counter() - began) * 1000
    return images


def decode(data: bytes) -> Image.Image:
    """
    Return the image in ``data`` as upright RGB.

    Raises ImageTooLargeError over the byte or pixel limit and ImageError when it can't be
    decoded.
    """
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageTooLargeError(f"over {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > MAX_PIXELS:
            raise ImageTooLargeError(f"over {MAX_PIXELS // 1_000_000} megapixels")
        image = ImageOps.exif_transpose(image)
        if image.mode == "P" and "transparency" in image.info:
            image = image.convert("RGBA")
        if image.mode in {"RGBA", "LA", "PA"}:
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            background.alpha_composite(image.convert("RGBA"))
            image = background
        return image.convert("RGB")
    except Image.DecompressionBombError as err:
        raise ImageTooLargeError(f"over {MAX_PIXELS // 1_000_000} megapixels") from err
    except UnidentifiedImageError as err:
        raise ImageError("not an image format the server can read (HEIC, for one, isn't)") from err
    except (OSError, ValueError, SyntaxError) as err:
        if isinstance(err, ImageError):
            raise
        raise ImageError(f"cannot decode: {err}") from err


class ImageClassifier:
    """Classifies images with an ONNX model that outputs one logit per label."""

    def __init__(self, session: ort.InferenceSession, meta: ModelMeta, variant: Variant) -> None:
        """Wrap a loaded session; labels and preprocessing come from the export metadata."""
        self._session = session
        self._batch_size = variant.runtime.batch_size
        self._preprocess = Preprocess.from_meta(meta)
        self.labels: tuple[str, ...] = tuple(meta.require("labels"))
        self.graded: bool = bool(meta.data.get("graded", False))
        self.safe_label: str = meta.require("safe_label")
        if self.safe_label not in self.labels:
            raise RegistryError(f"{variant.id} has no {self.safe_label!r} label")

    @classmethod
    def load(cls, directory: Path, meta: ModelMeta, variant: Variant) -> ImageClassifier:
        """Return a classifier for the exported variant in ``directory``."""
        return cls(create_session(directory / "model.onnx", variant), meta, variant)

    def warm_up(self) -> None:
        """Run one inference so the first real request pays no one-time costs."""
        self.predict([Image.new("RGB", (64, 64))], Timings())

    def signals(self, probabilities: np.ndarray) -> dict[str, float]:
        """
        Return one image's scores as named signals.

        Each label's probability; for graded models, ``at_least_<level>``, the probability of
        that level or worse; and ``nsfw``, 1 - P(safe label), comparable across models.
        """
        named = dict(zip(self.labels, probabilities.tolist(), strict=True))
        safe = self.labels.index(self.safe_label)
        if self.graded:
            # Labels run from safe to most explicit, so a reversed cumulative sum gives each
            # level or worse.
            cumulative = np.cumsum(probabilities[::-1])[::-1]
            named |= {
                f"at_least_{label}": float(cumulative[i])
                for i, label in enumerate(self.labels)
                if i != safe
            }
        named["nsfw"] = float(1 - probabilities[safe])
        return named

    def predict(self, images: list[Image.Image], timings: Timings) -> np.ndarray:
        """Return one row of probabilities per image, with columns in ``self.labels`` order."""
        rows = []
        for start in range(0, len(images), self._batch_size):
            began = perf_counter()
            pixels = self._preprocess.apply(images[start : start + self._batch_size])
            prepared = perf_counter()
            rows.append(self._session.run(None, {"pixel_values": pixels})[0])
            timings.preprocess += (prepared - began) * 1000
            timings.inference += (perf_counter() - prepared) * 1000
        return softmax(np.concatenate(rows))


class SiglipEncoder:
    """SigLIP 2's image and text towers, for embeddings and zero-shot classification."""

    labels: tuple[str, ...] = ()

    def __init__(
        self,
        vision: ort.InferenceSession,
        text: ort.InferenceSession,
        tokenizer: Tokenizer,
        meta: ModelMeta,
        variant: Variant,
    ) -> None:
        """Wrap both towers; scale, bias and preprocessing come from the export metadata."""
        self._vision = vision
        self._text = text
        self._tokenizer = tokenizer
        self._batch_size = variant.runtime.batch_size
        self._preprocess = Preprocess.from_meta(meta)
        self._lowercase = bool(meta.require("lowercase"))
        self._scale = float(np.exp(meta.require("logit_scale")))
        self._bias = float(meta.require("logit_bias"))
        self._text_cache: dict[str, np.ndarray] = {}

    @classmethod
    def load(cls, directory: Path, meta: ModelMeta, variant: Variant) -> SiglipEncoder:
        """Return the encoder for the exported variant in ``directory``."""
        # The text tower was trained on prompts padded to a fixed length; it pools the last
        # position, so the padding length changes the embedding.
        length = meta.require("text_length")
        tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        tokenizer.enable_truncation(length)
        tokenizer.enable_padding(
            pad_id=meta.require("pad_id"), pad_token=meta.require("pad_token"), length=length
        )
        return cls(
            create_session(directory / "vision.onnx", variant),
            create_session(directory / "text.onnx", variant),
            tokenizer,
            meta,
            variant,
        )

    def warm_up(self) -> None:
        """Run both towers once so the first real request pays no one-time costs."""
        self.embed_images([Image.new("RGB", (64, 64))], Timings())
        self._encode_texts(["warm-up"])

    def embed_images(self, images: list[Image.Image], timings: Timings) -> np.ndarray:
        """Return one L2-normalised embedding per image."""
        rows = []
        for start in range(0, len(images), self._batch_size):
            began = perf_counter()
            pixels = self._preprocess.apply(images[start : start + self._batch_size])
            prepared = perf_counter()
            rows.append(self._vision.run(None, {"pixel_values": pixels})[0])
            timings.preprocess += (prepared - began) * 1000
            timings.inference += (perf_counter() - prepared) * 1000
        return _normalise(np.concatenate(rows))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalised embedding per text, reusing those seen recently."""
        missing = list(dict.fromkeys(text for text in texts if text not in self._text_cache))
        if missing:
            if len(self._text_cache) + len(missing) > _TEXT_CACHE_SIZE:
                self._text_cache.clear()
            self._text_cache.update(zip(missing, self._encode_texts(missing), strict=True))
        return np.stack([self._text_cache[text] for text in texts])

    def classify(
        self, images: list[Image.Image], classes: dict[str, list[str]], timings: Timings
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return zero-shot scores per image and class, as (softmax, sigmoid) matrices.

        A class with several prompts is represented by the mean of their embeddings. The softmax
        picks one class among those given; the sigmoid is SigLIP's own per-class match
        probability, independent of the other classes.
        """
        image_embeddings = self.embed_images(images, timings)
        began = perf_counter()
        class_embeddings = _normalise(
            np.stack([self.embed_texts(prompts).mean(axis=0) for prompts in classes.values()])
        )
        logits = image_embeddings @ class_embeddings.T * self._scale + self._bias
        timings.inference += (perf_counter() - began) * 1000
        return softmax(logits), 1 / (1 + np.exp(-logits))

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        prepared = [text.lower() for text in texts] if self._lowercase else texts
        rows = []
        for start in range(0, len(prepared), self._batch_size):
            encodings = self._tokenizer.encode_batch(prepared[start : start + self._batch_size])
            ids = np.array([e.ids for e in encodings], dtype=np.int64)
            rows.append(self._text.run(None, {"input_ids": ids})[0])
        return _normalise(np.concatenate(rows))


def _normalise(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
