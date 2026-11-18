"""Synthetic parity images, and the reference decoding ``serve`` must match."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageOps

_SEED = 0
_EXIF_ORIENTATION = 0x0112
_ROTATE_CLOCKWISE = 6


@dataclass(frozen=True, slots=True)
class EncodedImage:
    """An image file's bytes, with an id saying which decoding case it exercises."""

    id: str
    data: bytes

    @property
    def base64(self) -> str:
        """Return the bytes base64-encoded, as the API takes them."""
        return base64.b64encode(self.data).decode()


def synthetic() -> list[EncodedImage]:
    """
    Return a fixed set of images covering the decoding cases serve handles.

    They are not realistic content: parity compares the served model against the original on
    identical inputs, so what matters is that resizing, transparency, palettes, EXIF
    orientation and colour modes are all exercised.
    """
    rng = np.random.default_rng(_SEED)
    photo = _smooth_noise(rng, 480, 640)
    images = [
        EncodedImage("rgb-jpeg", _save(Image.fromarray(photo), "JPEG", quality=90)),
        EncodedImage(
            "rgb-png-portrait", _save(Image.fromarray(_smooth_noise(rng, 500, 300)), "PNG")
        ),
        EncodedImage("rgba-png-transparent", _save(_transparent_shape(), "PNG")),
        EncodedImage("palette-gif-transparent", _save(_palette_shape(), "GIF", transparency=0)),
        EncodedImage("grayscale-jpeg", _save(Image.fromarray(photo).convert("L"), "JPEG")),
        EncodedImage("cmyk-jpeg", _save(Image.fromarray(photo).convert("CMYK"), "JPEG")),
        EncodedImage("tiny-png", _save(Image.fromarray(_smooth_noise(rng, 9, 17)), "PNG")),
        EncodedImage("large-jpeg", _save(Image.fromarray(_smooth_noise(rng, 2000, 3000)), "JPEG")),
        EncodedImage("webp", _save(Image.fromarray(_smooth_noise(rng, 256, 384)), "WEBP")),
    ]
    rotated = Image.fromarray(_smooth_noise(rng, 300, 600))
    exif = rotated.getexif()
    exif[_EXIF_ORIENTATION] = _ROTATE_CLOCKWISE
    images.append(EncodedImage("exif-rotated-jpeg", _save(rotated, "JPEG", exif=exif.tobytes())))
    return images


def reference_decode(data: bytes) -> Image.Image:
    """
    Return the image as the models expect it: upright, RGB, transparency on white.

    This is the documented contract, implemented independently of ``serve``.
    """
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    if image.mode in {"RGBA", "LA", "PA"}:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, rgba)
    return image.convert("RGB")


def _smooth_noise(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    """Return low-frequency colour noise, closer to photo statistics than white noise."""
    coarse = rng.integers(0, 256, size=(max(height // 32, 2), max(width // 32, 2), 3))
    image = Image.fromarray(coarse.astype(np.uint8)).resize(
        (width, height), Image.Resampling.BICUBIC
    )
    return np.asarray(image)


def _transparent_shape() -> Image.Image:
    image = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((60, 40, 340, 260), fill=(200, 40, 40, 255))
    draw.rectangle((150, 120, 250, 180), fill=(20, 90, 200, 128))
    return image


def _palette_shape() -> Image.Image:
    """Return a palette image whose index 0 (the background) is meant to be transparent."""
    image = Image.new("P", (320, 240), 0)
    image.putpalette([0, 0, 0, 30, 160, 60, 240, 200, 0])
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 30, 280, 210), fill=1)
    draw.rectangle((120, 90, 200, 150), fill=2)
    return image


def _save(image: Image.Image, image_format: str, **options: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **options)
    return buffer.getvalue()
