"""Request and response pieces shared by the task routes."""

from __future__ import annotations

from http import HTTPStatus
from time import perf_counter

from fastapi import HTTPException
from pydantic import BaseModel, Field

from .text import Timings

MAX_TEXTS = 128
MAX_TEXT_CHARS = 10_000
MAX_IMAGES = 32


class TextItem(BaseModel):
    """One text, with an id the response echoes back."""

    id: str
    text: str


class ImageItem(BaseModel):
    """One image, with an id the response echoes back."""

    id: str
    data: str = Field(description="The image file's bytes, base64-encoded")


class TimingsMs(BaseModel):
    """Where the request's time went, in milliseconds."""

    preprocess: float
    inference: float
    total: float


class ImageTimingsMs(TimingsMs):
    """Where an image request's time went, in milliseconds, including decoding."""

    decode: float


def check_texts(texts: list[TextItem], limit: int = MAX_TEXTS) -> None:
    """Raise a 413 HTTPException when there are too many texts or one is too long."""
    if len(texts) > limit:
        raise HTTPException(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"at most {limit} texts per request"
        )
    if too_long := [item.id for item in texts if len(item.text) > MAX_TEXT_CHARS]:
        raise HTTPException(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            f"texts over {MAX_TEXT_CHARS} characters: {', '.join(too_long)}",
        )


def check_images(images: list[ImageItem]) -> None:
    """Raise a 413 HTTPException when there are too many images."""
    if len(images) > MAX_IMAGES:
        raise HTTPException(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"at most {MAX_IMAGES} images per request"
        )


def text_timings(timings: Timings, began: float) -> TimingsMs:
    """Return a text request's timings, with ``total`` measured from ``began``."""
    return TimingsMs(
        preprocess=round(timings.preprocess, 3),
        inference=round(timings.inference, 3),
        total=round((perf_counter() - began) * 1000, 3),
    )


def image_timings(timings: Timings, began: float) -> ImageTimingsMs:
    """Return an image request's timings, with ``total`` measured from ``began``."""
    return ImageTimingsMs(
        decode=round(timings.decode, 3),
        preprocess=round(timings.preprocess, 3),
        inference=round(timings.inference, 3),
        total=round((perf_counter() - began) * 1000, 3),
    )
