"""The Civil Comments private test set, with the identity labels the bias metrics need."""

from __future__ import annotations

import csv
import gzip
import hashlib
from dataclasses import dataclass

import numpy as np
from huggingface_hub import hf_hub_download

from .datasets import TextItem

# Jigsaw's test_private_expanded.csv (CC0), mirrored on the Hub; the official copy is behind a
# Kaggle login. The hash pins the content regardless of the mirror.
REPO = "shuttie/jigsaw-unintended-bias"
REVISION = "9beb047ad342ac2c3ac209d2d74e0b7c833fe294"
FILENAME = "data/test_private_expanded.csv.gz"
SHA256 = "994cda8329554422d63a6691b66dcce838bc596f8975540f8e2d8b53cd993482"

ATTRIBUTES = (
    "toxicity",
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
    "sexual_explicit",
)
# The subgroups the Jigsaw Unintended Bias benchmark scores.
IDENTITIES = (
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness",
)
# Annotator fractions at or above this count as a positive label, as in the benchmark.
POSITIVE_FRACTION = 0.5
_HASH_CHUNK = 1 << 20


@dataclass(frozen=True, slots=True)
class CivilComments:
    """Comments with boolean labels per attribute and per identity subgroup."""

    version: str
    items: list[TextItem]
    labels: dict[str, np.ndarray]
    identities: dict[str, np.ndarray]


def load(limit: int | None = None, seed: int = 0) -> CivilComments:
    """
    Return the test set, downloading it into the Hugging Face cache on first use.

    ``limit`` takes a seeded random sample instead of every comment. Raises ValueError when the
    downloaded file does not match the pinned hash.
    """
    path = hf_hub_download(REPO, FILENAME, repo_type="dataset", revision=REVISION)
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(_HASH_CHUNK):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise ValueError(f"{FILENAME} has SHA-256 {digest.hexdigest()}, expected {SHA256}")

    with gzip.open(path, "rt", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    version = SHA256[:12]
    if limit is not None and limit < len(rows):
        picked = np.random.default_rng(seed).choice(len(rows), size=limit, replace=False)
        rows = [rows[i] for i in sorted(picked)]
        version = f"{version}-sample{limit}-seed{seed}"

    def flags(column: str) -> np.ndarray:
        # Identity columns are blank for comments that were not identity-annotated.
        return np.array([float(row[column] or 0) >= POSITIVE_FRACTION for row in rows])

    return CivilComments(
        version=version,
        items=[TextItem(f"civil-comments/{row['id']}", row["comment_text"]) for row in rows],
        labels={attribute: flags(attribute) for attribute in ATTRIBUTES},
        identities={identity: flags(identity) for identity in IDENTITIES},
    )
