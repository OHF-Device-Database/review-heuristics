"""Text models served through ONNX Runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .registry import ModelMeta, RegistryError, Variant

# A text, or a (first, second) pair such as an NLI premise and hypothesis.
type TextInput = str | tuple[str, str]


@dataclass(slots=True)
class Timings:
    """Time spent per stage, in milliseconds."""

    decode: float = 0.0
    preprocess: float = 0.0
    inference: float = 0.0


class SequenceClassifier:
    """Scores texts, or pairs of texts, with an ONNX sequence-classification model."""

    def __init__(
        self,
        session: ort.InferenceSession,
        tokenizer: Tokenizer,
        meta: ModelMeta,
        variant: Variant,
    ) -> None:
        """Wrap a loaded session and tokenizer; ``variant.labels`` picks the exposed heads."""
        self._session = session
        self._tokenizer = tokenizer
        self._output = meta.require("output")
        self._batch_size = variant.runtime.batch_size
        self._input_names = {node.name for node in session.get_inputs()}

        if self._output not in {"sigmoid", "softmax"}:
            raise RegistryError(f"{variant.id} has unknown output activation {self._output!r}")
        self.model_labels: tuple[str, ...] = tuple(meta.require("labels"))
        exposed = variant.labels or self.model_labels
        if unknown := [label for label in exposed if label not in self.model_labels]:
            raise RegistryError(f"{variant.id} exposes labels the model lacks: {unknown}")
        self.labels: tuple[str, ...] = tuple(exposed)
        self._columns = np.array([self.model_labels.index(label) for label in exposed])

    @classmethod
    def load(cls, directory: Path, meta: ModelMeta, variant: Variant) -> SequenceClassifier:
        """Return a classifier for the exported variant in ``directory``."""
        tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        tokenizer.enable_truncation(meta.require("max_length"))
        tokenizer.enable_padding(pad_id=meta.require("pad_id"), pad_token=meta.require("pad_token"))
        return cls(create_session(directory / "model.onnx", variant), tokenizer, meta, variant)

    def warm_up(self) -> None:
        """Run one inference so the first real request pays no one-time costs."""
        self.predict(["warm-up"])

    def predict(self, texts: list[TextInput]) -> tuple[np.ndarray, Timings]:
        """Return one row of probabilities per input, with columns in ``self.labels`` order."""
        logits, timings = self.logits(texts)
        # Softmax runs over every head, including those not exposed.
        probabilities = 1 / (1 + np.exp(-logits)) if self._output == "sigmoid" else softmax(logits)
        return probabilities[:, self._columns], timings

    def logits(self, texts: list[TextInput]) -> tuple[np.ndarray, Timings]:
        """Return the raw logits per input, with columns in ``self.model_labels`` order."""
        timings = Timings()
        rows = []
        for start in range(0, len(texts), self._batch_size):
            began = perf_counter()
            encodings = self._tokenizer.encode_batch(texts[start : start + self._batch_size])
            feeds = {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            }
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
            tokenized = perf_counter()
            rows.append(self._session.run(None, feeds)[0])
            timings.preprocess += (tokenized - began) * 1000
            timings.inference += (perf_counter() - tokenized) * 1000
        return np.concatenate(rows), timings


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return the softmax of ``logits`` along ``axis``."""
    exp = np.exp(logits - logits.max(axis=axis, keepdims=True))
    return exp / exp.sum(axis=axis, keepdims=True)


def create_session(path: Path, variant: Variant) -> ort.InferenceSession:
    """Return a CPU session using the variant's thread settings."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = variant.runtime.intra_op_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_cpu_mem_arena = variant.runtime.memory_arena
    options.enable_mem_pattern = variant.runtime.memory_arena
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
