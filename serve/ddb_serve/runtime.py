"""Loaded variants, their bounded executors, and the pool the routes draw from."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import onnxruntime as ort
from prometheus_client import Counter, Gauge

from .language_id import LanguageIdentifier
from .registry import ModelMeta, Variant, load_meta, load_registry, select, verify
from .text import SequenceClassifier
from .vision import ImageClassifier, SiglipEncoder

_LOGGER = logging.getLogger(__name__)

_QUEUE_DEPTH = Gauge("ddb_queue_depth", "Requests running or waiting, per variant.", ["variant"])
_REJECTIONS = Counter("ddb_rejections_total", "Requests refused with a full queue.", ["variant"])


class Model(Protocol):
    """Anything a variant can serve; ``labels`` names the output columns."""

    labels: tuple[str, ...]

    def warm_up(self) -> None:
        """Run one inference so the first real request pays no one-time costs."""


_LOADERS: dict[str, Callable[[Path, ModelMeta, Variant], Model]] = {
    "toxicity": SequenceClassifier.load,
    "nli": SequenceClassifier.load,
    "language": LanguageIdentifier.load,
    "image-embed": SiglipEncoder.load,
    "obscenity": ImageClassifier.load,
}


class ModelNotFoundError(Exception):
    """No loaded variant matches the request."""


class ModelRequiredError(Exception):
    """Several variants serve the task, so the request has to name one."""


class QueueFullError(Exception):
    """The variant's queue is full; the caller should retry later."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Where the registry and models live, and which variants to load."""

    config_dir: Path
    models_dir: Path
    serve_models: str
    cors_origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> Settings:
        """
        Return settings from the environment.

        ``CONFIG_DIR`` holds the configuration (``models.json``, ``checks.json`` and the rest);
        ``MODELS_DIR`` the exported variants; ``SERVE_MODELS`` selects which variants load.
        ``CORS_ORIGINS`` is a comma-separated list of origins whose pages may call the API
        (``*`` for any); empty by default, so only same-origin pages can.
        """
        return cls(
            config_dir=Path(os.environ.get("CONFIG_DIR", "/config")),
            models_dir=Path(os.environ.get("MODELS_DIR", "/models")),
            serve_models=os.environ.get("SERVE_MODELS", "*"),
            cors_origins=tuple(
                origin.strip()
                for origin in os.environ.get("CORS_ORIGINS", "").split(",")
                if origin.strip()
            ),
        )

    @property
    def registry_path(self) -> Path:
        """Return the model registry inside the configuration directory."""
        return self.config_dir / "models.json"


@dataclass(slots=True)
class LoadedVariant:
    """A variant ready to serve, with its own bounded executor."""

    variant: Variant
    meta: ModelMeta
    model: Any
    rss_delta_mb: float
    _executor: ThreadPoolExecutor = field(init=False)
    _inflight: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Create the executor sized by the variant's ``max_concurrency``."""
        self._executor = ThreadPoolExecutor(
            max_workers=self.variant.runtime.max_concurrency,
            thread_name_prefix=self.variant.id,
        )

    async def submit[T](self, function: Callable[..., T], *args: Any) -> T:
        """
        Run ``function`` on this variant's executor and return its result.

        Raises QueueFullError when ``max_concurrency + max_queue`` requests are already in flight.
        """
        runtime = self.variant.runtime
        if self._inflight >= runtime.max_concurrency + runtime.max_queue:
            _REJECTIONS.labels(self.variant.id).inc()
            raise QueueFullError(self.variant.id)
        self._inflight += 1
        _QUEUE_DEPTH.labels(self.variant.id).set(self._inflight)
        try:
            return await asyncio.get_running_loop().run_in_executor(self._executor, function, *args)
        finally:
            self._inflight -= 1
            _QUEUE_DEPTH.labels(self.variant.id).set(self._inflight)

    def close(self) -> None:
        """Stop the executor once running work finishes."""
        self._executor.shutdown(wait=True)


class ModelPool:
    """Every loaded variant, looked up by task and id."""

    def __init__(self, variants: list[LoadedVariant], load_seconds: float) -> None:
        """Hold already-loaded variants."""
        self.variants = {loaded.variant.id: loaded for loaded in variants}
        self.load_seconds = load_seconds

    @classmethod
    def load(cls, settings: Settings) -> ModelPool:
        """
        Load, verify and warm up every variant ``settings.serve_models`` selects.

        Raises RegistryError when a variant is missing, unexported or fails verification.
        """
        began = perf_counter()
        loaded = []
        for variant in select(load_registry(settings.registry_path), settings.serve_models):
            if variant.task not in _LOADERS:
                _LOGGER.warning("Skipping %s: no loader for task %s", variant.id, variant.task)
                continue
            directory = settings.models_dir / variant.directory
            meta = load_meta(directory)
            verify(directory, meta)
            rss_before = rss_mb()
            model = _LOADERS[variant.task](directory, meta, variant)
            model.warm_up()
            loaded.append(LoadedVariant(variant, meta, model, rss_mb() - rss_before))
            _LOGGER.info("Loaded %s", variant.id)
        _check_thread_budget([item.variant for item in loaded])
        return cls(loaded, perf_counter() - began)

    def resolve(self, task: str, model_id: str | None) -> LoadedVariant:
        """
        Return the variant serving ``task``, chosen by ``model_id``.

        ``model_id`` may be omitted when exactly one variant for the task is loaded. Raises
        ModelNotFoundError when no matching variant is loaded, and ModelRequiredError when
        ``model_id`` is omitted but several variants serve the task.
        """
        if model_id is not None:
            loaded = self.variants.get(model_id)
            if loaded is None or loaded.variant.task != task:
                raise ModelNotFoundError(f"no {task} variant {model_id!r} is loaded")
            return loaded
        candidates = [item for item in self.variants.values() if item.variant.task == task]
        if not candidates:
            raise ModelNotFoundError(f"no {task} variant is loaded")
        if len(candidates) > 1:
            raise ModelRequiredError(
                f"{len(candidates)} {task} variants are loaded; name one with 'model'"
            )
        return candidates[0]

    def close(self) -> None:
        """Shut down every variant's executor."""
        for loaded in self.variants.values():
            loaded.close()


def runtime_info() -> dict[str, Any]:
    """Return what a benchmark report needs to know about this process's environment."""
    return {
        "onnxruntime": ort.__version__,
        "python": sys.version.split()[0],
        "cpus_available": available_cpus(),
        "rss_mb": round(rss_mb(), 1),
    }


def available_cpus() -> int:
    """Return how many CPUs this process may run on, honouring a pinned CPU set."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def rss_mb() -> float:
    """Return this process's resident memory in MB (peak, where current isn't available)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    import resource  # noqa: PLC0415 - only reached off Linux

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def _check_thread_budget(variants: list[Variant]) -> None:
    wanted = sum(v.runtime.max_concurrency * v.runtime.intra_op_threads for v in variants)
    if wanted > (cpus := available_cpus()):
        _LOGGER.warning(
            "Loaded variants can use %d threads at once, but only %d CPUs are available",
            wanted,
            cpus,
        )
