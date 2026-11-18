"""Gate 3: model cost in-process, and endpoint latency under open-loop load."""

from __future__ import annotations

import asyncio
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .datasets import TextItem

CPU_FLAGS = (
    "avx2",
    "avx512f",
    "avx512_vnni",
    "avx512_bf16",
    "amx_tile",
    "amx_bf16",
    "amx_int8",
)
_WARM_UP_RUNS = 3
_REQUEST_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class ModelCost:
    """In-process ``session.run`` time for one batch size, in milliseconds."""

    batch_size: int
    iterations: int
    p50: float
    p95: float


@dataclass(frozen=True, slots=True)
class LoadStep:
    """Endpoint results at one fixed arrival rate."""

    rate: float
    sent: int
    ok: int
    rejected: int
    failed: int
    p50: float | None
    p95: float | None
    p99: float | None
    server_inference_p50: float | None
    server_preprocess_p50: float | None


def host_environment() -> dict[str, Any]:
    """Return the CPU model, relevant instruction-set flags and core count of this host."""
    model, flags = platform.processor() or platform.machine(), set()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "model name":
                model = value.strip()
            elif key.strip() in {"flags", "Features"}:
                flags = set(value.split())
    except OSError:
        pass
    return {
        "cpu_model": model,
        "cpu_flags": [flag for flag in CPU_FLAGS if flag in flags],
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
    }


def model_cost(
    model_dir: Path,
    intra_op_threads: int,
    items: list[TextItem],
    batch_sizes: tuple[int, ...] = (1, 8),
    iterations: int = 50,
) -> list[ModelCost]:
    """
    Time ``session.run`` alone, with the same session options ``serve`` uses.

    Tokenization happens once up front, so the numbers are pure model cost.
    """
    meta = json.loads((model_dir / "meta.json").read_text())
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(meta["max_length"])
    tokenizer.enable_padding(pad_id=meta["pad_id"], pad_token=meta["pad_token"])

    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"]
    )
    input_names = {node.name for node in session.get_inputs()}

    results = []
    for batch_size in batch_sizes:
        texts = [items[i % len(items)].text for i in range(batch_size)]
        encodings = tokenizer.encode_batch(texts)
        feeds = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        for _ in range(_WARM_UP_RUNS):
            session.run(None, feeds)
        samples = []
        for _ in range(iterations):
            began = perf_counter()
            session.run(None, feeds)
            samples.append((perf_counter() - began) * 1000)
        results.append(
            ModelCost(batch_size, iterations, _percentile(samples, 50), _percentile(samples, 95))
        )
    return results


async def open_loop(
    serve_url: str, variant_id: str, items: list[TextItem], rate: float, duration: float
) -> tuple[LoadStep, list[dict[str, Any]]]:
    """
    Send one-text requests at a fixed rate, whatever the server's response time.

    Latency is measured from each request's scheduled send time, so a server that falls
    behind can't hide its queueing delay (coordinated omission). Returns the summary and the
    per-request samples.
    """
    samples: list[dict[str, Any]] = []
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=100)
    async with httpx.AsyncClient(
        base_url=serve_url, timeout=_REQUEST_TIMEOUT_SECONDS, limits=limits
    ) as client:
        loop = asyncio.get_running_loop()

        async def send(index: int, scheduled: float) -> None:
            item = items[index % len(items)]
            sample: dict[str, Any] = {"rate": rate, "id": item.id}
            try:
                response = await client.post(
                    "/v1/toxicity",
                    json={"model": variant_id, "texts": [{"id": item.id, "text": item.text}]},
                )
                sample["status"] = response.status_code
                if response.status_code == httpx.codes.OK:
                    sample["server_ms"] = response.json()["timings_ms"]
            except httpx.HTTPError as err:
                sample["status"] = None
                sample["error"] = type(err).__name__
            sample["latency_ms"] = (loop.time() - scheduled) * 1000
            samples.append(sample)

        start = loop.time()
        tasks = []
        for index in range(int(rate * duration)):
            scheduled = start + index / rate
            await asyncio.sleep(max(0.0, scheduled - loop.time()))
            tasks.append(asyncio.create_task(send(index, scheduled)))
        await asyncio.gather(*tasks)

    ok = [s for s in samples if s["status"] == httpx.codes.OK]
    latencies = [s["latency_ms"] for s in ok]
    return (
        LoadStep(
            rate=rate,
            sent=len(samples),
            ok=len(ok),
            rejected=sum(s["status"] == httpx.codes.SERVICE_UNAVAILABLE for s in samples),
            failed=sum(
                s["status"] not in {httpx.codes.OK, httpx.codes.SERVICE_UNAVAILABLE}
                for s in samples
            ),
            p50=_percentile(latencies, 50),
            p95=_percentile(latencies, 95),
            p99=_percentile(latencies, 99),
            server_inference_p50=_percentile([s["server_ms"]["inference"] for s in ok], 50),
            server_preprocess_p50=_percentile([s["server_ms"]["preprocess"] for s in ok], 50),
        ),
        samples,
    )


def capacity(steps: list[LoadStep], ceiling_ms: float) -> float | None:
    """Return the highest rate served with no rejections or failures and p95 under the ceiling."""
    passing = [
        step.rate
        for step in steps
        if step.rejected == 0
        and step.failed == 0
        and step.p95 is not None
        and step.p95 <= ceiling_ms
    ]
    return max(passing, default=None)


def as_dicts(rows: list[Any]) -> list[dict[str, Any]]:
    """Return dataclass rows as plain dicts for JSON output."""
    return [asdict(row) for row in rows]


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(values, q)), 3)
