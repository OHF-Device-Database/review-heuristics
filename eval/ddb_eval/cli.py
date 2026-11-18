"""Command line: ``python -m ddb_eval {export,parity,bench,quality} …``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from . import civil_comments, cost, datasets, export, images, parity, quality, report
from .registry import VariantId, find

_READY_TIMEOUT_SECONDS = 600
# Zero-shot classes for SigLIP parity: one prompt each, so the original model's own
# logits_per_image is the reference.
_ZERO_SHOT_PROMPTS = [
    "a product photo of a device on a plain background",
    "a smart home device installed in a room",
    "the packaging box of an electronic device",
    "a screenshot of a phone app",
]
_READY_POLL_SECONDS = 2


def main(argv: list[str] | None = None) -> int:
    """Run the command line; returns the process exit code."""
    args = _parser().parse_args(argv)
    # Loading a .bin-only checkpoint otherwise starts a background thread that asks the Hub's
    # bot to convert the repo to safetensors: an outward action, and a source of extra
    # downloads from a ref other than the pinned revision.
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # One line per request would bury the progress messages.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.command == "export":
        for variant_id in args.variants:
            export.export(variant_id, _registry(args), args.models_dir)
        return 0
    if args.command == "quality":
        return _quality(args)
    if args.command == "parity":
        return _parity(args)
    return _bench(args)


def _bench(args: argparse.Namespace) -> int:
    variant_id = args.variant
    VariantId.parse(variant_id)
    entry = find(_registry(args), variant_id)
    dataset = datasets.load_text(args.dataset, with_long_text=True)
    started = datetime.now(UTC).isoformat(timespec="seconds")

    with httpx.Client(base_url=args.serve_url, timeout=120) as client:
        catalogue, served = _served(client, variant_id)
        if served is None:
            return 1

        logging.info("Parity: %d texts", len(dataset.items))
        labels = served["labels"]
        got = parity.served_scores(client, variant_id, labels, dataset.items)
        expected = parity.reference_scores(entry["source"], labels, dataset.items)
        comparison = parity.compare(expected, got)
        passed = parity.passes(VariantId.parse(variant_id).precision, comparison)

    logging.info("Model cost")
    model_dir = args.models_dir / VariantId.parse(variant_id).directory
    model_rows = cost.model_cost(model_dir, served["runtime"]["intra_op_threads"], dataset.items)

    steps, samples = [], []
    for rate in args.rates:
        logging.info("Endpoint load at %s/s for %ss", rate, args.duration)
        step, step_samples = asyncio.run(
            cost.open_loop(args.serve_url, variant_id, dataset.items, rate, args.duration)
        )
        steps.append(step)
        samples.extend(step_samples)

    metrics: dict[str, Any] = {
        "variant": variant_id,
        "started": started,
        "harness": report.harness_version(),
        "dataset": {"name": dataset.name, "version": dataset.version},
        "environment": {"host": cost.host_environment(), "server": catalogue["runtime"]},
        "served": {**served, "source": entry["source"]},
        "parity": {
            "items": len(dataset.items),
            "max_abs_diff": comparison.max_abs_diff,
            "mean_abs_diff": comparison.mean_abs_diff,
            "decision_agreement": comparison.decision_agreement,
            "passed": passed,
        },
        "cost": {
            "target_cpu": args.target_cpu,
            "model": cost.as_dicts(model_rows),
            "endpoint": cost.as_dicts(steps),
            "ceiling_ms": args.ceiling_ms,
            "capacity_per_second": cost.capacity(steps, args.ceiling_ms),
        },
    }
    results = {
        "parity": [
            {
                "id": item.id,
                "reference": dict(zip(labels, ref, strict=True)),
                "served": dict(zip(labels, srv, strict=True)),
            }
            for item, ref, srv in zip(dataset.items, expected.tolist(), got.tolist(), strict=True)
        ],
        "parity_worst": [
            {"id": dataset.items[row].id, "label": labels[col], "reference": ref, "served": srv}
            for row, col, ref, srv in comparison.worst
        ],
        "endpoint_samples": samples,
    }
    out = report.write(args.reports_dir, variant_id, metrics, results, report.render(metrics))
    print(f"Report written to {out}")
    return 0 if passed is not False else 1


def _parity(args: argparse.Namespace) -> int:
    variant_id = args.variant
    parsed = VariantId.parse(variant_id)
    entry = find(_registry(args), variant_id)
    if parsed.task == "language":
        print(
            f"{variant_id} runs the published file on its own runtime; there is no export to check"
        )
        return 0
    started = datetime.now(UTC).isoformat(timespec="seconds")

    with httpx.Client(base_url=args.serve_url, timeout=300) as client:
        catalogue, served = _served(client, variant_id)
        if served is None:
            return 1
        labels = served["labels"]
        cosine = None
        if parsed.task == "toxicity":
            text = datasets.load_text(
                args.datasets_dir / "parity-text-v0.jsonl", with_long_text=True
            )
            dataset = {"name": text.name, "version": text.version}
            ids = [item.id for item in text.items]
            got = parity.served_scores(client, variant_id, labels, text.items)
            expected = parity.reference_scores(entry["source"], labels, text.items)
        elif parsed.task == "nli":
            version, pairs = datasets.load_pairs(args.datasets_dir / "parity-nli-v0.jsonl")
            dataset = {"name": "parity-nli-v0.jsonl", "version": version}
            ids = [pair.id for pair in pairs]
            got = parity.served_nli(client, variant_id, labels, pairs)
            expected = parity.reference_nli(entry["source"], labels, pairs)
        elif parsed.task == "obscenity":
            synthetic = images.synthetic()
            dataset = {"name": "synthetic images", "version": report.harness_version()}
            ids = [image.id for image in synthetic]
            got = parity.served_obscenity(client, variant_id, labels, synthetic)
            expected = parity.reference_obscenity(entry["source"], labels, synthetic)
        elif parsed.task == "image-embed":
            synthetic = images.synthetic()
            dataset = {"name": "synthetic images", "version": report.harness_version()}
            ids = [image.id for image in synthetic]
            labels = _ZERO_SHOT_PROMPTS
            served_embeddings = parity.served_embeddings(client, variant_id, synthetic)
            got = parity.served_zero_shot(client, variant_id, labels, synthetic)
            reference_embeddings, expected = parity.reference_siglip(
                entry["source"], labels, synthetic
            )
            cosine = parity.min_cosine(reference_embeddings, served_embeddings)
        else:
            print(f"no parity check for task {parsed.task!r} yet")
            return 1

    comparison = parity.compare(expected, got)
    passed = parity.passes(parsed.precision, comparison)
    if passed is not None and cosine is not None:
        passed = passed and cosine >= parity.FP32_MIN_COSINE
    metrics: dict[str, Any] = {
        "variant": variant_id,
        "started": started,
        "harness": report.harness_version(),
        "dataset": dataset,
        "environment": {"host": cost.host_environment(), "server": catalogue["runtime"]},
        "served": {**served, "source": entry["source"], "labels": labels},
        "parity": {
            "items": len(ids),
            "max_abs_diff": comparison.max_abs_diff,
            "mean_abs_diff": comparison.mean_abs_diff,
            "decision_agreement": comparison.decision_agreement,
            "min_cosine": cosine,
            "passed": passed,
        },
    }
    results = {
        "parity": [
            {
                "id": item_id,
                "reference": dict(zip(labels, ref, strict=True)),
                "served": dict(zip(labels, srv, strict=True)),
            }
            for item_id, ref, srv in zip(ids, expected.tolist(), got.tolist(), strict=True)
        ],
        "parity_worst": [
            {"id": ids[row], "label": labels[col], "reference": ref, "served": srv}
            for row, col, ref, srv in comparison.worst
        ],
    }
    out = report.write(args.reports_dir, variant_id, metrics, results, report.render(metrics))
    print(f"Report written to {out}")
    return 0 if passed is not False else 1


def _registry(args: argparse.Namespace) -> Path:
    return args.config_dir / "models.json"


def _quality(args: argparse.Namespace) -> int:
    variant_id = args.variant
    VariantId.parse(variant_id)
    entry = find(_registry(args), variant_id)
    device = datasets.load_text(args.device_set)
    logging.info("Loading Civil Comments")
    civil = civil_comments.load(args.limit)
    started = datetime.now(UTC).isoformat(timespec="seconds")

    with httpx.Client(base_url=args.serve_url, timeout=300) as client:
        catalogue, served = _served(client, variant_id)
        if served is None:
            return 1
        heads = served["labels"]
        logging.info("Device text: %d texts", len(device.items))
        device_scores = parity.served_scores(client, variant_id, heads, device.items)
        logging.info("Civil Comments: %d texts", len(civil.items))
        civil_scores = parity.served_scores(client, variant_id, heads, civil.items)

    metrics: dict[str, Any] = {
        "variant": variant_id,
        "started": started,
        "harness": report.harness_version(),
        "datasets": {
            "device": {"name": device.name, "version": device.version},
            "civil_comments": {
                "name": f"{civil_comments.REPO}/{civil_comments.FILENAME}",
                "version": civil.version,
            },
        },
        "environment": {"host": cost.host_environment(), "server": catalogue["runtime"]},
        "served": {**served, "source": entry["source"]},
        "device": quality.device_metrics(device.items, heads, device_scores),
        "civil_comments": quality.civil_comments_metrics(civil, heads, civil_scores),
    }
    results = {
        "device": [
            {"id": item.id, "scores": dict(zip(heads, row, strict=True))}
            for item, row in zip(device.items, device_scores.tolist(), strict=True)
        ],
        # Columns follow metrics["served"]["labels"]; ids follow the dataset order.
        "civil_comments": {
            "ids": [item.id for item in civil.items],
            "scores": np.round(civil_scores, 6).tolist(),
        },
    }
    out = report.write(
        args.reports_dir, variant_id, metrics, results, report.render_quality(metrics)
    )
    print(f"Report written to {out}")
    return 0


def _served(client: httpx.Client, variant_id: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    _wait_ready(client)
    catalogue = client.get("/v1/models").json()
    served = next((m for m in catalogue["models"] if m["id"] == variant_id), None)
    if served is None:
        print(f"{variant_id} is not loaded; start serve with SERVE_MODELS={variant_id}")
    return catalogue, served


def _wait_ready(client: httpx.Client) -> None:
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    while True:
        try:
            if client.get("/readyz").status_code == httpx.codes.OK:
                return
        except httpx.TransportError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(f"serve was not ready within {_READY_TIMEOUT_SECONDS} s")
        time.sleep(_READY_POLL_SECONDS)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ddb_eval", description=__doc__)
    parser.add_argument(
        "--config-dir", type=Path, default=Path(os.environ.get("CONFIG_DIR", "config"))
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path(os.environ.get("MODELS_DIR", "models"))
    )
    commands = parser.add_subparsers(dest="command", required=True)

    export_cmd = commands.add_parser("export", help="Export variants to ONNX")
    export_cmd.add_argument("variants", nargs="+", metavar="VARIANT")

    parity_cmd = commands.add_parser("parity", help="Check one served variant against its original")
    parity_cmd.add_argument("variant", metavar="VARIANT")
    parity_cmd.add_argument(
        "--serve-url", default=os.environ.get("SERVE_URL", "http://localhost:8000")
    )
    parity_cmd.add_argument(
        "--reports-dir", type=Path, default=Path(os.environ.get("REPORTS_DIR", "reports"))
    )
    parity_cmd.add_argument(
        "--datasets-dir", type=Path, default=Path(os.environ.get("DATASETS_DIR", "eval/datasets"))
    )

    bench = commands.add_parser("bench", help="Run parity and cost for one served variant")
    bench.add_argument("variant", metavar="VARIANT")
    bench.add_argument("--serve-url", default=os.environ.get("SERVE_URL", "http://localhost:8000"))
    bench.add_argument(
        "--reports-dir", type=Path, default=Path(os.environ.get("REPORTS_DIR", "reports"))
    )
    bench.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.environ.get("DATASETS_DIR", "eval/datasets")) / "parity-text-v0.jsonl",
    )
    bench.add_argument(
        "--rates",
        type=lambda value: [float(rate) for rate in value.split(",")],
        default=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
        help="Comma-separated arrival rates per second (default: 1,2,5,10,20,50,100)",
    )
    bench.add_argument("--duration", type=float, default=15.0, help="Seconds per rate")
    bench.add_argument("--ceiling-ms", type=float, default=2000.0, help="p95 latency ceiling")
    bench.add_argument(
        "--target-cpu", action="store_true", help="Mark the run as measured on the target CPU"
    )

    quality_cmd = commands.add_parser(
        "quality", help="Score Civil Comments and the device set through one served variant"
    )
    quality_cmd.add_argument("variant", metavar="VARIANT")
    quality_cmd.add_argument(
        "--serve-url", default=os.environ.get("SERVE_URL", "http://localhost:8000")
    )
    quality_cmd.add_argument(
        "--reports-dir", type=Path, default=Path(os.environ.get("REPORTS_DIR", "reports"))
    )
    quality_cmd.add_argument(
        "--device-set",
        type=Path,
        default=Path(os.environ.get("DATASETS_DIR", "eval/datasets")) / "device-text-v0.jsonl",
    )
    quality_cmd.add_argument(
        "--limit", type=int, help="Score a seeded random sample of this many comments"
    )

    return parser


if __name__ == "__main__":
    sys.exit(main())
