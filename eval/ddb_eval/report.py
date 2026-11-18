"""Benchmark reports: one directory per run, with raw results, metrics and a summary."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def harness_version() -> str:
    """Return a short hash of the harness source, so a report names the code that made it."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def write(
    reports_dir: Path,
    variant_id: str,
    metrics: dict[str, Any],
    results: dict[str, Any],
    summary: str,
) -> Path:
    """Write a run's ``metrics.json``, ``results.json`` and ``report.md``; return the directory."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = reports_dir / f"{stamp}-{variant_id.replace('/', '-').replace('@', '-')}"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out / "results.json").write_text(json.dumps(results) + "\n")
    (out / "report.md").write_text(summary)
    return out


def render(metrics: dict[str, Any]) -> str:
    """Return the Markdown summary of a bench run."""
    env, served = metrics["environment"], metrics["served"]
    flags = ", ".join(env["host"]["cpu_flags"]) or "none of the tracked flags"
    lines = [
        f"# {metrics['variant']}",
        "",
        f"**Environment:** {env['host']['cpu_model']} ({env['host']['machine']}), "
        f"{env['server']['cpus_available']} CPUs available to serve, CPU flags: {flags}. "
        f"ONNX Runtime {env['server']['onnxruntime']}.",
        "",
        f"Model SHA-256 `{served['sha256']}` · dataset `{metrics['dataset']['name']}` "
        f"version `{metrics['dataset']['version']}` · harness `{metrics['harness']}` · "
        f"{metrics['started']}",
        "",
    ]

    if parity := metrics.get("parity"):
        verdict = {True: "pass", False: "FAIL", None: "reported only (not fp32)"}[parity["passed"]]
        cosine = parity.get("min_cosine")
        lines += [
            "## Gate 1: parity",
            "",
            f"Against `{served['source']}` in PyTorch, {parity['items']} items by "
            f"{len(served['labels'])} scores. **{verdict}**",
            "",
            "| Max abs diff | Mean abs diff | Decision agreement at 0.5 | Min embedding cosine |",
            "| --- | --- | --- | --- |",
            f"| {parity['max_abs_diff']:.2e} | {parity['mean_abs_diff']:.2e} | "
            f"{parity['decision_agreement']:.4%} | "
            f"{'n/a' if cosine is None else f'{cosine:.6f}'} |",
            "",
        ]

    if cost := metrics.get("cost"):
        lines += [
            "## Gate 3: cost",
            "",
            "Provisional: not measured on the target CPU."
            if not cost.get("target_cpu")
            else "Measured on the target CPU.",
            "",
            "**Model cost** (in-process `session.run`, "
            f"{served['runtime']['intra_op_threads']} intra-op threads):",
            "",
            "| Batch | p50 ms | p95 ms |",
            "| --- | --- | --- |",
            *(f"| {row['batch_size']} | {row['p50']} | {row['p95']} |" for row in cost["model"]),
            "",
            "**Endpoint latency** (open-loop, one text per request, latency from scheduled send):",
            "",
            "| Rate/s | Sent | OK | 503 | Failed | p50 ms | p95 ms | p99 ms "
            "| Server inference p50 | Server tokenize p50 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            *(
                f"| {s['rate']} | {s['sent']} | {s['ok']} | {s['rejected']} | {s['failed']} "
                f"| {s['p50']} | {s['p95']} | {s['p99']} | {s['server_inference_p50']} "
                f"| {s['server_preprocess_p50']} |"
                for s in cost["endpoint"]
            ),
            "",
            f"**Capacity** at a {cost['ceiling_ms']:g} ms p95 ceiling: {_capacity(cost)}.",
            "",
            f"**Memory:** {served['rss_delta_mb']} MB resident for this variant; "
            f"server total {env['server']['rss_mb']} MB. Startup (load, verify, warm up): "
            f"{env['server']['load_seconds']} s.",
            "",
        ]
    return "\n".join(lines)


def render_quality(metrics: dict[str, Any]) -> str:
    """Return the Markdown summary of a quality run."""
    served, device, civil = metrics["served"], metrics["device"], metrics["civil_comments"]
    sets = metrics["datasets"]
    lines = [
        f"# {metrics['variant']}: quality",
        "",
        f"Model SHA-256 `{served['sha256']}` · `{sets['device']['name']}` version "
        f"`{sets['device']['version']}` · Civil Comments version "
        f"`{sets['civil_comments']['version']}` · harness `{metrics['harness']}` · "
        f"{metrics['started']}",
        "",
        "## Device text",
        "",
        f"A text counts as flagged when any head scores at least {device['threshold']:g}. "
        "Labels are drafts until reviewed.",
        "",
        f"- **Benign flagged:** {_share(device['benign_flagged'])}",
        f"- **Toxic caught:** {_share(device['toxic_caught'])}",
        "",
        "| Category | Label | Texts | Flagged | Rate (95% CI) | Flagged by head |",
        "| --- | --- | --- | --- | --- | --- |",
        *(
            f"| {name} | {row['label']} | {row['texts']} | {row['flagged']} "
            f"| {_percent(row['flagged_rate'])} ({_interval(row['flagged_95'])}) "
            f"| {_by_head(row['flagged_by_head'])} |"
            for name, row in device["categories"].items()
        ),
        "",
        "**Errors** (benign flagged, toxic missed), highest score first:",
        "",
        "| Id | Label | Top head | Score | Text |",
        "| --- | --- | --- | --- | --- |",
        *(
            f"| {e['id']} | {e['label']} | {e['head']} | {e['score']:.3f} "
            f"| {e['text'].replace('|', '/')} |"
            for e in device["errors"]
        ),
        "",
        "## Civil Comments",
        "",
        f"{civil['texts']:,} comments. A label is positive when at least half the annotators "
        f"chose it; decisions at {civil['threshold']:g}.",
        "",
        "| Head | Attribute | Positives | AUROC (95% CI) | Average precision "
        "| Recall | False-positive rate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *(
            f"| {head} | {row['attribute']} | {row['positives']:,} "
            f"| {_number(row['auroc'])} ({_interval(row['auroc_95'], percent=False)}) "
            f"| {_number(row['average_precision'])} | {_percent(row['recall_at_threshold'])} "
            f"| {_percent(row['false_positive_rate_at_threshold'])} |"
            for head, row in civil["heads"].items()
        ),
        "",
        f"**Bias** on the `{civil['bias']['head']}` head: overall AUC "
        f"{_number(civil['bias']['overall_auc'])}, Jigsaw final score "
        f"{_number(civil['bias']['final'])}.",
        "",
        "| Subgroup | Size | Toxic | Subgroup AUC | BPSN AUC | BNSP AUC |",
        "| --- | --- | --- | --- | --- | --- |",
        *(
            f"| {name} | {row['size']:,} | {row['positives']:,} | {_number(row['subgroup_auc'])} "
            f"| {_number(row['bpsn_auc'])} | {_number(row['bnsp_auc'])} |"
            for name, row in civil["bias"]["subgroups"].items()
        ),
        "",
    ]
    return "\n".join(lines)


def _capacity(cost: dict[str, Any]) -> str:
    rates = [step["rate"] for step in cost["endpoint"]]
    found = cost["capacity_per_second"]
    if found is None:
        return "below the lowest rate tested"
    if found == max(rates):
        return f"at least {found:g} requests/s (the highest rate tested; raise --rates to find it)"
    return f"{found:g} requests/s"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _interval(bounds: list[float] | tuple[float, float] | None, *, percent: bool = True) -> str:
    if bounds is None:
        return "n/a"
    show = _percent if percent else _number
    return f"{show(bounds[0])}\N{EN DASH}{show(bounds[1])}"


def _share(summary: dict[str, Any]) -> str:
    return (
        f"{summary['count']} of {summary['of']}, {_percent(summary['rate'])} "
        f"(95% CI {_interval(summary['rate_95'])})"
    )


def _by_head(counts: dict[str, int]) -> str:
    return ", ".join(f"{head} {count}" for head, count in counts.items() if count) or "none"
