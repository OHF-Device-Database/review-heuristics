"""
Command line: ``python -m ddb_checks {scores,compare}``.

``compare`` runs two versions of the configuration over the benchmark cases and reports what
changed; ``scores`` fills the score cache it reads from a running model server.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

from . import benchmark
from . import config as checks_config


def main(argv: list[str] | None = None) -> int:
    """Run the command line; returns the process exit code."""
    args = _parser().parse_args(argv)
    if args.command == "scores":
        return asyncio.run(_scores(args))
    return asyncio.run(_compare(args))


async def _scores(args: argparse.Namespace) -> int:
    config = checks_config.load(args.config_dir)
    cases = benchmark.load_cases(sorted((args.cases / "cases").glob("*.jsonl")), args.root)
    cache = benchmark.ScoreCache.read(args.cases / "scores")
    async with httpx.AsyncClient(base_url=args.serve_url, timeout=600) as client:
        scorer = benchmark.CachingScorer(cache, benchmark.HttpScorer(client))
        result = await benchmark.run(cases, config, scorer)
    cache.write(args.cases / "scores")
    print(benchmark.summary(result))
    return 0


async def _compare(args: argparse.Namespace) -> int:
    try:
        base_files = _files_at(args.base, args.config_dir, ".json", recursive=True)
        base_scores = _files_at(args.base, args.cases / "scores", ".jsonl")
    except subprocess.CalledProcessError:
        print(f"Can't read {args.base}; fetch it first (git fetch origin) or pass BASE=<ref>.")
        return 2
    if "checks.json" not in base_files:
        print(f"{args.base} has no {args.config_dir}/checks.json, so there is nothing to compare.")
        return 2
    cases = benchmark.load_cases(sorted((args.cases / "cases").glob("*.jsonl")), args.root)
    cache = benchmark.ScoreCache.read(args.cases / "scores")
    cache.merge(base_scores)
    with tempfile.TemporaryDirectory() as scratch:
        base_dir = Path(scratch)
        for relative, text in base_files.items():
            (base_dir / relative).parent.mkdir(parents=True, exist_ok=True)
            (base_dir / relative).write_text(text)
        base_config = checks_config.load(base_dir)
    head_config = checks_config.load(args.config_dir)

    async def both(live: benchmark.Scorer | None) -> tuple[benchmark.Run, benchmark.Run]:
        scorer = benchmark.CachingScorer(cache, live)
        return (
            await benchmark.run(cases, base_config, scorer),
            await benchmark.run(cases, head_config, scorer),
        )

    try:
        if args.serve_url:
            async with httpx.AsyncClient(base_url=args.serve_url, timeout=600) as client:
                base, head = await both(benchmark.HttpScorer(client))
        else:
            base, head = await both(None)
    except benchmark.MissingScoresError as err:
        print(f"{err}. Pass --serve-url to fetch them.")
        return 2
    text, regressed = benchmark.compare(cases, base, head)
    print(text)
    if args.output:
        args.output.write_text(text)
    return 1 if regressed else 0


def _files_at(ref: str, directory: Path, suffix: str, recursive: bool = False) -> dict[str, str]:
    """
    Return the files under ``directory`` as they are at git ``ref``.

    Keys are paths relative to ``directory`` with ``recursive``, otherwise file stems.
    """
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("compare needs git on the PATH")
    root = directory.as_posix().rstrip("/")
    # A fixed git command; the only caller input is the ref, passed as a single argument.
    listing = subprocess.run(  # noqa: S603
        [git, "ls-tree", "-r", "--name-only", ref, f"{root}/"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    files = {}
    for name in listing:
        relative = Path(name).relative_to(root)
        if not name.endswith(suffix) or (not recursive and len(relative.parts) > 1):
            continue
        files[str(relative) if recursive else relative.stem] = subprocess.run(  # noqa: S603
            [git, "show", f"{ref}:{name}"], capture_output=True, text=True, check=True
        ).stdout
    return files


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ddb_checks", description=__doc__)
    parser.add_argument(
        "--config-dir", type=Path, default=Path(os.environ.get("CONFIG_DIR", "config"))
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("scores", "Fill the benchmark's score cache from a model server"),
        ("compare", "Compare the config with another git ref's on the benchmark cases"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--cases",
            type=Path,
            default=Path("benchmarks"),
            help="Directory with cases/*.jsonl and scores/ (default: benchmarks)",
        )
        command.add_argument(
            "--root", type=Path, default=Path("."), help="Image paths in cases are relative to this"
        )
        command.add_argument(
            "--serve-url",
            default=os.environ.get(
                "SERVE_URL", "http://localhost:8000" if name == "scores" else ""
            ),
            help="Model server for missing scores" + ("" if name == "scores" else " (optional)"),
        )
        if name == "compare":
            command.add_argument("--base", default="origin/main", help="Git ref to compare with")
            command.add_argument("--output", type=Path, help="Also write the report here")
    return parser


if __name__ == "__main__":
    sys.exit(main())
