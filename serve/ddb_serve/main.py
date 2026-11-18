"""The FastAPI application; run with ``uvicorn --factory ddb_serve.main:create_app``."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ddb_checks import config as checks_config

from . import contributions, images, language, nli, obscenity, problems, toxicity
from .runtime import ModelPool, Settings, runtime_info

_PLAYGROUND = Path(__file__).parent / "static" / "playground.html"


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Return the application, which loads its variants on startup.

    Startup fails if the configuration is invalid, or any selected variant is missing or fails
    verification, so a broken deployment never reports ready.
    """
    settings = settings or Settings.from_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.checks = checks_config.load(settings.config_dir)
        app.state.pool = ModelPool.load(settings)
        try:
            yield
        finally:
            app.state.pool.close()

    app = FastAPI(title="Device Database contribution checks", version="0.1.0", lifespan=lifespan)
    problems.install(app)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["content-type"],
        )
    for module in (contributions, toxicity, nli, language, images, obscenity):
        app.include_router(module.router)

    # The playground is a development tool; production images are built without it.
    if _PLAYGROUND.exists():

        @app.get("/playground", include_in_schema=False)
        async def playground() -> FileResponse:
            """Serve a page for trying every endpoint by hand."""
            return FileResponse(_PLAYGROUND, media_type="text/html")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Report that the process is up."""
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> dict[str, Any]:
        """Report that every selected variant is loaded, and the configuration version."""
        return {
            "status": "ready",
            "config": request.app.state.checks.version,
            "models": sorted(request.app.state.pool.variants),
        }

    @app.get("/v1/models")
    async def models(request: Request) -> dict[str, Any]:
        """List the loaded variants with their registry metadata and the process environment."""
        pool: ModelPool = request.app.state.pool
        return {
            "runtime": {**runtime_info(), "load_seconds": round(pool.load_seconds, 2)},
            "models": [
                {
                    "id": loaded.variant.id,
                    "task": loaded.variant.task,
                    "precision": loaded.variant.precision,
                    "source": loaded.variant.source,
                    "license": loaded.variant.license,
                    "sha256": loaded.meta.sha256,
                    "labels": list(loaded.model.labels),
                    "runtime": asdict(loaded.variant.runtime),
                    "rss_delta_mb": round(loaded.rss_delta_mb, 1),
                }
                for loaded in pool.variants.values()
            ],
        }

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        """Expose Prometheus metrics."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
