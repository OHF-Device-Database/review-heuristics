"""RFC 9457 problem-details responses for every error the API returns."""

from __future__ import annotations

from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from .runtime import ModelNotFoundError, ModelRequiredError, QueueFullError
from .vision import ImageError, ImageTooLargeError

_MEDIA_TYPE = "application/problem+json"
_RETRY_AFTER_SECONDS = "1"


def problem(status: int, detail: str, headers: dict[str, str] | None = None) -> JSONResponse:
    """Return a problem-details response for ``status``."""
    body = {"type": "about:blank", "title": HTTPStatus(status).phrase, "status": status}
    if detail:
        body["detail"] = detail
    return JSONResponse(body, status_code=status, headers=headers, media_type=_MEDIA_TYPE)


def install(app: FastAPI) -> None:
    """Make ``app`` answer every error with problem details."""

    @app.exception_handler(HTTPException)
    async def _http(_: Request, err: HTTPException) -> JSONResponse:
        return problem(err.status_code, str(err.detail), dict(err.headers or {}))

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, err: RequestValidationError) -> JSONResponse:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in err.errors()
        )
        return problem(HTTPStatus.UNPROCESSABLE_ENTITY, detail)

    @app.exception_handler(ModelNotFoundError)
    async def _not_found(_: Request, err: ModelNotFoundError) -> JSONResponse:
        return problem(HTTPStatus.NOT_FOUND, str(err))

    @app.exception_handler(ModelRequiredError)
    async def _model_required(_: Request, err: ModelRequiredError) -> JSONResponse:
        return problem(HTTPStatus.UNPROCESSABLE_ENTITY, str(err))

    @app.exception_handler(QueueFullError)
    async def _queue_full(_: Request, err: QueueFullError) -> JSONResponse:
        return problem(
            HTTPStatus.SERVICE_UNAVAILABLE,
            f"{err} is at capacity",
            {"Retry-After": _RETRY_AFTER_SECONDS},
        )

    @app.exception_handler(ImageError)
    async def _bad_image(_: Request, err: ImageError) -> JSONResponse:
        status = (
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE
            if isinstance(err, ImageTooLargeError)
            else HTTPStatus.UNPROCESSABLE_ENTITY
        )
        return problem(status, str(err))
