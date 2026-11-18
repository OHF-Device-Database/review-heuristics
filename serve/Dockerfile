# Build context: the repository root.
FROM ghcr.io/astral-sh/uv:0.12.0 AS uv

FROM python:3.12.13-slim AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Every workspace member's pyproject is needed to resolve the lockfile; only serve's
# dependencies, and the checks package, are installed.
COPY pyproject.toml uv.lock ./
COPY serve/pyproject.toml serve/
COPY eval/pyproject.toml eval/
COPY checks checks
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --package ddb-serve

FROM python:3.12.13-slim AS runtime
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY serve/ddb_serve ./ddb_serve
# The playground page is a development tool; production images leave it out.
ARG PLAYGROUND=false
RUN if [ "$PLAYGROUND" != "true" ]; then rm -rf ddb_serve/static; fi
COPY config /config
RUN useradd --system --uid 10001 --no-create-home serve
USER 10001
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CONFIG_DIR=/config MODELS_DIR=/models
EXPOSE 8000
CMD ["uvicorn", "--factory", "ddb_serve.main:create_app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

# Self-contained image with the exported models inside, for hosts without the models volume.
# Build with `make bundle`, which supplies the `models` build context.
FROM runtime AS bundle
COPY --from=models --chmod=a+rX . /models/
