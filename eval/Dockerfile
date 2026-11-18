# Build context: the repository root.
FROM ghcr.io/astral-sh/uv:0.12.0 AS uv

FROM python:3.12.13-slim AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY serve/pyproject.toml serve/
COPY eval/pyproject.toml eval/
COPY checks checks
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --package ddb-eval

FROM python:3.12.13-slim
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY eval/ddb_eval ./ddb_eval
COPY eval/datasets /datasets
COPY config /config
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CONFIG_DIR=/config MODELS_DIR=/models DATASETS_DIR=/datasets \
    REPORTS_DIR=/reports SERVE_URL=http://serve:8000 HF_HOME=/cache/huggingface \
    DISABLE_SAFETENSORS_CONVERSION=1
ENTRYPOINT ["python", "-m", "ddb_eval"]
