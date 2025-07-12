FROM python:3.11.4-slim-bullseye AS prod

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app/src

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/src/.venv/bin:$PATH"

CMD ["python", "-m", "diagnosis_service"]

FROM prod AS dev

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen