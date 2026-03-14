# Builder stage: install dependencies and build wheels
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock* ./
COPY common/ common/
COPY mcp/ mcp/

RUN uv build --wheel --out-dir /tmp/wheels ./common \
    && uv build --wheel --out-dir /tmp/wheels ./mcp \
    && uv pip install --python /usr/local/bin/python --target /opt/venv ./common ./mcp \
    && uv pip install --python /usr/local/bin/python --target /opt/venv \
       /tmp/wheels/common-*.whl /tmp/wheels/mcp_server-*.whl


# Runtime stage: minimal image with only the installed packages
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

ENV PYTHONPATH=/opt/venv \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "mcp_server.main"]


# Dev stage: source supplied via bind mounts for live reloading
FROM python:3.12-slim AS dev

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock* ./
COPY common/ common/
COPY mcp/ mcp/

RUN uv pip install --python /usr/local/bin/python --system \
    ./common ./mcp

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "mcp_server.main"]
