# Builder stage: Install dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install uv

# Copy workspace config and all packages needed by the server
COPY pyproject.toml uv.lock* ./
COPY common/ common/
COPY server/ server/

# Build wheels from local workspace packages (produces non-editable installs).
# Step 1: install everything (including third-party deps) via the workspace paths —
#         this may create editable .pth files for the local packages.
# Step 2: overwrite only the local packages with the pre-built wheels so the
#         .pth files are replaced by proper, self-contained package files.
RUN uv build --wheel --out-dir /tmp/wheels ./common \
    && uv build --wheel --out-dir /tmp/wheels ./server \
    && uv pip install --python /usr/local/bin/python --target /opt/venv ./common ./server \
    && uv pip install --python /usr/local/bin/python --target /opt/venv \
       /tmp/wheels/common-*.whl /tmp/wheels/server-*.whl


# Runtime stage: Minimal image with only installed packages
FROM python:3.12-slim

WORKDIR /app

# Copy only the installed packages from builder — all code is in the wheels,
# no source directories needed at runtime.
COPY --from=builder /opt/venv /opt/venv

# Add the installed packages to Python path so they can be imported
ENV PYTHONPATH=/opt/venv \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "server.main"]
