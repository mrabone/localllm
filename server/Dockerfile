# Builder stage: Install dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install uv

# Copy workspace config and all packages needed by the server
COPY pyproject.toml uv.lock* ./
COPY common/ common/
COPY server/ server/

# Install dependencies into /opt/venv using uv
RUN uv pip install --python /usr/local/bin/python --target /opt/venv ./common ./server


# Runtime stage: Minimal image with only installed packages
FROM python:3.12-slim

WORKDIR /app

# Copy only the installed packages from builder
COPY --from=builder /opt/venv /opt/venv

# Copy only the server source (needed for module execution)
COPY server/ server/

# Add the installed packages to Python path so they can be imported
ENV PYTHONPATH=/opt/venv \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "server.main"]
