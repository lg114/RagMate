FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Install curl for healthcheck and basic build utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. Install Python dependencies using pyproject.toml (Docker layer cached)
COPY backend/pyproject.toml /app/backend/
RUN python -c "import tomllib; p = tomllib.load(open('/app/backend/pyproject.toml', 'rb')); print('\n'.join(p['project']['dependencies']))" > /tmp/requirements.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# 2. Copy application source code and frontend assets
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Create directory for documents
RUN mkdir -p /app/documents

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
