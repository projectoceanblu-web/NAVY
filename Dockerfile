# GDAL/PROJ arrive as rasterio wheels, so the slim image is enough.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl is used by the compose healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so application edits do not invalidate the wheel layer.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY bob_sentinel/ ./bob_sentinel/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/

# Run as a non-root user; /data is a mounted volume so it needs to be writable.
RUN useradd --create-home --uid 10001 sentinel \
    && mkdir -p /data/scenes /data/work \
    && chown -R sentinel:sentinel /app /data
USER sentinel

ENV DATA_DIR=/data
EXPOSE 8000

CMD ["uvicorn", "bob_sentinel.main:app", "--host", "0.0.0.0", "--port", "8000"]
