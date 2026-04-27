# ETA Forge serving image.
#
# Serving needs psycopg, LightGBM and FastAPI but none of the ingestion stack, so the
# build installs from requirements.txt and drops the compiler afterwards. Model
# artifacts are mounted at /artifacts rather than baked in: a model is data with its
# own release cadence, and rebuilding an image to ship a retrained tree is the wrong
# unit of deployment.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends libgomp1 curl; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY eta_forge ./eta_forge
COPY configs ./configs
COPY sql ./sql
COPY scripts ./scripts

# The service reads models read-only; it never writes here.
RUN mkdir -p /artifacts/models /artifacts/eval

RUN useradd --create-home --uid 10001 eta && chown -R eta:eta /app
USER eta

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fsS http://localhost:8080/health || exit 1

ENTRYPOINT ["python", "-m", "eta_forge.server"]
CMD ["--config", "configs/eta_forge.yaml"]
