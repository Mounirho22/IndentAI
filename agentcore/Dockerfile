# ===========================================================================
# Amazon Bedrock AgentCore runtime container for CurriculumAI.
#
# Serves the FastAPI application (Strands Agents + Amazon Bedrock) and accepts
# structured agent-task payloads through agentcore/entrypoint.py.
#
# Build (from the repository root):
#   docker build --platform linux/amd64 -t curriculum-ai:latest \
#     -f agentcore/Dockerfile .
#
# AgentCore requires linux/amd64 images.
# ===========================================================================

# Official Python base, pinned to a minor version for reproducible builds.
FROM python:3.12-slim

# Run as a non-root user; the uid is fixed so volumes stay predictable.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    HOST=0.0.0.0 \
    MODEL_PROVIDER=bedrock \
    BEDROCK_REGION=us-east-1

WORKDIR /app

# curl is needed for the container HEALTHCHECK; ca-certificates for AWS endpoints.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so source edits never invalidate this layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Application source. scripts/ ships the Bedrock diagnostic helpers that
# main.py imports lazily; data/ seeds the demo database.
COPY main.py ./
COPY verify.py ./
COPY app/ ./app/
COPY agentcore/ ./agentcore/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Non-root runtime user owns the mutable state (SQLite DB + Strands session memory).
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/agent_sessions \
    && chown -R appuser:appuser /app
USER appuser

# AgentCore runtime port.
EXPOSE 8080

# Liveness probe — deliberately shallow (/health), so a transient Bedrock
# throttle never fails the container probe. Deep diagnostics: /api/bedrock-check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# Dual-mode entrypoint: with a JSON payload on stdin it runs one agent task
# (assess_submission / resolve_decision / status); without one it serves HTTP.
ENTRYPOINT ["python", "agentcore/entrypoint.py"]
