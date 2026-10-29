# Multi-stage image for the Job Match Agent REST API.
# Build:  docker build -t job-match-agent .
# Run:    docker run --rm -p 8000:8000 job-match-agent          (fixture mode, no keys)
ARG PYTHON_VERSION=3.14

# --- builder: resolve and install pinned dependencies into a virtualenv ------------
FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /build
COPY requirements.txt requirements-retrieval.txt constraints.txt ./
# Runtime requirements only (no pytest); constraints pin every transitive version.
RUN pip install -r requirements-retrieval.txt -c constraints.txt

# --- runtime: application code, synthetic fixtures, non-root user -------------------
FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    HOST=0.0.0.0 \
    PORT=8000 \
    APP_MODE=fixture \
    SAVED_JOBS_PATH=/app/data/private/saved_jobs.json
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY src ./src
COPY scripts/serve_api.py scripts/serve_mcp.py scripts/cv.py ./scripts/
# Synthetic, public demo data only; data/private and .env are excluded by .dockerignore.
COPY data/fixtures ./data/fixtures
RUN mkdir -p /app/data/private && chown app:app /app/data/private
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)"]
CMD ["python", "-m", "scripts.serve_api"]
