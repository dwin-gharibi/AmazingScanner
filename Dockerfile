FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu \
           "torch==2.13.0" \
    && /opt/venv/bin/pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN /opt/venv/bin/pip install --no-deps .

FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="AmazingScanner" \
      org.opencontainers.image.description="AmazingScanner - CNN document scanner: corner detection, rectification and enhancement" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    OMP_NUM_THREADS=4 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    DOCSCANNER_MODELS=/app/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
        tesseract-ocr-eng \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY docs/assets/examples ./docs/assets/examples
COPY model[s] ./models

RUN useradd --create-home --uid 10001 scanner \
    && mkdir -p /app/outputs /app/data \
    && chown -R scanner:scanner /app
USER scanner

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:7860/ >/dev/null || exit 1

ENTRYPOINT ["python", "-m", "docscanner.app.gradio_app"]
CMD ["--host", "0.0.0.0", "--port", "7860"]
