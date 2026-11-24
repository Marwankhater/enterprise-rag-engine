# syntax=docker/dockerfile:1.7
# Multi-stage build: dependencies and models are prepared in a builder stage, and the runtime
# image only carries the virtualenv, the model files and the application code.

ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE} AS builder
ARG INSTALL_MODELS=1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
ARG RERANKER_MODEL=BAAI/bge-reranker-base
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/models/hf TIKTOKEN_CACHE_DIR=/opt/models/tiktoken
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY app ./app
# CPU-only torch keeps the image around 2 GB instead of pulling CUDA libraries.
RUN if [ "$INSTALL_MODELS" = "1" ]; then \
        pip install --index-url "$TORCH_INDEX_URL" torch && pip install ".[models]"; \
    else \
        pip install .; \
    fi
# Bake model weights and the tokenizer table into the image so containers start offline.
RUN mkdir -p /opt/models/hf /opt/models/tiktoken && \
    (python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" || echo 'tiktoken table not cached; token counts will be approximate') && \
    if [ "$INSTALL_MODELS" = "1" ]; then \
        python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('$EMBEDDING_MODEL'); CrossEncoder('$RERANKER_MODEL')"; \
    fi

FROM ${PYTHON_IMAGE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/models/hf TIKTOKEN_CACHE_DIR=/opt/models/tiktoken HF_HUB_OFFLINE=1 \
    UPLOAD_DIR=/data/uploads
RUN groupadd --system app && useradd --system --gid app --home /srv app \
    && mkdir -p /data/uploads && chown -R app:app /data
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/models /opt/models
WORKDIR /srv
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app app ./app
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app examples ./examples
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers"]
