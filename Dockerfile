# HallucinationGuard-Env Dockerfile - HF Spaces optimized
# Single-stage build: avoids broken --target copy with compiled packages (torch, etc.)

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pin torch to CPU-only slim wheel FIRST — prevents pip from pulling the 2.4 GB CUDA build.
# Must be installed before sentence-transformers / bert-score resolve their torch dep.
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package itself
RUN pip install --no-cache-dir -e .

# Cache directory for datasets
RUN mkdir -p /tmp/halluguard_cache /tmp/transformers_cache /tmp/hf_cache

# HF Spaces default port
EXPOSE 7860

# Health check — generous start-period for dataset download on cold start
HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=10 \
    CMD curl -f http://localhost:7860/health || exit 1

ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/hf_cache

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
