# HallucinationGuard-Env Dockerfile - Optimized for HF Spaces
# Multi-stage build for faster cold starts

# Stage 1: Install dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY server/requirements.txt .
RUN pip install --no-cache-dir --target=/app/deps -r requirements.txt

# Stage 2: Runtime image
FROM python:3.10-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /app/deps /usr/local/lib/python3.10/site-packages

# Copy application code
COPY . .

# Create cache directory
RUN mkdir -p /tmp/halluguard_cache

# HF Spaces Docker default port is 7860
EXPOSE 7860

# Health check - give 5 minutes for cold start (dataset downloads + model loading)
# HF Spaces kills container if unhealthy for too long
HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=10 \
    CMD curl -f http://localhost:7860/health || exit 1

# Set environment variables for faster startup
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/hf_cache

# Run on port 7860 to match HF Spaces default
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]