# HallucinationGuard-Env Dockerfile - Optimized for HF Spaces
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (curl needed for HEALTHCHECK, git for huggingface_hub)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ML models download on first request (faster build, slower first request)
# Models: all-MiniLM-L6-v2 (~90MB), nli-deberta-v3-base (~400MB)

# Copy application code
COPY . .

# Install the package in editable mode
RUN pip install -e .

# HF Spaces Docker default port is 7860
EXPOSE 7860

# Health check - give 5 minutes for cold start (dataset downloads + model loading)
# HF Spaces kills container if unhealthy for too long
HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=10 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run on port 7860 to match HF Spaces default
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]