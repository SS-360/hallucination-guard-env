# HallucinationGuard-Env Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (curl needed for HEALTHCHECK)
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package in editable mode
RUN pip install -e .

# HF Spaces Docker default port is 7860 — match it so no app_port override needed
EXPOSE 7860

# Health check so HF Spaces knows when the container is truly ready
HEALTHCHECK --interval=15s --timeout=10s --start-period=120s --retries=6 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run on port 7860 to match HF Spaces default
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
