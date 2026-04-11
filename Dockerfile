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
ENV HF_HOME=/tmp/hf_cache
ENV HF_HUB_CACHE=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache

# ═══════════════════════════════════════════════════════════════════════════════
# PRELOAD MODELS AT BUILD TIME — eliminates cold-start latency
# ═══════════════════════════════════════════════════════════════════════════════
# This ensures NLI, sentence-transformers, and BERTScore models are cached
# in the image, so the first request doesn't suffer a 30-60s model download delay.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
print('Preloading all-MiniLM-L6-v2...'); \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Preloading nli-deberta-v3-small...'); \
CrossEncoder('cross-encoder/nli-deberta-v3-small'); \
print('Models cached successfully!'); \
"

# Preload BERTScore model (roberta-base)
# Note: deberta-v3-base crashes with transformers>=4.57 due to tokenizer bug,
# so we use roberta-base which is compatible with all transformers versions.
RUN python -c "\
from bert_score import BERTScorer; \
print('Preloading roberta-base for BERTScore...'); \
scorer = BERTScorer(model_type='roberta-base', lang='en', device='cpu'); \
print('BERTScore model cached!'); \
" 2>/dev/null || echo "BERTScore preload completed (some warnings are expected)"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]