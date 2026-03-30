# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HallucinationGuard-Env is an OpenEnv RL environment for training LLMs to avoid hallucinations. It runs as a FastAPI server on HuggingFace Spaces with 3 graded tasks (beginner → advanced) and a 9-component reward system.

## Key Commands

```bash
# Install dependencies
pip install -r server/requirements.txt

# Run server locally (port 7860)
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Run heuristic baseline (no API key needed)
python inference.py --heuristic --env-url http://localhost:7860

# Run tests
pytest tests/                           # All tests
pytest tests/test_grader.py -v         # Specific test file
pytest tests/test_grader.py::TestGraderScoreRange -v  # Specific test class

# Lint (CI uses this)
ruff check . --ignore E501,F401,F403

# Docker build
docker build -t hallucination-guard-env .
docker run -p 7860:7860 hallucination-guard-env
```

## Running with LLM APIs

### Groq (cloud)
```bash
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
export HF_TOKEN=gsk_your_key_here
python inference.py --env-url http://localhost:7860 --episodes 3 --steps 5 --seed 42
```

### Ollama (local)
```bash
ollama pull qwen2.5:7b
export API_BASE_URL=http://localhost:11434/v1
export MODEL_NAME=qwen2.5:7b
export HF_TOKEN=ollama  # Any non-empty value triggers LLM mode
python inference.py --env-url http://localhost:7860 --episodes 3 --steps 5 --seed 42
```

## Critical Dependencies

- **NumPy must be <2.0.0** — Pre-compiled packages (sentence-transformers, bert-score) crash with NumPy 2.x. Pinned in requirements.
- **Protobuf required** — BERTScore dependency; explicitly listed in requirements.

## Architecture

```
server/
├── app.py           # FastAPI endpoints
├── environment.py   # HallucinationEnvironment class (OpenEnv step/reset/state)
├── grader.py        # 9-component reward calculation + refusal handling + explanations
├── dataset_loader.py # Loads 38 datasets from HF cache
└── tasks.py         # Task registry with difficulty-weighted graders

models.py            # Pydantic models: HallucinationAction, HallucinationObservation, HallucinationState
inference.py         # Hackathon submission script (OpenAI-compatible client)
```

### Data Flow

1. **reset()** → Samples question from dataset_loader, returns HallucinationObservation
2. **step(HallucinationAction)** → Grades answer via grader.py, returns reward + feedback
3. **grader.calculate_reward()** → 9 components (see Reward System below)
4. **tasks.compute_task_score()** → Aggregates per-step rewards into 0.0-1.0 task score

### API Endpoints

| Category | Method | Endpoint | Description |
|----------|--------|----------|-------------|
| Environment | POST | `/reset` | Start new episode |
| Environment | POST | `/step` | Submit answer |
| Environment | GET | `/state` | Get episode state |
| Batch | POST | `/batch/evaluate` | Evaluate multiple Q&A pairs |
| Batch | POST | `/batch/stream` | Streaming batch (NDJSON) |
| Metrics | GET | `/metrics/timing` | Time-per-step latency stats |
| Leaderboard | GET | `/leaderboard/viz` | Chart data (bar, scatter, tiers) |
| OpenEnv | GET | `/tasks` | List tasks + action schema |
| OpenEnv | POST | `/grader` | Score completed episode |
| OpenEnv | POST | `/baseline` | Run heuristic baseline |

### Dataset Loading

Datasets load from `SamSankar/hallucination-guard-cache` HF Dataset repo. Core datasets load synchronously on startup; extended datasets load in background thread. Cached locally at `/tmp/halluguard_cache/`.

### Model Preloading

ML models (sentence-transformers, CrossEncoder/NLI, ROUGE, BERTScore) preload at server startup in `lifespan()` to avoid 30-60s cold start delays. Environment variable `HF_HOME=/tmp/hf_cache` replaces deprecated `TRANSFORMERS_CACHE`.

## Reward System (grader.py)

9-component reward system:

| Component | Weight | Description |
|-----------|--------|-------------|
| factual_correctness | 0.35 | Exact/fuzzy match + semantic similarity to ground truth |
| source_grounding | 0.20 | Answer supported by context (reduced for wrong answers) |
| citation_accuracy | 0.10 | source_quote found verbatim in context |
| confidence_calibration | 0.10 | ECE between stated confidence and correctness |
| semantic_consistency | 0.10 | NLI entailment score (DeBERTa-v3) |
| hallucination_penalty | 0.10 | Penalizes detected hallucinations |
| rouge_score | 0.02 | ROUGE-1/2/L overlap with reference |
| bertscore | 0.02 | Token-level semantic similarity |
| alignscore | 0.01 | Faithfulness to context (RoBERTa) |

**Key behavior:**
- Wrong answers capped at ~0.4 reward regardless of grounding
- Grounding contribution reduced for incorrect answers
- Difficulty multiplier: beginner×0.9, intermediate×1.0, advanced×1.1, expert×1.2

## Refusal Handling

The grader detects when models appropriately refuse to answer unanswerable questions:

| Scenario | Reward | Behavior |
|----------|--------|----------|
| Proper refusal on unanswerable | 0.65–0.80 | Rewarded for honesty |
| Refusal with low confidence | 0.50 | Partial credit |
| Underconfident refusal (answer exists) | 0.30 | Penalized for not trying |

Detected phrases: "I cannot answer", "not in the context", "I don't know", "cannot determine", "insufficient information". See `is_refusal_answer()` in grader.py.

## Pydantic Models

All models inherit from `openenv.core.env_server.Action`, `Observation`, `State` (Pydantic BaseModel, not dataclass). When modifying:
- Use `Field(default_factory=...)` not `field(default_factory=...)`
- Use `str` for enum values in model fields (e.g., `difficulty: str = "intermediate"`)
- Serialization uses `_safe_dict()` in app.py which handles Pydantic models via `model_dump()`

## Test Structure

```
tests/
├── test_grader.py        # 20 tests: reward calculation, refusal handling, hallucination detection
├── test_adversarial.py   # 18 tests: HaluEval, TruthfulQA edge cases
├── test_endpoints.py     # 15 tests: batch eval, metrics, leaderboard endpoints
├── test_environment.py   # 13 tests: reset/step behavior
└── test_dataset_loader.py # 14 tests: dataset loading, caching
```

Run with `pytest tests/ -v`. CI runs automatically via `.github/workflows/test.yml`.

## Repositories

- **GitHub:** https://github.com/SS-360/hallucination-guard-env
- **HuggingFace Space:** https://huggingface.co/spaces/SamSankar/hallucination-guard-env

Changes pushed to GitHub automatically sync to HuggingFace Spaces via `.github/workflows/sync-to-hf.yml`. Requires `HF_TOKEN` secret with write permissions in GitHub repo settings.

## Baseline Scores

Heuristic agent (seed=42, 3 episodes × 5 steps):
- task_1_factual_grounding: 0.29 (±0.15)
- task_2_multi_hop_synthesis: 0.25 (±0.14)
- task_3_adversarial_resistance: 0.22 (±0.16)
- Overall: 0.25