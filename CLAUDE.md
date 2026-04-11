# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Key Commands

```bash
# Install (editable, includes ML deps)
pip install -e .

# Run server locally (port 7860)
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Run heuristic baseline (no API key needed)
python inference.py --heuristic --env-url http://localhost:7860

# Run with LLM (Groq/Ollama/OpenAI-compatible)
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
export HF_TOKEN=your_key
python inference.py --env-url http://localhost:7860 --episodes 3 --steps 5

# Run tests
pytest tests/ -v
pytest tests/test_grader.py::TestRewardRange -v          # Specific class
pytest tests/test_grader.py::TestRewardRange::test_reward_in_range_correct_answer -v  # Single test

# Validate OpenEnv compliance
openenv validate                          # Local structure check
openenv validate --url http://localhost:7860  # Runtime check against live server

# Lint
ruff check . --ignore E501,F401,F403

# Docker
docker build -t hallucination-guard-env .
docker run -p 7860:7860 hallucination-guard-env
```

## Architecture

OpenEnv RL environment for training LLMs to avoid hallucinations. FastAPI server on HuggingFace Spaces with 3 graded tasks and a 9-component reward system.

**Data flow:** `POST /reset` → sample question from dataset → `POST /step(answer)` → grade via 9-component grader → `POST /grader` → 0.0–1.0 per episode

### Core Modules

| Module | Role |
|--------|------|
| `server/app.py` | FastAPI endpoints, session management, inline Gradio-style HTML docs |
| `server/environment.py` | `HallucinationEnvironment` — reset/step/state, curriculum, early stopping |
| `server/grader.py` | `calculate_reward()` — 9 components, hallucination detection, refusal handling |
| `server/dataset_loader.py` | Loads 38 datasets from `SamSankar/hallucination-guard-cache` HF Dataset repo |
| `server/tasks.py` | 3 `TaskDefinition` objects + `compute_task_score()` per-episode grader |
| `models.py` | Pydantic models inheriting from `openenv.core.env_server` base classes |
| `inference.py` | Hackathon submission script with `[START]/[STEP]/[END]` log format |

### Session Isolation

`/reset` creates a per-session `HallucinationEnvironment` that shares the global dataset loader (expensive) but has its own episode state. Sessions tracked by `episode_id`/`session_id` in `_sessions` dict. `/step` looks up the session — **must pass `session_id`** or it falls back to a shared default env that may be in stale state. The `EnvClient` in `inference.py` stores `_session_id` from reset and passes it on step.

### Reward System (grader.py)

`calculate_reward()` returns `(reward: float, info: dict)`. 9 components with weights:

| Component | Weight | Implementation |
|-----------|--------|---------------|
| factual_correctness | 0.35 | `check_factual_accuracy_advanced()` — exact/fuzzy/semantic match |
| source_grounding | 0.20 | `check_quote_in_context_advanced()` — reduced for wrong answers |
| citation_accuracy | 0.10 | Verbatim quote match in context |
| confidence_calibration | 0.10 | `compute_calibration_error()` — overconfidence penalized more |
| semantic_consistency | 0.10 | NLI entailment (DeBERTa-v3 CrossEncoder) |
| hallucination_penalty | 0.10 | `detect_hallucination_advanced()` — type + severity classification |
| rouge_score | 0.02 | ROUGE-1/2/L |
| bertscore | 0.02 | Uses `roberta-base` |
| alignscore | 0.01 | NLI CrossEncoder-based alignment (no separate model — reuses `_get_nli()`) |

**Key grader behavior:**
- Wrong answers capped at ~0.4 regardless of grounding (`factual_cap`)
- Refusal on unanswerable questions: rewarded 0.65–0.80
- Refusal when answer exists: penalized to 0.30
- Numerical fabrication checked on original text (before `normalize_text` replaces numbers with `NUM`)
- Thinking traces (Nemotron `◸`, `<reasoning>`) stripped via `_strip_thinking()` before grading

### ML Model Loading (lazy singletons)

All ML models load on first use via module-level globals in `grader.py`:
- `_get_embedder()` → `all-MiniLM-L6-v2` (sentence-transformers)
- `_get_nli()` → `cross-encoder/nli-deberta-v3-small` (also used by `compute_alignscore`)
- `_get_rouge()` → `rouge_score.rouge_scorer.RougeScorer`
- `_check_bertscore()` → `bert_score` package with `roberta-base`

`app.py` lifespan preloads these in a background thread so first request isn't slow. Dockerfile pre-caches them at build time.

### Task Scoring (tasks.py)

`compute_task_score()` aggregates per-step rewards into 0.0–1.0:
- Base = mean step reward - hallucination penalty
- Completion bonus (+0.02) for episodes with ≥5 steps
- Task-3 overconfidence penalty: `max(0, avg_calibration - 0.7) * avg_hallucination * 0.1`

### Dataset Loading

Core datasets (squad, halueval, boolq, openbookqa, sciq) load synchronously at startup. Extended datasets download in a background thread from `SamSankar/hallucination-guard-cache` on HF Hub. Cache at `/tmp/halluguard_cache/`.

## Pydantic Models

All inherit from `openenv.core.env_server` (`Action`, `Observation`, `State`). Use `Field(default_factory=...)` not `field(default_factory=...)`. Use `str` for enum values in model fields.

Serialization uses `_safe_dict()` in app.py which handles Pydantic models, dataclasses, and enums recursively.

## Hackathon Log Format (inference.py)

`inference.py` emits structured logs required by the hackathon evaluator:
```
[START] task=<task_name> env=hallucination-guard-env model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

## Critical Constraints

- **NumPy <2.0.0** — Pre-compiled packages crash with NumPy 2.x
- **BERTScore model** — Must use `roberta-base`, NOT `microsoft/deberta-v3-base` (fast tokenizer crashes with transformers>=4.57 due to `vocab_file=None`)
- **AlignScore** — Replaced with NLI CrossEncoder-based `compute_alignscore()` (no separate package/model needed; do NOT add `alignscore` to requirements)
- **`/metadata` must include `description`** — OpenEnv validator fails without it
- **`/schema` must include `state`** — OpenEnv validator fails without it
- **Port is 7860** — HF Spaces config, Dockerfile, and client.py all use this
- **session_id** — `inference.py` and `client.py` must pass `session_id` from `/reset` to `/step` or episodes end after 1 step

## Repositories

- **GitHub:** https://github.com/SS-360/hallucination-guard-env
- **HuggingFace Space:** https://huggingface.co/spaces/SamSankar/hallucination-guard-env