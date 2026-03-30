---
title: HallucinationGuard-Env
emoji: 🛡️
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - hallucination-detection
  - grounded-generation
  - question-answering
  - fact-checking
  - llm-training
  - llm-evaluation
  - benchmark
  - ai-safety
---

# 🛡️ HallucinationGuard-Env v4.1

> **The production-grade OpenEnv RL environment for training and evaluating LLMs on hallucination avoidance.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![PyPI](https://img.shields.io/badge/PyPI-openenv--halluguard-orange)](https://pypi.org/project/openenv-halluguard/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-1M%2B_examples-orange)](#datasets)
[![Grader](https://img.shields.io/badge/Grader-research--grade-purple)](#-hallucination-detection)

---

## 💡 The Inspiration

During research for a Hackathon, an AI model confidently hallucinated a **"golden ticket backdoor"** — claiming that Ideathon winners could skip directly to the Grand Finale. This information existed nowhere in the official sources. The AI stated it with high confidence and even fabricated a supporting quote.

That moment made one thing clear: hallucination isn't just an academic problem. It causes real confusion in high-stakes situations.

**HallucinationGuard-Env** was built to fix that — training AI models to say *"I don't know"* when they don't, cite real sources when they do, and never fabricate with confidence.

---

## 🚀 Quick Start

### Python SDK

```bash
pip install openenv-halluguard
```

```python
from hallucination_guard_env import HallucinationEnv

# Define your model
def my_model(question, context):
    # Call your LLM here — answer from context only
    return "your answer based on context"

# Evaluate in 5 lines
env = HallucinationEnv()
obs = env.reset()
action = my_model(obs.question, obs.context)
result = env.step(answer=action, confidence=0.8)
print(f"Reward: {result.reward}, Hallucinated: {result.is_hallucination}")
```

### Raw HTTP

```python
import requests

BASE = "https://samsankar-hallucination-guard-env.hf.space"

# 1. Start episode
obs = requests.post(f"{BASE}/reset").json()
print(obs["question"], obs["context"])

# 2. Answer from context only
result = requests.post(f"{BASE}/step", json={"answer": "your answer"}).json()
print(f"Reward: {result['reward']}, Hallucinated: {result['is_hallucination']}")

# 3. View leaderboard
print(requests.get(f"{BASE}/leaderboard").json())
```

### Run Locally

```bash
git clone https://huggingface.co/spaces/SamSankar/hallucination-guard-env
cd hallucination-guard-env
pip install -r server/requirements.txt
uvicorn server.app:app --reload --port 7860
curl http://localhost:7860/health
```

---

## 📁 Project Structure

```
hallucination-guard-env/
├── Dockerfile                    # HF Spaces Docker config
├── pyproject.toml                # Package metadata
├── openenv.yaml                  # OpenEnv manifest
├── README.md                     # This file
│
├── server/                       # FastAPI backend
│   ├── app.py                    # Main FastAPI application (endpoints)
│   ├── environment.py            # Core RL environment logic
│   ├── grader.py                 # 9-component reward system
│   ├── dataset_loader.py         # 38 dataset loader with HF cache
│   ├── tasks.py                  # Task registry (3 tasks)
│   ├── metrics.py                # Real-time metrics tracker
│   ├── requirements.txt          # Python dependencies
│   └── Dockerfile                # Server Docker image
│
├── models.py                     # Data models (Action, Observation, State)
├── client.py                     # HTTP/WebSocket client
└── inference.py                  # Baseline inference script (hackathon submission)
```

---

## 🎯 Tasks

HallucinationGuard-Env exposes **3 named tasks** in difficulty order:

| # | task_id | Difficulty | Primary Datasets | Frontier LLM Score |
|---|---------|-----------|-----------------|-------------------|
| 1 | `task_1_factual_grounding` | 🟢 Beginner | SQuAD, BoolQ, ARC, OpenBookQA | 0.70–0.85 |
| 2 | `task_2_multi_hop_synthesis` | 🟡 Intermediate | HotpotQA, CoQA, NQ-Open, MS-MARCO | 0.55–0.70 |
| 3 | `task_3_adversarial_resistance` | 🔴 Advanced | HaluEval, TruthfulQA, FEVER, AdversarialQA | 0.40–0.60 |

---

## 🎮 How The Environment Works

The agent receives a **question** and a **source document**. It must answer using only what the document says, provide a direct quote supporting its answer, and state how confident it is.

### Action Space

Every `POST /step` call accepts this JSON body (only `answer` is required):

```json
{
    "answer":           "string — derived ONLY from the provided context",
    "confidence":       0.5,     // float 0.0–1.0, calibrated estimate
    "source_quote":     "string — verbatim phrase from context supporting the answer",
    "reasoning":        "string — optional chain-of-thought",
    "uncertainty_flags": []      // list of aspects the agent is unsure about
}
```

### Observation Space

```python
@dataclass
class HallucinationObservation:
    question: str                  # The question to answer
    context: str                   # Source document to answer from
    reward: float                  # Step reward (-1.0 to 1.0)
    feedback: str                  # Detailed human-readable feedback
    is_hallucination: bool         # Was hallucination detected?
    hallucination_type: str        # Type of hallucination detected
    hallucination_severity: str    # NONE / MINOR / MODERATE / SEVERE / CRITICAL
    grounding_score: float         # How well answer is grounded in context
    accuracy_so_far: float         # Running accuracy this episode
    skill_rating: float            # ELO-style skill rating
    attempts_remaining: int        # Steps left in episode
    done: bool                     # Episode complete?
```

### Episode Flow

```
reset()
  → Sample question + context from dataset (curriculum-aware)
  → Return initial observation

step(action)
  → Grade answer across 9 research-grade components
  → Detect hallucination type and severity
  → Compute multi-factor reward with ROUGE + BERTScore + AlignScore
  → Adapt difficulty based on performance
  → Return observation with reward + rich feedback

state()
  → Return episode metadata: ID, step count, skill rating, curriculum stage
```

---

## 📊 Reward System (v4.1 — Research-Grade)

| Component | Weight | Description |
|-----------|--------|-------------|
| Factual correctness | 0.30 | Exact/fuzzy match + semantic similarity to ground truth |
| Source grounding | 0.20 | Verifies answer is supported by context |
| Citation accuracy | 0.15 | `source_quote` found verbatim in context |
| Confidence calibration | 0.15 | ECE between stated confidence and correctness |
| Semantic consistency | 0.10 | NLI entailment score (DeBERTa-v3-base) |
| Hallucination penalty | 0.10 | Penalises detected hallucinations |
| ROUGE (1/2/L) | 0.05 | Surface-form overlap with reference answer |
| BERTScore (DeBERTa) | 0.05 | Token-level semantic similarity |
| AlignScore | 0.05 | Faithfulness to context (RoBERTa, ACL 2023) |

Difficulty multiplier: `beginner × 0.9`, `intermediate × 1.0`, `advanced × 1.1`, `expert × 1.2`

```
reward = clamp(Σ(weight × score) × difficulty_multiplier + consistency_bonus, 0.0, 1.0)
```

---

## 🔬 Hallucination Detection

### 8 Types Classified

| Type | What It Catches |
|---|---|
| `FABRICATED_FACT` | Information stated that is not in the source |
| `FALSE_CITATION` | `source_quote` that does not exist in the document |
| `OVERCONFIDENT_WRONG` | High confidence on an incorrect answer |
| `CONTEXT_DRIFT` | Answer gradually drifts away from source |
| `NUMERICAL_FABRICATION` | Made-up statistics or numbers |
| `ENTITY_CONFUSION` | Wrong names, organisations, or places |
| `TEMPORAL_ERROR` | Incorrect dates or timelines |
| `RELATIONSHIP_ERROR` | Incorrect relationships between entities |

### 5 Severity Levels

| Level | Score | Meaning |
|---|---|---|
| NONE | 0.0 | Fully grounded answer |
| MINOR | 0.1–0.3 | Slight deviation from source |
| MODERATE | 0.3–0.5 | Noticeable unsupported claims |
| SEVERE | 0.5–0.7 | Significantly fabricated content |
| CRITICAL | 0.7+ | Answer largely invented |

---

## 📚 Datasets

**1,090,163 total examples** across 38 real-world QA datasets — cached permanently, instant boot:

| Source | Examples | Domain |
|---|---|---|
| SQuAD + SQuAD-v2 | 100,000 | Reading comprehension |
| TriviaQA | 50,000 | Open-domain factual QA |
| HotpotQA | 50,000 | Multi-hop reasoning |
| DROP | 50,000 | Numerical reasoning |
| RACE | 50,000 | Exam reading comprehension |
| NewsQA | 50,000 | News article QA |
| FaithDial / HH-RLHF | 49,649 | Faithful dialogue |
| FEVER / SNLI | 49,947 | Fact verification |
| NQ Open | 50,000 | Natural questions |
| AQUA-RAT | 97,467 | Math word problems |
| XSum | 49,994 | Extreme summarisation |
| CNN/DailyMail | 50,000 | News summarisation |
| HellaSwag | 39,905 | Commonsense completion |
| AdversarialQA | 30,000 | Adversarial reading comprehension |
| WinoGrande | 40,398 | Commonsense inference |
| CommonsenseQA | 9,741 | Commonsense reasoning |
| BoolQ | 9,427 | Boolean yes/no QA |
| CoQA | 7,199 | Conversational QA |
| MedQA | 10,000 | Medical licensing exam |
| MedMCQA | 20,000 | Medical entrance exam |
| SciTail | 23,596 | Science entailment |
| HaluEval | 10,000 | Hallucination evaluation |
| TruthfulQA | 817 | Factuality benchmark |
| QASC | 8,134 | Multi-hop science |
| QUAIL | 10,246 | Reading comprehension |
| SciQ | 11,679 | Science QA |
| Circa | 31,525 | Social context QA |
| ARC | 2,590 | Science exam |
| OpenBookQA | 4,957 | Common knowledge |
| AG News | 50,000 | News classification |
| QuaRTz | 2,696 | Qualitative science |
| Climate-FEVER | 881 | Climate fact verification |
| PubMedQA | 1,000 | Biomedical QA |
| Medical QA Pairs | 3,000 | Medical question similarity |
| MS MARCO | 30,568 | Web search QA |

---

## 🔧 OpenEnv Required Endpoints

### `GET /tasks`
Returns all 3 task definitions and the complete action schema.

### `POST /grader`
Score a completed episode. Pass the per-step rewards and info dicts collected during the episode.

### `POST /baseline`
Run the built-in heuristic agent across all 3 tasks. No API key required.

---

## 🌐 Deployment (HuggingFace Spaces)

### Startup Optimization

The environment uses a **two-phase loading strategy**:

1. **Core datasets** (~50K examples) load synchronously at startup
2. **Extended datasets** (~1M examples) load in background after server is healthy

This ensures fast cold starts while maintaining full dataset availability.

### Configuration

```dockerfile
# Dockerfile optimized for HF Spaces
FROM python:3.10-slim

# Pre-download ML models during build (saves ~2min startup)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/nli-deberta-v3-base')"

# Health check with 5-minute start-period for cold boot
HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=10 \
    CMD curl -f http://localhost:7860/health || exit 1
```

---

## 📀 API Endpoints

### Environment

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit an answer |
| `GET` | `/state` | Get current episode state |

### Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/session/reset` | Create a stateful session |
| `POST` | `/session/step` | Step in a session |
| `DELETE` | `/session` | Close a session |

### OpenEnv

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/tasks` | List all tasks + action schema |
| `POST` | `/grader` | Score a completed episode |
| `POST` | `/baseline` | Run baseline agent |
| `GET` | `/metadata` | Environment metadata |
| `GET` | `/schema` | Action/Observation/State schemas |
| `POST` | `/mcp` | MCP JSON-RPC endpoint |

### Leaderboard

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/leaderboard` | Model leaderboard |
| `POST` | `/leaderboard/submit` | Submit evaluation results |

### Metrics

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/metrics` | Real-time metrics |
| `GET` | `/metrics/summary` | Metrics summary report |

---

## 📋 Baseline Scores

### Heuristic Baseline (no LLM required)

The heuristic baseline extracts the first sentence of the context as the answer. This establishes a floor that real LLMs should beat.

```bash
python inference.py --heuristic --episodes 3 --steps 5 --seed 42 --env-url http://localhost:7860
```

| Task | Score | Std Dev |
|------|-------|---------|
| task_1_factual_grounding | 0.29 | ±0.15 |
| task_2_multi_hop_synthesis | 0.25 | ±0.14 |
| task_3_adversarial_resistance | 0.22 | ±0.16 |
| **Overall** | **0.25** | - |

> **Note**: Scores are reproducible with `--seed 42`. The heuristic intentionally returns the first sentence of context — it's meant to be a weak baseline, not a competitive benchmark. It ignores the question, uses a fixed confidence (0.6), and provides an irrelevant source quote (first 80 chars). Real LLMs should score 1.5-2x higher by actually reading questions and finding relevant context.

### LLM Baseline (requires API key)

```bash
# Set required environment variables
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_...

# Run inference
python inference.py --episodes 3 --steps 5 --seed 42
```

### Tested LLM Results

We tested multiple LLMs on this benchmark (3 episodes × 5 steps, seed=42):

| Rank | Model | Provider | Overall Score | Task 1 | Task 2 | Task 3 | Time |
|------|-------|----------|---------------|--------|--------|--------|------|
| 🥇 | Llama 3.3 70B | Groq (cloud) | **0.462** | 0.499 | 0.431 | 0.458 | 95s |
| 🥈 | Qwen2.5:7B | Ollama (local CPU) | **0.427** | 0.497 | 0.339 | 0.444 | 555s |
| 🥉 | Nemotron 3 Super | OpenRouter (cloud) | **0.418** | 0.612 | 0.397 | ~0.38 | 771s |
| 4 | Llama 3.1 8B | Groq (cloud) | 0.406 | 0.437 | 0.419 | 0.363 | 145s |
| 5 | Heuristic baseline | — | 0.25 | 0.29 | 0.25 | 0.22 | 61s |

*\*Nemotron Task 3 score is estimated based on Task 1-2 performance.*

**Key Findings:**
- **Groq Llama 3.3 70B** is the fastest and most accurate for cloud inference
- **Local Qwen2.5:7B on CPU** achieves 92% of Groq's performance — excellent for privacy-sensitive applications
- Local inference requires ~6x longer than cloud but runs entirely offline

#### Running with Ollama (Local)

```bash
# Install and pull model
ollama pull qwen2.5:7b

# Run inference (local)
API_BASE_URL=http://localhost:11434/v1 MODEL_NAME=qwen2.5:7b HF_TOKEN=ollama \
  python inference.py --env-url http://localhost:7860 --episodes 3 --steps 5
```

#### Running with Groq (Cloud)

```bash
# Set environment variables
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
export HF_TOKEN=gsk_your_key_here

# Run inference
python inference.py --env-url http://localhost:7860 --episodes 3 --steps 5 --seed 42
```

---

## 💻 Development

### Run Locally

```bash
# Clone and install
git clone https://huggingface.co/spaces/SamSankar/hallucination-guard-env
cd hallucination-guard-env
pip install -r server/requirements.txt

# Run server
uvicorn server.app:app --reload --port 7860

# Run tests
pytest tests/

# Run baseline (heuristic, no API key)
python inference.py --heuristic
```

### Run with LLM API

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_...
python inference.py --episodes 3 --steps 5
```

---

## 🔗 Links

| | |
|---|---|
| 🤗 HuggingFace Space | https://huggingface.co/spaces/SamSankar/hallucination-guard-env |
| 📦 PyPI Package | https://pypi.org/project/openenv-halluguard/ |
| 📖 Interactive API Docs | https://samsankar-hallucination-guard-env.hf.space/docs |
| 🏆 Leaderboard | https://samsankar-hallucination-guard-env.hf.space/leaderboard |
| 🔧 OpenEnv Framework | https://github.com/meta-pytorch/OpenEnv |

---

## 🏆 Why This Environment Stands Out

| | |
|---|---|
| **Real-world origin** | Born from an actual AI hallucination experience during hackathon research |
| **Solves the #1 LLM problem** | Hallucination is the most critical reliability issue in production AI |
| **Novel** | First OpenEnv environment targeting hallucination and grounding |
| **Research-grade grader** | ROUGE + BERTScore + AlignScore + nli-deberta-v3-base — publication quality |
| **1M+ diverse examples** | 38 real-world datasets: SQuAD, HaluEval, TruthfulQA, HotpotQA, MedQA and more |
| **Model-agnostic** | Works with GPT-4, Claude, Llama, Mistral, Gemma, Phi, or any LLM |
| **PyPI package** | `pip install openenv-halluguard` for instant SDK access |
| **Production-ready** | Session management, leaderboard, persistent cache, Dockerfile |
| **Adaptive** | ELO-based curriculum scales difficulty with the agent's skill |

---

## Changelog

### v4.1.0 (2026-03)

- **Fixed** HF Spaces restart loop — optimized startup with lazy dataset loading
- **Fixed** Missing `_torch_available()` function in grader
- **Fixed** Reproducibility — seed now properly resets dataset sampling for consistent results
- **Reduced** core datasets from 15 to 5 for faster cold starts
- **Increased** healthcheck start-period to 300s for dataset downloads
- **Added** stderr logging for progress visibility in HF Space logs
- **Added** `GET /tasks` — lists all 3 tasks + action schema (OpenEnv required)
- **Added** `POST /grader` — per-episode task scoring 0.0–1.0 (OpenEnv required)
- **Added** `POST /baseline` — built-in heuristic baseline runner (OpenEnv required)
- **Added** `inference.py` — baseline inference script for hackathon submission
- **Added** `server/tasks.py` — task registry with difficulty-mapped graders
- **Updated** `openenv.yaml` to v4.1.0 with task declarations

### v4.0.0

- 9-component reward system (ROUGE + BERTScore + AlignScore)
- NLI upgraded to nli-deberta-v3-base (optimized for HF Spaces)
- 38 datasets, 1,090,163 examples

---

*Built to train models to stop hallucination · MIT License*