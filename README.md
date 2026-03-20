---
title: HallucinationGuard-Env
emoji: 🛡️
colorFrom: red
colorTo: indigo
sdk: docker
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - hallucination-detection
  - grounded-generation
  - question-answering
  - fact-checking
  - llm-training
---

# 🛡️ HallucinationGuard-Env

> **An OpenEnv reinforcement learning environment that trains AI models to answer only from verified context — penalizing hallucination and rewarding factual grounding.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-2000%2B_examples-orange)](#datasets)

---

## 💡 The Inspiration

During research for the Meta PyTorch OpenEnv Hackathon, an AI model confidently hallucinated a **"golden ticket backdoor"** — claiming that Ideathon winners could skip directly to the Grand Finale. This information existed nowhere in the official sources. The AI stated it with high confidence and even fabricated a supporting quote.

That moment made one thing clear: hallucination isn't just an academic problem. It causes real confusion in high-stakes situations.

**HallucinationGuard-Env** was built to fix that — training AI models to say *"I don't know"* when they don't, cite real sources when they do, and never fabricate with confidence.

---

## 🚀 Quick Start

```bash
# Install
pip install -e .

# Run locally
uvicorn server.app:app --reload

# Health check
curl http://localhost:8000/health
# → {"status": "healthy", "service": "HallucinationGuard-Env"}

# Deploy to HuggingFace Spaces
openenv push --repo-id your-username/hallucination-guard-env
```

---

## 🎮 How The Environment Works

The agent receives a **question** and a **source document**. It must answer using only what the document says, provide a direct quote supporting its answer, and state how confident it is.

### Action Space

```python
@dataclass
class HallucinationAction(Action):
    answer: str          # The agent's answer
    confidence: float    # Certainty 0.0 → 1.0
    source_quote: str    # Direct quote from context supporting the answer
```

### Observation Space

```python
@dataclass
class HallucinationObservation(Observation):
    question: str                  # The question to answer
    context: str                   # Source document to answer from
    reward: float                  # Step reward
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
  → Grade answer across 6 components
  → Detect hallucination type and severity
  → Compute multi-factor reward
  → Adapt difficulty based on performance
  → Return observation with reward + rich feedback

state()
  → Return episode metadata: ID, step count, skill rating, curriculum stage
```

---

## 🏆 Reward System

Six components combine into a single reward signal in **[0.0, 1.0]**:

| Component | Weight | What It Measures |
|---|---|---|
| **Factual Correctness** | 30% | Semantic similarity + entity overlap vs ground truth |
| **Source Grounding** | 20% | Word coverage and context matching |
| **Citation Accuracy** | 15% | Is source_quote actually in the document? |
| **Confidence Calibration** | 15% | Does stated confidence match actual accuracy? |
| **Semantic Consistency** | 10% | Logical coherence with context |
| **Hallucination Penalty** | 10% | Penalty for fabricated content |

**Difficulty multipliers:** beginner 0.9× → expert 1.2×
**Consistency bonus:** up to +0.05 for sustained high performance

```
reward = clamp(Σ(weight × score) × difficulty_multiplier + consistency_bonus, 0.0, 1.0)
```

**In practice:**
- Hallucinated answer with false citation → reward ≈ **0.002–0.10**, CRITICAL severity
- Grounded correct answer with real quote → reward ≈ **0.85–1.00**

---

## 🔬 Hallucination Detection

### 8 Types Classified

| Type | What It Catches |
|---|---|
| `FABRICATED_FACT` | Information stated that is not in the source |
| `FALSE_CITATION` | source_quote that does not exist in the document |
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

### Detection Algorithms

- **Word coverage** — fraction of meaningful content words in answer found in context
- **Entity hallucination** — novel entities in answer not found in source
- **Numerical fabrication** — numbers in answer absent from context
- **Sliding window fuzzy matching** — citation verification (threshold 0.7)
- **Negation mismatch** — contradiction detection via negation word analysis
- **Confidence calibration error** — `|confidence − correctness|` with 50% overconfidence surcharge

---

## 📚 Datasets

2,140+ total examples loaded at runtime across four difficulty levels:

| Source | Examples | Type | Difficulty |
|---|---|---|---|
| Synthetic (built-in) | 140 | Hallucination traps, edge cases | All levels |
| **SQuAD** | ~500 | Reading comprehension | Intermediate |
| **TriviaQA** | ~500 | Open-domain factual QA | Intermediate |
| **HaluEval** | ~500 | Hallucination evaluation | Advanced |
| **TruthfulQA** | ~500 | Factuality benchmark | Advanced/Expert |

Datasets load from Hugging Face automatically on first start (`pip install datasets`).
A local disk cache (`server/cache/`) is used on subsequent starts for instant loading.

### Built-in Synthetic Dataset Breakdown

| Difficulty | Count | Focus |
|---|---|---|
| Beginner | 60 | Simple factual recall, API concepts, basic science |
| Intermediate | 60 | Multi-hop reasoning, history, technology, biology |
| Advanced | 10 | Hallucination traps, common misconceptions |
| Expert | 10 | System mechanics, algorithms, quantum physics |

### Add Custom Datasets

```python
from server.dataset_loader import DatasetLoader

loader = DatasetLoader()
loader.load_from_json("my_dataset.json")   # Custom JSON
loader.load_from_huggingface("squad")      # Any HF dataset
```

Custom JSON format:
```json
[
  {
    "question": "What is the prize pool?",
    "context": "The hackathon has a total prize pool of $30,000 USD...",
    "answer": "$30,000 USD",
    "id": "q001",
    "source": "custom",
    "difficulty": "intermediate",
    "category": "factual_recall"
  }
]
```

---

## 🎓 Curriculum Learning

The environment adapts difficulty in real-time using an ELO-style skill rating:

| Trigger | Action |
|---|---|
| Recent avg reward > 0.7 | Increase difficulty |
| Recent avg reward < 0.3 | Decrease difficulty |
| Overall accuracy > 0.8 | EXPERT ceiling |
| Overall accuracy > 0.6 | ADVANCED ceiling |
| Overall accuracy > 0.4 | INTERMEDIATE ceiling |

Episodes can use progressive difficulty mixing (beginner → expert within one episode) for maximum learning efficiency.

---

## 🔌 Model-Agnostic Adapters

Works with any LLM out of the box:

```python
from model_adapters import create_adapter

# OpenAI
adapter = create_adapter("openai", model_name="gpt-4", api_key="sk-...")

# Anthropic Claude
adapter = create_adapter("anthropic", model_name="claude-sonnet-4-6", api_key="sk-ant-...")

# HuggingFace (Llama, Mistral, Qwen...)
adapter = create_adapter("huggingface", model_name="meta-llama/Llama-3-8B-Instruct")

# Local Ollama
adapter = create_adapter("ollama", model_name="llama3", api_base="http://localhost:11434")

# Use it
response = adapter.generate_response(
    question="What is the prize pool?",
    context="The hackathon has $30,000 USD in prizes...",
    require_citation=True,
    require_confidence=True
)
```

---

## 📊 Metrics & Monitoring

```bash
curl http://localhost:8000/metrics                     # Live metrics
curl http://localhost:8000/metrics/training-curves     # Reward curves
curl http://localhost:8000/metrics/heatmap             # Hallucination heatmap
curl http://localhost:8000/metrics/export?format=json  # Export data
```

Sample output after training:
```
Episodes: 15  |  Steps: 150
Accuracy: 78.5%  |  Avg Reward: 0.742  |  Hallucination Rate: 12.3%
Reward Trend: IMPROVING ↑   |  Recent Hallucination Rate: 8.2%
```

---

## 🏗️ Project Structure

```
hallucination_guard_env/
├── models.py               # HallucinationAction, Observation, State, Config
├── client.py               # HTTP/WebSocket client
├── model_adapters.py       # OpenAI, Anthropic, HuggingFace, Ollama adapters
├── test_env.py             # Full test suite
├── openenv.yaml            # Manifest
├── pyproject.toml          # Package metadata
└── server/
    ├── environment.py      # Core RL environment logic
    ├── app.py              # FastAPI server (stateless + session endpoints)
    ├── grader.py           # 6-component reward + hallucination detection
    ├── dataset_loader.py   # Multi-source dataset loader with caching
    ├── metrics.py          # Real-time metrics tracker
    ├── cache/              # Pre-built dataset cache (instant startup)
    ├── requirements.txt
    └── Dockerfile
```

---

## ⚙️ Configuration

```python
from models import EnvironmentConfig

config = EnvironmentConfig(
    max_questions_per_episode=10,
    reward_weights={
        "factual_correctness":    0.30,
        "source_grounding":       0.20,
        "citation_accuracy":      0.15,
        "confidence_calibration": 0.15,
        "semantic_consistency":   0.10,
        "hallucination_penalty":  0.10,
    },
    adaptive_difficulty=True,
    difficulty_threshold_increase=0.7,
    difficulty_threshold_decrease=0.3,
    curriculum_enabled=True,
)

env = HallucinationEnvironment(config=config)
```

---

## 🧪 Tests

```bash
python test_env.py
```

Covers: dataset loading, grader components, reset/step/state, episode completion, hallucination type classification, curriculum difficulty, metrics tracking, model adapter factory.

---

## 🔗 Links

| | |
|---|---|
| 📖 OpenEnv Docs | https://github.com/meta-pytorch/OpenEnv |
| 🎓 OpenEnv Course | https://github.com/huggingface/openenv-course |

---

## 🏆 Why This Environment Stands Out

| | |
|---|---|
| **Real-world origin** | Born from an actual AI hallucination experience during hackathon research |
| **Solves the #1 LLM problem** | Hallucination is the most critical reliability issue in production AI |
| **Novel** | First OpenEnv environment targeting hallucination and grounding |
| **Rich reward signal** | 6-component system gives models precise, actionable feedback |
| **2,140+ diverse examples** | SQuAD, TriviaQA, HaluEval, TruthfulQA + curated synthetic traps |
| **Model-agnostic** | Works with GPT-4, Claude, Llama, Mistral, or any LLM |
| **Production-ready** | Session management, metrics, caching, Dockerfile included |
| **Adaptive** | ELO-based curriculum scales difficulty with the agent's skill |

---

*Built for the Meta PyTorch OpenEnv Hackathon 2026 · MIT License*
