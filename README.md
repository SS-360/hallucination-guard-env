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
[![PyPI](https://img.shields.io/pypi/v/openenv-halluguard)](https://pypi.org/project/openenv-halluguard/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-1M%2B_examples-orange)](#datasets)

---

## 💡 The Inspiration

During research for the Meta PyTorch OpenEnv Hackathon, an AI model confidently hallucinated a **"golden ticket backdoor"** — claiming that Ideathon winners could skip directly to the Grand Finale. This information existed nowhere in the official sources. The AI stated it with high confidence and even fabricated a supporting quote.

That moment made one thing clear: hallucination isn't just an academic problem. It causes real confusion in high-stakes situations.

**HallucinationGuard-Env** was built to fix that — training AI models to say *"I don't know"* when they don't, cite real sources when they do, and never fabricate with confidence.

---

## 🚀 Quick Start

### Python SDK

```bash
pip install openenv-halluguard
```

```python
from openenv_halluguard import HallucinationGuardEnv

# Define your model
def my_model(question, context):
    # Call your LLM here — answer from context only
    return "your answer based on context"

# Evaluate
env = HallucinationGuardEnv()
results = env.evaluate(my_model, episodes=5, model_name="my-model")
env.print_report(results)
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
print(result["reward"], result["is_hallucination"])
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
- **NLI cross-encoder** — `nli-deberta-v3-small` for semantic entailment checking

---

## 📚 Datasets

**1,090,163 total examples** loaded at runtime across 38 real-world QA datasets:

| Source | Examples | Domain |
|---|---|---|
| SQuAD + SQuAD-v2 | 100,000 | Reading comprehension |
| TriviaQA | 50,000 | Open-domain factual QA |
| HotpotQA | 50,000 | Multi-hop reasoning |
| DROP | 50,000 | Numerical reasoning |
| RACE | 50,000 | Exam reading comprehension |
| NewsQA | 50,000 | News article QA |
| FaithDial | 49,649 | Faithful dialogue |
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

### Add Custom Datasets

```python
from server.dataset_loader import DatasetLoader

loader = DatasetLoader()
loader.load_from_json("my_dataset.json")   # Custom JSON
loader.load_from_huggingface("squad")      # Any HF dataset
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

---

## 🔌 Works With Any LLM

```python
from openenv_halluguard import HallucinationGuardEnv

# OpenAI
import openai
client = openai.OpenAI(api_key="sk-...")
def gpt4(question, context):
    r = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer from context only."}]
    )
    return r.choices[0].message.content

# Anthropic Claude
import anthropic
client = anthropic.Anthropic(api_key="sk-ant-...")
def claude(question, context):
    msg = client.messages.create(
        model="claude-3-haiku-20240307", max_tokens=256,
        messages=[{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer from context only."}]
    )
    return msg.content[0].text

# Evaluate any model
env = HallucinationGuardEnv()
results = env.evaluate(gpt4, episodes=5, model_name="gpt-4")
env.print_report(results)
```

---

## 📊 Metrics & Monitoring

```bash
curl https://samsankar-hallucination-guard-env.hf.space/metrics
curl https://samsankar-hallucination-guard-env.hf.space/metrics/training-curves
curl https://samsankar-hallucination-guard-env.hf.space/metrics/export?format=json
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
    ├── requirements.txt
    └── Dockerfile
```

---

## 🔗 Links

| | |
|---|---|
| 🤗 HuggingFace Space | https://huggingface.co/spaces/SamSankar/hallucination-guard-env |
| 📦 PyPI Package | https://pypi.org/project/openenv-halluguard/ |
| 📖 API Docs | https://samsankar-hallucination-guard-env.hf.space/docs |
| 🔧 OpenEnv Docs | https://github.com/meta-pytorch/OpenEnv |
| 🎓 OpenEnv Course | https://github.com/huggingface/openenv-course |

---

## 🏆 Why This Environment Stands Out

| | |
|---|---|
| **Real-world origin** | Born from an actual AI hallucination experience during hackathon research |
| **Solves the #1 LLM problem** | Hallucination is the most critical reliability issue in production AI |
| **Novel** | First OpenEnv environment targeting hallucination and grounding |
| **Rich reward signal** | 6-component system gives models precise, actionable feedback |
| **1,090,163 diverse examples** | 15 real-world datasets including SQuAD, HaluEval, TruthfulQA, HotpotQA |
| **Model-agnostic** | Works with GPT-4, Claude, Llama, Mistral, or any LLM |
| **PyPI package** | `pip install openenv-halluguard` for instant SDK access |
| **Production-ready** | NLI grader, session management, metrics, caching, Dockerfile |
| **Adaptive** | ELO-based curriculum scales difficulty with the agent's skill |

---

*Built for the Meta PyTorch OpenEnv Hackathon 2026 · MIT License*
