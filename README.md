---
title: HallucinationGuard-Env
emoji: 🛡️
colorFrom: red
colorTo: indigo
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

# 🛡️ HallucinationGuard-Env v4.0

> **The production-grade OpenEnv RL environment for training and evaluating LLMs on hallucination avoidance.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![PyPI](https://img.shields.io/pypi/v/openenv-halluguard)](https://pypi.org/project/openenv-halluguard/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-1M%2B_examples-orange)](#datasets)
[![Grader](https://img.shields.io/badge/Grader-research--grade-purple)](#grader)

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
from openenv_halluguard import HallucinationGuardEnv

# Define your model
def my_model(question, context):
    # Call your LLM here — answer from context only
    return "your answer based on context"

# Evaluate in 3 lines
env = HallucinationGuardEnv()
results = env.evaluate(my_model, episodes=5, model_name="my-model")
env.print_report(results)
env.submit_to_leaderboard(results, organization="MyCompany")
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

## 🏆 Reward System (v4.0 — Research-Grade)

Nine components combine into a single reward signal in **[0.0, 1.0]**:

| Component | Weight | What It Measures |
|---|---|---|
| **Factual Correctness** | 30% | Semantic similarity + entity overlap vs ground truth |
| **Source Grounding** | 25% | Word coverage and context matching |
| **Citation Accuracy** | 10% | Is `source_quote` actually in the document? |
| **Confidence Calibration** | 8% | Does stated confidence match actual accuracy? |
| **Semantic Consistency** | 7% | NLI-based logical coherence with context |
| **Hallucination Penalty** | 5% | Penalty for fabricated content |
| **ROUGE-L** | 5% | Token overlap with ground truth (Lin 2004) |
| **BERTScore** | 5% | Contextual embedding similarity (Zhang et al. 2020) |
| **AlignScore** | 5% | Faithfulness to source context (Zha et al. ACL 2023) |

**Difficulty multipliers:** beginner 0.9× → expert 1.2×
**Consistency bonus:** up to +0.05 for sustained high performance

```
reward = clamp(Σ(weight × score) × difficulty_multiplier + consistency_bonus, 0.0, 1.0)
```

**In practice:**
- Hallucinated answer with false citation → reward ≈ **0.002–0.10**, CRITICAL severity
- Grounded correct answer with real quote → reward ≈ **0.85–1.00**

---

## 🔬 Hallucination Detection <a name="grader"></a>

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

### Detection Algorithms (v4.0)

- **Word coverage** — fraction of meaningful content words in answer found in context
- **Entity hallucination** — novel entities in answer not found in source
- **Numerical fabrication** — numbers in answer absent from context
- **Sliding window fuzzy matching** — citation verification (threshold 0.7)
- **Negation mismatch** — contradiction detection via negation word analysis
- **Confidence calibration error** — `|confidence − correctness|` with 50% overconfidence surcharge
- **NLI cross-encoder** — `cross-encoder/nli-deberta-v3-large` for semantic entailment (upgraded from small in v4.0)
- **ROUGE-1/2/L** — standard token overlap metric in QA/summarisation research (Lin 2004)
- **BERTScore** — contextual embedding similarity via DeBERTa-v3-base (Zhang et al. 2020)
- **AlignScore** — faithfulness/grounding scorer via RoBERTa (Zha et al. ACL 2023)

---

## 📚 Datasets <a name="datasets"></a>

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

# OpenAI GPT-4
import openai
client = openai.OpenAI(api_key="sk-...")
def gpt4(question, context):
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "Answer ONLY from the context provided."},
                  {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}]
    )
    return r.choices[0].message.content

# Anthropic Claude
import anthropic
client = anthropic.Anthropic(api_key="sk-ant-...")
def claude(question, context):
    msg = client.messages.create(
        model="claude-3-5-sonnet-20241022", max_tokens=256,
        messages=[{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer from context only."}]
    )
    return msg.content[0].text

# Groq (free tier)
from groq import Groq
gclient = Groq(api_key="YOUR_GROQ_KEY")
def llama(question, context):
    r = gclient.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": f"Context: {context}\n\nQ: {question}"}],
        max_tokens=200
    )
    return r.choices[0].message.content

# Ollama (local, free)
import requests as req
def ollama(question, context):
    r = req.post("http://localhost:11434/api/generate",
                 json={"model": "llama3", "stream": False,
                       "prompt": f"Context: {context}\n\nQ: {question}\n\nAnswer from context only."})
    return r.json()["response"]

# Evaluate and submit to leaderboard
env = HallucinationGuardEnv()
results = env.evaluate(gpt4, episodes=10, model_name="gpt-4o")
env.submit_to_leaderboard(results, organization="OpenAI")
```

---

## 📊 Metrics & Monitoring

```bash
curl https://samsankar-hallucination-guard-env.hf.space/metrics
curl https://samsankar-hallucination-guard-env.hf.space/metrics/summary
curl https://samsankar-hallucination-guard-env.hf.space/leaderboard
curl https://samsankar-hallucination-guard-env.hf.space/datasets
```

---

## 🏗️ Project Structure

```
hallucination_guard_env/
├── models.py               # HallucinationAction, Observation, State, Config
├── client.py               # HTTP/WebSocket client
├── model_adapters.py       # OpenAI, Anthropic, HuggingFace, Ollama adapters
├── hallucination_guard_sdk.py  # Python SDK (evaluate + leaderboard)
├── test_env.py             # Full test suite
├── openenv.yaml            # Manifest
├── pyproject.toml          # Package metadata
└── server/
    ├── environment.py      # Core RL environment logic + curriculum learning
    ├── app.py              # FastAPI server (stateless + session + leaderboard)
    ├── grader.py           # 9-component reward: ROUGE + BERTScore + AlignScore + NLI-large
    ├── dataset_loader.py   # 38 dataset loader with persistent JSON cache
    ├── metrics.py          # Real-time metrics tracker
    ├── requirements.txt    # Dependencies including rouge-score, bert-score, alignscore
    ├── cache/              # Pre-built dataset cache (instant boot, no downloads)
    └── Dockerfile
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
| **Research-grade grader** | ROUGE + BERTScore + AlignScore + nli-deberta-v3-large — publication quality |
| **1,090,163 diverse examples** | 38 real-world datasets: SQuAD, HaluEval, TruthfulQA, HotpotQA, MedQA and more |
| **Model-agnostic** | Works with GPT-4, Claude, Llama, Mistral, Gemma, Phi, or any LLM |
| **PyPI package** | `pip install openenv-halluguard` for instant SDK access |
| **Production-ready** | Session management, leaderboard, persistent cache, Dockerfile |
| **Adaptive** | ELO-based curriculum scales difficulty with the agent's skill |
| **Paper-ready** | Designed for EMNLP 2026 system demonstration submission |

---

## 🗺️ Roadmap

| Phase | Status | Description |
|---|---|---|
| Phase 1 — Deployment | ✅ Done | v4.0 live, 1M+ examples, 38 datasets, instant boot |
| Phase 2 — Research-grade grader | ✅ Done | ROUGE, BERTScore, AlignScore, nli-deberta-v3-large |
| Phase 3 — Experiments | 🔄 In progress | Qwen3-1.7B, Llama3-8B, Mistral-7B baseline + GRPO training |
| Phase 4 — Paper | 📝 Planned | EMNLP 2026 system demonstration paper |

---

*Built for to Train Models to Stop Hallucination · MIT License*
