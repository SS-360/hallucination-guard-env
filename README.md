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
  - llm-evaluation
  - llm-training
  - benchmark
  - grounded-generation
  - question-answering
  - fact-checking
---

<div align="center">

# 🛡️ HallucinationGuard-Env

### The Production-Grade RL Environment for LLM Hallucination Detection & Prevention

[![Status](https://img.shields.io/badge/status-live-brightgreen)](https://huggingface.co/spaces/SamSankar/hallucination-guard-env)
[![Version](https://img.shields.io/badge/version-3.0.0-blue)](https://huggingface.co/spaces/SamSankar/hallucination-guard-env)
[![Datasets](https://img.shields.io/badge/dataset-100k%2B%20examples-orange)](#datasets)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-purple)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

**[Live Demo](https://huggingface.co/spaces/SamSankar/hallucination-guard-env) · [API Docs](https://samsankar-hallucination-guard-env.hf.space/docs) · [Leaderboard](https://samsankar-hallucination-guard-env.hf.space/leaderboard)**

</div>

---

## Overview

HallucinationGuard-Env is an open, standardized reinforcement learning environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. It provides a rigorous, reproducible benchmark for measuring and training LLMs to avoid hallucination — the tendency to generate plausible-sounding but factually incorrect information.

### The Problem

Large language models frequently generate confident, well-formed responses that are factually wrong. This is one of the most critical barriers to deploying LLMs in production for healthcare, legal, financial, and enterprise applications.

### The Solution

HallucinationGuard-Env provides:
- A **standardized RL loop** (reset → observe → answer → reward) that any model can plug into
- **100,033 real-world QA examples** across 15 diverse datasets with factual ground truth
- A **multi-dimensional reward signal** that distinguishes grounded answers from hallucinated ones
- A **public leaderboard** for comparing models objectively
- A **Python SDK** for zero-friction integration

---

## Quick Start

### 3-Line Evaluation

```python
pip install requests
```

```python
from hallucination_guard_sdk import HallucinationGuardEnv

env = HallucinationGuardEnv()
results = env.evaluate(your_model_fn, episodes=10, model_name="your-model")
env.submit_to_leaderboard(results, organization="YourCompany")
```

### REST API

```bash
BASE="https://samsankar-hallucination-guard-env.hf.space"

# Start episode — get question + context
curl -X POST $BASE/reset

# Submit answer — get reward + hallucination verdict
curl -X POST $BASE/step \
  -H "Content-Type: application/json" \
  -d '{"answer": "Your answer based only on the context"}'

# View leaderboard
curl $BASE/leaderboard
```

---

## Integration Examples

### OpenAI

```python
from openai import OpenAI
from hallucination_guard_sdk import HallucinationGuardEnv

client = OpenAI(api_key="YOUR_KEY")

def gpt_model(question: str, context: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer ONLY using the provided context. Never use outside knowledge."},
            {"role": "user",   "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content

env = HallucinationGuardEnv()
results = env.evaluate(gpt_model, episodes=10, model_name="gpt-4o")
env.submit_to_leaderboard(results, organization="OpenAI")
```

### Anthropic Claude

```python
import anthropic
from hallucination_guard_sdk import HallucinationGuardEnv

client = anthropic.Anthropic(api_key="YOUR_KEY")

def claude_model(question: str, context: str) -> str:
    msg = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer using ONLY the context above."
        }]
    )
    return msg.content[0].text

env = HallucinationGuardEnv()
results = env.evaluate(claude_model, episodes=10, model_name="claude-3-5-sonnet")
env.submit_to_leaderboard(results, organization="Anthropic")
```

### Groq (Free Tier)

```python
from groq import Groq
from hallucination_guard_sdk import HallucinationGuardEnv

client = Groq(api_key="YOUR_GROQ_KEY")  # Free at console.groq.com

def llama_model(question: str, context: str) -> str:
    r = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Answer ONLY from the provided context."},
            {"role": "user",   "content": f"Context: {context}\n\nQ: {question}"}
        ],
        max_tokens=200
    )
    return r.choices[0].message.content

env = HallucinationGuardEnv()
results = env.evaluate(llama_model, episodes=10, model_name="llama-3.1-8b")
env.submit_to_leaderboard(results)
```

### Local Models via Ollama

```python
import requests
from hallucination_guard_sdk import HallucinationGuardEnv

def ollama_model(question: str, context: str) -> str:
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": f"Context: {context}\n\nQuestion: {question}\n\nAnswer using ONLY the context.",
        "stream": False
    })
    return r.json()["response"]

env = HallucinationGuardEnv()
results = env.evaluate(ollama_model, episodes=10, model_name="llama3-local")
```

### HuggingFace Transformers

```python
from transformers import pipeline
from hallucination_guard_sdk import HallucinationGuardEnv

pipe = pipeline("text-generation", model="microsoft/phi-2")

def phi_model(question: str, context: str) -> str:
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    result = pipe(prompt, max_new_tokens=100, do_sample=False)
    return result[0]["generated_text"].split("Answer:")[-1].strip()

env = HallucinationGuardEnv()
results = env.evaluate(phi_model, episodes=10, model_name="phi-2")
```

---

## API Reference

### Environment Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start a new episode. Returns first question + context |
| `POST` | `/step` | Submit an answer. Returns reward, hallucination verdict, next question |
| `GET`  | `/state` | Current episode state: step count, accuracy, skill rating, streaks |
| `GET`  | `/health` | Health check |

### Session Endpoints (Stateful)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/session/reset` | Create a persistent named session |
| `POST` | `/session/step` | Step within a session (pass `X-Session-Id` header) |
| `DELETE` | `/session` | Close a session |
| `GET`  | `/session/list` | List active sessions |

### Leaderboard Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/leaderboard` | Ranked leaderboard of all model evaluations |
| `POST` | `/leaderboard/submit` | Submit your model's results |
| `DELETE` | `/leaderboard/{model}` | Remove an entry |

### Info Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/datasets` | Dataset statistics by source, difficulty, category |
| `GET`  | `/metrics` | Real-time usage metrics |
| `GET`  | `/environment/info` | Full environment specification |
| `GET`  | `/docs` | Interactive Swagger UI |

---

## Reward System

Each answer is evaluated across six dimensions:

| Component | Weight | Description |
|-----------|:------:|-------------|
| Factual correctness | 35% | Semantic similarity between answer and ground truth |
| Source grounding | 30% | Is the answer explicitly supported by the context? |
| Citation accuracy | 15% | Does the answer cite specific phrases from context? |
| Confidence calibration | 10% | Is expressed confidence appropriate to accuracy? |
| Semantic consistency | 5% | Coherence and logical consistency |
| Hallucination penalty | 5% | Fabricated content detection |

**Reward range:** `-1.0` (complete hallucination) → `+1.0` (perfect grounded answer)

**Hallucination types detected:**
- `fabricated_fact` — invented information not in context
- `false_citation` — citing non-existent sources
- `overconfident_wrong` — high confidence incorrect response
- `context_drift` — straying from source material
- `numerical_fabrication` — invented numbers or statistics
- `entity_confusion` — mixing up named entities

---

## Datasets

**100,033 examples across 15 real-world QA datasets** — cached permanently, no download on startup:

| Dataset | Examples | Category | Difficulty |
|---------|:--------:|----------|:----------:|
| SQuAD | 10,000 | Reading comprehension | Intermediate |
| TriviaQA | 10,000 | General knowledge trivia | Intermediate |
| HotpotQA | 10,000 | Multi-hop reasoning | Advanced |
| HH-RLHF | 9,945 | Grounded dialogue | Advanced |
| BoolQ | 9,427 | Yes/No Wikipedia QA | Beginner |
| NQ Open | 8,000 | Real Google search questions | Intermediate |
| CommonsenseQA | 8,000 | Commonsense MCQ | Intermediate |
| HaluEval | 5,000 | Hallucination detection | Advanced |
| WinoGrande | 5,000 | Commonsense fill-in-the-blank | Intermediate |
| CoQA | 7,199 | Conversational QA | Intermediate |
| OpenBookQA | 4,957 | Elementary science facts | Intermediate |
| MS MARCO | 6,050 | Web search QA | Intermediate |
| ARC-Challenge | 2,590 | Science exam (hard) | Advanced |
| Medical QA | 3,048 | Medical fact verification | Advanced |
| TruthfulQA | 817 | Factuality benchmark | Expert |
| **Total** | **100,033** | **15 datasets** | |

---

## Curriculum Learning

The environment implements adaptive difficulty that automatically adjusts based on agent performance:

```
Skill 0.0–0.3    Skill 0.3–0.6    Skill 0.6–0.8    Skill 0.8–1.0
─────────────    ─────────────    ─────────────    ─────────────
   Beginner        Intermediate      Advanced           Expert
  BoolQ, NQ       SQuAD, CoQA     HotpotQA, Arc    TruthfulQA
  (yes/no)       (reading comp)   (multi-hop)      (factuality)
```

Difficulty multipliers: `0.9×` (beginner) → `1.0×` → `1.1×` → `1.2×` (expert)

---

## Leaderboard Submission

```python
# Automatic via SDK
env = HallucinationGuardEnv()
results = env.evaluate(my_model, episodes=10)
env.submit_to_leaderboard(results, organization="MyCompany", notes="GPT-4o baseline")
```

```bash
# Manual via API
curl -X POST https://samsankar-hallucination-guard-env.hf.space/leaderboard/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gpt-4o",
    "avg_reward": 0.72,
    "avg_accuracy": 0.81,
    "hallucination_rate": 0.19,
    "total_episodes": 10,
    "total_steps": 100,
    "organization": "OpenAI",
    "notes": "Zero-shot evaluation, no fine-tuning"
  }'
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        REST API (FastAPI)                        │
│     /reset · /step · /leaderboard · /datasets · /metrics        │
├─────────────────────────────────────────────────────────────────┤
│                   HallucinationEnvironment                       │
│        Episode management · Curriculum learning · Sessions       │
├─────────────────────────────────────────────────────────────────┤
│                           Grader                                 │
│   Semantic similarity (all-MiniLM-L6-v2) · NLI contradiction    │
│   detection (nli-deberta-v3-small) · Citation analysis          │
├─────────────────────────────────────────────────────────────────┤
│                       Dataset Loader                             │
│    15 datasets · 100k+ examples · Pre-built JSON cache           │
│    No startup downloads — instant boot from cached files         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Use Cases

**AI Safety & Alignment Research**
Standardized hallucination benchmarking across model families. Track how fine-tuning affects grounding behavior. Compare zero-shot vs few-shot vs RLHF approaches.

**Enterprise LLM Evaluation**
Before deploying an LLM in a regulated industry (healthcare, legal, finance), benchmark its hallucination rate across 100k real-world questions. Get a single score you can track over model versions.

**RL Training**
Use the reward signal to train anti-hallucination behavior via GRPO, PPO, or DPO. The environment is fully OpenEnv-compatible and supports concurrent sessions for parallel training.

**Model Comparison**
Submit any model to the public leaderboard and compare head-to-head on the same benchmark. Works with GPT, Claude, Llama, Mistral, Gemma, Phi — any model accessible via Python.

---

## Environment Specification

```json
{
  "name": "HallucinationGuard-Env",
  "version": "3.0.0",
  "observation_space": {
    "question": "string",
    "context": "string",
    "difficulty_level": "beginner | intermediate | advanced | expert",
    "attempts_remaining": "int",
    "skill_rating": "float [0, 1]"
  },
  "action_space": {
    "answer": "string"
  },
  "reward_range": [-1.0, 1.0],
  "max_steps_per_episode": 10,
  "supported_frameworks": ["OpenEnv", "REST API", "Python SDK"]
}
```

---

## License

MIT License — free for research and commercial use.

---

<div align="center">

Built for the **Meta PyTorch OpenEnv Hackathon 2026**

*HallucinationGuard-Env — Making AI tell the truth, one reward at a time.*

</div>
