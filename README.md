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
  - llm-evaluation
  - llm-training
  - benchmark
---

# 🛡️ HallucinationGuard-Env v3.0

> **The production-grade OpenEnv RL environment for training and evaluating LLMs on hallucination avoidance.**

[![Running](https://img.shields.io/badge/status-running-brightgreen)](https://huggingface.co/spaces/SamSankar/hallucination-guard-env)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Datasets](https://img.shields.io/badge/datasets-50k%2B%20examples-orange)](#datasets)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Why HallucinationGuard?

Large language models hallucinate — they confidently state false information not supported by any evidence. This is a critical problem for companies deploying LLMs in production.

**HallucinationGuard-Env** provides a standardized, reproducible RL environment to:

- 📊 **Benchmark** any LLM's hallucination rate across 50,000+ real-world QA examples
- 🎯 **Train** models to stay grounded in provided context
- 🏆 **Compare** models on a public leaderboard
- 🔧 **Integrate** into any ML pipeline via REST API or Python SDK

---

## Quick Start

### Option 1 — Python SDK (recommended)

```python
pip install requests
```

```python
from hallucination_guard_sdk import HallucinationGuardEnv
import anthropic

client = anthropic.Anthropic(api_key="YOUR_KEY")

def my_model(question: str, context: str) -> str:
    """Your model function — takes question + context, returns answer."""
    msg = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer using ONLY the context above."
        }]
    )
    return msg.content[0].text

# Evaluate in 3 lines
env = HallucinationGuardEnv()
results = env.evaluate(my_model, episodes=10, model_name="claude-3-haiku")
env.submit_to_leaderboard(results, organization="Anthropic")
```

### Option 2 — REST API

```bash
BASE="https://samsankar-hallucination-guard-env.hf.space"

# Start episode
curl -X POST $BASE/reset

# Submit answer
curl -X POST $BASE/step \
  -H "Content-Type: application/json" \
  -d '{"answer": "Your answer based only on the context"}'

# View leaderboard
curl $BASE/leaderboard
```

### Option 3 — OpenAI compatible

```python
from openai import OpenAI
from hallucination_guard_sdk import HallucinationGuardEnv

client = OpenAI(api_key="YOUR_KEY")

def gpt4_model(question, context):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer ONLY from the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQ: {question}"}
        ]
    )
    return response.choices[0].message.content

env = HallucinationGuardEnv()
results = env.evaluate(gpt4_model, episodes=10, model_name="gpt-4o-mini")
env.submit_to_leaderboard(results, organization="OpenAI")
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start a new episode, receive first question + context |
| `POST` | `/step` | Submit answer, receive reward + next question |
| `GET`  | `/state` | Current episode state |
| `GET`  | `/health` | Health check |
| `POST` | `/session/reset` | Create a stateful multi-turn session |
| `POST` | `/session/step` | Step within a named session |
| `GET`  | `/leaderboard` | Public model leaderboard |
| `POST` | `/leaderboard/submit` | Submit evaluation results |
| `GET`  | `/datasets` | Dataset statistics |
| `GET`  | `/metrics` | Real-time usage metrics |
| `GET`  | `/docs` | Interactive Swagger UI |

---

## Reward System

Each answer is scored across 6 dimensions:

| Component | Weight | Description |
|-----------|--------|-------------|
| Factual correctness | 35% | Does the answer match the ground truth? |
| Source grounding | 30% | Is the answer supported by the context? |
| Citation accuracy | 15% | Does the answer cite specific context passages? |
| Confidence calibration | 10% | Is confidence appropriate to accuracy? |
| Semantic consistency | 5% | Is the answer semantically coherent? |
| Hallucination penalty | 5% | Was any fabricated content detected? |

**Reward range:** -1.0 (complete hallucination) to +1.0 (perfect grounded answer)

---

## Datasets

50,000+ examples across 13 real-world QA datasets:

| Dataset | Size | Category | Difficulty |
|---------|------|----------|------------|
| SQuAD | 5,000 | Reading comprehension | Intermediate |
| TriviaQA | 5,000 | Trivia / general knowledge | Intermediate |
| HaluEval | 2,000 | Hallucination detection | Advanced |
| TruthfulQA | 817 | Factuality benchmark | Expert |
| Natural Questions | 5,000 | Open-domain QA | Intermediate |
| HotpotQA | 5,000 | Multi-hop reasoning | Advanced |
| BoolQ | 5,000 | Yes/No questions | Beginner |
| FaithDial | 5,000 | Hallucination in dialogue | Advanced |
| FEVER | 5,000 | Fact verification | Advanced |
| ARC-Challenge | 2,000 | Science exam | Advanced |
| OpenBookQA | 2,000 | Science facts | Intermediate |
| MS MARCO | 5,000 | Web search QA | Intermediate |
| CoQA | 5,000 | Conversational QA | Intermediate |

---

## Curriculum Learning

The environment implements adaptive difficulty:

```
Beginner → Intermediate → Advanced → Expert
  BoolQ      SQuAD          HotpotQA    TruthfulQA
  (yes/no)   (reading)      (multi-hop) (factuality)
```

Difficulty adjusts automatically based on the agent's rolling skill rating.

---

## Leaderboard

Submit your model's results to the public leaderboard:

```python
env = HallucinationGuardEnv()
results = env.evaluate(my_model, episodes=10)
env.submit_to_leaderboard(results, organization="YourCompany")
```

Or via API:
```bash
curl -X POST https://samsankar-hallucination-guard-env.hf.space/leaderboard/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gpt-4o",
    "avg_reward": 0.72,
    "avg_accuracy": 0.81,
    "hallucination_rate": 0.19,
    "total_episodes": 10,
    "total_steps": 100,
    "organization": "OpenAI"
  }'
```

---

## Use Cases

### For AI Companies
Benchmark your models before deployment. Compare across model versions. Track hallucination regression.

### For Researchers
Standardized evaluation protocol. 50k+ diverse examples. Reproducible results via seed parameter.

### For Developers
REST API — works with any language. Python SDK — 3 lines to evaluate. Per-dataset caching for fast iteration.

### For RL Training
Full OpenEnv-compatible interface. Curriculum learning built-in. Reward signal optimized for RL training loops.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                FastAPI Server                    │
│  /reset → /step → reward signal → /leaderboard  │
├─────────────────────────────────────────────────┤
│              HallucinationEnvironment            │
│  Episode management · Curriculum learning        │
├─────────────────────────────────────────────────┤
│                 Grader                           │
│  Semantic similarity · NLI · Citation detection  │
├─────────────────────────────────────────────────┤
│              Dataset Loader                      │
│  13 datasets · 50k+ examples · Per-file cache   │
└─────────────────────────────────────────────────┘
```

---

## License

MIT License — free for research and commercial use.

---

*Built for the Meta PyTorch OpenEnv Hackathon 2026*
