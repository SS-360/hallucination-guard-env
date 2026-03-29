"""
HallucinationGuard-Env v4.1 — Production FastAPI Server

Endpoints:
  Standard   : POST /reset  POST /step  GET /state  GET /health
  Session    : POST /session/reset  POST /session/step  DELETE /session
  Leaderboard: GET /leaderboard  POST /leaderboard/submit  DELETE /leaderboard/{model}
  Info       : GET /  GET /docs  GET /environment/info  GET /datasets
               GET /metrics  GET /metrics/summary
  ── OpenEnv Required ──────────────────────────────────────────────────────
  Tasks      : GET  /tasks                    ← list tasks + action schema
  Grader     : POST /grader                   ← score a completed episode
  Baseline   : POST /baseline                 ← run baseline agent, return scores
"""

import sys, os, uuid, logging, dataclasses, enum, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from models import HallucinationAction, HallucinationObservation, HallucinationState
from environment import HallucinationEnvironment
from metrics import get_tracker

# ── NEW: task registry import ─────────────────────────────────────────────────
from tasks import (
    ALL_TASKS,
    get_task,
    task_id_for_difficulty,
    compute_task_score,
    ACTION_SCHEMA,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load datasets once at startup so requests are instant."""
    global _default_env
    logger.info("Pre-loading datasets at startup...")
    try:
        _default_env = HallucinationEnvironment()
        logger.info(f"Startup complete — {_default_env.dataset_loader.get_total_examples():,} examples loaded.")
    except Exception as e:
        logger.warning(f"Startup pre-load failed ({e}); will load on first request.")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="HallucinationGuard-Env",
    description="""
## 🛡️ HallucinationGuard-Env v4.1

**The production-grade OpenEnv RL environment for training and evaluating LLMs on hallucination avoidance.**

Built on 1,000,000+ examples across 38 real-world QA datasets:
SQuAD · TriviaQA · HaluEval · TruthfulQA · HotpotQA · BoolQ · HH-RLHF ·
NQ Open · CommonsenseQA · WinoGrande · CoQA · OpenBookQA · MS MARCO · ARC · Medical QA

### PyPI Package

```bash
pip install openenv-halluguard
```

```python
from hallucination_guard_env import HallucinationEnv, HallucinationAction

env = HallucinationEnv()
obs = env.reset()
result = env.step(HallucinationAction(answer="your answer", confidence=0.8))
print(f"Reward: {result.reward}, Hallucinated: {result.is_hallucination}")
```

### Quick Start (HTTP)

```python
import requests

BASE = "https://samsankar-hallucination-guard-env.hf.space"

# 1. Start episode
obs = requests.post(f"{BASE}/reset").json()
print(obs["question"], obs["context"])

# 2. Answer from context only
result = requests.post(f"{BASE}/step", json={"answer": "your answer"}).json()
print(f"Reward: {result['reward']}, Hallucinated: {result['is_hallucination']}")
```

### OpenEnv Required Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tasks` | GET | List all 3 tasks and the action schema |
| `/grader` | POST | Score a completed episode (0.0–1.0) |
| `/baseline` | POST | Run built-in baseline agent across all tasks |

### Links

- **HuggingFace Space**: https://huggingface.co/spaces/SamSankar/hallucination-guard-env
- **PyPI Package**: https://pypi.org/project/openenv-halluguard/
- **Interactive Docs**: https://samsankar-hallucination-guard-env.hf.space/docs
- **Leaderboard**: https://samsankar-hallucination-guard-env.hf.space/leaderboard
    """,
    version="4.1.0",
    contact={"name": "HallucinationGuard", "url": "https://huggingface.co/spaces/SamSankar/hallucination-guard-env"},
    license_info={"name": "MIT"},
)

# CORS — allow all origins so any company/researcher can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State ──────────────────────────────────────────────────────────────────────
_sessions: Dict[str, HallucinationEnvironment] = {}
_default_env: Optional[HallucinationEnvironment] = None

# Leaderboard — file-backed so it survives requests within the container lifetime
import json as _json

_LEADERBOARD_FILE = "/tmp/hallucination_guard_leaderboard.json"

def _load_leaderboard() -> Dict[str, Any]:
    if os.path.exists(_LEADERBOARD_FILE):
        try:
            with open(_LEADERBOARD_FILE) as f:
                return _json.load(f)
        except Exception:
            pass
    return {}

def _save_leaderboard(lb: Dict[str, Any]) -> None:
    try:
        with open(_LEADERBOARD_FILE, "w") as f:
            _json.dump(lb, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not persist leaderboard: {e}")

_leaderboard: Dict[str, Dict[str, Any]] = _load_leaderboard()


def _get_default_env() -> HallucinationEnvironment:
    global _default_env
    if _default_env is None:
        _default_env = HallucinationEnvironment()
    return _default_env


def _safe_dict(obj):
    if dataclasses.is_dataclass(obj):
        return {f.name: _safe_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    elif isinstance(obj, enum.Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {k: _safe_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_safe_dict(i) for i in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


# ── Root ───────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


# ── Standard stateless endpoints ──────────────────────────────────────────────

@app.post("/reset", summary="Start a new episode", tags=["Environment"])
async def reset(body: Dict[str, Any] = {}):
    """
    Reset the environment and receive the first question + context.

    **Returns:** question, context, difficulty, attempts_remaining, skill_rating

    **Optional body params:**
    - `seed` (int): reproducible episode
    - `difficulty` (str): beginner | intermediate | advanced | expert
    - `episode_id` (str): custom episode ID
    """
    try:
        env = _get_default_env()
        obs = env.reset(**{k: v for k, v in body.items()
                           if k in ("seed", "episode_id", "difficulty",
                                    "enable_multi_turn", "enable_context_retrieval")})
        return JSONResponse(content=_safe_dict(obs))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", summary="Submit an answer", tags=["Environment"])
async def step(action_data: Dict[str, Any]):
    """
    Submit an answer to the current question.

    **Body:**
    ```json
    {"answer": "Your answer based ONLY on the provided context"}
    ```

    **Returns:** reward (-1 to 1), is_hallucination, hallucination_type,
    grounding_score, feedback, next question + context
    """
    try:
        env = _get_default_env()
        valid = {f.name for f in dataclasses.fields(HallucinationAction)}
        action = HallucinationAction(**{k: v for k, v in action_data.items() if k in valid})
        obs = env.step(action)
        return JSONResponse(content=_safe_dict(obs))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", summary="Get current episode state", tags=["Environment"])
async def get_state():
    """Returns full episode state: step count, accuracy, skill rating, streaks."""
    try:
        return JSONResponse(content=_safe_dict(_get_default_env().state()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Session endpoints ──────────────────────────────────────────────────────────

@app.post("/session/reset", summary="Create a stateful session", tags=["Sessions"])
async def session_reset(body: Dict[str, Any] = {},
                        x_session_id: Optional[str] = Header(None)):
    """
    Create a persistent session for multi-turn evaluation.
    Pass `X-Session-Id` header to reuse an existing session.
    Returns a `session_id` to use in subsequent calls.
    """
    session_id = x_session_id or str(uuid.uuid4())
    if session_id in _sessions:
        _sessions[session_id].close()
    _sessions[session_id] = HallucinationEnvironment(session_id=session_id)
    obs = _sessions[session_id].reset(**{k: v for k, v in body.items()
                                          if k in ("seed", "episode_id", "difficulty",
                                                   "enable_multi_turn", "enable_context_retrieval")})
    result = _safe_dict(obs)
    result["session_id"] = session_id
    return result


@app.post("/session/step", summary="Step in a session", tags=["Sessions"])
async def session_step(action_data: Dict[str, Any],
                       x_session_id: str = Header(...)):
    """Submit an answer within a named session. Requires `X-Session-Id` header."""
    if x_session_id not in _sessions:
        raise HTTPException(status_code=404,
                            detail=f"Session {x_session_id} not found. Call /session/reset first.")
    valid = {f.name for f in dataclasses.fields(HallucinationAction)}
    action = HallucinationAction(**{k: v for k, v in action_data.items() if k in valid})
    obs = _sessions[x_session_id].step(action)
    result = _safe_dict(obs)
    result["session_id"] = x_session_id
    return result


@app.delete("/session", summary="Close a session", tags=["Sessions"])
async def close_session(x_session_id: str = Header(...)):
    """Close and clean up a session."""
    if x_session_id in _sessions:
        _sessions[x_session_id].close()
        del _sessions[x_session_id]
    return {"status": "closed", "session_id": x_session_id}


@app.get("/session/list", summary="List active sessions", tags=["Sessions"])
async def list_sessions():
    return {"active_sessions": len(_sessions), "session_ids": list(_sessions.keys())}


# ── Leaderboard ────────────────────────────────────────────────────────────────

@app.get("/leaderboard", summary="Model leaderboard", tags=["Leaderboard"])
async def get_leaderboard():
    """
    Returns ranked leaderboard of all submitted model evaluations.
    Ranked by avg_reward descending.
    """
    if not _leaderboard:
        return {"leaderboard": [], "total_models": 0,
                "message": "No models submitted yet. Use POST /leaderboard/submit"}
    ranked = sorted(_leaderboard.values(), key=lambda x: x.get("avg_reward", 0), reverse=True)
    for i, entry in enumerate(ranked):
        entry["rank"] = i + 1
    return {
        "leaderboard": ranked,
        "total_models": len(ranked),
        "last_updated": max(e.get("submitted_at", 0) for e in ranked),
    }


@app.post("/leaderboard/submit", summary="Submit model evaluation results", tags=["Leaderboard"])
async def submit_to_leaderboard(data: Dict[str, Any]):
    """
    Submit your model's evaluation results to the leaderboard.

    **Required fields:**
    ```json
    {
      "model_name": "gpt-4o",
      "avg_reward": 0.72,
      "avg_accuracy": 0.81,
      "hallucination_rate": 0.19,
      "total_episodes": 10,
      "total_steps": 100
    }
    ```
    **Optional:** `organization`, `model_version`, `notes`
    """
    required = ["model_name", "avg_reward", "avg_accuracy",
                "hallucination_rate", "total_episodes", "total_steps"]
    missing = [f for f in required if f not in data]
    if missing:
        raise HTTPException(status_code=422,
                            detail=f"Missing required fields: {missing}")
    model_name = data["model_name"]
    _leaderboard[model_name] = {
        "model_name":        model_name,
        "organization":      data.get("organization", ""),
        "model_version":     data.get("model_version", ""),
        "avg_reward":        round(float(data["avg_reward"]), 4),
        "avg_accuracy":      round(float(data["avg_accuracy"]), 4),
        "hallucination_rate": round(float(data["hallucination_rate"]), 4),
        "total_episodes":    int(data["total_episodes"]),
        "total_steps":       int(data["total_steps"]),
        "notes":             data.get("notes", ""),
        "submitted_at":      time.time(),
    }
    logger.info(f"Leaderboard submission: {model_name} reward={data['avg_reward']:.3f}")
    _save_leaderboard(_leaderboard)
    return {"status": "submitted", "model_name": model_name,
            "message": f"'{model_name}' added to leaderboard. View at /leaderboard"}


@app.delete("/leaderboard/{model_name}", summary="Remove from leaderboard", tags=["Leaderboard"])
async def remove_from_leaderboard(model_name: str):
    """Remove a model entry from the leaderboard."""
    if model_name not in _leaderboard:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    del _leaderboard[model_name]
    _save_leaderboard(_leaderboard)
    return {"status": "removed", "model_name": model_name}


# ═══════════════════════════════════════════════════════════════════════════════
# ── OPENENV REQUIRED ENDPOINTS ─────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/tasks",
    summary="List all tasks and the action schema",
    tags=["OpenEnv"],
    response_description=(
        "Array of task objects. Each object contains task_id, name, description, "
        "difficulty, datasets, and the full action_schema (JSON Schema) that "
        "describes every field an agent can send to POST /step."
    ),
)
async def list_tasks():
    """
    ## OpenEnv required endpoint — GET /tasks

    Returns all 3 tasks in difficulty order (easy → medium → hard) and the
    complete action schema that governs what an agent must include in each
    `POST /step` call.

    ### Tasks
    | task_id | difficulty | primary datasets |
    |---------|-----------|-----------------|
    | `task_1_factual_grounding` | beginner | SQuAD, BoolQ, ARC |
    | `task_2_multi_hop_synthesis` | intermediate | HotpotQA, CoQA, NQ-Open |
    | `task_3_adversarial_resistance` | advanced | HaluEval, TruthfulQA, FEVER |
    """
    ordered = ["task_1_factual_grounding", "task_2_multi_hop_synthesis",
               "task_3_adversarial_resistance"]
    tasks_list = [ALL_TASKS[tid].to_dict() for tid in ordered if tid in ALL_TASKS]
    return {
        "tasks": tasks_list,
        "total": len(tasks_list),
        "action_schema": ACTION_SCHEMA,
        "notes": (
            "Run POST /reset with {\"difficulty\": \"<difficulty>\"} to start an episode "
            "for a specific task. Then call POST /step with the action schema fields. "
            "Use POST /grader to score the completed episode."
        ),
    }


@app.post(
    "/grader",
    summary="Score a completed episode (0.0 – 1.0)",
    tags=["OpenEnv"],
    response_description=(
        "task_id, score (0.0–1.0), breakdown of component scores, "
        "and episode metadata."
    ),
)
async def grade_episode(body: Dict[str, Any]):
    """
    ## OpenEnv required endpoint — POST /grader

    Computes a deterministic score in **[0.0, 1.0]** for a completed episode.

    ### Body
    ```json
    {
        "task_id": "task_1_factual_grounding",
        "step_rewards": [0.82, 0.55, 0.91, 0.43, 0.78],
        "step_infos": [
            {
                "correctness": 0.9, "grounding": 0.8,
                "calibration": 0.7, "hallucination_score": 0.1,
                "is_hallucination": false
            }
        ]
    }
    ```

    `step_rewards` and `step_infos` are the per-step values returned by
    `POST /step` during the episode.
    If you only have `step_rewards` (no `step_infos`), the grader falls back
    to using the mean reward as the score.

    ### Scoring

    Each task uses task-specific weights across four components:
    - **Factual correctness** (highest weight for beginner tasks)
    - **Source grounding** (citation accuracy)
    - **Confidence calibration** (lower weight for beginner, higher for adversarial)
    - **Hallucination penalty** (highest weight for adversarial task)

    A **completion bonus** of +0.05 is applied to full episodes (≥ 5 steps).
    """
    task_id = body.get("task_id")
    if not task_id:
        raise HTTPException(status_code=422, detail="'task_id' is required.")

    task = get_task(task_id)
    if task is None:
        valid_ids = list(ALL_TASKS.keys())
        raise HTTPException(
            status_code=404,
            detail=f"task_id '{task_id}' not found. Valid values: {valid_ids}",
        )

    step_rewards: List[float] = body.get("step_rewards", [])
    step_infos: List[Dict[str, Any]] = body.get("step_infos", [])

    if not isinstance(step_rewards, list):
        raise HTTPException(status_code=422, detail="'step_rewards' must be a list of floats.")

    # Fallback: if no step_infos provided, use mean reward directly
    if not step_infos and step_rewards:
        mean_reward = sum(step_rewards) / len(step_rewards)
        return {
            "task_id": task_id,
            "score": round(min(1.0, max(0.0, mean_reward)), 4),
            "breakdown": {"mean_reward": round(mean_reward, 4)},
            "metadata": {
                "task_id": task.task_id,
                "difficulty": task.difficulty,
                "steps": len(step_rewards),
                "note": "Scored from step_rewards only (no step_infos provided).",
            },
        }

    result = compute_task_score(task, step_rewards, step_infos)
    return result


@app.post(
    "/baseline",
    summary="Run baseline agent across all 3 tasks",
    tags=["OpenEnv"],
    response_description=(
        "Per-task baseline scores plus an aggregate summary. "
        "Uses a simple heuristic agent (no LLM API required)."
    ),
)
async def run_baseline(body: Dict[str, Any] = {}):
    """
    ## OpenEnv required endpoint — POST /baseline

    Runs the built-in **heuristic baseline agent** across all 3 tasks and
    returns reproducible scores.  No external API key is required — the agent
    uses deterministic context-extraction heuristics.

    ### Optional body params
    | field | default | description |
    |-------|---------|-------------|
    | `steps_per_task` | `5` | Questions per task episode (min 3, max 10) |
    | `seed` | `42` | Random seed for reproducibility |
    | `model` | `"heuristic_baseline"` | Label to record in results |

    ### What the heuristic baseline does
    1. Extracts the first sentence of the context as the answer.
    2. Sets confidence to 0.6.
    3. Uses the first 80 characters of the context as the source quote.

    This is intentionally weak — it establishes a reproducible **floor** that
    any real LLM should beat. Expected scores:

    | Task | Expected baseline score |
    |------|------------------------|
    | task_1_factual_grounding | 0.35 – 0.50 |
    | task_2_multi_hop_synthesis | 0.25 – 0.40 |
    | task_3_adversarial_resistance | 0.15 – 0.30 |

    To run a real LLM baseline instead, use `run_baseline.py`.
    """
    steps_per_task: int = max(3, min(10, int(body.get("steps_per_task", 5))))
    seed: int = int(body.get("seed", 42))
    model_label: str = str(body.get("model", "heuristic_baseline"))

    task_order = [
        ("task_1_factual_grounding",      "beginner"),
        ("task_2_multi_hop_synthesis",    "intermediate"),
        ("task_3_adversarial_resistance", "advanced"),
    ]

    results: List[Dict[str, Any]] = []
    all_rewards: List[float] = []
    all_hallucination_flags: List[bool] = []
    total_steps = 0

    for task_id, difficulty in task_order:
        task = get_task(task_id)
        if task is None:
            continue

        # Create an isolated session for this task
        session_id = f"baseline_{task_id}_{seed}"
        if session_id in _sessions:
            _sessions[session_id].close()
        _sessions[session_id] = HallucinationEnvironment(session_id=session_id)

        try:
            obs_raw = _sessions[session_id].reset(
                seed=seed, difficulty=difficulty
            )
            obs = _safe_dict(obs_raw)
        except Exception as e:
            logger.warning(f"Baseline reset failed for {task_id}: {e}")
            results.append({"task_id": task_id, "score": 0.0, "error": str(e)})
            continue

        step_rewards: List[float] = []
        step_infos: List[Dict[str, Any]] = []

        for _ in range(steps_per_task):
            if obs.get("done", False):
                break

            # ── Heuristic baseline agent ──────────────────────────────────
            context: str = obs.get("context", "")
            sentences = [s.strip() for s in context.replace("\n", " ").split(".") if s.strip()]
            answer = sentences[0] if sentences else context[:100]
            source_quote = context[:80] if context else ""
            confidence = 0.6
            # ─────────────────────────────────────────────────────────────

            valid = {f.name for f in dataclasses.fields(HallucinationAction)}
            action = HallucinationAction(
                answer=answer,
                confidence=confidence,
                source_quote=source_quote,
            )

            try:
                obs_raw = _sessions[session_id].step(action)
                obs = _safe_dict(obs_raw)
            except Exception as e:
                logger.warning(f"Baseline step failed for {task_id}: {e}")
                break

            reward = obs.get("reward") or 0.0
            step_rewards.append(float(reward))
            step_infos.append({
                "correctness":        obs.get("grounding_score", 0.0),
                "grounding":          obs.get("grounding_score", 0.0),
                "calibration":        0.5,  # heuristic agent has no calibration signal
                "hallucination_score": 1.0 if obs.get("is_hallucination") else 0.0,
                "is_hallucination":   bool(obs.get("is_hallucination", False)),
            })
            all_hallucination_flags.append(bool(obs.get("is_hallucination", False)))
            total_steps += 1

        # Grade this task
        grade = compute_task_score(task, step_rewards, step_infos)
        grade["model"] = model_label
        grade["seed"]  = seed
        results.append(grade)
        all_rewards.extend(step_rewards)

        # Cleanup session
        try:
            _sessions[session_id].close()
            del _sessions[session_id]
        except Exception:
            pass

    # Aggregate summary
    overall_score = sum(r.get("score", 0.0) for r in results) / max(len(results), 1)
    hallucination_rate = sum(all_hallucination_flags) / max(len(all_hallucination_flags), 1)
    avg_reward = sum(all_rewards) / max(len(all_rewards), 1)

    return {
        "model": model_label,
        "seed": seed,
        "steps_per_task": steps_per_task,
        "tasks": results,
        "summary": {
            "overall_score":     round(overall_score, 4),
            "avg_reward":        round(avg_reward, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "total_steps":       total_steps,
            "total_tasks":       len(results),
        },
        "note": (
            "These are heuristic baseline scores. Run run_baseline.py with "
            "OPENAI_API_KEY set to benchmark a real LLM."
        ),
    }


# ── Info & metrics ─────────────────────────────────────────────────────────────

# ── OpenEnv Required Endpoints ───────────────────────────────────────────────

@app.get("/metadata", summary="Environment metadata", tags=["OpenEnv"])
async def metadata():
    """
    GET /metadata — Required by OpenEnv validator.
    Returns environment name, description, version and author.
    """
    return {
        "name":        "hallucination-guard-env",
        "description": (
            "An OpenEnv RL environment that trains AI models to answer questions "
            "ONLY from verified context documents — penalizing hallucination and "
            "rewarding factual grounding. Built on 1,090,163 examples across 38 "
            "real-world QA datasets."
        ),
        "version":     "4.1.0",
        "author":      "SamSankar",
        "license":     "MIT",
        "tags": [
            "hallucination-detection",
            "question-answering",
            "grounded-generation",
            "fact-checking",
            "rl-environment",
        ],
        "links": {
            "space":  "https://huggingface.co/spaces/SamSankar/hallucination-guard-env",
            "pypi":   "https://pypi.org/project/openenv-halluguard/",
            "docs":   "https://samsankar-hallucination-guard-env.hf.space/docs",
        },
    }


@app.get("/schema", summary="Action, observation and state schemas", tags=["OpenEnv"])
async def schema():
    """
    GET /schema — Required by OpenEnv validator.
    Returns typed schemas for Action, Observation, and State spaces.
    """
    return {
        "action": {
            "type": "object",
            "description": "The agent response for one step.",
            "required": ["answer"],
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Answer derived ONLY from the provided context.",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "Calibrated confidence (0=unsure, 1=certain).",
                },
                "source_quote": {
                    "type": "string",
                    "default": "",
                    "description": "Verbatim snippet from the context supporting the answer.",
                },
                "reasoning": {
                    "type": "string",
                    "default": "",
                    "description": "Optional chain-of-thought explanation.",
                },
            },
        },
        "observation": {
            "type": "object",
            "description": "What the agent receives after each step.",
            "properties": {
                "question":              {"type": "string"},
                "context":               {"type": "string"},
                "reward":                {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "done":                  {"type": "boolean"},
                "is_hallucination":      {"type": "boolean"},
                "hallucination_type":    {"type": "string"},
                "hallucination_severity":{"type": "string"},
                "grounding_score":       {"type": "number"},
                "accuracy_so_far":       {"type": "number"},
                "skill_rating":          {"type": "number"},
                "attempts_remaining":    {"type": "integer"},
                "feedback":              {"type": "string"},
            },
        },
        "state": {
            "type": "object",
            "description": "Current episode state returned by GET /state.",
            "properties": {
                "episode_id":      {"type": "string"},
                "step":            {"type": "integer"},
                "max_steps":       {"type": "integer"},
                "done":            {"type": "boolean"},
                "skill_rating":    {"type": "number"},
                "difficulty":      {"type": "string"},
                "task_id":         {"type": "string"},
                "accuracy_so_far": {"type": "number"},
            },
        },
    }


@app.post("/mcp", summary="MCP JSON-RPC endpoint", tags=["OpenEnv"])
async def mcp(body: Dict[str, Any] = {}):
    """
    POST /mcp — Required by OpenEnv validator.
    Implements the Model Context Protocol (MCP) JSON-RPC 2.0 interface.
    Exposes environment tools: reset, step, state, tasks, grader.
    """
    jsonrpc = body.get("jsonrpc", "2.0")
    method  = body.get("method", "")
    params  = body.get("params", {})
    req_id  = body.get("id", 1)

    # MCP tools/list — advertise available tools
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "reset",
                        "description": "Start a new episode. Returns question + context.",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "step",
                        "description": "Submit an answer. Returns reward + hallucination info.",
                        "inputSchema": {
                            "type": "object",
                            "required": ["answer"],
                            "properties": {
                                "answer":       {"type": "string"},
                                "confidence":   {"type": "number"},
                                "source_quote": {"type": "string"},
                            },
                        },
                    },
                    {
                        "name": "state",
                        "description": "Get current episode state.",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "tasks",
                        "description": "List all 3 tasks with action schemas.",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ]
            },
        }

    # MCP tools/call — route to actual endpoint logic
    if method == "tools/call":
        tool_name   = params.get("name", "")
        tool_params = params.get("arguments", {})

        try:
            if tool_name == "reset":
                result = await reset(tool_params)
            elif tool_name == "step":
                result = await step(tool_params)
            elif tool_name == "state":
                result = await get_state()
            elif tool_name == "tasks":
                result = await list_tasks()
            else:
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": str(result)}],
                    "isError": False,
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": str(e)},
            }

    # Default — return server info for any unknown method
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "name":        "hallucination-guard-env",
            "version":     "4.1.0",
            "description": "OpenEnv RL environment for hallucination detection",
            "capabilities": {"tools": {}},
        },
    }


@app.get("/health", summary="Health check", tags=["Info"])
async def health():
    return {"status": "healthy", "service": "HallucinationGuard-Env", "version": "4.1.0", "pypi_package": "openenv-halluguard", "pypi_version": "2.0.1"}


@app.get("/environment/info", summary="Full environment spec", tags=["Info"])
async def env_info():
    return {
        "name":    "HallucinationGuard-Env",
        "version": "4.1.0",
        "description": "Production RL environment for hallucination detection & prevention",
        "pypi": {
            "package": "openenv-halluguard",
            "version": "2.0.1",
            "install": "pip install openenv-halluguard",
        },
        "datasets": {
            "count": 38,
            "total_examples": "1,090,163",
            "sources": [
                "squad", "squad_v2", "trivia_qa", "halueval", "truthful_qa",
                "hotpotqa", "boolq", "faithdial", "fever", "arc",
                "openbookqa", "ms_marco", "coqa", "nq_open",
                "commonsense_qa", "winogrande", "drop", "race", "newsqa",
                "hellaswag", "adversarial_qa", "ag_news", "aqua_rat", "circa",
                "climate_fever", "cnn_dailymail", "medqa", "medmcqa",
                "medical_questions", "pubmedqa", "qasc", "quartz", "quail",
                "sciq", "scitail", "xsum",
            ],
        },
        "tasks": {
            "count": 3,
            "ids": list(ALL_TASKS.keys()),
            "endpoint": "/tasks",
        },
        "endpoints": {
            "environment": ["/reset", "/step", "/state"],
            "openenv":     ["/tasks", "/grader", "/baseline"],
            "sessions":    ["/session/reset", "/session/step", "/session/list", "/session"],
            "leaderboard": ["/leaderboard", "/leaderboard/submit"],
            "info":        ["/health", "/environment/info", "/datasets", "/metrics"],
        },
        "difficulty_levels":    ["beginner", "intermediate", "advanced", "expert"],
        "hallucination_types":  [
            "fabricated_fact", "false_citation", "overconfident_wrong",
            "context_drift", "numerical_fabrication", "entity_confusion",
        ],
        "reward_components": 9,
        "reward_range":    [-1.0, 1.0],
        "supported_frameworks": ["OpenAI Gym", "OpenEnv", "custom Python", "REST API"],
    }


@app.get("/datasets", summary="Dataset statistics", tags=["Info"])
async def dataset_info():
    """Returns breakdown of loaded datasets by source, difficulty, and category."""
    try:
        env = _get_default_env()
        stats = env.dataset_loader.get_statistics()
        return {
            "total_examples":         stats.total_examples,
            "by_source":              stats.examples_by_source,
            "by_difficulty":          stats.examples_by_difficulty,
            "by_category":            stats.examples_by_category,
            "avg_context_length":     round(stats.average_context_length, 1),
            "avg_question_length":    round(stats.average_question_length, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", summary="Real-time metrics", tags=["Metrics"])
async def get_metrics():
    try:
        return get_tracker().get_real_time_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/summary", summary="Metrics summary report", tags=["Metrics"])
async def metrics_summary():
    try:
        return {"summary": get_tracker().generate_summary_report()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Middleware ─────────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request, call_next):
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} → {response.status_code}")
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
