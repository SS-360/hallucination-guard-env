"""
HallucinationGuard-Env v4.0 — Production FastAPI Server

Endpoints:
  Standard  : POST /reset  POST /step  GET /state  GET /health
  Session   : POST /session/reset  POST /session/step  DELETE /session
  Leaderboard: GET /leaderboard  POST /leaderboard/submit  DELETE /leaderboard/{model}
  Info      : GET /  GET /docs  GET /environment/info  GET /datasets
              GET /metrics  GET /metrics/summary
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
## 🛡️ HallucinationGuard-Env v4.0

**The production-grade OpenEnv RL environment for training and evaluating LLMs on hallucination avoidance.**

Built on 1,000,000+ examples across 38 real-world QA datasets:
SQuAD · TriviaQA · HaluEval · TruthfulQA · HotpotQA · BoolQ · HH-RLHF ·
NQ Open · CommonsenseQA · WinoGrande · CoQA · OpenBookQA · MS MARCO · ARC · Medical QA

### Quick Start

```python
pip install requests
import requests

BASE = "https://samsankar-hallucination-guard-env.hf.space"

# 1. Start episode
obs = requests.post(f"{BASE}/reset").json()
print(obs["question"], obs["context"])

# 2. Answer from context only
result = requests.post(f"{BASE}/step", json={"answer": "your answer"}).json()
print(result["reward"], result["is_hallucination"])
```

### Python SDK

```python
pip install openenv-halluguard
from openenv_halluguard import HallucinationGuardEnv
env = HallucinationGuardEnv()
results = env.evaluate(your_model_fn, episodes=5)
env.print_report(results)
```

### HallucinationGuard - [Website](https://huggingface.co/spaces/SamSankar/hallucination-guard-env) · [PyPI](https://pypi.org/project/openenv-halluguard/) · [Docs](https://samsankar-hallucination-guard-env.hf.space/docs)
    """,
    version="4.0.0",
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


# ── Info & metrics ─────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check", tags=["Info"])
async def health():
    return {"status": "healthy", "service": "HallucinationGuard-Env", "version": "4.0.0"}


@app.get("/environment/info", summary="Full environment spec", tags=["Info"])
async def env_info():
    return {
        "name":    "HallucinationGuard-Env",
        "version": "4.0.0",
        "description": "Production RL environment for hallucination detection & prevention",
        "datasets": {
            "count": 15,
            "total_examples": "1,000,000+",
            "sources": [
                "squad", "trivia_qa", "halueval", "truthful_qa",
                "hotpotqa", "boolq", "faithdial", "fever", "arc",
                "openbookqa", "ms_marco", "coqa", "nq_open",
                "commonsense_qa", "winogrande",
            ],
        },
        "endpoints": {
            "environment": ["/reset", "/step", "/state"],
            "sessions":    ["/session/reset", "/session/step", "/session/list", "/session"],
            "leaderboard": ["/leaderboard", "/leaderboard/submit"],
            "info":        ["/health", "/environment/info", "/datasets", "/metrics"],
        },
        "difficulty_levels":    ["beginner", "intermediate", "advanced", "expert"],
        "hallucination_types":  [
            "fabricated_fact", "false_citation", "overconfident_wrong",
            "context_drift", "numerical_fabrication", "entity_confusion",
        ],
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
