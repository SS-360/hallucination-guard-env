"""FastAPI server for HallucinationGuard-Env with session management.

Standard endpoints (/reset, /step, /state, /health) — stateless, new env per request.
Session endpoints (/session/reset, /session/step) — stateful, env persists across calls.
"""

import sys, os, uuid, logging, dataclasses, enum
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse, RedirectResponse
from typing import Dict, Any, Optional

from models import HallucinationAction, HallucinationObservation, HallucinationState
from environment import HallucinationEnvironment
from metrics import get_tracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HallucinationGuard-Env",
    description="OpenEnv RL environment for training AI to avoid hallucinations",
    version="2.0.0",
)

# Session storage for stateful HTTP interactions
_sessions: Dict[str, HallucinationEnvironment] = {}
# Shared stateless env instance for standard endpoints
_default_env: Optional[HallucinationEnvironment] = None


def _get_default_env() -> HallucinationEnvironment:
    global _default_env
    if _default_env is None:
        _default_env = HallucinationEnvironment()
    return _default_env


def _safe_dict(obj):
    """Recursively convert dataclass/enum/dict to JSON-safe structure."""
    if dataclasses.is_dataclass(obj):
        result = {}
        for f in dataclasses.fields(obj):
            result[f.name] = _safe_dict(getattr(obj, f.name))
        return result
    elif isinstance(obj, enum.Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {k: _safe_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_safe_dict(i) for i in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


# ── Standard stateless endpoints ──────────────────────────────────────────────

@app.post("/reset")
async def reset(body: Dict[str, Any] = {}):
    """Reset environment and return initial observation."""
    try:
        env = _get_default_env()
        obs = env.reset(**{k: v for k, v in body.items()
                           if k in ("seed", "episode_id", "difficulty",
                                    "enable_multi_turn", "enable_context_retrieval")})
        return JSONResponse(content=_safe_dict(obs))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(action_data: Dict[str, Any]):
    """Take a step with the provided action."""
    try:
        env = _get_default_env()
        valid = {f.name for f in dataclasses.fields(HallucinationAction)}
        action = HallucinationAction(**{k: v for k, v in action_data.items() if k in valid})
        obs = env.step(action)
        return JSONResponse(content=_safe_dict(obs))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state():
    """Get current environment state."""
    try:
        return JSONResponse(content=_safe_dict(_get_default_env().state()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Session-based stateful endpoints ──────────────────────────────────────────

@app.post("/session/reset")
async def session_reset(
    body: Dict[str, Any] = {},
    x_session_id: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """Create or reset a named session."""
    session_id = x_session_id or str(uuid.uuid4())
    if session_id in _sessions:
        _sessions[session_id].close()
    _sessions[session_id] = HallucinationEnvironment(session_id=session_id)
    obs = _sessions[session_id].reset(**{k: v for k, v in body.items()
                                          if k in ("seed", "episode_id", "difficulty",
                                                   "enable_multi_turn", "enable_context_retrieval")})
    result = _safe_dict(obs)
    result["session_id"] = session_id
    logger.info(f"Created session {session_id}")
    return result


@app.post("/session/step")
async def session_step(
    action_data: Dict[str, Any],
    x_session_id: str = Header(...),
) -> Dict[str, Any]:
    """Execute a step in an existing session."""
    if x_session_id not in _sessions:
        raise HTTPException(status_code=404,
                            detail=f"Session {x_session_id} not found. Call /session/reset first.")
    valid = {f.name for f in dataclasses.fields(HallucinationAction)}
    action = HallucinationAction(**{k: v for k, v in action_data.items() if k in valid})
    obs = _sessions[x_session_id].step(action)
    result = _safe_dict(obs)
    result["session_id"] = x_session_id
    return result


@app.delete("/session")
async def close_session(x_session_id: str = Header(...)) -> Dict[str, str]:
    """Close and clean up a session."""
    if x_session_id in _sessions:
        _sessions[x_session_id].close()
        del _sessions[x_session_id]
    return {"status": "closed", "session_id": x_session_id}


@app.get("/session/list")
async def list_sessions() -> Dict[str, Any]:
    return {"active_sessions": len(_sessions), "session_ids": list(_sessions.keys())}


# ── Utility endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "HallucinationGuard-Env", "version": "2.0.0"}


@app.get("/metrics")
async def get_metrics():
    try:
        return get_tracker().get_real_time_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/summary")
async def metrics_summary():
    try:
        return {"summary": get_tracker().generate_summary_report()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/environment/info")
async def env_info():
    return {
        "name": "HallucinationGuard-Env",
        "version": "2.0.0",
        "endpoints": {
            "standard": ["/reset", "/step", "/state", "/health"],
            "session":  ["/session/reset", "/session/step", "/session", "/session/list"],
            "metrics":  ["/metrics", "/metrics/summary"],
        },
        "difficulty_levels": ["beginner", "intermediate", "advanced", "expert"],
        "hallucination_types": [
            "fabricated_fact", "false_citation", "overconfident_wrong",
            "context_drift", "numerical_fabrication", "entity_confusion",
        ],
        "supported_models": ["openai", "anthropic", "huggingface", "ollama", "generic"],
    }


@app.middleware("http")
async def log_requests(request, call_next):
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} → {response.status_code}")
    return response



@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
