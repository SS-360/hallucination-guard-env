"""
OpenEnv compatibility shim for HallucinationGuard-Env.
This provides base classes that match the openenv-core interface.
When deploying to HuggingFace Spaces, openenv-core will be installed
and this shim won't be needed.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, Type, TypeVar
import json

# Type variables for generics
ActionType = TypeVar("ActionType", bound="Action")
ObservationType = TypeVar("ObservationType", bound="Observation")
StateType = TypeVar("StateType", bound="State")


@dataclass
class Action:
    """Base class for environment actions."""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


@dataclass
class Observation:
    """Base class for environment observations."""
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


@dataclass
class State:
    """Base class for environment state."""
    episode_id: Optional[str] = None
    step_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


class Environment(Generic[ActionType, ObservationType, StateType]):
    """Base class for OpenEnv environments."""

    def __init__(self, transform=None, **kwargs):
        self.transform = transform

    def reset(self, **kwargs) -> ObservationType:
        raise NotImplementedError

    def step(self, action: ActionType, **kwargs) -> ObservationType:
        raise NotImplementedError

    def state(self) -> StateType:
        raise NotImplementedError

    def close(self) -> None:
        pass


def create_fastapi_app(env, action_cls=None, observation_cls=None):
    """
    Create a FastAPI app for the given environment class.
    Implements the standard OpenEnv HTTP interface:
    - POST /reset
    - POST /step
    - GET /state
    - GET /health
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
        import dataclasses
        import uvicorn
    except ImportError:
        raise ImportError("fastapi and uvicorn are required. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="HallucinationGuard-Env",
        description="OpenEnv environment for training AI to avoid hallucinations",
        version="2.0.0"
    )

    # Create a single environment instance
    _env_instance = env()

    def _to_dict(obj):
        """Convert dataclass or object to dict safely."""
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, dict):
            return obj
        return {}

    def _safe_dict(obj):
        """Recursively convert to JSON-safe dict."""
        import enum
        if dataclasses.is_dataclass(obj):
            result = {}
            for f in dataclasses.fields(obj):
                val = getattr(obj, f.name)
                result[f.name] = _safe_dict(val)
            return result
        elif isinstance(obj, enum.Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {k: _safe_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_safe_dict(i) for i in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    @app.post("/reset")
    async def reset(request: dict = {}):
        """Reset the environment."""
        try:
            obs = _env_instance.reset(**request)
            return JSONResponse(content=_safe_dict(obs))
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )

    @app.post("/step")
    async def step(action_data: dict):
        """Take a step in the environment."""
        try:
            if action_cls:
                # Build action from dict, filtering to known fields
                import dataclasses as dc
                valid_fields = {f.name for f in dc.fields(action_cls)}
                filtered = {k: v for k, v in action_data.items() if k in valid_fields}
                action = action_cls(**filtered)
            else:
                action = action_data
            obs = _env_instance.step(action)
            return JSONResponse(content=_safe_dict(obs))
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )

    @app.get("/state")
    async def get_state():
        """Get the current environment state."""
        try:
            state = _env_instance.state()
            return JSONResponse(content=_safe_dict(state))
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "service": "HallucinationGuard-Env"}

    return app
