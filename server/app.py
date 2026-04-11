"""
HallucinationGuard-Env v4.2 — Production FastAPI Server

Endpoints:
  Standard   : POST /reset  POST /step  GET /state  GET /health
  OpenEnv    : GET /tasks  POST /grader  POST /baseline
  Extra      : GET /leaderboard  POST /leaderboard/submit  GET /datasets

"""

import sys, os, uuid, logging, dataclasses, enum, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from models import HallucinationAction, HallucinationObservation, HallucinationState
from environment import HallucinationEnvironment
from metrics import get_tracker

from tasks import (
    ALL_TASKS, get_task, task_id_for_difficulty, compute_task_score, ACTION_SCHEMA,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# MINIMALIST GRADIO-STYLE DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

STUNNING_DOCS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HallucinationGuard-Env</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',system-ui,sans-serif;background:#f7f7f8;color:#1a1a2e;line-height:1.6}
.mono{font-family:'JetBrains Mono',monospace}
header{background:#fff;border-bottom:1px solid #e5e7eb;padding:16px 32px;display:flex;align-items:center;gap:12px}
header h1{font-size:20px;font-weight:700;color:#1a1a2e}
header span{font-size:13px;color:#6b7280;margin-left:8px}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;background:#ecfdf5;color:#059669}
main{max-width:1100px;margin:0 auto;padding:24px 32px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}
.stat-card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:16px;text-align:center}
.stat-card .num{font-size:28px;font-weight:700;color:#1a1a2e}
.stat-card .lbl{font-size:12px;color:#6b7280;margin-top:4px}
.tabs{display:flex;border-bottom:2px solid #e5e7eb;margin-bottom:20px;gap:0}
.tab{padding:10px 20px;cursor:pointer;font-size:14px;font-weight:500;color:#6b7280;border-bottom:2px solid transparent;margin-bottom:-2px;transition:all .15s}
.tab:hover{color:#1a1a2e}
.tab.active{color:#2563eb;border-bottom-color:#2563eb}
.panel{display:none}
.panel.active{display:block}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:20px;margin-bottom:16px}
.card h3{font-size:15px;font-weight:600;margin-bottom:8px}
.card p{font-size:13px;color:#4b5563}
.task-card{border-left:4px solid #e5e7eb}
.task-card.beginner{border-left-color:#10b981}
.task-card.intermediate{border-left-color:#3b82f6}
.task-card.advanced{border-left-color:#ef4444}
.diff{font-size:11px;font-weight:600;padding:2px 8px;border-radius:4px;text-transform:uppercase;letter-spacing:.5px}
.diff.beginner{background:#ecfdf5;color:#059669}
.diff.intermediate{background:#eff6ff;color:#2563eb}
.diff.advanced{background:#fef2f2;color:#dc2626}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:10px 12px;background:#f9fafb;border-bottom:1px solid #e5e7eb;font-weight:600;font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:.5px}
td{padding:10px 12px;border-bottom:1px solid #f3f4f6}
.method{padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;font-family:'JetBrains Mono',monospace}
.method.get{background:#ecfdf5;color:#059669}
.method.post{background:#eff6ff;color:#2563eb}
.btn{padding:8px 16px;border-radius:6px;border:1px solid #d1d5db;background:#fff;font-size:13px;font-weight:500;cursor:pointer;transition:all .15s}
.btn:hover{background:#f9fafb;border-color:#9ca3af}
.btn.primary{background:#2563eb;color:#fff;border-color:#2563eb}
.btn.primary:hover{background:#1d4ed8}
.input-group{margin-bottom:12px}
.input-group label{display:block;font-size:12px;font-weight:500;color:#374151;margin-bottom:4px}
.input-group input,.input-group textarea,.input-group select{width:100%;padding:8px 12px;border:1px solid #d1d5db;border-radius:6px;font-size:13px;font-family:'Inter',sans-serif}
.input-group textarea{min-height:60px;resize:vertical}
.response-box{background:#f9fafb;border:1px solid #e5e7eb;border-radius:6px;padding:12px;font-family:'JetBrains Mono',monospace;font-size:12px;white-space:pre-wrap;max-height:300px;overflow-y:auto;color:#374151}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:768px){.stats{grid-template-columns:repeat(2,1fr)}.grid-2{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
<h1>HallucinationGuard-Env</h1>
<span class="badge">OpenEnv</span>
<span>v4.2.0</span>
</header>
<main>
<div class="stats">
<div class="stat-card"><div class="num">1M+</div><div class="lbl">Examples</div></div>
<div class="stat-card"><div class="num">38</div><div class="lbl">Datasets</div></div>
<div class="stat-card"><div class="num">3</div><div class="lbl">Tasks</div></div>
<div class="stat-card"><div class="num">9</div><div class="lbl">Reward Components</div></div>
</div>
<div class="tabs">
<div class="tab active" onclick="showTab('overview')">Overview</div>
<div class="tab" onclick="showTab('tasks')">Tasks</div>
<div class="tab" onclick="showTab('api')">API</div>
<div class="tab" onclick="showTab('playground')">Playground</div>
</div>
<div id="overview" class="panel active">
<div class="card">
<h3>What is this?</h3>
<p>An OpenEnv RL environment that trains AI models to answer questions <strong>only from verified context documents</strong> &mdash; penalizing hallucination and rewarding factual grounding. The 9-component reward system evaluates factual correctness, source grounding, citation accuracy, confidence calibration, semantic consistency, hallucination detection, ROUGE-L, BERTScore, and AlignScore.</p>
</div>
<div class="card">
<h3>How it works</h3>
<p><strong>reset()</strong> &mdash; Sample a question + context from one of 38 datasets.<br>
<strong>step(answer)</strong> &mdash; Grade the answer across 9 components, return reward + feedback.<br>
<strong>state()</strong> &mdash; Get episode metadata, accuracy, skill rating.<br>
The curriculum progresses from beginner (single-hop factual QA) through intermediate (multi-hop synthesis) to advanced (adversarial hallucination resistance).</p>
</div>
<div class="card">
<h3>Quick Start</h3>
<p class="mono" style="font-size:12px;background:#f9fafb;padding:12px;border-radius:6px;border:1px solid #e5e7eb">
pip install -e .<br>
uvicorn server.app:app --port 7860<br>
python inference.py --heuristic --env-url http://localhost:7860
</p>
</div>
</div>
<div id="tasks" class="panel">
<div class="card task-card beginner">
<h3>Factual Grounding <span class="diff beginner">Beginner</span></h3>
<p>Answer straightforward factual questions from a short context passage. SQuAD, BoolQ, OpenBookQA, ARC &mdash; single-hop retrieval with unambiguous ground-truth. Grader rewards correct citation and penalizes fabrication.</p>
</div>
<div class="card task-card intermediate">
<h3>Multi-Hop Synthesis <span class="diff intermediate">Intermediate</span></h3>
<p>Synthesize evidence from multiple context sentences. HotpotQA, CoQA, NQ-Open, MS-MARCO &mdash; requires connecting disparate facts without fabricating bridge claims.</p>
</div>
<div class="card task-card advanced">
<h3>Adversarial Resistance <span class="diff advanced">Advanced</span></h3>
<p>Resist adversarial prompts designed to elicit hallucinations. HaluEval, TruthfulQA, FEVER, Climate-FEVER &mdash; many questions are unanswerable; confident refusals score well.</p>
</div>
</div>
<div id="api" class="panel">
<div class="card">
<table>
<thead><tr><th>Method</th><th>Endpoint</th><th>Description</th></tr></thead>
<tbody>
<tr><td><span class="method post">POST</span></td><td class="mono">/reset</td><td>Start episode, return question + context</td></tr>
<tr><td><span class="method post">POST</span></td><td class="mono">/step</td><td>Submit answer, receive reward + feedback</td></tr>
<tr><td><span class="method get">GET</span></td><td class="mono">/state</td><td>Get current episode state</td></tr>
<tr><td><span class="method get">GET</span></td><td class="mono">/tasks</td><td>List 3 tasks + action schema</td></tr>
<tr><td><span class="method post">POST</span></td><td class="mono">/grader</td><td>Score completed episode 0.0-1.0</td></tr>
<tr><td><span class="method post">POST</span></td><td class="mono">/baseline</td><td>Run heuristic baseline</td></tr>
<tr><td><span class="method get">GET</span></td><td class="mono">/metadata</td><td>Environment metadata</td></tr>
<tr><td><span class="method get">GET</span></td><td class="mono">/schema</td><td>Action, observation, state schemas</td></tr>
<tr><td><span class="method get">GET</span></td><td class="mono">/health</td><td>Health check</td></tr>
<tr><td><span class="method post">POST</span></td><td class="mono">/mcp</td><td>JSON-RPC tool discovery</td></tr>
</tbody>
</table>
</div>
</div>
<div id="playground" class="panel">
<div class="grid-2">
<div>
<div class="card">
<h3>Episode Controls</h3>
<div class="input-group">
<label>Difficulty</label>
<select id="difficulty">
<option value="beginner">Beginner</option>
<option value="intermediate">Intermediate</option>
<option value="advanced">Advanced</option>
</select>
</div>
<div style="display:flex;gap:8px;margin-bottom:12px">
<button class="btn primary" onclick="doReset()">Reset</button>
<button class="btn" onclick="doStep()">Step</button>
</div>
<div class="input-group">
<label>Answer</label>
<textarea id="answer" placeholder="Your answer..."></textarea>
</div>
<div class="input-group">
<label>Confidence (0-1)</label>
<input id="confidence" type="number" min="0" max="1" step="0.1" value="0.7">
</div>
<div class="input-group">
<label>Source Quote</label>
<input id="source_quote" type="text" placeholder="Verbatim quote from context...">
</div>
</div>
</div>
<div>
<div class="card">
<h3>Observation</h3>
<div id="obs-response" class="response-box">Click Reset to start an episode.</div>
</div>
<div class="card">
<h3>Step Response</h3>
<div id="step-response" class="response-box">Submit an answer to see the response.</div>
</div>
</div>
</div>
</div>
</main>
<script>
let sessionId=null;
function showTab(id){document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));event.target.classList.add('active');document.getElementById(id).classList.add('active')}
async function doReset(){try{const d=document.getElementById('difficulty').value;const r=await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({difficulty:d,seed:42})});const data=await r.json();sessionId=data.session_id||null;document.getElementById('obs-response').textContent=JSON.stringify(data,null,2);document.getElementById('step-response').textContent='Ready for step.'}catch(e){document.getElementById('obs-response').textContent='Error: '+e.message}}
async function doStep(){try{if(!sessionId){document.getElementById('step-response').textContent='Reset first!';return}const body={answer:document.getElementById('answer').value,confidence:parseFloat(document.getElementById('confidence').value)||0.5,source_quote:document.getElementById('source_quote').value,session_id:sessionId};const r=await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();document.getElementById('step-response').textContent=JSON.stringify(data,null,2);if(data.done){sessionId=null;document.getElementById('obs-response').textContent='Episode done. Reset to start a new one.'}}catch(e){document.getElementById('step-response').textContent='Error: '+e.message}}
</script>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP — session-isolated environments for thread safety
# ═══════════════════════════════════════════════════════════════════════════════

_default_env: Optional[HallucinationEnvironment] = None
_env_loading = False
_env_lock = threading.Lock()

def _get_default_env() -> HallucinationEnvironment:
    """Get or create the shared dataset-loader environment (used only for dataset access)."""
    global _default_env, _env_loading
    if _default_env is not None:
        return _default_env
    with _env_lock:
        if _default_env is not None:
            return _default_env
        _env_loading = True
        try:
            logger.info("Creating HallucinationEnvironment (dataset loader)...")
            _default_env = HallucinationEnvironment()
            logger.info(f"Environment ready — {_default_env.dataset_loader.get_total_examples():,} examples loaded.")
            return _default_env
        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            from dataset_loader import DatasetLoader
            class MinimalEnv:
                def __init__(self):
                    self.dataset_loader = DatasetLoader()
                    self.dataset_loader.examples = []
                def reset(self, **kwargs):
                    return type('Obs', (), {'question': 'Placeholder', 'context': 'Context', 'reward': 0.0, 'done': False, 'info': {}})()
                def step(self, action):
                    return type('Obs', (), {'reward': 0.0, 'done': False, 'is_hallucination': False, 'info': {}})()
                def state(self): return {}
                def close(self): pass
            _default_env = MinimalEnv()
            return _default_env
        finally:
            _env_loading = False


def _create_session_env(session_id: str) -> HallucinationEnvironment:
    """Create a fresh per-session environment that shares the dataset loader
    (expensive to load) but has its own episode state (safe for concurrent use)."""
    loader_env = _get_default_env()
    # Pass the shared loader directly into __init__ so we skip the expensive
    # DatasetLoader() construction and dataset loading that would otherwise
    # happen inside HallucinationEnvironment.__init__
    env = HallucinationEnvironment(session_id=session_id, dataset_loader=loader_env.dataset_loader)
    return env


_sessions: Dict[str, HallucinationEnvironment] = {}
_session_lock = threading.Lock()


def _get_session(session_id: str) -> Optional[HallucinationEnvironment]:
    """Retrieve an existing session environment."""
    with _session_lock:
        return _sessions.get(session_id)


def _cleanup_session(session_id: str):
    """Remove and clean up a session environment."""
    with _session_lock:
        env = _sessions.pop(session_id, None)
    if env:
        try: env.close()
        except: pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _default_env

    def preload_models():
        try:
            logger.info("Preloading ML models...")
            from sentence_transformers import SentenceTransformer, CrossEncoder
            SentenceTransformer('all-MiniLM-L6-v2')
            CrossEncoder('cross-encoder/nli-deberta-v3-small')
            from rouge_score import rouge_scorer
            rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            try:
                from grader import _get_bert_scorer
                _get_bert_scorer()
            except: pass
            logger.info("All ML models preloaded!")
        except Exception as e:
            logger.error(f"Model preload failed: {e}")

    threading.Thread(target=preload_models, daemon=True).start()

    def background_load():
        try:
            logger.info("Background dataset loading...")
            env = _get_default_env()
            logger.info(f"Loaded {env.dataset_loader.get_total_examples():,} examples.")
        except Exception as e:
            logger.error(f"Background loading failed: {e}")

    threading.Thread(target=background_load, daemon=True).start()
    yield
    if _default_env:
        try: _default_env.close()
        except: pass

app = FastAPI(
    lifespan=lifespan,
    title="HallucinationGuard-Env",
    version="4.2.0",
    docs_url="/swagger",
    redoc_url="/redoc",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

import json as _json
_LEADERBOARD_FILE = "/tmp/hallucination_guard_leaderboard.json"

def _load_leaderboard():
    if os.path.exists(_LEADERBOARD_FILE):
        try:
            with open(_LEADERBOARD_FILE, "r", encoding="utf-8") as f:
                return _json.load(f)
        except Exception:
            pass
    return {}

def _save_leaderboard(lb):
    try:
        with open(_LEADERBOARD_FILE, "w", encoding="utf-8") as f:
            _json.dump(lb, f, indent=2)
    except Exception:
        pass

_leaderboard: Dict[str, Dict[str, Any]] = _load_leaderboard()

def _safe_dict(obj):
    if hasattr(obj, 'model_dump'): return _safe_dict(obj.model_dump())
    if hasattr(obj, 'dict'): return _safe_dict(obj.dict())
    if dataclasses.is_dataclass(obj): return {f.name: _safe_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum): return obj.value
    if isinstance(obj, dict): return {k: _safe_dict(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_safe_dict(i) for i in obj]
    if isinstance(obj, (str, int, float, bool, type(None))): return obj
    return str(obj)

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def root(): return STUNNING_DOCS_HTML

@app.get("/docs", include_in_schema=False, response_class=HTMLResponse)
async def docs(): return STUNNING_DOCS_HTML

@app.post("/reset", tags=["Environment"])
async def reset(body: Dict[str, Any] = {}):
    try:
        # Create a per-session environment for thread safety
        session_id = body.get("session_id") or f"ses_{uuid.uuid4().hex[:8]}"
        env = _create_session_env(session_id)
        obs = env.reset(**{k: v for k, v in body.items() if k in ("seed", "episode_id", "difficulty")})
        # Store the episode_id -> session mapping so /step can find this env
        episode_id = getattr(obs, 'episode_id', None) or body.get("episode_id") or session_id
        with _session_lock:
            _sessions[episode_id] = env
            _sessions[session_id] = env
        result = _safe_dict(obs)
        result["session_id"] = session_id
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        logger.error(f"Reset error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

@app.post("/step", tags=["Environment"])
async def step(action_data: Dict[str, Any]):
    try:
        # Look up session by episode_id or session_id for thread safety
        session_id = action_data.pop("session_id", None) or action_data.pop("episode_id", None)
        env = _get_session(session_id) if session_id else None
        if env is None:
            # Fallback: use default env (single-user mode)
            env = _get_default_env()
        valid = set(HallucinationAction.model_fields.keys()) if hasattr(HallucinationAction, 'model_fields') else set(HallucinationAction.__fields__.keys())
        action = HallucinationAction(**{k: v for k, v in action_data.items() if k in valid})
        result = _safe_dict(env.step(action))
        # If episode is done, clean up session
        if result.get("done", False) and session_id:
            _cleanup_session(session_id)
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        logger.error(f"Step error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

@app.get("/state", tags=["Environment"])
async def get_state(session_id: Optional[str] = None):
    try:
        env = _get_session(session_id) if session_id else _get_default_env()
        return JSONResponse(content=_safe_dict(env.state()))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/tasks", tags=["OpenEnv"])
async def list_tasks():
    ordered = ["task_1_factual_grounding", "task_2_multi_hop_synthesis", "task_3_adversarial_resistance"]
    return {"tasks": [ALL_TASKS[t].to_dict() for t in ordered if t in ALL_TASKS], "action_schema": ACTION_SCHEMA}

@app.post("/grader", tags=["OpenEnv"])
async def grade_episode(body: Dict[str, Any]):
    task_id = body.get("task_id")
    if not task_id: raise HTTPException(422, "'task_id' required")
    task = get_task(task_id)
    if not task: raise HTTPException(404, f"task_id '{task_id}' not found")
    rewards, infos = body.get("step_rewards", []), body.get("step_infos", [])
    if not infos and rewards: return {"task_id": task_id, "score": round(sum(rewards)/len(rewards), 4)}
    return compute_task_score(task, rewards, infos)

@app.post("/baseline", tags=["OpenEnv"])
async def run_baseline(body: Dict[str, Any] = {}):
    steps = max(3, min(10, int(body.get("steps_per_task", 5))))
    seed = int(body.get("seed", 42))
    results = []
    for task_id, diff in [("task_1_factual_grounding","beginner"),("task_2_multi_hop_synthesis","intermediate"),("task_3_adversarial_resistance","advanced")]:
        task = get_task(task_id)
        if not task: continue
        sid = f"bl_{task_id}_{seed}"
        # Use session-based env with shared dataset loader
        env = _create_session_env(sid)
        obs_dict = _safe_dict(env.reset(seed=seed, difficulty=diff))
        rewards, infos = [], []
        for _ in range(steps):
            if obs_dict.get("done"): break
            ctx = obs_dict.get("context", "")
            action = HallucinationAction(answer=ctx[:100], confidence=0.6, source_quote=ctx[:80])
            obs_dict = _safe_dict(env.step(action))
            rewards.append(float(obs_dict.get("reward") or 0))
            obs_meta = obs_dict.get("metadata", {})
            if isinstance(obs_meta, dict):
                obs_correctness = obs_meta.get("correctness", 0.0)
            else:
                obs_correctness = 0.0
            infos.append({
                "correctness": obs_correctness,
                "grounding": obs_dict.get("grounding_score", 0),
                "calibration": 0.6,
                "hallucination_score": 1.0 if obs_dict.get("is_hallucination") else 0.0,
                "is_hallucination": bool(obs_dict.get("is_hallucination", False)),
            })
        results.append(compute_task_score(task, rewards, infos))
        try: env.close()
        except: pass
    return {"tasks": results, "summary": {"overall_score": round(sum(r["score"] for r in results)/max(len(results),1), 4)}}

@app.post("/batch/evaluate", tags=["Evaluation"])
async def batch_evaluate(body: Dict[str, Any]):
    items = body.get("items", [])
    if not items: raise HTTPException(422, "'items' required")
    from server.grader import calculate_reward
    results = []
    for i, item in enumerate(items):
        r, info = calculate_reward(item.get("answer",""), item.get("confidence",0.5), item.get("source_quote",""), item.get("context",""), item.get("ground_truth",""))
        results.append({"index": i, "reward": round(r,4), "is_hallucination": info.get("is_hallucination", False)})
    return {"total_items": len(results), "results": results}

@app.get("/leaderboard", tags=["Leaderboard"])
async def leaderboard():
    if not _leaderboard: return {"leaderboard": [], "message": "No submissions"}
    ranked = sorted(_leaderboard.values(), key=lambda x: x.get("avg_reward",0), reverse=True)
    for i, e in enumerate(ranked): e["rank"] = i+1
    return {"leaderboard": ranked}

@app.post("/leaderboard/submit", tags=["Leaderboard"])
async def submit_leaderboard(data: Dict[str, Any]):
    required = ["model_name", "avg_reward", "avg_accuracy", "hallucination_rate", "total_episodes", "total_steps"]
    if missing := [f for f in required if f not in data]: raise HTTPException(422, f"Missing: {missing}")
    _leaderboard[data["model_name"]] = {**data, "submitted_at": time.time()}
    _save_leaderboard(_leaderboard)
    return {"status": "submitted", "model_name": data["model_name"]}

@app.get("/health", tags=["Info"])
async def health(): return {"status": "healthy", "version": "4.2.0"}

@app.get("/metadata", tags=["OpenEnv"])
async def metadata():
    return {
        "name": "hallucination-guard-env",
        "version": "4.2.0",
        "license": "MIT",
        "description": (
            "An OpenEnv RL environment that trains AI models to answer questions "
            "ONLY from verified context documents — penalizing hallucination and "
            "rewarding factual grounding."
        ),
    }

@app.get("/schema", tags=["OpenEnv"])
async def schema():
    return {
        "action": {
            "type": "object",
            "required": ["answer"],
            "properties": {
                "answer":       {"type": "string",  "description": "Answer derived ONLY from the provided context document."},
                "confidence":   {"type": "number",  "minimum": 0.0, "maximum": 1.0, "default": 0.5},
                "source_quote": {"type": "string",  "default": ""},
                "reasoning":    {"type": "string",  "default": ""},
            },
        },
        "observation": {
            "type": "object",
            "properties": {
                "question":           {"type": "string"},
                "context":            {"type": "string"},
                "ground_truth":       {"type": "string"},
                "done":               {"type": "boolean"},
                "reward":             {"type": "number"},
                "feedback":           {"type": "string"},
                "is_hallucination":   {"type": "boolean"},
                "grounding_score":    {"type": "number"},
                "difficulty_level":   {"type": "string"},
                "attempts_remaining": {"type": "integer"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "episode_id":            {"type": "string"},
                "step_count":            {"type": "integer"},
                "accuracy":              {"type": "number"},
                "hallucination_rate":    {"type": "number"},
                "average_reward":        {"type": "number"},
                "current_difficulty":    {"type": "string"},
                "skill_rating":          {"type": "number"},
                "current_streak":        {"type": "integer"},
                "best_streak":           {"type": "integer"},
            },
        },
    }

@app.get("/datasets", tags=["Info"])
async def datasets():
    try: return {"total_examples": _get_default_env().dataset_loader.get_total_examples()}
    except: return {"total_examples": 0}

@app.post("/mcp", tags=["OpenEnv"])
async def mcp(body: Dict[str, Any]):
    if body.get("method") == "tools/list":
        return {"jsonrpc": "2.0", "id": body.get("id",1), "result": {"tools": [{"name": "reset", "inputSchema": {"type": "object"}}, {"name": "step", "inputSchema": {"type": "object"}}]}}
    return {"jsonrpc": "2.0", "id": body.get("id",1), "result": {"name": "hallucination-guard-env", "version": "4.2.0"}}

@app.middleware("http")
async def log_req(request, call_next):
    resp = await call_next(request)
    logger.info(f"{request.method} {request.url.path} → {resp.status_code}")
    return resp

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
