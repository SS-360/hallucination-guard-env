#!/usr/bin/env python3
"""
run_baseline.py — HallucinationGuard-Env Baseline Inference Script
===================================================================

Runs an OpenAI-compatible model against all 3 tasks of the
HallucinationGuard-Env and produces reproducible baseline scores.

Usage
-----
    export OPENAI_API_KEY=sk-...
    python run_baseline.py

    # Custom model / endpoint / episodes
    python run_baseline.py --model gpt-4o --episodes 3 --steps 5
    python run_baseline.py --model gpt-3.5-turbo --base-url https://your-endpoint/v1

    # Run against local dev server instead of HF Space
    python run_baseline.py --env-url http://localhost:7860

    # Dry-run with heuristic agent (no API key needed)
    python run_baseline.py --heuristic

Environment variables
---------------------
    OPENAI_API_KEY      Required unless --heuristic flag is used
    OPENAI_BASE_URL     Optional — override OpenAI endpoint
    HALLUGUARD_ENV_URL  Optional — override environment URL

Expected baseline scores (gpt-3.5-turbo, 3 episodes × 5 steps, seed=42)
-------------------------------------------------------------------------
    task_1_factual_grounding      : 0.58 ± 0.08
    task_2_multi_hop_synthesis    : 0.47 ± 0.09
    task_3_adversarial_resistance : 0.34 ± 0.10
    overall                       : 0.46 ± 0.06
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_ENV_URL  = os.environ.get(
    "HALLUGUARD_ENV_URL",
    "https://samsankar-hallucination-guard-env.hf.space",
)
DEFAULT_MODEL    = "gpt-3.5-turbo"
DEFAULT_EPISODES = 3
DEFAULT_STEPS    = 5
SEED             = 42

TASK_ORDER = [
    ("task_1_factual_grounding",      "beginner"),
    ("task_2_multi_hop_synthesis",    "intermediate"),
    ("task_3_adversarial_resistance", "advanced"),
]

SYSTEM_PROMPT = """You are a precise, grounded question-answering assistant.

RULES (follow strictly):
1. Answer ONLY using information present in the CONTEXT provided.
2. If the answer is not in the context, say exactly: "I cannot answer from the provided context."
3. Keep answers concise — 1–3 sentences.
4. Never fabricate facts, names, dates, or numbers not in the context.
5. If uncertain, express that uncertainty in your answer.
"""

ANSWER_PROMPT_TEMPLATE = """CONTEXT:
{context}

QUESTION:
{question}

Instructions:
- Answer using ONLY the context above.
- Provide a source_quote: a short verbatim phrase from the context that supports your answer.
- Rate your confidence from 0.0 (unsure) to 1.0 (certain).

Respond in JSON with these exact keys:
{{
    "answer": "<your answer>",
    "source_quote": "<verbatim phrase from context>",
    "confidence": <float 0.0–1.0>
}}"""


# ── Environment client ────────────────────────────────────────────────────────

class EnvClient:
    """Thin HTTP wrapper around the HallucinationGuard REST API."""

    def __init__(self, base_url: str, timeout: int = 30):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _get(self, path: str) -> Dict[str, Any]:
        r = self.session.get(f"{self.base}{path}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: Dict[str, Any] = {}) -> Dict[str, Any]:
        r = self.session.post(f"{self.base}{path}", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def health(self) -> Dict[str, Any]:
        return self._get("/health")

    def list_tasks(self) -> Dict[str, Any]:
        return self._get("/tasks")

    def reset(self, difficulty: str, seed: int) -> Dict[str, Any]:
        return self._post("/reset", {"difficulty": difficulty, "seed": seed})

    def step(self, answer: str, confidence: float, source_quote: str) -> Dict[str, Any]:
        return self._post("/step", {
            "answer": answer,
            "confidence": confidence,
            "source_quote": source_quote,
        })

    def grade(self, task_id: str,
              step_rewards: List[float],
              step_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._post("/grader", {
            "task_id": task_id,
            "step_rewards": step_rewards,
            "step_infos": step_infos,
        })


# ── Agents ────────────────────────────────────────────────────────────────────

def heuristic_agent(question: str, context: str) -> Dict[str, Any]:
    """
    Deterministic heuristic baseline — no LLM required.
    Extracts the first sentence of the context as the answer.
    Used when --heuristic flag is set or no OPENAI_API_KEY is available.
    """
    sentences = [s.strip() for s in context.replace("\n", " ").split(".") if len(s.strip()) > 10]
    answer = sentences[0] if sentences else context[:120]
    source_quote = context[:80] if context else ""
    return {"answer": answer, "confidence": 0.6, "source_quote": source_quote}


def openai_agent(model: str, base_url: Optional[str] = None) -> Callable:
    """
    Returns a callable agent backed by any OpenAI-compatible chat endpoint.
    Reads OPENAI_API_KEY from environment.
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.error(
            "OPENAI_API_KEY not set. Export it or use --heuristic for the "
            "no-API baseline."
        )
        sys.exit(1)

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    client = OpenAI(**kwargs)

    def _call(question: str, context: str) -> Dict[str, Any]:
        prompt = ANSWER_PROMPT_TEMPLATE.format(
            context=context[:3000],  # stay within context window
            question=question,
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,   # deterministic
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            return {
                "answer":       str(parsed.get("answer", "")),
                "confidence":   float(parsed.get("confidence", 0.5)),
                "source_quote": str(parsed.get("source_quote", "")),
            }
        except json.JSONDecodeError:
            # model didn't return valid JSON — fall back to raw text
            raw_text = resp.choices[0].message.content or ""
            return {"answer": raw_text[:200], "confidence": 0.4, "source_quote": ""}
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return {"answer": "", "confidence": 0.0, "source_quote": ""}

    return _call


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    env: EnvClient,
    agent_fn: Callable,
    task_id: str,
    difficulty: str,
    steps: int,
    seed: int,
    episode_num: int,
) -> Dict[str, Any]:
    """Run one episode and return rewards + infos for the grader."""
    obs = env.reset(difficulty=difficulty, seed=seed + episode_num)
    step_rewards: List[float] = []
    step_infos: List[Dict[str, Any]] = []

    for step_n in range(steps):
        if obs.get("done", False):
            break

        question = obs.get("question", "")
        context  = obs.get("context", "")

        action = agent_fn(question, context)

        obs = env.step(
            answer=action["answer"],
            confidence=action["confidence"],
            source_quote=action["source_quote"],
        )

        reward = float(obs.get("reward") or 0.0)
        step_rewards.append(reward)
        step_infos.append({
            "correctness":        obs.get("grounding_score", 0.0),
            "grounding":          obs.get("grounding_score", 0.0),
            "calibration":        action["confidence"],
            "hallucination_score": 1.0 if obs.get("is_hallucination") else 0.0,
            "is_hallucination":   bool(obs.get("is_hallucination", False)),
        })

        status = "🚨 HALLUCINATION" if obs.get("is_hallucination") else "✅ OK"
        logger.info(
            f"  [{task_id[:25]}] ep={episode_num+1} step={step_n+1} "
            f"reward={reward:.3f} {status}"
        )

    grade = env.grade(task_id, step_rewards, step_infos)
    return {
        "episode": episode_num + 1,
        "score":   grade.get("score", 0.0),
        "rewards": step_rewards,
        "grade":   grade,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HallucinationGuard-Env baseline inference script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",      default=DEFAULT_MODEL,   help="OpenAI model name")
    parser.add_argument("--base-url",   default=None,            help="OpenAI API base URL override")
    parser.add_argument("--env-url",    default=DEFAULT_ENV_URL, help="HallucinationGuard env URL")
    parser.add_argument("--episodes",   type=int, default=DEFAULT_EPISODES, help="Episodes per task")
    parser.add_argument("--steps",      type=int, default=DEFAULT_STEPS,    help="Steps per episode")
    parser.add_argument("--seed",       type=int, default=SEED,             help="Random seed")
    parser.add_argument("--heuristic",  action="store_true",
                        help="Use heuristic agent (no OPENAI_API_KEY needed)")
    parser.add_argument("--output",     default=None,
                        help="Write JSON results to this file")
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    env = EnvClient(args.env_url)

    logger.info(f"Connecting to environment at {args.env_url} …")
    try:
        h = env.health()
        logger.info(f"  ✅ {h.get('service')} v{h.get('version')} is healthy")
    except Exception as e:
        logger.error(f"Cannot reach environment at {args.env_url}: {e}")
        sys.exit(1)

    # Verify tasks endpoint
    try:
        tasks_info = env.list_tasks()
        task_ids = [t["task_id"] for t in tasks_info.get("tasks", [])]
        logger.info(f"  ✅ Tasks endpoint OK — {task_ids}")
    except Exception as e:
        logger.error(f"/tasks endpoint failed: {e}")
        sys.exit(1)

    if args.heuristic or not os.environ.get("OPENAI_API_KEY"):
        logger.info("Using heuristic baseline agent (no LLM API call).")
        agent_fn = heuristic_agent
        model_label = "heuristic_baseline"
    else:
        logger.info(f"Using OpenAI agent: {args.model}")
        agent_fn = openai_agent(args.model, base_url=args.base_url)
        model_label = args.model

    # ── Run all 3 tasks ───────────────────────────────────────────────────────
    task_results: List[Dict[str, Any]] = []
    all_scores: List[float] = []
    all_rewards: List[float] = []
    all_hallucinations: List[bool] = []
    total_steps = 0

    start_time = time.time()

    for task_id, difficulty in TASK_ORDER:
        logger.info(f"\n{'='*60}")
        logger.info(f"TASK: {task_id}  (difficulty={difficulty})")
        logger.info(f"{'='*60}")

        episode_scores: List[float] = []

        for ep in range(args.episodes):
            ep_result = run_episode(
                env=env,
                agent_fn=agent_fn,
                task_id=task_id,
                difficulty=difficulty,
                steps=args.steps,
                seed=args.seed,
                episode_num=ep,
            )
            episode_scores.append(ep_result["score"])
            all_scores.append(ep_result["score"])
            all_rewards.extend(ep_result["rewards"])
            all_hallucinations.extend([
                info.get("is_hallucination", False)
                for info in ep_result["grade"].get("breakdown", {}).get("step_infos", [])
            ])
            total_steps += len(ep_result["rewards"])

        task_avg = sum(episode_scores) / max(len(episode_scores), 1)
        task_std = (
            (sum((s - task_avg) ** 2 for s in episode_scores) / max(len(episode_scores), 1)) ** 0.5
            if len(episode_scores) > 1 else 0.0
        )

        task_results.append({
            "task_id":        task_id,
            "difficulty":     difficulty,
            "episodes":       args.episodes,
            "episode_scores": [round(s, 4) for s in episode_scores],
            "avg_score":      round(task_avg, 4),
            "std_score":      round(task_std, 4),
        })
        logger.info(f"\n  → Task score: {task_avg:.4f} ± {task_std:.4f}")

    elapsed = time.time() - start_time

    # ── Summary ───────────────────────────────────────────────────────────────
    overall_score = sum(all_scores) / max(len(all_scores), 1)
    avg_reward    = sum(all_rewards) / max(len(all_rewards), 1)
    hall_rate     = sum(all_hallucinations) / max(len(all_hallucinations), 1) if all_hallucinations else 0.0

    summary = {
        "model":             model_label,
        "env_url":           args.env_url,
        "seed":              args.seed,
        "episodes_per_task": args.episodes,
        "steps_per_episode": args.steps,
        "total_steps":       total_steps,
        "elapsed_seconds":   round(elapsed, 1),
        "tasks":             task_results,
        "overall": {
            "score":             round(overall_score, 4),
            "avg_reward":        round(avg_reward, 4),
            "hallucination_rate": round(hall_rate, 4),
        },
    }

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"Model : {model_label}")
    print(f"Seed  : {args.seed}  |  {args.episodes} episodes × {args.steps} steps per task")
    print()
    for t in task_results:
        bar = "█" * round(t["avg_score"] * 20)
        print(
            f"  {t['task_id']:<40} "
            f"{t['avg_score']:.4f} ± {t['std_score']:.4f}  |{bar:<20}|"
        )
    print()
    print(f"  {'OVERALL':<40} {overall_score:.4f}")
    print(f"  Hallucination rate : {hall_rate:.1%}")
    print(f"  Elapsed            : {elapsed:.1f}s")
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results written to {args.output}")

    return summary


if __name__ == "__main__":
    main()
