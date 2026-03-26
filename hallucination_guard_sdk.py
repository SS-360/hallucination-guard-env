"""
HallucinationGuard SDK v3.0
===========================
The easiest way to evaluate any LLM for hallucination using HallucinationGuard-Env.

Install:
    pip install requests

Usage (3 lines):
    from hallucination_guard_sdk import HallucinationGuardEnv
    env = HallucinationGuardEnv()
    results = env.evaluate(your_model_fn, episodes=5)

Full example:
    import anthropic
    client = anthropic.Anthropic(api_key="...")

    def my_model(question, context):
        msg = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=256,
            messages=[{"role": "user", "content": f"Context: {context}\\n\\nQuestion: {question}\\n\\nAnswer using ONLY the context."}]
        )
        return msg.content[0].text

    env = HallucinationGuardEnv()
    results = env.evaluate(my_model, episodes=5, model_name="claude-3-haiku")
    env.print_report(results)
    env.submit_to_leaderboard(results)
"""

import time
import json
import sys
from typing import Callable, Optional, Dict, Any, List

try:
    import requests
except ImportError:
    print("Run: pip install requests")
    sys.exit(1)


class HallucinationGuardEnv:
    """
    Python SDK for HallucinationGuard-Env.

    Parameters
    ----------
    base_url : str
        URL of the deployed environment. Defaults to the live HF Space.
    verbose : bool
        Print step-by-step output during evaluation.
    """

    BASE_URL = "https://samsankar-hallucination-guard-env.hf.space"

    def __init__(
        self,
        base_url: str = BASE_URL,
        verbose: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.verbose  = verbose
        self._check_health()

    # ── Core methods ──────────────────────────────────────────────────────────

    def reset(self, difficulty: Optional[str] = None, seed: Optional[int] = None) -> Dict:
        """Reset the environment. Returns the first observation."""
        body = {}
        if difficulty: body["difficulty"] = difficulty
        if seed is not None: body["seed"] = seed
        return self._post("/reset", body)

    def step(self, answer: str) -> Dict:
        """Submit an answer. Returns reward, hallucination flag, feedback, next question."""
        return self._post("/step", {"answer": answer})

    def health(self) -> Dict:
        """Check if the environment is running."""
        return self._get("/health")

    def leaderboard(self) -> Dict:
        """Get the current leaderboard."""
        return self._get("/leaderboard")

    def dataset_info(self) -> Dict:
        """Get statistics about loaded datasets."""
        return self._get("/datasets")

    # ── High-level evaluate() ─────────────────────────────────────────────────

    def evaluate(
        self,
        model_fn: Callable[[str, str], str],
        episodes: int = 3,
        difficulty: Optional[str] = None,
        model_name: str = "my_model",
        delay: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Run a full evaluation of your model against the environment.

        Parameters
        ----------
        model_fn : callable
            Function that takes (question: str, context: str) → answer: str
        episodes : int
            Number of episodes to run (default: 3)
        difficulty : str, optional
            Force a difficulty level: beginner | intermediate | advanced | expert
        model_name : str
            Name for the leaderboard
        delay : float
            Seconds to wait between API calls (be gentle with free tier)

        Returns
        -------
        dict with summary stats and full episode logs

        Example
        -------
        >>> def my_model(question, context):
        ...     # call your LLM here
        ...     return "answer from context"
        >>> env = HallucinationGuardEnv()
        >>> results = env.evaluate(my_model, episodes=5)
        """
        if self.verbose:
            print(f"\n🛡️  HallucinationGuard-Env — Evaluating: {model_name}")
            print(f"   Episodes : {episodes}")
            print(f"   Difficulty: {difficulty or 'mixed'}")
            print(f"   Endpoint : {self.base_url}\n")

        all_episodes = []

        for ep_num in range(1, episodes + 1):
            if self.verbose:
                print(f"{'='*60}")
                print(f"  EPISODE {ep_num}/{episodes}")
                print(f"{'='*60}")

            ep_result = self._run_episode(model_fn, ep_num, difficulty, delay)
            all_episodes.append(ep_result)

            if self.verbose:
                print(f"  ─ Episode {ep_num} complete │ "
                      f"accuracy: {ep_result['accuracy']*100:.0f}% │ "
                      f"reward: {ep_result['avg_reward']:.3f} │ "
                      f"hallucinations: {ep_result['hallucinations']}/{ep_result['steps']}")

            time.sleep(delay)

        # ── Aggregate ─────────────────────────────────────────────────────────
        total_steps  = sum(e["steps"]          for e in all_episodes)
        total_halluc = sum(e["hallucinations"] for e in all_episodes)
        avg_accuracy = sum(e["accuracy"]       for e in all_episodes) / len(all_episodes)
        avg_reward   = sum(e["avg_reward"]     for e in all_episodes) / len(all_episodes)
        avg_skill    = sum(e["final_skill"]    for e in all_episodes) / len(all_episodes)
        best_streak  = max(e["best_streak"]    for e in all_episodes)
        halluc_rate  = total_halluc / max(total_steps, 1)

        results = {
            "model_name":        model_name,
            "episodes":          episodes,
            "total_steps":       total_steps,
            "avg_accuracy":      round(avg_accuracy, 4),
            "avg_reward":        round(avg_reward, 4),
            "hallucination_rate": round(halluc_rate, 4),
            "best_streak":       best_streak,
            "avg_skill_rating":  round(avg_skill, 4),
            "episode_logs":      all_episodes,
        }

        if self.verbose:
            self.print_report(results)

        return results

    def _run_episode(self, model_fn, ep_num, difficulty, delay) -> Dict:
        obs = self.reset(difficulty=difficulty)
        step_logs = []
        step = 0

        while not obs.get("done", False):
            question = obs.get("question", "")
            context  = obs.get("context", "")
            step    += 1

            if not question:
                break

            if self.verbose:
                q_display = question[:75] + "..." if len(question) > 75 else question
                print(f"\n  Step {step} [{obs.get('source_dataset','?')}]")
                print(f"  Q: {q_display}")

            # Call the model
            try:
                answer = model_fn(question, context)
            except Exception as e:
                answer = f"Error calling model: {e}"

            if self.verbose:
                a_display = answer[:90] + "..." if len(answer) > 90 else answer
                print(f"  A: {a_display}")

            obs = self.step(answer)

            reward    = obs.get("reward", 0) or 0
            is_halluc = obs.get("is_hallucination", False)
            status    = "❌ HALLUCINATION" if is_halluc else "✅ OK"

            if self.verbose:
                print(f"  {status} │ reward: {reward:.3f} │ skill: {obs.get('skill_rating', 0):.3f}")

            step_logs.append({
                "step":               step,
                "question":           question,
                "answer":             answer,
                "reward":             reward,
                "is_hallucination":   is_halluc,
                "hallucination_type": obs.get("hallucination_type"),
                "source":             obs.get("source_dataset", ""),
            })

            time.sleep(delay)

        accuracy    = obs.get("accuracy_so_far", 0)
        best_streak = obs.get("best_streak", 0)
        final_skill = obs.get("skill_rating", 0)
        avg_reward  = sum(s["reward"] for s in step_logs) / max(len(step_logs), 1)
        hallucinations = sum(1 for s in step_logs if s["is_hallucination"])

        return {
            "episode":       ep_num,
            "steps":         len(step_logs),
            "accuracy":      accuracy,
            "avg_reward":    avg_reward,
            "best_streak":   best_streak,
            "hallucinations": hallucinations,
            "final_skill":   final_skill,
            "step_logs":     step_logs,
        }

    # ── Reporting ──────────────────────────────────────────────────────────────

    def print_report(self, results: Dict) -> None:
        """Print a formatted evaluation report."""
        print(f"\n{'='*60}")
        print(f"  📊 EVALUATION REPORT — {results['model_name']}")
        print(f"{'='*60}")
        print(f"  Episodes run        : {results['episodes']}")
        print(f"  Total steps         : {results['total_steps']}")
        print(f"  Avg accuracy        : {results['avg_accuracy']*100:.1f}%")
        print(f"  Avg reward          : {results['avg_reward']:.4f}")
        print(f"  Hallucination rate  : {results['hallucination_rate']*100:.1f}%")
        print(f"  Best answer streak  : {results['best_streak']}")
        print(f"  Avg skill rating    : {results['avg_skill_rating']:.4f}")
        print(f"{'='*60}\n")

    def save_results(self, results: Dict, filepath: str = "evaluation_results.json") -> None:
        """Save evaluation results to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {filepath}")

    def submit_to_leaderboard(
        self,
        results: Dict,
        organization: str = "",
        notes: str = "",
    ) -> Dict:
        """
        Submit your evaluation results to the public leaderboard.

        Parameters
        ----------
        results : dict
            Output from evaluate()
        organization : str
            Your company/institution name
        notes : str
            Any notes about the evaluation setup
        """
        payload = {
            "model_name":        results["model_name"],
            "avg_reward":        results["avg_reward"],
            "avg_accuracy":      results["avg_accuracy"],
            "hallucination_rate": results["hallucination_rate"],
            "total_episodes":    results["episodes"],
            "total_steps":       results["total_steps"],
            "organization":      organization,
            "notes":             notes,
        }
        response = self._post("/leaderboard/submit", payload)
        if self.verbose:
            print(f"🏆 Submitted to leaderboard: {results['model_name']}")
            print(f"   View at: {self.base_url}/leaderboard")
        return response

    # ── HTTP helpers ───────────────────────────────────────────────────────────

    def _get(self, path: str) -> Dict:
        try:
            r = requests.get(f"{self.base_url}{path}", timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise ConnectionError(f"GET {path} failed: {e}")

    def _post(self, path: str, body: Dict = {}) -> Dict:
        try:
            r = requests.post(f"{self.base_url}{path}", json=body, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise ConnectionError(f"POST {path} failed: {e}")

    def _check_health(self) -> None:
        try:
            h = self._get("/health")
            if self.verbose:
                print(f"✅ Connected to HallucinationGuard-Env ({h.get('version','?')})")
        except Exception as e:
            print(f"⚠️  Could not reach {self.base_url}: {e}")


# ── CLI quick-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Quick smoke-test using a simple rule-based 'model'."""

    def dummy_model(question: str, context: str) -> str:
        """Answers only from context — extracts a key phrase."""
        words = context.split()
        if len(words) > 5:
            return " ".join(words[:10])
        return context

    env = HallucinationGuardEnv()
    results = env.evaluate(dummy_model, episodes=2, model_name="dummy-baseline")
    env.save_results(results, "dummy_results.json")
    env.submit_to_leaderboard(results, organization="Test Org", notes="Baseline run")
