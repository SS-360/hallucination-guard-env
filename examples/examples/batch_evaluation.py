"""
Batch Evaluation Script for HallucinationGuard-Env.

This script demonstrates how to run batch evaluations across multiple
tasks and difficulties, generating comprehensive benchmark reports.

Requirements:
    pip install requests matplotlib pandas
"""

import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests


class BatchEvaluator:
    """
    Run batch evaluations across tasks and difficulties.

    Features:
    - Multi-task evaluation (Factual Grounding, Multi-hop, Adversarial)
    - Multiple difficulty levels
    - Performance metrics and calibration analysis
    - JSON report generation
    """

    TASKS = [
        "task_1_factual_grounding",
        "task_2_multi_hop_synthesis",
        "task_3_adversarial_resistance"
    ]

    DIFFICULTIES = ["beginner", "intermediate", "advanced"]

    def __init__(self, env_base_url: str = "https://samsankar-hallucination-guard-env.hf.space"):
        """Initialize evaluator with environment URL."""
        self.env_base_url = env_base_url.rstrip('/')
        self.session = requests.Session()

    def get_tasks(self) -> List[Dict]:
        """Get available tasks from environment."""
        response = self.session.get(f"{self.env_base_url}/tasks")
        response.raise_for_status()
        return response.json().get("tasks", [])

    def evaluate_baseline(
        self,
        task_id: str,
        num_episodes: int = 3,
        difficulty: str = "intermediate"
    ) -> Dict[str, Any]:
        """
        Run baseline evaluation for a specific task.

        Uses a simple heuristic baseline:
        - Extract key entities from context
        - Match entities to question
        - Provide confidence based on match quality

        Args:
            task_id: Task identifier
            num_episodes: Number of episodes to run
            difficulty: Difficulty level

        Returns:
            Evaluation results
        """
        results = {
            "task_id": task_id,
            "difficulty": difficulty,
            "episodes": [],
            "summary": {}
        }

        all_rewards = []
        all_hallucinations = []
        all_correct = []

        for episode_num in range(num_episodes):
            # Reset environment
            reset_data = self._reset(task_id=task_id, difficulty=difficulty)

            episode_rewards = []
            episode_hallucinations = 0
            episode_correct = 0

            steps = 0
            max_steps = 10

            while steps < max_steps:
                # Get current observation
                question = reset_data.get("question", "")
                context = reset_data.get("context", "")

                # Generate baseline answer
                answer_data = self._generate_baseline_answer(question, context)

                # Step environment
                step_data = self._step(**answer_data)

                # Track metrics
                reward = step_data.get("reward", 0.0)
                episode_rewards.append(reward)

                if step_data.get("is_hallucination", False):
                    episode_hallucinations += 1

                if step_data.get("grounding_score", 0) > 0.7:
                    episode_correct += 1

                steps += 1

                if step_data.get("done", False):
                    break

                # Get next question
                reset_data = step_data

            # Episode statistics
            episode_avg_reward = sum(episode_rewards) / max(1, len(episode_rewards))
            all_rewards.append(episode_avg_reward)
            all_hallucinations.append(episode_hallucinations / max(1, steps))
            all_correct.append(episode_correct / max(1, steps))

            results["episodes"].append({
                "episode_num": episode_num + 1,
                "avg_reward": episode_avg_reward,
                "hallucination_rate": episode_hallucinations / max(1, steps),
                "accuracy": episode_correct / max(1, steps),
                "total_steps": steps
            })

            print(f"Episode {episode_num + 1}: Reward={episode_avg_reward:.3f}, "
                  f"Hallucinations={episode_hallucinations}/{steps}")

        # Aggregate results
        results["summary"] = {
            "avg_reward": sum(all_rewards) / len(all_rewards),
            "avg_hallucination_rate": sum(all_hallucinations) / len(all_hallucinations),
            "avg_accuracy": sum(all_correct) / len(all_correct),
            "total_episodes": num_episodes,
            "timestamp": datetime.now().isoformat()
        }

        return results

    def _reset(self, task_id: str = None, difficulty: str = "intermediate") -> dict:
        """Reset environment."""
        payload = {"difficulty": difficulty}
        if task_id:
            payload["task_id"] = task_id

        response = self.session.post(f"{self.env_base_url}/reset", json=payload)
        response.raise_for_status()
        return response.json()

    def _step(self, answer: str, confidence: float, source_quote: str = "") -> dict:
        """Submit step."""
        response = self.session.post(
            f"{self.env_base_url}/step",
            json={
                "answer": answer,
                "confidence": confidence,
                "source_quote": source_quote
            }
        )
        response.raise_for_status()
        return response.json()

    def _generate_baseline_answer(self, question: str, context: str) -> dict:
        """
        Generate a simple baseline answer.

        Strategy:
        1. Extract sentences from context
        2. Find sentence most similar to question
        3. Use that as answer with moderate confidence
        4. Use sentence as source quote
        """
        import re

        # Split context into sentences
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return {
                "answer": "I cannot find the answer in the provided context.",
                "confidence": 0.3,
                "source_quote": ""
            }

        # Find most relevant sentence (simple keyword matching)
        question_words = set(question.lower().split())

        best_sentence = sentences[0]
        best_overlap = 0

        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words & sentence_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence

        # Check if answer is likely in context
        if best_overlap < 2:
            return {
                "answer": "The answer does not appear to be in the provided context.",
                "confidence": 0.4,
                "source_quote": ""
            }

        # Extract key part of sentence as answer
        answer = best_sentence[:200] if len(best_sentence) > 200 else best_sentence

        return {
            "answer": answer,
            "confidence": 0.5 + (best_overlap / 20),  # Higher confidence with more overlap
            "source_quote": best_sentence[:150]
        }

    def run_full_evaluation(
        self,
        episodes_per_task: int = 3,
        difficulties: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run full evaluation across all tasks and difficulties.

        Args:
            episodes_per_task: Episodes per task configuration
            difficulties: List of difficulties to test

        Returns:
            Complete evaluation report
        """
        difficulties = difficulties or ["beginner", "intermediate", "advanced"]

        report = {
            "evaluation_date": datetime.now().isoformat(),
            "environment_url": self.env_base_url,
            "configuration": {
                "episodes_per_task": episodes_per_task,
                "difficulties": difficulties
            },
            "results": {}
        }

        print("Starting Full Evaluation")
        print("=" * 60)

        for task_id in self.TASKS:
            print(f"\nEvaluating: {task_id}")
            print("-" * 40)

            report["results"][task_id] = {}

            for difficulty in difficulties:
                print(f"  Difficulty: {difficulty}")

                task_results = self.evaluate_baseline(
                    task_id=task_id,
                    num_episodes=episodes_per_task,
                    difficulty=difficulty
                )

                report["results"][task_id][difficulty] = task_results

                # Brief pause between evaluations
                time.sleep(1)

        # Generate summary
        report["summary"] = self._generate_summary(report)

        return report

    def _generate_summary(self, report: dict) -> dict:
        """Generate cross-task summary."""
        summary = {
            "overall_avg_reward": 0.0,
            "overall_avg_hallucination_rate": 0.0,
            "overall_avg_accuracy": 0.0,
            "best_task": "",
            "best_difficulty": ""
        }

        all_rewards = []
        all_hallucinations = []
        all_accuracies = []
        task_performances = {}

        for task_id, difficulties in report.get("results", {}).items():
            task_rewards = []
            for difficulty, results in difficulties.items():
                task_summary = results.get("summary", {})
                all_rewards.append(task_summary.get("avg_reward", 0))
                all_hallucinations.append(task_summary.get("avg_hallucination_rate", 0))
                all_accuracies.append(task_summary.get("avg_accuracy", 0))
                task_rewards.append(task_summary.get("avg_reward", 0))

            task_performances[task_id] = sum(task_rewards) / len(task_rewards)

        if all_rewards:
            summary["overall_avg_reward"] = sum(all_rewards) / len(all_rewards)
        if all_hallucinations:
            summary["overall_avg_hallucination_rate"] = sum(all_hallucinations) / len(all_hallucinations)
        if all_accuracies:
            summary["overall_avg_accuracy"] = sum(all_accuracies) / len(all_accuracies)

        if task_performances:
            summary["best_task"] = max(task_performances, key=task_performances.get)

        return summary

    def save_report(self, report: dict, filename: str = None) -> str:
        """Save report to JSON file."""
        if filename is None:
            filename = f"hallucination_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Report saved to: {filename}")
        return filename


def main():
    """Run batch evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run batch hallucination evaluation")
    parser.add_argument("--env-url", default="https://samsankar-hallucination-guard-env.hf.space",
                        help="Environment server URL")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per task")
    parser.add_argument("--output", default=None, help="Output file name")

    args = parser.parse_args()

    evaluator = BatchEvaluator(env_base_url=args.env_url)

    # Run full evaluation
    report = evaluator.run_full_evaluation(
        episodes_per_task=args.episodes
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    summary = report.get("summary", {})
    print(f"Overall Average Reward: {summary.get('overall_avg_reward', 0):.3f}")
    print(f"Overall Hallucination Rate: {summary.get('overall_avg_hallucination_rate', 0):.1%}")
    print(f"Overall Accuracy: {summary.get('overall_avg_accuracy', 0):.1%}")
    print(f"Best Performing Task: {summary.get('best_task', 'N/A')}")

    # Save report
    evaluator.save_report(report, args.output)


if __name__ == "__main__":
    main()