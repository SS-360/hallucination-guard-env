"""
Anthropic Claude SDK Integration Example for HallucinationGuard-Env.

This example demonstrates how to evaluate Claude models
(Claude 3.5 Sonnet, Claude 3 Opus) using the HallucinationGuard environment.

Requirements:
    pip install anthropic requests
"""

import os
from typing import Optional
import requests

# Anthropic SDK
try:
    from anthropic import Anthropic
except ImportError:
    print("Install Anthropic SDK: pip install anthropic")
    raise


class ClaudeHallucinationEvaluator:
    """
    Evaluate Claude models for hallucination resistance.

    Features:
    - Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
    - Uses structured prompts for consistent responses
    - Tracks calibration and grounding scores
    """

    def __init__(
        self,
        env_base_url: str = "https://samsankar-hallucination-guard-env.hf.space",
        anthropic_api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514"
    ):
        """
        Initialize evaluator.

        Args:
            env_base_url: HallucinationGuard-Env server URL
            anthropic_api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Claude model name
        """
        self.env_base_url = env_base_url.rstrip('/')
        self.model = model
        self.client = Anthropic(api_key=anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"))

        # Session tracking
        self.episode_id = None

    def reset_environment(self, difficulty: str = "intermediate") -> dict:
        """Start a new evaluation episode."""
        response = requests.post(
            f"{self.env_base_url}/reset",
            json={"difficulty": difficulty}
        )
        response.raise_for_status()
        data = response.json()
        self.episode_id = data.get("episode_id")
        return data

    def generate_answer(self, question: str, context: str) -> dict:
        """
        Generate an answer using Claude model.

        Claude is instructed to:
        1. Answer ONLY from the context
        2. Provide calibrated confidence
        3. Cite verbatim source quotes
        """
        prompt = f"""I need you to answer a question using ONLY the provided context.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the context below
2. If the answer is not in the context, respond: "I cannot determine the answer from the provided context."
3. Provide a confidence score (0.0 to 1.0) for your answer
4. Include a direct quote from the context that supports your answer

CONTEXT:
{context}

QUESTION:
{question}

Respond in this exact JSON format:
{{
    "answer": "your answer based solely on the context",
    "confidence": 0.XX,
    "source_quote": "exact verbatim quote from the context"
}}

Remember: Only use information from the context. Do not use outside knowledge."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            content = response.content[0].text

            # Parse JSON response
            import json
            import re

            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Fallback if no JSON found
                result = {
                    "answer": content.split('"answer"')[1].split('"')[1] if '"answer"' in content else content[:200],
                    "confidence": 0.5,
                    "source_quote": ""
                }

            return {
                "answer": result.get("answer", ""),
                "confidence": float(result.get("confidence", 0.5)),
                "source_quote": result.get("source_quote", "")
            }

        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                "answer": "I cannot determine the answer from the provided context.",
                "confidence": 0.3,
                "source_quote": ""
            }

    def step(self, answer: str, confidence: float, source_quote: str = "") -> dict:
        """Submit an answer to the environment."""
        response = requests.post(
            f"{self.env_base_url}/step",
            json={
                "answer": answer,
                "confidence": confidence,
                "source_quote": source_quote
            }
        )
        response.raise_for_status()
        return response.json()

    def evaluate_episode(
        self,
        num_questions: int = 10,
        difficulty: str = "intermediate",
        verbose: bool = True
    ) -> dict:
        """
        Run a complete evaluation episode.

        Args:
            num_questions: Number of questions to evaluate
            difficulty: Starting difficulty level
            verbose: Print progress

        Returns:
            Episode statistics
        """
        obs = self.reset_environment(difficulty=difficulty)

        total_reward = 0.0
        hallucinations = 0
        correct = 0
        calibration_errors = []

        for step_num in range(num_questions):
            question = obs.get("question", "")
            context = obs.get("context", "")

            if verbose:
                print(f"\n--- Question {step_num + 1}/{num_questions} ---")
                print(f"Q: {question[:100]}...")

            # Generate answer with Claude
            answer_data = self.generate_answer(question, context)

            if verbose:
                print(f"A: {answer_data['answer'][:100]}...")
                print(f"Confidence: {answer_data['confidence']:.2f}")

            # Submit to environment
            obs = self.step(
                answer=answer_data["answer"],
                confidence=answer_data["confidence"],
                source_quote=answer_data["source_quote"]
            )

            # Track statistics
            reward = obs.get("reward", 0.0)
            total_reward += reward

            if obs.get("is_hallucination", False):
                hallucinations += 1

            if obs.get("grounding_score", 0) > 0.7:
                correct += 1

            # Track calibration
            correctness = obs.get("metadata", {}).get("correctness", 0.5)
            calibration_error = abs(answer_data["confidence"] - correctness)
            calibration_errors.append(calibration_error)

            if verbose:
                print(f"Reward: {reward:.3f}")
                print(f"Hallucination: {obs.get('is_hallucination', False)}")

            if obs.get("done", False):
                break

        # Calculate statistics
        avg_reward = total_reward / max(1, step_num + 1)
        hallucination_rate = hallucinations / max(1, step_num + 1)
        accuracy = correct / max(1, step_num + 1)
        avg_calibration = sum(calibration_errors) / max(1, len(calibration_errors))

        results = {
            "model": self.model,
            "avg_reward": avg_reward,
            "hallucination_rate": hallucination_rate,
            "accuracy": accuracy,
            "avg_calibration_error": avg_calibration,
            "total_steps": step_num + 1,
            "difficulty": difficulty
        }

        if verbose:
            print(f"\n=== Episode Complete ===")
            print(f"Model: {self.model}")
            print(f"Average Reward: {avg_reward:.3f}")
            print(f"Hallucination Rate: {hallucination_rate:.1%}")
            print(f"Accuracy: {accuracy:.1%}")
            print(f"Avg Calibration Error: {avg_calibration:.3f}")

        return results


def compare_models(models: list, num_questions: int = 5, difficulty: str = "intermediate") -> dict:
    """
    Compare multiple Claude models.

    Args:
        models: List of model names to compare
        num_questions: Questions per model
        difficulty: Difficulty level

    Returns:
        Comparison results
    """
    results = {}

    for model in models:
        print(f"\n{'='*50}")
        print(f"Evaluating: {model}")
        print(f"{'='*50}")

        evaluator = ClaudeHallucinationEvaluator(model=model)
        model_results = evaluator.evaluate_episode(
            num_questions=num_questions,
            difficulty=difficulty
        )
        results[model] = model_results

    # Print comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Reward':>10} {'Halluc%':>12} {'Accuracy':>10}")
    print("-" * 60)
    for model, res in results.items():
        print(f"{model:<30} {res['avg_reward']:>10.3f} {res['hallucination_rate']*100:>11.1f}% {res['accuracy']*100:>9.1f}%")

    return results


def main():
    """Run evaluation demo."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Claude models for hallucination resistance")
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                        help="Claude model name")
    parser.add_argument("--difficulty", default="intermediate", help="Difficulty level")
    parser.add_argument("--num-questions", type=int, default=5, help="Number of questions")
    parser.add_argument("--env-url", default="https://samsankar-hallucination-guard-env.hf.space",
                        help="Environment server URL")
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple models")

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: Set ANTHROPIC_API_KEY environment variable")
        return

    if args.compare:
        # Compare multiple models
        models = [
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307"
        ]
        compare_models(models, num_questions=args.num_questions, difficulty=args.difficulty)
    else:
        # Single model evaluation
        evaluator = ClaudeHallucinationEvaluator(
            env_base_url=args.env_url,
            model=args.model
        )

        results = evaluator.evaluate_episode(
            num_questions=args.num_questions,
            difficulty=args.difficulty
        )

        print(f"\nFinal Results: {results}")


if __name__ == "__main__":
    main()