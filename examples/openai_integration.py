"""
OpenAI SDK Integration Example for HallucinationGuard-Env.

This example demonstrates how to evaluate OpenAI models
(GPT-4, GPT-4o, GPT-3.5) using the HallucinationGuard environment.

Requirements:
    pip install openai requests
"""

import os
from typing import Optional
import requests

# OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    print("Install OpenAI SDK: pip install openai")
    raise


class HallucinationGuardEvaluator:
    """
    Evaluate OpenAI models for hallucination resistance.

    Features:
    - Supports all OpenAI chat models
    - Handles rate limiting gracefully
    - Tracks calibration and grounding scores
    """

    def __init__(
        self,
        env_base_url: str = "https://samsankar-hallucination-guard-env.hf.space",
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize evaluator.

        Args:
            env_base_url: HallucinationGuard-Env server URL
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: OpenAI model name
        """
        self.env_base_url = env_base_url.rstrip('/')
        self.model = model
        self.client = OpenAI(api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"))

        # Session for environment
        self.session_id = None
        self.episode_id = None

    def reset_environment(self, difficulty: str = "intermediate") -> dict:
        """
        Start a new evaluation episode.

        Args:
            difficulty: Starting difficulty (beginner, intermediate, advanced)

        Returns:
            Initial observation with question and context
        """
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
        Generate an answer using OpenAI model.

        Prompts the model to:
        1. Answer ONLY from the provided context
        2. Provide a confidence score
        3. Cite the source quote

        Args:
            question: The question to answer
            context: The source context

        Returns:
            dict with answer, confidence, source_quote
        """
        prompt = f"""Answer the following question using ONLY the provided context.

IMPORTANT RULES:
1. Answer ONLY from the context - do not use outside knowledge
2. If the answer is not in the context, say "I cannot answer from the provided context"
3. Provide your confidence level (0.0-1.0)
4. Quote the exact passage from the context that supports your answer

CONTEXT:
{context}

QUESTION:
{question}

Respond in JSON format:
{{
    "answer": "your answer here",
    "confidence": 0.85,
    "source_quote": "exact quote from context"
}}

JSON Response:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise QA assistant. Always respond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for factual tasks
                max_tokens=500,
                response_format={"type": "json_object"}
            )

            import json
            content = response.choices[0].message.content
            result = json.loads(content)

            return {
                "answer": result.get("answer", ""),
                "confidence": float(result.get("confidence", 0.5)),
                "source_quote": result.get("source_quote", "")
            }

        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                "answer": "I cannot answer from the provided context.",
                "confidence": 0.3,
                "source_quote": ""
            }

    def step(self, answer: str, confidence: float, source_quote: str = "") -> dict:
        """
        Submit an answer to the environment.

        Args:
            answer: The answer text
            confidence: Confidence level (0.0-1.0)
            source_quote: Verbatim quote from context

        Returns:
            Observation with reward and feedback
        """
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
        difficulty: str = "intermediate"
    ) -> dict:
        """
        Run a complete evaluation episode.

        Args:
            num_questions: Number of questions to evaluate
            difficulty: Starting difficulty level

        Returns:
            Episode statistics
        """
        # Reset environment
        obs = self.reset_environment(difficulty=difficulty)

        total_reward = 0.0
        hallucinations = 0
        correct = 0

        for step_num in range(num_questions):
            # Get current question and context
            question = obs.get("question", "")
            context = obs.get("context", "")

            print(f"\n--- Question {step_num + 1}/{num_questions} ---")
            print(f"Q: {question[:100]}...")

            # Generate answer with OpenAI
            answer_data = self.generate_answer(question, context)
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

            print(f"Reward: {reward:.3f}")
            print(f"Feedback: {obs.get('feedback', '')[:100]}...")

            if obs.get("done", False):
                break

        # Calculate final statistics
        avg_reward = total_reward / max(1, step_num + 1)
        hallucination_rate = hallucinations / max(1, step_num + 1)
        accuracy = correct / max(1, step_num + 1)

        print(f"\n=== Episode Complete ===")
        print(f"Average Reward: {avg_reward:.3f}")
        print(f"Hallucination Rate: {hallucination_rate:.1%}")
        print(f"Accuracy: {accuracy:.1%}")

        return {
            "avg_reward": avg_reward,
            "hallucination_rate": hallucination_rate,
            "accuracy": accuracy,
            "total_steps": step_num + 1
        }


def main():
    """Run evaluation demo."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate OpenAI models for hallucination resistance")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--difficulty", default="intermediate", help="Difficulty level")
    parser.add_argument("--num-questions", type=int, default=5, help="Number of questions")
    parser.add_argument("--env-url", default="https://samsankar-hallucination-guard-env.hf.space",
                        help="Environment server URL")

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable")
        return

    # Run evaluation
    evaluator = HallucinationGuardEvaluator(
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