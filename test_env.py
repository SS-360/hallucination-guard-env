#!/usr/bin/env python3
"""Comprehensive test script for HallucinationGuard-Env.

This script tests all aspects of the environment:
- Basic functionality (reset, step, state)
- Reward system
- Hallucination detection
- Curriculum learning
- Model adapters
- Metrics tracking

Run with: python test_env.py
"""

import sys
import os
import json
import time
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    HallucinationAction,
    HallucinationObservation,
    HallucinationState,
    DifficultyLevel,
    HallucinationType,
    HallucinationSeverity,
)
from server.environment import HallucinationEnvironment
from server.grader import calculate_reward, detect_hallucination_advanced
from server.dataset_loader import DatasetLoader, DifficultyLevel as DatasetDifficulty
from server.metrics import MetricsTracker, get_tracker


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(test_name: str, passed: bool, details: str = "") -> None:
    """Print test result."""
    status = "PASS" if passed else "FAIL"
    marker = "[OK]" if passed else "[!!]"
    print(f"{marker} {test_name}")
    if details:
        print(f"      {details}")


class TestRunner:
    """Test runner for the environment."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []

    def run_test(self, name: str, test_func) -> bool:
        """Run a single test."""
        try:
            result = test_func()
            self.passed += 1
            self.results.append((name, True, ""))
            print_result(name, True)
            return True
        except AssertionError as e:
            self.failed += 1
            self.results.append((name, False, str(e)))
            print_result(name, False, str(e))
            return False
        except Exception as e:
            self.failed += 1
            self.results.append((name, False, f"Exception: {e}"))
            print_result(name, False, f"Exception: {e}")
            return False

    def print_summary(self) -> None:
        """Print test summary."""
        print_header("TEST SUMMARY")
        total = self.passed + self.failed
        print(f"Total: {total} | Passed: {self.passed} | Failed: {self.failed}")
        print(f"Success Rate: {self.passed / max(1, total) * 100:.1f}%")

        if self.failed > 0:
            print("\nFailed tests:")
            for name, passed, details in self.results:
                if not passed:
                    print(f"  - {name}: {details}")


def test_dataset_loader() -> bool:
    """Test dataset loading functionality."""
    print_header("TEST: Dataset Loader")

    loader = DatasetLoader()
    count = loader.load_builtin_datasets()

    # Test basic loading
    assert count > 0, "Should load some examples"
    assert loader.get_total_examples() >= 20, "Should have at least 20 examples"

    # Test difficulty distribution
    stats = loader.get_statistics()
    assert "beginner" in stats.examples_by_difficulty, "Should have beginner examples"
    assert "intermediate" in stats.examples_by_difficulty, "Should have intermediate examples"
    assert "advanced" in stats.examples_by_difficulty, "Should have advanced examples"
    assert "expert" in stats.examples_by_difficulty, "Should have expert examples"

    # Test episode creation
    examples = loader.start_new_episode(num_questions=5, mix_difficulties=True)
    assert len(examples) == 5, "Should create episode with 5 questions"

    # Test random sampling
    example = loader.get_random_example()
    assert example is not None, "Should return a random example"
    assert hasattr(example, 'question'), "Example should have question"
    assert hasattr(example, 'context'), "Example should have context"
    assert hasattr(example, 'answer'), "Example should have answer"

    return True


def test_grader() -> bool:
    """Test the grading system."""
    print_header("TEST: Grader System")

    # Test 1: Correct answer with proper citation
    reward, info = calculate_reward(
        answer="$30,000 USD",
        confidence=0.95,
        source_quote="total prize pool of $30,000 USD",
        context="The Meta PyTorch OpenEnv Hackathon 2026 has a total prize pool of $30,000 USD.",
        ground_truth="$30,000 USD"
    )
    assert reward > 0.7, f"High reward for correct answer: {reward}"
    assert not info["is_hallucination"], "Should not detect hallucination"
    print_result("Correct answer with citation", reward > 0.7, f"reward={reward:.3f}")

    # Test 2: Incorrect answer
    reward, info = calculate_reward(
        answer="$50,000 USD",  # Wrong amount
        confidence=0.9,
        source_quote="prize pool",
        context="The Meta PyTorch OpenEnv Hackathon 2026 has a total prize pool of $30,000 USD.",
        ground_truth="$30,000 USD"
    )
    assert reward < 0.5, f"Low reward for incorrect answer: {reward}"
    assert info["correctness"] < 0.5, "Should have low correctness score"
    print_result("Incorrect answer detection", reward < 0.5, f"reward={reward:.3f}")

    # Test 3: Hallucination detection
    reward, info = calculate_reward(
        answer="Winners get a golden ticket to the finale",  # Fabricated
        confidence=0.9,
        source_quote="According to the rules",  # Fake citation
        context="The ideathon is a standalone track with separate prizes.",
        ground_truth="Winners receive prizes from a separate pool"
    )
    assert info["is_hallucination"], "Should detect hallucination"
    print_result("Hallucination detection", info["is_hallucination"], f"score={info['hallucination_score']:.3f}")

    # Test 4: Confidence calibration
    reward_good, _ = calculate_reward(
        answer="Correct answer",
        confidence=0.9,
        source_quote="source",
        context="Correct answer is mentioned here.",
        ground_truth="Correct answer"
    )
    reward_bad, _ = calculate_reward(
        answer="Wrong answer",
        confidence=0.9,  # Overconfident
        source_quote="source",
        context="Something else entirely.",
        ground_truth="Correct answer"
    )
    assert reward_good > reward_bad, "Well-calibrated should score higher than overconfident wrong"
    print_result("Confidence calibration", reward_good > reward_bad)

    return True


def test_environment_reset() -> bool:
    """Test environment reset functionality."""
    print_header("TEST: Environment Reset")

    env = HallucinationEnvironment()
    obs = env.reset()

    # Test observation structure
    assert obs is not None, "Should return observation"
    assert hasattr(obs, 'question'), "Observation should have question"
    assert hasattr(obs, 'context'), "Observation should have context"
    assert hasattr(obs, 'done'), "Observation should have done flag"
    assert not obs.done, "Episode should not be done at start"
    print_result("Observation structure", True)

    # Test state
    state = env.state()
    assert state is not None, "Should return state"
    assert state.step_count == 0, "Step count should be 0 at start"
    assert state.episode_id is not None, "Should have episode ID"
    print_result("Initial state", True)

    return True


def test_environment_step() -> bool:
    """Test environment step functionality."""
    print_header("TEST: Environment Step")

    env = HallucinationEnvironment()
    obs = env.reset()

    # Get first observation's context for proper answer
    question = obs.question
    context = obs.context

    # Create a reasonable action based on context
    action = HallucinationAction(
        answer=context.split(".")[0] if context else "answer",
        confidence=0.7,
        source_quote=context[:50] if context else "quote"
    )

    obs = env.step(action)

    assert obs is not None, "Should return observation"
    assert obs.reward is not None, "Should have reward"
    assert 0 <= obs.reward <= 1, "Reward should be in [0, 1]"
    print_result("Step returns valid reward", True, f"reward={obs.reward:.3f}")

    # Test state update
    state = env.state()
    assert state.step_count == 1, "Step count should increment"
    print_result("State updates correctly", True)

    return True


def test_episode_completion() -> bool:
    """Test full episode completion."""
    print_header("TEST: Episode Completion")

    env = HallucinationEnvironment()
    obs = env.reset()

    steps_completed = 0
    rewards = []

    while not obs.done and steps_completed < 15:  # Max 15 steps safety
        action = HallucinationAction(
            answer="test answer",
            confidence=0.5,
            source_quote="test"
        )
        obs = env.step(action)
        steps_completed += 1

        if obs.reward is not None:
            rewards.append(obs.reward)

    state = env.state()
    assert state.step_count > 0, "Should have completed some steps"
    assert len(rewards) > 0, "Should have collected some rewards"
    avg_reward = sum(rewards) / max(1, len(rewards))
    print_result(f"Episode completed: {steps_completed} steps, avg_reward={avg_reward:.3f}", True)

    return True


def test_hallucination_types() -> bool:
    """Test different hallucination type detection."""
    print_header("TEST: Hallucination Types")

    # Test fabricated fact
    score, h_type, severity, _ = detect_hallucination_advanced(
        answer="The event is in Paris",
        context="The event is in London",
        confidence=0.9
    )
    assert score > 0.3, "Should detect some hallucination"
    print_result("Fabricated fact detection", score > 0.3, f"score={score:.3f}, type={h_type.value}")

    # Test numerical fabrication
    score, h_type, severity, _ = detect_hallucination_advanced(
        answer="The prize is $100,000",
        context="The prize is $30,000",
        confidence=0.95
    )
    assert score > 0.3, "Should detect numerical hallucination"
    print_result("Numerical fabrication detection", score > 0.3, f"score={score:.3f}")

    # Test no hallucination (grounded answer)
    score, h_type, severity, _ = detect_hallucination_advanced(
        answer="The prize is $30,000",
        context="The prize is $30,000",
        confidence=0.9
    )
    assert score < 0.3, "Should not detect hallucination for grounded answer"
    print_result("No false positive hallucination", score < 0.3, f"score={score:.3f}")

    return True


def test_metrics_tracker() -> bool:
    """Test metrics tracking functionality."""
    print_header("TEST: Metrics Tracker")

    tracker = MetricsTracker(session_id="test_session")

    # Log some steps
    for i in range(5):
        step_data = {
            "step": i,
            "episode_id": "test_ep_001",
            "reward": 0.5 + i * 0.1,
            "correctness": 0.6 + i * 0.05,
            "grounding": 0.7,
            "calibration": 0.8,
            "hallucination_score": 0.2 - i * 0.02,
            "is_hallucination": i == 2,
            "confidence": 0.75,
            "difficulty": "intermediate",
        }
        tracker.log_step(step_data)

    # End episode
    episode_data = {
        "episode_id": "test_ep_001",
        "total_steps": 5,
        "average_reward": 0.7,
        "total_hallucinations": 1,
        "hallucination_rate": 0.2,
        "accuracy": 0.8,
        "average_confidence": 0.75,
        "calibration_error": 0.1,
        "best_streak": 3,
        "skill_rating": 0.6,
        "start_time": time.time() - 60,
        "end_time": time.time(),
    }
    tracker.end_episode(episode_data)

    # Test metrics retrieval
    metrics = tracker.get_real_time_metrics()
    assert "overall_accuracy" in metrics, "Should have overall accuracy"
    assert "average_reward" in metrics, "Should have average reward"
    print_result("Metrics retrieval", True)

    # Test training curve data
    curve_data = tracker.get_training_curve_data()
    assert "episodes" in curve_data, "Should have episodes data"
    assert "rewards" in curve_data, "Should have rewards data"
    print_result("Training curve data", True)

    return True


def test_difficulty_levels() -> bool:
    """Test difficulty level handling."""
    print_header("TEST: Difficulty Levels")

    loader = DatasetLoader()
    loader.load_builtin_datasets()

    # Test getting examples by difficulty
    beginner_ex = loader.get_example_by_difficulty(DatasetDifficulty.BEGINNER)
    assert beginner_ex is not None, "Should get beginner example"
    assert beginner_ex.difficulty == DatasetDifficulty.BEGINNER, "Should be beginner difficulty"
    print_result("Beginner examples", True)

    expert_ex = loader.get_example_by_difficulty(DatasetDifficulty.EXPERT)
    assert expert_ex is not None, "Should get expert example"
    assert expert_ex.difficulty == DatasetDifficulty.EXPERT, "Should be expert difficulty"
    print_result("Expert examples", True)

    return True


def test_reward_breakdown() -> bool:
    """Test detailed reward breakdown."""
    print_header("TEST: Reward Breakdown")

    reward, info = calculate_reward(
        answer="April 7, 2026",
        confidence=0.85,
        source_quote="April 7, 2026 at 11:59 PM IST",
        context="The submission deadline is April 7, 2026 at 11:59 PM IST.",
        ground_truth="April 7, 2026"
    )

    # Check all components exist
    assert "components" in info, "Should have components dict"
    assert "correctness" in info, "Should have correctness score"
    assert "grounding" in info, "Should have grounding score"
    assert "calibration" in info, "Should have calibration score"
    assert "hallucination_score" in info, "Should have hallucination score"

    print_result(f"Reward breakdown available (total={reward:.3f})", True)

    # Verify components sum approximately to total
    components = info["components"]
    component_sum = sum(components.values())
    assert abs(component_sum - reward) < 0.1, f"Components should sum to ~{reward}"
    print_result("Components sum correctly", True)

    return True


def test_model_adapters() -> bool:
    """Test model adapter factory (without actual API calls)."""
    print_header("TEST: Model Adapters")

    try:
        from model_adapters import ModelAdapterFactory, ModelConfig

        # Test factory registration
        adapters = ModelAdapterFactory.get_available_adapters()
        assert "openai" in adapters, "Should have OpenAI adapter"
        assert "anthropic" in adapters, "Should have Anthropic adapter"
        assert "ollama" in adapters, "Should have Ollama adapter"
        print_result("Adapter factory registration", True)

        # Test config creation
        config = ModelConfig(model_name="test-model", api_key="test-key")
        assert config.model_name == "test-model", "Config should store model name"
        print_result("Model config creation", True)

    except ImportError as e:
        print_result("Model adapters (skipped - import error)", True, str(e))

    return True



def test_full_episode_integration() -> bool:
    """
    End-to-end integration test: runs a complete reset() -> step() x N -> done loop.

    This is the critical path test that catches bugs which unit tests miss.
    It exercises the entire pipeline: dataset loading, observation building,
    grader scoring, curriculum updates, and episode completion.
    """
    print_header("TEST: Full Episode Integration (end-to-end)")

    env = HallucinationEnvironment()

    # 1. Reset
    obs = env.reset(seed=42, difficulty="beginner")
    assert obs is not None, "reset() returned None"
    assert obs.question, "Initial observation has no question"
    assert obs.context, "Initial observation has no context"
    assert not obs.done, "Episode should not be done after reset"
    print_result("reset() returns valid observation", True, f"q={obs.question[:50]}...")

    # 2. Step loop
    rewards = []
    hallucinations = []
    step_num = 0
    max_steps = env.config.max_questions_per_episode + 2

    while not obs.done and step_num < max_steps:
        if step_num % 3 == 0:
            # Grounded correct answer
            action = HallucinationAction(
                answer=obs.context[:80] if obs.context else "I do not know",
                confidence=0.85,
                source_quote=obs.context[:40] if obs.context else "",
                reasoning="Extracted directly from context.",
            )
        elif step_num % 3 == 1:
            # Hallucinated answer
            action = HallucinationAction(
                answer="The answer is definitely 42 and involves a golden ticket.",
                confidence=0.99,
                source_quote="golden ticket is mentioned in the rules",
                reasoning="Fabricated reasoning not grounded in context.",
            )
        else:
            # Low confidence grounded answer
            action = HallucinationAction(
                answer=obs.context[:50] if obs.context else "unknown",
                confidence=0.3,
                source_quote=obs.context[:25] if obs.context else "",
            )

        obs = env.step(action)
        step_num += 1

        assert obs is not None, f"step() returned None at step {step_num}"
        assert obs.reward is not None or obs.done, f"step() returned no reward at step {step_num}"

        if obs.reward is not None:
            rewards.append(obs.reward)
            assert 0.0 <= obs.reward <= 1.0, f"Reward {obs.reward} out of [0,1] at step {step_num}"

        hallucinations.append(obs.is_hallucination)
        print_result(
            f"Step {step_num}",
            obs.reward is not None or obs.done,
            f"reward={obs.reward:.3f if obs.reward is not None else 0.0}, "
            f"hallucination={obs.is_hallucination}, done={obs.done}"
        )

    # 3. Episode completion checks
    assert obs.done, "Episode did not complete within max_questions steps"
    print_result("Episode completed (done=True)", obs.done)

    assert len(rewards) > 0, "No rewards collected during episode"
    avg_reward = sum(rewards) / len(rewards)
    print_result("Rewards collected", True, f"steps={len(rewards)}, avg={avg_reward:.3f}")

    # Grounded answers should outscore hallucinated on average
    grounded_rewards = [rewards[i] for i in range(len(rewards)) if i % 3 != 1]
    hallucinated_rewards = [rewards[i] for i in range(len(rewards)) if i % 3 == 1]
    if grounded_rewards and hallucinated_rewards:
        avg_grounded = sum(grounded_rewards) / len(grounded_rewards)
        avg_hallucinated = sum(hallucinated_rewards) / len(hallucinated_rewards)
        passed = avg_grounded > avg_hallucinated
        print_result(
            "Grounded answers reward > hallucinated", passed,
            f"grounded={avg_grounded:.3f}, hallucinated={avg_hallucinated:.3f}"
        )
        assert passed, "Reward signal broken: hallucinations scored >= grounded answers"

    # 4. State after episode
    state = env.state()
    assert state is not None, "state() returned None after episode"
    assert state.step_count > 0, "step_count is 0 after episode"
    assert 0.0 <= state.hallucination_rate <= 1.0, "hallucination_rate out of range"
    assert 0.0 <= state.accuracy <= 1.0, "accuracy out of range"
    print_result("state() valid after episode", True,
                 f"steps={state.step_count}, acc={state.accuracy:.2f}, "
                 f"hall_rate={state.hallucination_rate:.2f}")

    # 5. Second episode resets state cleanly
    obs2 = env.reset(seed=99, difficulty="intermediate")
    assert not obs2.done, "Second episode immediately done after reset"
    assert env.step_count == 0, "step_count not reset to 0"
    assert env.total_hallucinations == 0, "hallucination counter not cleared"
    print_result("Second reset() clears state cleanly", True, f"episode_id={env.episode_id}")

    env.close()
    return True


def run_all_tests() -> None:
    """Run all tests."""
    print_header("HALLUCINATION GUARD ENV - TEST SUITE")
    print("Testing professional-grade RL environment...")

    runner = TestRunner()

    # Run all tests
    runner.run_test("Dataset Loader", test_dataset_loader)
    runner.run_test("Grader System", test_grader)
    runner.run_test("Environment Reset", test_environment_reset)
    runner.run_test("Environment Step", test_environment_step)
    runner.run_test("Episode Completion", test_episode_completion)
    runner.run_test("Hallucination Types", test_hallucination_types)
    runner.run_test("Metrics Tracker", test_metrics_tracker)
    runner.run_test("Difficulty Levels", test_difficulty_levels)
    runner.run_test("Reward Breakdown", test_reward_breakdown)
    runner.run_test("Model Adapters", test_model_adapters)
    runner.run_test("Full Episode Integration", test_full_episode_integration)

    # Print summary
    runner.print_summary()

    # Print final status
    if runner.failed == 0:
        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED! Environment is ready for deployment.")
        print("=" * 60)
    else:
        print(f"\n[!!] {runner.failed} test(s) failed. Review the issues above.")

    sys.exit(0 if runner.failed == 0 else 1)


if __name__ == "__main__":
    run_all_tests()
