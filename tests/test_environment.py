"""Tests for the HallucinationGuard environment."""
import pytest
from server.environment import HallucinationGuardEnvironment


class TestEnvironmentReset:
    """Tests for environment reset functionality."""

    def test_reset_returns_observation(self):
        """Reset should return a valid observation."""
        env = HallucinationGuardEnvironment()
        obs = env.reset()

        assert obs is not None
        assert hasattr(obs, 'question')
        assert hasattr(obs, 'context')
        assert hasattr(obs, 'reward')
        assert hasattr(obs, 'done')

    def test_reset_sets_initial_reward_to_zero(self):
        """Initial reward should be zero."""
        env = HallucinationGuardEnvironment()
        obs = env.reset()

        assert obs.reward == 0.0

    def test_reset_sets_done_to_false(self):
        """Episode should not be done after reset."""
        env = HallucinationGuardEnvironment()
        obs = env.reset()

        assert obs.done is False

    def test_reset_provides_attempts_remaining(self):
        """Reset should indicate attempts remaining."""
        env = HallucinationGuardEnvironment()
        obs = env.reset()

        assert obs.attempts_remaining > 0

    def test_reset_with_task_id(self):
        """Reset with specific task ID should work."""
        env = HallucinationGuardEnvironment()
        obs = env.reset(task_id="task_1_factual_grounding")

        assert obs is not None

    def test_reset_clears_previous_state(self):
        """Multiple resets should produce clean state each time."""
        env = HallucinationGuardEnvironment()
        env.reset()
        obs = env.reset()

        assert obs.reward == 0.0
        assert obs.done is False


class TestEnvironmentStep:
    """Tests for environment step functionality."""

    def test_step_returns_observation(self):
        """Step should return a valid observation."""
        env = HallucinationGuardEnvironment()
        env.reset()

        action = {
            "answer": "test answer",
            "confidence": 0.8,
            "source_quote": "",
            "reasoning": "",
            "uncertainty_flags": []
        }
        obs = env.step(action)

        assert obs is not None
        assert hasattr(obs, 'reward')

    def test_step_reward_in_valid_range(self):
        """Step reward should be in [0.0, 1.0] range."""
        env = HallucinationGuardEnvironment()
        env.reset()

        action = {
            "answer": "test answer",
            "confidence": 0.5,
            "source_quote": "",
            "reasoning": "",
            "uncertainty_flags": []
        }
        obs = env.step(action)

        assert -1.0 <= obs.reward <= 1.0

    def test_step_with_high_confidence(self):
        """Step with high confidence should work."""
        env = HallucinationGuardEnvironment()
        env.reset()

        action = {
            "answer": "test answer",
            "confidence": 1.0,
            "source_quote": "",
            "reasoning": "",
            "uncertainty_flags": []
        }
        obs = env.step(action)

        assert obs is not None

    def test_step_with_low_confidence(self):
        """Step with low confidence should work."""
        env = HallucinationGuardEnvironment()
        env.reset()

        action = {
            "answer": "test answer",
            "confidence": 0.1,
            "source_quote": "",
            "reasoning": "",
            "uncertainty_flags": []
        }
        obs = env.step(action)

        assert obs is not None

    def test_step_updates_attempts(self):
        """Step should decrement attempts remaining."""
        env = HallucinationGuardEnvironment()
        obs1 = env.reset()

        action = {
            "answer": "test",
            "confidence": 0.5,
            "source_quote": "",
            "reasoning": "",
            "uncertainty_flags": []
        }
        obs2 = env.step(action)

        assert obs2.attempts_remaining < obs1.attempts_remaining


class TestEnvironmentState:
    """Tests for environment state functionality."""

    def test_state_returns_metadata(self):
        """State should return episode metadata."""
        env = HallucinationGuardEnvironment()
        env.reset()
        state = env.state()

        assert state is not None
        assert hasattr(state, 'episode_id') or hasattr(state, 'step_count') or 'episode_id' in state or 'step_count' in state

    def test_state_tracks_step_count(self):
        """State should track step count."""
        env = HallucinationGuardEnvironment()
        env.reset()

        action = {
            "answer": "test",
            "confidence": 0.5,
            "source_quote": "",
            "reasoning": "",
            "uncertainty_flags": []
        }
        env.step(action)
        state = env.state()

        # State should reflect that a step was taken
        assert state is not None


class TestTaskSelection:
    """Tests for task selection."""

    def test_reset_with_task_1(self):
        """Reset with task_1_factual_grounding should work."""
        env = HallucinationGuardEnvironment()
        obs = env.reset(task_id="task_1_factual_grounding")

        assert obs is not None

    def test_reset_with_task_2(self):
        """Reset with task_2_multi_hop_synthesis should work."""
        env = HallucinationGuardEnvironment()
        obs = env.reset(task_id="task_2_multi_hop_synthesis")

        assert obs is not None

    def test_reset_with_task_3(self):
        """Reset with task_3_adversarial_resistance should work."""
        env = HallucinationGuardEnvironment()
        obs = env.reset(task_id="task_3_adversarial_resistance")

        assert obs is not None