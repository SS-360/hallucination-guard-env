"""Tests for the OpenEnv task registry and grader."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from server.tasks import (
    ALL_TASKS, TASK_1, TASK_2, TASK_3,
    get_task, task_id_for_difficulty,
    compute_task_score, ACTION_SCHEMA,
)


class TestTaskRegistry:
    """Task registry should contain all required tasks."""

    def test_three_tasks_exist(self):
        assert len(ALL_TASKS) == 3
        assert "task_1_factual_grounding" in ALL_TASKS
        assert "task_2_multi_hop_synthesis" in ALL_TASKS
        assert "task_3_adversarial_resistance" in ALL_TASKS

    def test_task_difficulties(self):
        assert TASK_1.difficulty == "beginner"
        assert TASK_2.difficulty == "intermediate"
        assert TASK_3.difficulty == "advanced"

    def test_get_task(self):
        assert get_task("task_1_factual_grounding") is TASK_1
        assert get_task("nonexistent") is None

    def test_difficulty_mapping(self):
        assert task_id_for_difficulty("beginner") == TASK_1.task_id
        assert task_id_for_difficulty("intermediate") == TASK_2.task_id
        assert task_id_for_difficulty("advanced") == TASK_3.task_id
        assert task_id_for_difficulty("expert") == TASK_3.task_id

    def test_action_schema_has_required_fields(self):
        props = ACTION_SCHEMA["properties"]
        assert "answer" in props
        assert "confidence" in props
        assert "source_quote" in props
        assert ACTION_SCHEMA["required"] == ["answer"]


class TestTaskGrader:
    """Per-episode task grader should produce scores in [0, 1]."""

    def test_empty_steps_score_zero(self):
        result = compute_task_score(TASK_1, [], [])
        assert result["score"] == 0.0

    def test_perfect_scores(self):
        step_rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
        step_infos = [
            {"correctness": 1.0, "grounding": 1.0, "calibration": 1.0,
             "hallucination_score": 0.0, "is_hallucination": False}
            for _ in range(5)
        ]
        result = compute_task_score(TASK_1, step_rewards, step_infos)
        assert 0.9 <= result["score"] <= 1.05, f"Perfect answers should score ~1.0, got {result['score']}"

    def test_zero_rewards_score_low(self):
        step_rewards = [0.0, 0.0, 0.0]
        step_infos = [
            {"correctness": 0.0, "grounding": 0.0, "calibration": 0.0,
             "hallucination_score": 1.0, "is_hallucination": True}
            for _ in range(3)
        ]
        result = compute_task_score(TASK_1, step_rewards, step_infos)
        assert result["score"] <= 0.1, f"All-wrong should score ~0, got {result['score']}"

    def test_task3_overconfidence_penalty(self):
        """Task 3 should penalize overconfident wrong answers."""
        step_rewards = [0.3, 0.3, 0.3]
        step_infos = [
            {"correctness": 0.2, "grounding": 0.3, "calibration": 0.9,
             "hallucination_score": 0.7, "is_hallucination": True}
            for _ in range(3)
        ]
        result_t3 = compute_task_score(TASK_3, step_rewards, step_infos)
        # Same data on task 1 should score higher than task 3
        result_t1 = compute_task_score(TASK_1, step_rewards, step_infos)
        assert result_t3["score"] <= result_t1["score"], \
            f"Task 3 should penalize overconfidence more than Task 1"

    def test_completion_bonus(self):
        """5+ steps should get a completion bonus."""
        short_infos = [{"correctness": 0.5, "grounding": 0.5, "calibration": 0.5,
                        "hallucination_score": 0.0, "is_hallucination": False}]
        long_infos = [{"correctness": 0.5, "grounding": 0.5, "calibration": 0.5,
                       "hallucination_score": 0.0, "is_hallucination": False}
                      for _ in range(6)]
        short_result = compute_task_score(TASK_1, [0.5], short_infos)
        long_result = compute_task_score(TASK_1, [0.5] * 6, long_infos)
        assert long_result["score"] > short_result["score"], \
            "Longer episodes should get completion bonus"

    def test_score_always_in_range(self):
        """Any combination of rewards/infos should produce score in [0, 1]."""
        import random
        random.seed(42)
        for _ in range(100):
            n = random.randint(1, 10)
            rewards = [random.uniform(0, 1) for _ in range(n)]
            infos = [{
                "correctness": random.uniform(0, 1),
                "grounding": random.uniform(0, 1),
                "calibration": random.uniform(0, 1),
                "hallucination_score": random.uniform(0, 1),
                "is_hallucination": random.random() > 0.5,
            } for _ in range(n)]
            for task in [TASK_1, TASK_2, TASK_3]:
                result = compute_task_score(task, rewards, infos)
                assert 0.0 <= result["score"] <= 1.0, \
                    f"Score {result['score']} out of range for {task.task_id}"