"""Tests for the HallucinationGuard grader system.

Tests the calculate_reward function and related grading components.
"""
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.grader import (
    calculate_reward,
    detect_hallucination_advanced,
    check_factual_accuracy_advanced,
    check_quote_in_context_advanced,
    compute_semantic_consistency,
    HallucinationType,
    HallucinationSeverity,
)


class TestGraderScoreRange:
    """Tests that grader returns valid score ranges."""

    def test_grader_returns_score_in_range(self):
        """Grader should return score between 0.0 and 1.0."""
        reward, info = calculate_reward(
            answer="4",
            confidence=0.8,
            source_quote="2+2 equals 4",
            context="2+2 equals 4.",
            ground_truth="4",
            difficulty_level="beginner"
        )
        assert 0.0 <= reward <= 1.0
        assert 0.0 <= info["correctness"] <= 1.0

    def test_grader_with_exact_match(self):
        """Exact match should score high."""
        reward, info = calculate_reward(
            answer="Paris",
            confidence=0.9,
            source_quote="capital of France is Paris",
            context="The capital of France is Paris.",
            ground_truth="Paris",
            difficulty_level="beginner"
        )
        assert reward >= 0.6
        assert info["correctness"] >= 0.8

    def test_grader_with_wrong_answer(self):
        """Wrong answer should score low."""
        reward, info = calculate_reward(
            answer="London",
            confidence=0.8,
            source_quote="capital of France",
            context="The capital of France is Paris.",
            ground_truth="Paris",
            difficulty_level="beginner"
        )
        assert reward < 0.5
        assert info["is_hallucination"] == True

    def test_grader_with_partial_match(self):
        """Partial match should score moderately."""
        reward, info = calculate_reward(
            answer="Shakespeare",
            confidence=0.7,
            source_quote="written by William Shakespeare",
            context="Romeo and Juliet was written by William Shakespeare.",
            ground_truth="William Shakespeare",
            difficulty_level="intermediate"
        )
        assert 0.3 <= reward <= 0.9


class TestHallucinationDetection:
    """Tests for hallucination detection."""

    def test_detects_fabricated_fact(self):
        """Grader should detect fabricated facts."""
        halluc_score, h_type, severity, analysis = detect_hallucination_advanced(
            answer="Tokyo has 50 million people.",
            context="Tokyo is a major city in Japan.",
            ground_truth="Not mentioned",
            confidence=0.8
        )
        # Should detect hallucination due to fabricated numbers
        assert halluc_score > 0.0 or h_type != HallucinationType.NONE

    def test_detects_false_citation(self):
        """Grader should detect false citations."""
        score, analysis = check_quote_in_context_advanced(
            source_quote="The ocean is made of chocolate milk",
            context="The sky appears blue during clear days."
        )
        # Quote not found in context
        assert score < 0.5

    def test_overconfident_wrong_answer(self):
        """High confidence on wrong answer should be penalized."""
        reward_confident, _ = calculate_reward(
            answer="20",
            confidence=0.95,
            source_quote="",
            context="Basic arithmetic.",
            ground_truth="10",
            difficulty_level="beginner"
        )
        reward_uncertain, _ = calculate_reward(
            answer="20",
            confidence=0.3,
            source_quote="",
            context="Basic arithmetic.",
            ground_truth="10",
            difficulty_level="beginner"
        )
        # Confident wrong answer should score lower
        assert reward_confident < reward_uncertain


class TestSourceGrounding:
    """Tests for source grounding verification."""

    def test_answer_grounded_in_context(self):
        """Answer supported by context should score higher."""
        reward, info = calculate_reward(
            answer="Python is a programming language.",
            confidence=0.8,
            source_quote="Python is a programming language",
            context="Python is a programming language created by Guido van Rossum.",
            ground_truth="programming language",
            difficulty_level="beginner"
        )
        assert reward >= 0.4

    def test_answer_not_in_context(self):
        """Answer not supported by context should be penalized."""
        reward, info = calculate_reward(
            answer="Guido van Rossum created Python in 1991.",
            confidence=0.9,
            source_quote="Python is a programming language",
            context="Python is a programming language.",
            ground_truth="Not mentioned",
            difficulty_level="beginner"
        )
        # Fabricated details not in context
        assert reward < 0.7

    def test_citation_exact_match(self):
        """Exact citation match should score high."""
        score, analysis = check_quote_in_context_advanced(
            source_quote="The capital of France is Paris",
            context="The capital of France is Paris. It has a population of 2 million."
        )
        assert analysis["exact_match"] == True
        assert score >= 0.9


class TestConfidenceCalibration:
    """Tests for confidence calibration."""

    def test_confident_correct_answer(self):
        """High confidence on correct answer should be rewarded."""
        reward, info = calculate_reward(
            answer="2",
            confidence=0.95,
            source_quote="1+1 equals 2",
            context="1+1 equals 2.",
            ground_truth="2",
            difficulty_level="beginner"
        )
        assert reward >= 0.6
        assert info["calibration"] >= 0.8

    def test_uncertain_correct_answer(self):
        """Low confidence on correct answer should be slightly penalized."""
        reward_high, _ = calculate_reward(
            answer="2",
            confidence=0.95,
            source_quote="1+1 equals 2",
            context="1+1 equals 2.",
            ground_truth="2",
            difficulty_level="beginner"
        )
        reward_low, _ = calculate_reward(
            answer="2",
            confidence=0.3,
            source_quote="1+1 equals 2",
            context="1+1 equals 2.",
            ground_truth="2",
            difficulty_level="beginner"
        )
        # High confidence on correct answer should score higher or equal
        assert reward_high >= reward_low


class TestGraderDeterminism:
    """Tests for grader determinism."""

    def test_grader_is_deterministic(self):
        """Same inputs should produce same output."""
        args = {
            "answer": "Paris",
            "confidence": 0.8,
            "source_quote": "capital of France is Paris",
            "context": "The capital of France is Paris.",
            "ground_truth": "Paris",
            "difficulty_level": "beginner"
        }
        reward1, info1 = calculate_reward(**args)
        reward2, info2 = calculate_reward(**args)
        assert reward1 == reward2
        assert info1["correctness"] == info2["correctness"]

    def test_grader_handles_empty_answer(self):
        """Grader should handle empty answer gracefully."""
        reward, info = calculate_reward(
            answer="",
            confidence=0.5,
            source_quote="",
            context="The capital of France is Paris.",
            ground_truth="Paris",
            difficulty_level="beginner"
        )
        assert 0.0 <= reward <= 1.0

    def test_grader_handles_empty_context(self):
        """Grader should handle empty context gracefully."""
        reward, info = calculate_reward(
            answer="Paris",
            confidence=0.5,
            source_quote="",
            context="",
            ground_truth="Paris",
            difficulty_level="beginner"
        )
        assert 0.0 <= reward <= 1.0


class TestDifficultyMultipliers:
    """Tests for difficulty-based scoring."""

    def test_beginner_has_lower_multiplier(self):
        """Beginner difficulty should have 0.9 multiplier."""
        args = {
            "answer": "Paris",
            "confidence": 0.8,
            "source_quote": "capital of France is Paris",
            "context": "The capital of France is Paris.",
            "ground_truth": "Paris"
        }
        reward_beginner, _ = calculate_reward(**args, difficulty_level="beginner")
        reward_advanced, _ = calculate_reward(**args, difficulty_level="advanced")
        # Advanced should get higher multiplier (1.1 vs 0.9)
        assert reward_advanced >= reward_beginner

    def test_expert_has_highest_multiplier(self):
        """Expert difficulty should have highest multiplier."""
        args = {
            "answer": "Paris",
            "confidence": 0.8,
            "source_quote": "capital of France is Paris",
            "context": "The capital of France is Paris.",
            "ground_truth": "Paris"
        }
        reward_expert, _ = calculate_reward(**args, difficulty_level="expert")
        reward_beginner, _ = calculate_reward(**args, difficulty_level="beginner")
        assert reward_expert >= reward_beginner


class TestRefusalHandling:
    """Tests for 'I don't know' refusal handling."""

    def test_proper_refusal_on_unanswerable(self):
        """Proper refusal on unanswerable question should be rewarded."""
        reward, info = calculate_reward(
            answer="I cannot answer from the provided context.",
            confidence=0.3,
            source_quote="",
            context="The sky is blue.",
            ground_truth="not mentioned",
            difficulty_level="advanced"
        )
        assert reward >= 0.6
        assert info.get("is_refusal") == True

    def test_refusal_with_low_confidence(self):
        """Refusal with low confidence should score well."""
        reward, info = calculate_reward(
            answer="The answer is not in the context.",
            confidence=0.2,
            source_quote="",
            context="Some unrelated text.",
            ground_truth="unknown",
            difficulty_level="beginner"
        )
        assert reward >= 0.5

    def test_underconfident_refusal(self):
        """Refusal when answer exists should be penalized."""
        reward, info = calculate_reward(
            answer="I don't know",
            confidence=0.3,
            source_quote="",
            context="The capital of France is Paris.",
            ground_truth="Paris",  # Answer exists!
            difficulty_level="beginner"
        )
        assert reward < 0.5
        assert info.get("is_refusal") == True