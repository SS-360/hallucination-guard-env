"""Tests for the HallucinationGrader scoring system."""
import pytest
from server.grader import HallucinationGrader


class TestGraderScoreRange:
    """Tests that grader returns valid score ranges."""

    def test_grader_returns_score_in_range(self):
        """Grader should return score between 0.0 and 1.0."""
        grader = HallucinationGrader()
        result = grader.grade(
            question="What is 2+2?",
            context="2+2 equals 4.",
            answer="4",
            ground_truth="4"
        )

        assert 0.0 <= result <= 1.0

    def test_grader_with_exact_match(self):
        """Exact match should score high."""
        grader = HallucinationGrader()
        result = grader.grade(
            question="What is the capital of France?",
            context="The capital of France is Paris.",
            answer="Paris",
            ground_truth="Paris"
        )

        assert result >= 0.7

    def test_grader_with_wrong_answer(self):
        """Wrong answer should score low."""
        grader = HallucinationGrader()
        result = grader.grade(
            question="What is the capital of France?",
            context="The capital of France is Paris.",
            answer="London",
            ground_truth="Paris"
        )

        assert result < 0.5

    def test_grader_with_partial_match(self):
        """Partial match should score moderately."""
        grader = HallucinationGrader()
        result = grader.grade(
            question="Who wrote Romeo and Juliet?",
            context="Romeo and Juliet was written by William Shakespeare.",
            answer="Shakespeare",
            ground_truth="William Shakespeare"
        )

        assert 0.3 <= result <= 0.9


class TestHallucinationDetection:
    """Tests for hallucination detection."""

    def test_detects_fabricated_fact(self):
        """Grader should detect fabricated facts."""
        grader = HallucinationGrader()
        result = grader.grade(
            question="What is the population of Tokyo?",
            context="Tokyo is a major city in Japan.",
            answer="Tokyo has 50 million people.",  # Not in context
            ground_truth="Not mentioned"
        )

        assert result < 0.5

    def test_detects_false_citation(self):
        """Grader should detect false citations."""
        grader = HallucinationGrader()
        result = grader.grade(
            question="What color is the sky?",
            context="The sky appears blue during clear days.",
            answer="The sky is green.",
            ground_truth="blue",
            source_quote="The sky appears green"  # Not in context
        )

        assert result < 0.5

    def test_overconfident_wrong_answer(self):
        """High confidence on wrong answer should be penalized."""
        grader = HallucinationGrader()
        result_confident = grader.grade(
            question="What is 5+5?",
            context="Basic arithmetic.",
            answer="20",  # Wrong
            ground_truth="10",
            confidence=0.95  # High confidence
        )
        result_uncertain = grader.grade(
            question="What is 5+5?",
            context="Basic arithmetic.",
            answer="20",  # Wrong
            ground_truth="10",
            confidence=0.3  # Low confidence
        )

        # Confident wrong answer should score lower
        assert result_confident < result_uncertain


class TestSourceGrounding:
    """Tests for source grounding verification."""

    def test_answer_grounded_in_context(self):
        """Answer supported by context should score higher."""
        grader = HallucinationGrader()
        result_grounded = grader.grade(
            question="What is Python?",
            context="Python is a programming language created by Guido van Rossum.",
            answer="Python is a programming language.",
            ground_truth="programming language"
        )

        assert result_grounded >= 0.5

    def test_answer_not_in_context(self):
        """Answer not supported by context should be penalized."""
        grader = HallucinationGrader()
        result = grader.grade(
            question="Who created Python?",
            context="Python is a programming language.",
            answer="Guido van Rossum created Python in 1991.",  # Details not in context
            ground_truth="Not mentioned"
        )

        assert result < 0.7


class TestConfidenceCalibration:
    """Tests for confidence calibration."""

    def test_confident_correct_answer(self):
        """High confidence on correct answer should be rewarded."""
        grader = HallucinationGrader()
        result = grader.grade(
            question="What is 1+1?",
            context="Basic math.",
            answer="2",
            ground_truth="2",
            confidence=0.95
        )

        assert result >= 0.7

    def test_uncertain_correct_answer(self):
        """Low confidence on correct answer should be slightly penalized."""
        grader = HallucinationGrader()
        result_high_conf = grader.grade(
            question="What is 1+1?",
            context="Basic math.",
            answer="2",
            ground_truth="2",
            confidence=0.95
        )
        result_low_conf = grader.grade(
            question="What is 1+1?",
            context="Basic math.",
            answer="2",
            ground_truth="2",
            confidence=0.3
        )

        # High confidence on correct answer should score higher
        assert result_high_conf >= result_low_conf


class TestGraderDeterminism:
    """Tests for grader determinism."""

    def test_grader_is_deterministic(self):
        """Same inputs should produce same output."""
        grader = HallucinationGrader()

        result1 = grader.grade(
            question="What is the capital of France?",
            context="The capital of France is Paris.",
            answer="Paris",
            ground_truth="Paris"
        )
        result2 = grader.grade(
            question="What is the capital of France?",
            context="The capital of France is Paris.",
            answer="Paris",
            ground_truth="Paris"
        )

        assert result1 == result2

    def test_grader_handles_empty_answer(self):
        """Grader should handle empty answer gracefully."""
        grader = HallucinationGrader()
        result = grader.grade(
            question="What is the capital of France?",
            context="The capital of France is Paris.",
            answer="",
            ground_truth="Paris"
        )

        assert 0.0 <= result <= 1.0

    def test_grader_handles_empty_context(self):
        """Grader should handle empty context gracefully."""
        grader = HallucinationGrader()
        result = grader.grade(
            question="What is the capital of France?",
            context="",
            answer="Paris",
            ground_truth="Paris"
        )

        assert 0.0 <= result <= 1.0