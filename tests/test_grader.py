"""Tests for the 9-component reward system and hallucination detection."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from server.grader import (
    calculate_reward,
    detect_hallucination_advanced,
    compute_calibration_error,
    is_refusal_answer,
    normalize_text,
    check_quote_in_context_advanced,
    check_factual_accuracy_advanced,
    compute_rouge,
    compute_bertscore,
    HallucinationType,
    HallucinationSeverity,
)


class TestRewardRange:
    """Rewards must always be in [0, 1]."""

    @pytest.mark.parametrize("difficulty", ["beginner", "intermediate", "advanced", "expert"])
    def test_reward_in_range_correct_answer(self, difficulty):
        reward, info = calculate_reward(
            answer="Paris is the capital of France.",
            confidence=0.9,
            source_quote="Paris is the capital of France.",
            context="Paris is the capital of France. It is located in northern France.",
            ground_truth="Paris",
            difficulty_level=difficulty,
        )
        assert 0.0 <= reward <= 1.0, f"Reward {reward} out of range for {difficulty}"

    def test_reward_in_range_wrong_answer(self):
        reward, info = calculate_reward(
            answer="London is the capital of France.",
            confidence=0.9,
            source_quote="London is the capital of France.",
            context="Paris is the capital of France.",
            ground_truth="Paris",
        )
        assert 0.0 <= reward <= 1.0

    def test_reward_in_range_empty_answer(self):
        reward, info = calculate_reward(
            answer="",
            confidence=0.5,
            source_quote="",
            context="Some context here.",
            ground_truth="Some answer",
        )
        assert 0.0 <= reward <= 1.0

    def test_reward_in_range_refusal(self):
        reward, info = calculate_reward(
            answer="I cannot answer from the provided context.",
            confidence=0.3,
            source_quote="",
            context="Some unrelated context.",
            ground_truth="not mentioned in context",
        )
        assert 0.0 <= reward <= 1.0


class TestRefusalHandling:
    """Proper refusals on unanswerable questions should be rewarded."""

    def test_proper_refusal_rewarded(self):
        reward, info = calculate_reward(
            answer="I cannot answer from the provided context.",
            confidence=0.3,
            source_quote="",
            context="The sky is blue.",
            ground_truth="not mentioned in context",
        )
        assert reward >= 0.5, f"Proper refusal should get reward >= 0.5, got {reward}"
        assert info.get("is_refusal") is True

    def test_underconfident_refusal_penalized(self):
        """Refusing when the answer IS in context should be penalized."""
        reward, info = calculate_reward(
            answer="I cannot determine the answer from the context.",
            confidence=0.3,
            source_quote="",
            context="The capital of France is Paris.",
            ground_truth="Paris",
        )
        assert reward <= 0.4, f"Underconfident refusal should be penalized, got {reward}"

    def test_overconfident_refusal(self):
        """High confidence refusal on answerable question should be penalized."""
        reward, info = calculate_reward(
            answer="I don't know the answer.",
            confidence=0.9,
            source_quote="",
            context="The capital of France is Paris.",
            ground_truth="Paris",
        )
        assert reward <= 0.5


class TestHallucinationDetection:
    """Hallucination detection should classify types correctly."""

    def test_no_hallucination_for_grounded_answer(self):
        score, htype, severity, analysis = detect_hallucination_advanced(
            answer="Paris is the capital of France.",
            context="Paris is the capital of France.",
            ground_truth="Paris",
            confidence=0.9,
        )
        assert score < 0.3, f"Grounded answer should have low hallucination score, got {score}"

    def test_fabricated_fact_detected(self):
        score, htype, severity, analysis = detect_hallucination_advanced(
            answer="Berlin is the capital of France.",
            context="Paris is the capital of France.",
            ground_truth="Paris",
            confidence=0.9,
        )
        assert score > 0.3, f"Fabricated fact should have high hallucination score, got {score}"

    def test_numerical_fabrication_detected(self):
        score, htype, severity, analysis = detect_hallucination_advanced(
            answer="The population is 8.7 million.",
            context="The population is 2.1 million people.",
            ground_truth="2.1 million",
            confidence=0.8,
        )
        assert analysis.get("numerical_fabrication", 0) > 0, \
            f"Fabricated number 8.7 should be detected, got {analysis}"


class TestCitationAccuracy:
    """Source quote verification should work correctly."""

    def test_exact_quote_match(self):
        score, analysis = check_quote_in_context_advanced(
            "Paris is the capital of France.",
            "Paris is the capital of France. It is a beautiful city.",
        )
        assert score == 1.0, f"Exact quote should score 1.0, got {score}"

    def test_no_quote(self):
        score, analysis = check_quote_in_context_advanced(
            "",
            "Some context here.",
        )
        assert score == 0.0

    def test_partial_quote(self):
        score, analysis = check_quote_in_context_advanced(
            "capital of France",
            "Paris is the capital of France.",
        )
        assert score > 0.5, f"Partial quote should score > 0.5, got {score}"


class TestCalibrationError:
    """Calibration error should penalize overconfidence."""

    def test_perfect_calibration(self):
        error = compute_calibration_error(0.9, 0.9)
        assert error == 0.0

    def test_overconfidence_penalized(self):
        error = compute_calibration_error(0.95, 0.3)
        assert error > 0.5, f"Overconfidence should be heavily penalized, got {error}"

    def test_underconfidence_safe(self):
        error = compute_calibration_error(0.3, 0.9)
        assert error < compute_calibration_error(0.95, 0.3), \
            "Overconfidence should be penalized more than underconfidence"


class TestBERTScoreEdgeCases:
    """BERTScore should not crash on edge cases."""

    def test_empty_strings(self):
        result = compute_bertscore("", "")
        assert result["f1"] == 0.0

    def test_identical_strings(self):
        result = compute_bertscore("The cat sat on the mat.", "The cat sat on the mat.")
        assert result["f1"] > 0.8, f"Identical strings should have high BERTScore, got {result['f1']}"

    def test_short_strings(self):
        result = compute_bertscore("yes", "no")
        assert "f1" in result  # Should not crash


class TestROUGE:
    """ROUGE scores should be computed correctly."""

    def test_identical_strings(self):
        result = compute_rouge("The cat sat on the mat.", "The cat sat on the mat.")
        assert result["rougeL"] == 1.0

    def test_completely_different(self):
        result = compute_rouge("The cat sat on the mat.", "Dogs run in the park.")
        assert result["rougeL"] < 0.5

    def test_empty_strings(self):
        result = compute_rouge("", "")
        assert result["rouge1"] == 0.0


class TestFactualAccuracy:
    """Factual accuracy should handle various answer types."""

    def test_exact_match(self):
        score, analysis = check_factual_accuracy_advanced(
            "Paris", "Paris", "Paris is the capital of France."
        )
        assert score >= 0.9, f"Exact match should score high, got {score}"

    def test_wrong_answer(self):
        score, analysis = check_factual_accuracy_advanced(
            "London", "Paris", "Paris is the capital of France."
        )
        assert score < 0.5, f"Wrong answer should score low, got {score}"

    def test_contains_truth(self):
        score, analysis = check_factual_accuracy_advanced(
            "The capital is Paris, which is in northern France.",
            "Paris",
            "Paris is the capital of France.",
        )
        assert score >= 0.8, f"Answer containing truth should score high, got {score}"


class TestNormalizeText:
    """Text normalization should handle edge cases."""

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_whitespace_normalization(self):
        result = normalize_text("  The   cat   sat  ")
        assert "  " not in result

    def test_case_normalization(self):
        result = normalize_text("PARIS IS THE CAPITAL")
        assert result == result.lower()