"""Research-grade hallucination detection and grading system v4.0.

Upgrades in v4.0:
- NLI model upgraded: nli-deberta-v3-small -> nli-deberta-v3-large
- ROUGE-1/2/L added (Lin 2004)
- BERTScore added via DeBERTa-v3-base (Zhang et al. 2020)
- AlignScore faithfulness added via RoBERTa (Zha et al. ACL 2023)
- Reward expanded from 6 to 9 components
"""

import re
import math
import logging
from typing import Tuple, Optional, Dict, Any, List, Set
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

# ── Embedding model (sentence-transformers) ───────────────────────────────────
_embedder = None
_embedder_available = False

def _get_embedder():
    global _embedder, _embedder_available
    if _embedder is not None:
        return _embedder
    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        _embedder_available = True
        logger.info("sentence-transformers loaded: all-MiniLM-L6-v2")
    except Exception as e:
        logger.warning(f"sentence-transformers not available ({e}); using SequenceMatcher fallback")
        _embedder = None
        _embedder_available = False
    return _embedder


def _cosine_similarity(a, b) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ── NLI cross-encoder ─────────────────────────────────────────────────────────
_nli_model = None
_nli_available = False

def _get_nli():
    global _nli_model, _nli_available
    if _nli_model is not None:
        return _nli_model
    try:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-large")
        _nli_available = True
        logger.info("NLI cross-encoder loaded: nli-deberta-v3-large")
    except Exception as e:
        logger.warning(f"NLI cross-encoder not available ({e}); using heuristic fallback")
        _nli_model = None
        _nli_available = False
    return _nli_model



# ── ROUGE scorer ──────────────────────────────────────────────────────────────
_rouge_scorer = None

def _get_rouge():
    global _rouge_scorer
    if _rouge_scorer is not None:
        return _rouge_scorer
    try:
        from rouge_score import rouge_scorer as rs
        _rouge_scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        logger.info("ROUGE scorer loaded")
    except Exception as e:
        logger.warning(f"rouge-score not available ({e})")
        _rouge_scorer = None
    return _rouge_scorer


def compute_rouge(hypothesis: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    scorer = _get_rouge()
    if scorer is None or not hypothesis or not reference:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    try:
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }
    except Exception as e:
        logger.warning(f"ROUGE computation failed: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


# ── BERTScore ─────────────────────────────────────────────────────────────────
_bertscore_available = None

def _check_bertscore():
    global _bertscore_available
    if _bertscore_available is not None:
        return _bertscore_available
    try:
        import bert_score  # noqa: F401
        _bertscore_available = True
        logger.info("BERTScore available")
    except Exception:
        _bertscore_available = False
    return _bertscore_available


def compute_bertscore(hypothesis: str, reference: str) -> Dict[str, float]:
    """Compute BERTScore P/R/F1 using DeBERTa-v3-base."""
    if not _check_bertscore():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not hypothesis or not reference:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    try:
        from bert_score import score as bscore
        P, R, F = bscore(
            [hypothesis], [reference],
            model_type="microsoft/deberta-v3-base",
            lang="en", verbose=False, device="cpu"
        )
        return {
            "precision": round(float(P[0]), 4),
            "recall":    round(float(R[0]), 4),
            "f1":        round(float(F[0]), 4),
        }
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


# ── AlignScore ────────────────────────────────────────────────────────────────
_alignscore_model = None
_alignscore_checked = False

def _get_alignscore():
    global _alignscore_model, _alignscore_checked
    if _alignscore_checked:
        return _alignscore_model
    _alignscore_checked = True
    try:
        from alignscore import AlignScore
        _alignscore_model = AlignScore(
            model="roberta-base", batch_size=8,
            device="cpu", ckpt_path=None,
            evaluation_mode="nli_sp"
        )
        logger.info("AlignScore loaded (roberta-base, nli_sp)")
    except Exception as e:
        logger.warning(f"AlignScore not available ({e}); using neutral fallback 0.5")
        _alignscore_model = None
    return _alignscore_model


def compute_alignscore(context: str, claim: str) -> float:
    """Faithfulness of claim w.r.t. context. Returns [0,1]; 0.5 = unavailable."""
    model = _get_alignscore()
    if model is None or not context or not claim:
        return 0.5
    try:
        scores = model.score(contexts=[context], claims=[claim])
        return round(float(scores[0]), 4)
    except Exception as e:
        logger.warning(f"AlignScore inference failed: {e}")
        return 0.5


class HallucinationSeverity(Enum):
    """Severity levels for hallucinations."""
    NONE = 0
    MINOR = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4


class HallucinationType(Enum):
    """Types of hallucinations."""
    NONE = "none"
    FABRICATED_FACT = "fabricated_fact"
    FALSE_CITATION = "false_citation"
    OVERCONFIDENT_WRONG = "overconfident_wrong"
    CONTEXT_DRIFT = "context_drift"
    NUMERICAL_FABRICATION = "numerical_fabrication"
    ENTITY_CONFUSION = "entity_confusion"
    TEMPORAL_ERROR = "temporal_error"
    RELATIONSHIP_ERROR = "relationship_error"


@dataclass
class GradingResult:
    """Comprehensive grading result with all metrics."""
    # Core scores (0-1)
    factual_correctness: float = 0.0
    source_grounding: float = 0.0
    citation_accuracy: float = 0.0
    confidence_calibration: float = 0.0
    semantic_consistency: float = 0.0
    hallucination_score: float = 0.0

    # Combined reward
    total_reward: float = 0.0

    # Classification
    is_hallucination: bool = False
    hallucination_type: HallucinationType = HallucinationType.NONE
    hallucination_severity: HallucinationSeverity = HallucinationSeverity.NONE

    # Detailed analysis
    semantic_analysis: Dict[str, Any] = field(default_factory=dict)
    citation_analysis: Dict[str, Any] = field(default_factory=dict)
    entity_analysis: Dict[str, Any] = field(default_factory=dict)

    # Feedback
    feedback: str = ""
    detailed_explanation: str = ""

    # Metadata
    difficulty_adjustment: float = 0.0
    bonus_points: float = 0.0


def normalize_text(text: str, preserve_numbers: bool = False) -> str:
    """Normalize text for comparison with advanced preprocessing."""
    if not text:
        return ""

    # Lowercase and strip
    text = text.lower().strip()

    # Remove extra whitespace
    text = " ".join(text.split())

    # Remove common punctuation for comparison
    text = re.sub(r'[^\w\s\.]', ' ', text)

    # Normalize numbers only if not preserving (for factual comparison)
    if not preserve_numbers:
        text = re.sub(r'\b\d+\.\d+\b', 'NUM', text)
        text = re.sub(r'\b\d+\b', 'NUM', text)

    return text.strip()


def extract_entities(text: str) -> Set[str]:
    """Extract potential entities (capitalized words, numbers, dates) from text."""
    entities = set()

    # Extract capitalized words (potential proper nouns)
    capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
    entities.update(capitalized)

    # Extract numbers with units
    numbers = re.findall(r'\b\d+(?:\.\d+)?\s*(?:%|USD|dollars|people|times)?\b', text, re.IGNORECASE)
    entities.update(numbers)

    # Extract dates
    dates = re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d+\b', text, re.IGNORECASE)
    entities.update(dates)

    return entities


def extract_key_claims(text: str) -> List[str]:
    """Extract key claims/statements from text for detailed analysis."""
    if not text:
        return []

    # Split into sentences
    sentences = re.split(r'[.!?]+', text)

    # Filter out very short sentences and stop-word-only sentences
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'it', 'this', 'that'}
    claims = []

    for sentence in sentences:
        sentence = sentence.strip()
        words = sentence.split()
        if len(words) >= 3 and len([w for w in words if w.lower() not in stop_words]) >= 2:
            claims.append(sentence)

    return claims


def compute_string_similarity(s1: str, s2: str) -> float:
    """Compute semantic similarity between two strings.

    Uses sentence-transformers (all-MiniLM-L6-v2) cosine similarity when
    available, with SequenceMatcher + Jaccard as a graceful fallback.
    """
    if not s1 or not s2:
        return 0.0

    s1_norm = normalize_text(s1)
    s2_norm = normalize_text(s2)

    if s1_norm == s2_norm:
        return 1.0
    if s1_norm in s2_norm or s2_norm in s1_norm:
        return 0.9

    # ── Try embedding-based cosine similarity first ───────────────────────
    embedder = _get_embedder()
    if embedder is not None:
        try:
            vecs = embedder.encode([s1, s2], convert_to_numpy=True, show_progress_bar=False)
            score = _cosine_similarity(vecs[0], vecs[1])
            # Clip to [0,1] — cosine can be slightly negative for unrelated text
            return max(0.0, min(1.0, float(score)))
        except Exception as e:
            logger.warning(f"Embedding similarity failed: {e}; using fallback")

    # ── Fallback: SequenceMatcher + Jaccard ──────────────────────────────
    seq_sim = SequenceMatcher(None, s1_norm, s2_norm).ratio()
    s1_words = set(s1_norm.split())
    s2_words = set(s2_norm.split())
    if not s1_words or not s2_words:
        return seq_sim
    jaccard = len(s1_words & s2_words) / len(s1_words | s2_words)
    return max(seq_sim, jaccard)


def check_quote_in_context_advanced(source_quote: str, context: str) -> Tuple[float, Dict[str, Any]]:
    """
    Advanced citation verification with detailed analysis.

    Returns:
        Tuple of (score, analysis_dict)
    """
    analysis = {
        "exact_match": False,
        "partial_matches": [],
        "best_match_score": 0.0,
        "match_location": None,
        "surrounding_context": "",
        "quote_length": len(source_quote),
        "context_length": len(context)
    }

    if not source_quote or not context:
        return 0.0, analysis

    normalized_quote = normalize_text(source_quote)
    normalized_context = normalize_text(context)

    # Exact match check
    if normalized_quote in normalized_context:
        analysis["exact_match"] = True
        analysis["best_match_score"] = 1.0
        analysis["match_location"] = normalized_context.find(normalized_quote)

        # Get surrounding context
        start = max(0, analysis["match_location"] - 50)
        end = min(len(context), analysis["match_location"] + len(source_quote) + 50)
        analysis["surrounding_context"] = context[start:end]

        return 1.0, analysis

    # Sliding window for fuzzy matching
    quote_words = normalized_quote.split()
    context_words = normalized_context.split()

    if len(quote_words) == 0:
        return 0.0, analysis

    best_match_score = 0.0
    best_match_window = None
    best_match_idx = 0

    window_size = len(quote_words)
    for i in range(len(context_words) - window_size + 1):
        window = context_words[i:i + window_size]
        window_text = " ".join(window)
        similarity = SequenceMatcher(None, normalized_quote, window_text).ratio()

        if similarity > best_match_score:
            best_match_score = similarity
            best_match_window = window_text
            best_match_idx = i

    if best_match_score > 0.7:
        analysis["partial_matches"].append({
            "text": best_match_window,
            "score": best_match_score,
            "position": best_match_idx
        })
        analysis["best_match_score"] = best_match_score

        # Get surrounding context
        char_pos = sum(len(w) + 1 for w in context_words[:best_match_idx])
        start = max(0, char_pos - 50)
        end = min(len(context), char_pos + len(best_match_window) + 50)
        analysis["surrounding_context"] = context[start:end]

        return best_match_score, analysis

    # Try matching key phrases (relaxed matching)
    quote_key_phrases = [p for p in normalized_quote.split() if len(p) > 3]
    context_set = set(normalized_context.split())

    if quote_key_phrases:
        phrase_match_ratio = sum(1 for p in quote_key_phrases if p in context_set) / len(quote_key_phrases)
        if phrase_match_ratio > 0.5:
            analysis["partial_matches"].append({
                "type": "key_phrase_match",
                "ratio": phrase_match_ratio
            })
            return 0.5 + 0.3 * phrase_match_ratio, analysis

    return 0.0, analysis


def check_factual_accuracy_advanced(answer: str, ground_truth: str, context: str = "") -> Tuple[float, Dict[str, Any]]:
    """
    Advanced factual accuracy checking with semantic understanding.

    Returns:
        Tuple of (score, analysis_dict)
    """
    analysis = {
        "exact_match": False,
        "semantic_similarity": 0.0,
        "entity_overlap": 0.0,
        "key_info_present": False,
        "contradictions": [],
        "answer_length": len(answer),
        "truth_length": len(ground_truth)
    }

    if not answer or not ground_truth:
        return 0.0, analysis

    # First check for number mismatches BEFORE normalization
    answer_nums = set(re.findall(r'\d+(?:\.\d+)?', answer.lower()))
    truth_nums = set(re.findall(r'\d+(?:\.\d+)?', ground_truth.lower()))

    # If numbers differ significantly, this is a factual error
    number_mismatch = False
    if truth_nums and answer_nums and truth_nums != answer_nums:
        number_mismatch = True
        analysis["number_mismatch"] = True

    normalized_answer = normalize_text(answer)
    normalized_truth = normalize_text(ground_truth)

    # Exact match
    if normalized_answer == normalized_truth:
        analysis["exact_match"] = True
        # But if numbers mismatch, reduce score
        if number_mismatch:
            return 0.2, analysis
        return 1.0, analysis

    # Check if truth is contained in answer
    if normalized_truth in normalized_answer:
        analysis["key_info_present"] = True
        if number_mismatch:
            return 0.3, analysis
        return 0.95, analysis

    # Entity overlap analysis
    answer_entities = extract_entities(answer)
    truth_entities = extract_entities(ground_truth)

    if truth_entities:
        entity_overlap = len(answer_entities & truth_entities) / len(truth_entities)
        analysis["entity_overlap"] = entity_overlap
    else:
        analysis["entity_overlap"] = 1.0  # No entities to check

    # Word-level analysis
    truth_words = set(normalized_truth.split())
    answer_words = set(normalized_answer.split())

    # Remove stop words for content comparison
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'to', 'of',
                  'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into'}

    truth_content = truth_words - stop_words
    answer_content = answer_words - stop_words

    if truth_content:
        content_overlap = len(answer_content & truth_content) / len(truth_content)
        analysis["key_info_present"] = content_overlap > 0.5
    else:
        content_overlap = 1.0

    # Sequence similarity
    sequence_sim = SequenceMatcher(None, normalized_answer, normalized_truth).ratio()

    # Entity overlap is critical - if different entities, penalize heavily
    entity_penalty = 1.0 if analysis["entity_overlap"] > 0.5 else 0.3

    # Check for numerical contradiction BEFORE normalization (e.g., $50,000 vs $30,000)
    answer_nums = set(re.findall(r'\d+(?:\.\d+)?', answer.lower()))
    truth_nums = set(re.findall(r'\d+(?:\.\d+)?', ground_truth.lower()))
    number_mismatch = truth_nums and answer_nums and answer_nums != truth_nums
    number_penalty = 0.2 if number_mismatch else 1.0

    # Combine scores
    semantic_similarity = max(
        content_overlap * entity_penalty * number_penalty,
        sequence_sim * 0.8 * entity_penalty * number_penalty,
        analysis["entity_overlap"] * 0.7 * number_penalty
    )

    # Apply contradiction penalty for conflicting info
    if sequence_sim < 0.5 and content_overlap < 0.5:
        semantic_similarity *= 0.3  # Strong penalty for wrong answers

    analysis["semantic_similarity"] = semantic_similarity

    # Check for contradictions (simple heuristic: negation words)
    negation_words = {'not', 'no', 'never', 'none', 'neither', 'nobody', 'nothing'}
    has_negation_answer = any(w in answer_words for w in negation_words)
    has_negation_truth = any(w in truth_words for w in negation_words)

    if has_negation_answer != has_negation_truth:
        analysis["contradictions"].append("negation_mismatch")
        semantic_similarity *= 0.5

    # Number mismatch already handled above
    if number_mismatch:
        analysis["number_mismatch"] = True
        semantic_similarity = min(semantic_similarity, 0.3)

    return semantic_similarity, analysis


def detect_hallucination_advanced(
    answer: str,
    context: str,
    ground_truth: str = "",
    confidence: float = 0.5
) -> Tuple[float, HallucinationType, HallucinationSeverity, Dict[str, Any]]:
    """
    Advanced hallucination detection with type classification and severity scoring.

    Returns:
        Tuple of (hallucination_score, hallucination_type, severity, analysis)
    """
    analysis = {
        "word_coverage": 0.0,
        "entity_hallucination": 0.0,
        "numerical_fabrication": 0.0,
        "temporal_errors": 0.0,
        "relationship_errors": 0.0,
        "confidence_mismatch": 0.0
    }

    if not answer:
        return 0.0, HallucinationType.NONE, HallucinationSeverity.NONE, analysis

    normalized_answer = normalize_text(answer)
    normalized_context = normalize_text(context)

    # Stop words
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'shall',
        'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
        'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
        'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'and', 'but', 'or', 'yet', 'so',
        'if', 'because', 'although', 'though', 'while', 'where',
        'when', 'that', 'which', 'who', 'whom', 'whose', 'what',
        'this', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'them', 'their', 'there', 'then', 'than'
    }

    # Word coverage analysis
    answer_words = set(normalized_answer.split())
    context_words = set(normalized_context.split())

    content_words = answer_words - stop_words
    context_content = context_words - stop_words

    if content_words:
        words_in_context = content_words & context_content
        analysis["word_coverage"] = len(words_in_context) / len(content_words)
    else:
        analysis["word_coverage"] = 1.0

    # Entity hallucination check - more aggressive detection
    answer_entities = extract_entities(answer)
    context_entities = extract_entities(context)

    novel_entities = answer_entities - context_entities
    missing_entities = context_entities - answer_entities

    if answer_entities:
        # Base entity hallucination ratio
        analysis["entity_hallucination"] = len(novel_entities) / len(answer_entities)
        # Boost if there are ANY novel entities (fabricated facts)
        if novel_entities:
            analysis["entity_hallucination"] = max(analysis["entity_hallucination"], 0.5)
        # Also penalize if context entities are missing from answer
        if missing_entities and context_entities:
            analysis["entity_hallucination"] += 0.2 * (len(missing_entities) / len(context_entities))
            analysis["entity_hallucination"] = min(1.0, analysis["entity_hallucination"])
    else:
        analysis["entity_hallucination"] = 0.0

    # Numerical fabrication check
    answer_numbers = set(re.findall(r'\d+(?:\.\d+)?', normalized_answer))
    context_numbers = set(re.findall(r'\d+(?:\.\d+)?', normalized_context))
    novel_numbers = answer_numbers - context_numbers

    if answer_numbers:
        analysis["numerical_fabrication"] = min(1.0, len(novel_numbers) / len(answer_numbers))
    else:
        analysis["numerical_fabrication"] = 0.0

    # Compute base hallucination score
    hallucination_score = 0.0

    # Check for contradiction with ground truth (most important signal)
    if ground_truth:
        truth_sim, _ = check_factual_accuracy_advanced(answer, ground_truth, "")
        if truth_sim < 0.3:  # Answer contradicts truth
            hallucination_score += 0.5
            analysis["contradiction_with_truth"] = True
        # Additional penalty for very low similarity
        if truth_sim < 0.2:
            hallucination_score += 0.3

    # Low word coverage indicates hallucination
    if analysis["word_coverage"] < 0.5:
        hallucination_score += 0.3 * (1.0 - analysis["word_coverage"])

    # Entity hallucination is strong signal - boost the contribution
    if analysis["entity_hallucination"] > 0.3:
        hallucination_score += 0.4 * analysis["entity_hallucination"]
    elif analysis["entity_hallucination"] > 0.1:
        hallucination_score += 0.2 * analysis["entity_hallucination"]

    # Numerical fabrication
    if analysis["numerical_fabrication"] > 0:
        hallucination_score += 0.35 * analysis["numerical_fabrication"]

    # Confidence mismatch penalty (overconfident wrong answers)
    if (analysis["word_coverage"] < 0.5 or analysis["entity_hallucination"] > 0.3) and confidence > 0.7:
        analysis["confidence_mismatch"] = confidence - 0.5
        hallucination_score += 0.3 * analysis["confidence_mismatch"]

    # Cap at 1.0
    hallucination_score = min(1.0, hallucination_score)

    # Classify hallucination type
    hallucination_type = HallucinationType.NONE

    if analysis["numerical_fabrication"] > 0.5:
        hallucination_type = HallucinationType.NUMERICAL_FABRICATION
    elif analysis["entity_hallucination"] > 0.5:
        hallucination_type = HallucinationType.ENTITY_CONFUSION
    elif analysis["word_coverage"] < 0.3:
        hallucination_type = HallucinationType.FABRICATED_FACT
    elif analysis["confidence_mismatch"] > 0.3:
        hallucination_type = HallucinationType.OVERCONFIDENT_WRONG

    # Determine severity
    severity = HallucinationSeverity.NONE

    if hallucination_score >= 0.7:
        severity = HallucinationSeverity.CRITICAL
    elif hallucination_score >= 0.5:
        severity = HallucinationSeverity.SEVERE
    elif hallucination_score >= 0.3:
        severity = HallucinationSeverity.MODERATE
    elif hallucination_score > 0.1:
        severity = HallucinationSeverity.MINOR

    return hallucination_score, hallucination_type, severity, analysis


def compute_calibration_error(confidence: float, correctness: float) -> float:
    """
    Compute calibration error between confidence and actual correctness.

    Perfect calibration: confidence == correctness
    Overconfidence: confidence > correctness (dangerous)
    Underconfidence: confidence < correctness (safe but inefficient)
    """
    base_error = abs(confidence - correctness)

    # Penalize overconfidence more heavily
    if confidence > correctness:
        overconfidence_penalty = (confidence - correctness) * 0.5
        base_error += overconfidence_penalty

    return min(1.0, base_error)


def compute_semantic_consistency(answer: str, context: str, ground_truth: str) -> Tuple[float, Dict[str, Any]]:
    """
    Compute semantic consistency between answer, context, and ground truth.

    When sentence-transformers is available:
      - Uses NLI cross-encoder (nli-deberta-v3-small) for entailment/contradiction
        detection between (context, answer) and (ground_truth, answer) pairs.
      - Scores: entailment -> high consistency, contradiction -> low, neutral -> mid.

    Falls back to embedding cosine similarity + negation heuristics when the
    cross-encoder is not installed.
    """
    analysis = {
        "context_answer_similarity": 0.0,
        "truth_answer_similarity": 0.0,
        "key_claim_overlap": 0.0,
        "contradiction_detected": False,
        "entailment_score": 0.0,
        "nli_used": False,
    }

    if not answer:
        return 0.0, analysis

    # ── NLI cross-encoder path ────────────────────────────────────────────
    nli = _get_nli()
    if nli is not None and context and ground_truth:
        try:
            # Pairs: (premise, hypothesis). Labels: contradiction=0, entailment=1, neutral=2
            pairs = [(context, answer), (ground_truth, answer)]
            scores = nli.predict(pairs, apply_softmax=True)
            # scores shape: (2, 3) — [contradiction, entailment, neutral]
            ctx_entail  = float(scores[0][1])   # context entails answer
            ctx_contra  = float(scores[0][0])   # context contradicts answer
            truth_entail = float(scores[1][1])  # ground_truth entails answer
            truth_contra = float(scores[1][0])

            analysis["entailment_score"] = (ctx_entail + truth_entail) / 2.0
            analysis["contradiction_detected"] = (ctx_contra > 0.5) or (truth_contra > 0.5)
            analysis["nli_used"] = True

            # Consistency = average entailment, penalised by contradiction
            consistency_score = (ctx_entail * 0.5 + truth_entail * 0.5)
            if analysis["contradiction_detected"]:
                consistency_score *= max(0.1, 1.0 - max(ctx_contra, truth_contra))

            analysis["context_answer_similarity"] = ctx_entail
            analysis["truth_answer_similarity"]   = truth_entail
            return max(0.0, min(1.0, consistency_score)), analysis
        except Exception as e:
            logger.warning(f"NLI inference failed: {e}; falling back to similarity")

    # ── Embedding similarity path ─────────────────────────────────────────
    analysis["context_answer_similarity"] = compute_string_similarity(answer, context)
    analysis["truth_answer_similarity"]   = compute_string_similarity(answer, ground_truth)

    answer_claims  = extract_key_claims(answer)
    context_claims = extract_key_claims(context)
    if answer_claims and context_claims:
        matching = sum(
            1 for ac in answer_claims
            if any(compute_string_similarity(ac, cc) > 0.6 for cc in context_claims)
        )
        analysis["key_claim_overlap"] = matching / len(answer_claims)

    # Improved negation-based contradiction detection:
    # Check for negation asymmetry around the same key noun phrase
    negation_re = re.compile(r"\b(not|no|never|none|neither|isn't|aren't|wasn't|weren't|doesn't|don't)\b")
    answer_negated   = bool(negation_re.search(answer.lower()))
    truth_negated    = bool(negation_re.search(ground_truth.lower()))
    # Only flag contradiction if they share vocabulary but negate differently
    shared_words = set(normalize_text(answer).split()) & set(normalize_text(ground_truth).split())
    if answer_negated != truth_negated and len(shared_words) >= 2:
        analysis["contradiction_detected"] = True

    consistency_score = (
        0.4 * analysis["key_claim_overlap"] +
        0.3 * analysis["context_answer_similarity"] +
        0.3 * analysis["truth_answer_similarity"]
    )
    if analysis["contradiction_detected"]:
        consistency_score *= 0.5

    return max(0.0, min(1.0, consistency_score)), analysis


def calculate_reward(
    answer: str,
    confidence: float,
    source_quote: str,
    context: str,
    ground_truth: str,
    difficulty_level: str = "intermediate",
    difficulty: str = None,
    previous_performance: float = 0.5,
    recent_rewards: list = None,
    reward_weights: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate comprehensive multi-factor reward.

    This is the main entry point for reward calculation, combining:
    1. Factual correctness (30%)
    2. Source grounding (20%)
    3. Citation accuracy (15%)
    4. Confidence calibration (15%)
    5. Semantic consistency (10%)
    6. Hallucination penalty (10%)

    Plus difficulty bonuses and consistency bonuses.

    Args:
        answer: The AI's answer
        confidence: AI's confidence level (0-1)
        source_quote: Quote cited from context
        context: The source document
        ground_truth: The correct answer
        difficulty_level: Question difficulty
        previous_performance: Running performance metric
        reward_weights: Optional custom weights

    Returns:
        Tuple of (total_reward, info_dict)
    """
    # Resolve parameter aliases for README compatibility
    if difficulty is not None:
        difficulty_level = difficulty
    if recent_rewards is not None and len(recent_rewards) > 0:
        previous_performance = sum(recent_rewards) / len(recent_rewards)

    # Default weights - tuned for proper reward calibration
    # Grounded correct answers should receive 0.6+ rewards
    if reward_weights is None:
        reward_weights = {
            "factual_correctness":    0.30,
            "source_grounding":       0.25,
            "citation_accuracy":      0.10,
            "confidence_calibration": 0.08,
            "semantic_consistency":   0.07,
            "hallucination_penalty":  0.05,
            "rouge_score":            0.05,
            "bertscore":              0.05,
            "alignscore":             0.05,
        }

    # Component 1: Factual correctness
    correctness, correctness_analysis = check_factual_accuracy_advanced(answer, ground_truth, context)

    # Component 2 & 3: Source grounding and citation accuracy
    grounding_score, citation_analysis = check_quote_in_context_advanced(source_quote, context)

    # Component 4: Confidence calibration
    calibration_error = compute_calibration_error(confidence, correctness)
    calibration_score = 1.0 - calibration_error

    # Component 5: Semantic consistency
    semantic_score, semantic_analysis = compute_semantic_consistency(answer, context, ground_truth)

    # Component 6: Hallucination detection
    hallucination_score, hallucination_type, hallucination_severity, hallucination_analysis = \
        detect_hallucination_advanced(answer, context, ground_truth, confidence)

    hallucination_penalty_score = 1.0 - hallucination_score

    # Component 7: ROUGE
    rouge_scores = compute_rouge(answer, ground_truth)
    rouge_combined = (
        0.2 * rouge_scores["rouge1"] +
        0.3 * rouge_scores["rouge2"] +
        0.5 * rouge_scores["rougeL"]
    )

    # Component 8: BERTScore
    bs_scores = compute_bertscore(answer, ground_truth)
    bertscore_f1 = bs_scores["f1"]

    # Component 9: AlignScore
    align_score = compute_alignscore(context, answer)

    # Factual correctness gate: if answer is factually wrong, cap the reward
    # This prevents high rewards for well-grounded but incorrect answers
    factual_cap = correctness  # Wrong answers can't score higher than their correctness

    # Calculate base reward
    base_reward = (
        reward_weights["factual_correctness"]    * correctness +
        reward_weights["source_grounding"]       * min(grounding_score, factual_cap) +
        reward_weights["citation_accuracy"]      * min(citation_analysis.get("best_match_score", 0.0), factual_cap) +
        reward_weights["confidence_calibration"] * calibration_score +
        reward_weights["semantic_consistency"]   * min(semantic_score, factual_cap) +
        reward_weights["hallucination_penalty"]  * hallucination_penalty_score +
        reward_weights.get("rouge_score", 0.05)  * min(rouge_combined, factual_cap) +
        reward_weights.get("bertscore",   0.05)  * min(bertscore_f1, factual_cap) +
        reward_weights.get("alignscore",  0.05)  * min(align_score, factual_cap)
    )

    # Difficulty bonus
    difficulty_multipliers = {
        "beginner": 0.9,
        "intermediate": 1.0,
        "advanced": 1.1,
        "expert": 1.2
    }
    difficulty_multiplier = difficulty_multipliers.get(difficulty_level.lower(), 1.0)

    # Consistency bonus (for maintaining good performance)
    consistency_bonus = 0.0
    if previous_performance > 0.7:
        consistency_bonus = 0.05 * (previous_performance - 0.7) / 0.3

    # Apply adjustments
    total_reward = base_reward * difficulty_multiplier + consistency_bonus
    total_reward = max(0.0, min(1.0, total_reward))  # Clamp to [0, 1]

    # Determine if hallucination occurred
    is_hallucination = hallucination_score > 0.5

    # Build comprehensive info dict
    info = {
        # Core scores
        "correctness": correctness,
        "grounding": grounding_score,
        "calibration": calibration_score,
        "semantic_consistency": semantic_score,
        "hallucination_score": hallucination_score,
        "hallucination_penalty": hallucination_penalty_score,

        # Classification
        "is_hallucination": is_hallucination,
        "hallucination_type": hallucination_type.value,
        "hallucination_severity": hallucination_severity.name,

        # Reward breakdown
        "total_reward": total_reward,
        "base_reward": base_reward,
        "difficulty_multiplier": difficulty_multiplier,
        "consistency_bonus": consistency_bonus,

        # Component contributions
        "components": {
            "correctness_contrib": reward_weights["factual_correctness"] * correctness,
            "grounding_contrib": reward_weights["source_grounding"] * grounding_score,
            "citation_contrib": reward_weights["citation_accuracy"] * citation_analysis.get("best_match_score", 0.0),
            "calibration_contrib": reward_weights["confidence_calibration"] * calibration_score,
            "semantic_contrib": reward_weights["semantic_consistency"] * semantic_score,
            "hallucination_contrib": reward_weights["hallucination_penalty"] * hallucination_penalty_score,
        },

        # Research-grade metrics (v4.0)
        "rouge": rouge_scores,
        "rouge_combined": round(rouge_combined, 4),
        "bertscore": bs_scores,
        "alignscore": align_score,

        # Component contributions
        "rouge_contrib":      reward_weights.get("rouge_score", 0.05) * rouge_combined,
        "bertscore_contrib":  reward_weights.get("bertscore",   0.05) * bertscore_f1,
        "alignscore_contrib": reward_weights.get("alignscore",  0.05) * align_score,

        # Detailed analyses
        "correctness_analysis": correctness_analysis,
        "citation_analysis": citation_analysis,
        "semantic_analysis": semantic_analysis,
        "hallucination_analysis": hallucination_analysis,

        # Confidence info
        "confidence": confidence,
        "calibration_error": calibration_error,
    }

    return total_reward, info


def generate_feedback(
    answer: str,
    ground_truth: str,
    is_hallucination: bool,
    hallucination_type: HallucinationType,
    hallucination_severity: HallucinationSeverity,
    grounding_score: float,
    correctness: float,
    calibration_score: float,
    total_reward: float
) -> str:
    """Generate detailed, actionable feedback for the AI."""

    feedback_parts = []

    # Correctness feedback
    if correctness > 0.8:
        feedback_parts.append("Excellent! Answer is factually accurate.")
    elif correctness > 0.5:
        feedback_parts.append("Answer is partially correct but could be improved.")
    else:
        feedback_parts.append("Answer is factually incorrect.")

    # Grounding feedback
    if grounding_score > 0.8:
        feedback_parts.append("Source citation is verified in context.")
    elif grounding_score > 0.5:
        feedback_parts.append("Source citation partially matches context.")
    else:
        feedback_parts.append("WARNING: Source citation NOT found in context.")

    # Hallucination feedback
    if is_hallucination:
        severity_str = hallucination_severity.name.lower()
        type_str = hallucination_type.value.replace("_", " ")
        feedback_parts.append(f"HALLUCINATION DETECTED ({severity_str}): {type_str}.")

        if hallucination_severity in [HallucinationSeverity.SEVERE, HallucinationSeverity.CRITICAL]:
            feedback_parts.append("This is a serious hallucination that significantly undermines trust.")

    # Calibration feedback
    if calibration_score > 0.8:
        feedback_parts.append("Confidence level is well-calibrated.")
    elif calibration_score < 0.5:
        feedback_parts.append("WARNING: Confidence is poorly calibrated to accuracy.")

    # Summary
    if total_reward > 0.8:
        feedback_parts.append("Overall: OUTSTANDING performance!")
    elif total_reward > 0.6:
        feedback_parts.append("Overall: Good performance with room for improvement.")
    elif total_reward > 0.4:
        feedback_parts.append("Overall: Adequate but needs significant improvement.")
    else:
        feedback_parts.append("Overall: Poor performance - review and recalibrate.")

    return " ".join(feedback_parts)


def generate_feedback_from_info(info: Dict[str, Any]) -> str:
    """Convenience wrapper: generate feedback from the info dict returned by calculate_reward."""
    return generate_feedback(
        answer=info.get("answer", ""),
        ground_truth=info.get("ground_truth", ""),
        is_hallucination=info.get("is_hallucination", False),
        hallucination_type=info.get("hallucination_type", HallucinationType.NONE),
        hallucination_severity=info.get("hallucination_severity", HallucinationSeverity.NONE),
        grounding_score=info.get("grounding_score", 0.0),
        correctness=info.get("correctness", 0.0),
        calibration_score=info.get("calibration_score", 0.0),
        total_reward=info.get("total_reward", 0.0),
    )


def get_reward_breakdown(info: Dict[str, Any]) -> Dict[str, Any]:
    """Convert info dict to RewardBreakdown dataclass format."""
    from models import RewardBreakdown

    return {
        "factual_correctness": info.get("correctness", 0.0),
        "source_grounding": info.get("grounding", 0.0),
        "citation_accuracy": info.get("citation_analysis", {}).get("best_match_score", 0.0),
        "confidence_calibration": info.get("calibration", 0.0),
        "semantic_consistency": info.get("semantic_consistency", 0.0),
        "hallucination_penalty": info.get("hallucination_penalty", 0.0),
        "total": info.get("total_reward", 0.0),
    }
