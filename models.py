"""Professional-grade data contracts for HallucinationGuard-Env.

This module defines the core data structures for a complex RL environment
that trains AI models to avoid hallucinations and stay grounded in verified context.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Literal
from enum import Enum
import uuid

from openenv.core.env_server import Action, Observation, State


class HallucinationSeverity(Enum):
    """Severity levels for detected hallucinations."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class HallucinationType(Enum):
    """Types of hallucinations that can be detected."""
    NONE = "none"
    FABRICATED_FACT = "fabricated_fact"
    FALSE_CITATION = "false_citation"
    OVERCONFIDENT_WRONG = "overconfident_wrong"
    CONTEXT_DRIFT = "context_drift"
    TEMPORAL_HALLUCINATION = "temporal_hallucination"
    NUMERICAL_FABRICATION = "numerical_fabrication"
    ENTITY_CONFUSION = "entity_confusion"
    RELATIONSHIP_ERROR = "relationship_error"


class DifficultyLevel(Enum):
    """Difficulty levels for questions."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components."""
    factual_correctness: float = 0.0
    source_grounding: float = 0.0
    citation_accuracy: float = 0.0
    confidence_calibration: float = 0.0
    semantic_consistency: float = 0.0
    hallucination_penalty: float = 0.0
    difficulty_bonus: float = 0.0
    consistency_bonus: float = 0.0
    total: float = 0.0


@dataclass
class SemanticAnalysis:
    """Results of semantic analysis on the answer."""
    embedding_similarity: float = 0.0
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    neutral_score: float = 0.0
    key_entity_overlap: float = 0.0
    semantic_density: float = 0.0


@dataclass
class CitationAnalysis:
    """Results of citation verification."""
    exact_match: bool = False
    partial_matches: List[Dict[str, Any]] = field(default_factory=list)
    citation_location: Optional[str] = None
    surrounding_context: str = ""
    citation_confidence: float = 0.0


@dataclass
class HallucinationAction(Action):
    """
    Comprehensive action space for the AI agent.

    The AI must provide:
    - An answer to the question
    - Confidence level (calibrated)
    - Source citation from the context
    - Optional reasoning/chain-of-thought
    - Optional follow-up questions for clarification
    """
    answer: str = ""
    confidence: float = 0.5
    source_quote: str = ""
    reasoning: str = ""
    alternative_answers: List[str] = field(default_factory=list)
    uncertainty_flags: List[str] = field(default_factory=list)
    requires_clarification: bool = False
    clarification_questions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiTurnDialogue:
    """Track multi-turn conversation state."""
    turn_number: int = 0
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    unresolved_queries: List[str] = field(default_factory=list)
    context_shifts: List[str] = field(default_factory=list)


@dataclass
class HallucinationObservation(Observation):
    """
    Comprehensive observation space with rich feedback signals.

    Provides the AI with detailed information about:
    - The current question and context
    - Previous performance metrics
    - Detailed reward breakdown
    - Hallucination detection results
    - Curriculum progress
    """
    # Core QA elements
    question: str = ""
    context: str = ""
    ground_truth: str = ""
    question_id: str = ""
    source_dataset: str = ""

    # Episode state
    done: bool = False
    reward: Optional[float] = None

    # Feedback and evaluation
    feedback: str = ""
    is_hallucination: bool = False
    hallucination_type: Optional[HallucinationType] = None
    hallucination_severity: HallucinationSeverity = HallucinationSeverity.NONE
    grounding_score: float = 0.0

    # Performance metrics
    accuracy_so_far: float = 0.0
    attempts_remaining: int = 10
    current_streak: int = 0
    best_streak: int = 0

    # Detailed reward breakdown
    reward_breakdown: Optional[RewardBreakdown] = None
    semantic_analysis: Optional[SemanticAnalysis] = None
    citation_analysis: Optional[CitationAnalysis] = None

    # Curriculum and difficulty
    difficulty_level: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    curriculum_progress: float = 0.0
    skill_rating: float = 0.5

    # Multi-turn support
    dialogue: Optional[MultiTurnDialogue] = None

    # Extended metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeStatistics:
    """Comprehensive statistics for an episode."""
    episode_id: str = ""
    total_questions: int = 0
    questions_answered: int = 0
    correct_answers: int = 0
    hallucinated_answers: int = 0
    partially_correct: int = 0
    average_confidence: float = 0.0
    average_reward: float = 0.0
    calibration_error: float = 0.0
    hallucination_types: Dict[HallucinationType, int] = field(default_factory=dict)
    difficulty_distribution: Dict[DifficultyLevel, int] = field(default_factory=dict)
    time_per_question: List[float] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)


@dataclass
class AgentSkillProfile:
    """Long-term skill profile for an agent."""
    overall_accuracy: float = 0.0
    grounding_skill: float = 0.0
    calibration_skill: float = 0.0
    hallucination_rate: float = 0.0
    difficulty_ceiling: DifficultyLevel = DifficultyLevel.BEGINNER
    weak_areas: List[str] = field(default_factory=list)
    strong_areas: List[str] = field(default_factory=list)
    total_episodes: int = 0
    total_steps: int = 0


@dataclass
class HallucinationState(State):
    """
    Comprehensive state tracking for the RL environment.

    Tracks episode-level and agent-level state for:
    - Current episode progress
    - Historical performance
    - Curriculum positioning
    - Skill development
    """
    # Episode identification
    episode_id: Optional[str] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Step tracking
    step_count: int = 0
    max_questions: int = 10

    # Hallucination tracking
    total_hallucinations: int = 0
    hallucination_rate: float = 0.0
    hallucination_types_detected: Dict[str, int] = field(default_factory=dict)

    # Performance tracking
    total_correct: int = 0
    total_partial: int = 0
    accuracy: float = 0.0
    average_reward: float = 0.0

    # Confidence tracking
    average_confidence: float = 0.0
    calibration_error: float = 0.0

    # Curriculum state
    current_difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    curriculum_stage: int = 0
    skill_rating: float = 0.5

    # Streak tracking
    current_streak: int = 0
    best_streak: int = 0

    # Extended statistics
    episode_stats: Optional[EpisodeStatistics] = None
    agent_profile: Optional[AgentSkillProfile] = None

    # Environment configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    episode_start_time: Optional[float] = None
    last_step_time: Optional[float] = None

    # Metadata for extensibility
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "step_count": self.step_count,
            "max_questions": self.max_questions,
            "total_hallucinations": self.total_hallucinations,
            "hallucination_rate": self.hallucination_rate,
            "total_correct": self.total_correct,
            "accuracy": self.accuracy,
            "average_reward": self.average_reward,
            "current_difficulty": self.current_difficulty.value,
            "curriculum_stage": self.curriculum_stage,
            "skill_rating": self.skill_rating,
            "current_streak": self.current_streak,
            "best_streak": self.best_streak,
            **self.metadata
        }


@dataclass
class TrainingMetrics:
    """Metrics for tracking training progress over time."""
    episode_rewards: List[float] = field(default_factory=list)
    hallucination_rates: List[float] = field(default_factory=list)
    accuracy_curve: List[float] = field(default_factory=list)
    calibration_errors: List[float] = field(default_factory=list)
    difficulty_progression: List[str] = field(default_factory=list)
    moving_average_reward: float = 0.0
    trend_direction: Literal["improving", "stable", "declining"] = "stable"


@dataclass
class EnvironmentConfig:
    """Configuration for the hallucination detection environment."""
    # Episode configuration
    max_questions_per_episode: int = 10
    min_questions_for_completion: int = 5

    # Reward configuration
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "factual_correctness": 0.30,
        "source_grounding": 0.20,
        "citation_accuracy": 0.15,
        "confidence_calibration": 0.15,
        "semantic_consistency": 0.10,
        "hallucination_penalty": 0.10,
    })

    # Difficulty configuration
    initial_difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    adaptive_difficulty: bool = True
    difficulty_threshold_increase: float = 0.7
    difficulty_threshold_decrease: float = 0.4

    # Hallucination detection thresholds
    hallucination_threshold: float = 0.5
    severe_hallucination_threshold: float = 0.7

    # Curriculum configuration
    curriculum_enabled: bool = True
    min_steps_per_curriculum_stage: int = 50

    # Multi-turn configuration
    enable_multi_turn: bool = False
    max_turns_per_question: int = 3

    # Model compatibility
    supported_model_types: List[str] = field(default_factory=lambda: [
        "openai", "anthropic", "huggingface", "ollama", "llama", "generic"
    ])
