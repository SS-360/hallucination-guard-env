"""Professional-grade HallucinationGuard RL Environment.

This module implements a sophisticated, production-ready RL environment with:
- Curriculum learning with adaptive difficulty
- Multi-turn conversation support
- Context retrieval challenges
- Comprehensive episode management
- Model-agnostic design (works with any LLM)
- Real-time metrics and logging
- Session management for concurrent users
"""

import uuid
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Add directories to path for imports to work in both local and HF Spaces
import sys
import os
_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_dir)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from openenv.core.env_server import Environment

from models import (
    HallucinationAction,
    HallucinationObservation,
    HallucinationState,
    EpisodeStatistics,
    AgentSkillProfile,
    RewardBreakdown,
    SemanticAnalysis,
    CitationAnalysis,
    HallucinationSeverity,
    HallucinationType,
    DifficultyLevel,
    EnvironmentConfig,
    MultiTurnDialogue,
)
# Import from same directory for HF Spaces deployment compatibility
from grader import (
    calculate_reward,
    generate_feedback,
    detect_hallucination_advanced,
    HallucinationType as GraderHallucinationType,
    HallucinationSeverity as GraderHallucinationSeverity,
)
from dataset_loader import DatasetLoader, QAExample, DifficultyLevel as DatasetDifficulty


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EpisodePhase(Enum):
    """Phases of an episode."""
    INITIALIZATION = "initialization"
    ACTIVE = "active"
    MULTI_TURN_CLARIFICATION = "multi_turn_clarification"
    CONTEXT_RETRIEVAL = "context_retrieval"
    COMPLETION = "completion"


class HallucinationEnvironment(Environment[HallucinationAction, HallucinationObservation, HallucinationState]):
    """
    Professional-grade OpenEnv environment for training AI to avoid hallucinations.

    Features:
    - Curriculum learning with progressive difficulty
    - Adaptive difficulty based on performance
    - Multi-turn conversation support
    - Context retrieval challenges
    - Comprehensive metrics tracking
    - Model-agnostic design
    - Session management
    """

    SUPPORTS_CONCURRENT_SESSIONS = True
    VERSION = "2.0.0"

    def __init__(
        self,
        transform=None,
        config: Optional[EnvironmentConfig] = None,
        session_id: Optional[str] = None
    ):
        super().__init__(transform=transform)

        # Configuration
        self.config = config or EnvironmentConfig()
        self.session_id = session_id or str(uuid.uuid4())[:8]

        # Dataset management — load synthetic baseline, then augment with real HF data
        self.dataset_loader = DatasetLoader()
        self.dataset_loader.load_builtin_datasets()
        logger.info(f"Synthetic dataset: {self.dataset_loader.get_total_examples()} examples")

        # Attempt to load real HuggingFace datasets (SQuAD, TriviaQA, HaluEval, TruthfulQA).
        # Uses disk cache after first download so restarts are instant.
        # Gracefully skips if the `datasets` package is not installed.
        try:
            real_added = self.dataset_loader.load_real_datasets(max_per_dataset=500, cache=True)
            if real_added > 0:
                logger.info(f"Added {real_added} real examples — total: {self.dataset_loader.get_total_examples()}")
            else:
                logger.info("HuggingFace datasets unavailable; using synthetic data only")
        except Exception as _ds_err:
            logger.warning(f"Dataset loading failed ({_ds_err}); continuing with synthetic data only")

        # Episode state
        self.episode_id: Optional[str] = None
        self.episode_phase: EpisodePhase = EpisodePhase.INITIALIZATION
        self.step_count: int = 0
        self.total_hallucinations: int = 0
        self.total_correct: int = 0
        self.total_partial: int = 0

        # Current data
        self.current_example: Optional[QAExample] = None
        self.episode_examples: List[QAExample] = []
        self.episode_start_time: Optional[float] = None
        self.last_step_time: Optional[float] = None

        # Performance tracking
        self.reward_history: List[float] = []
        self.confidence_history: List[float] = []
        self.hallucination_history: List[bool] = []
        self.current_streak: int = 0
        self.best_streak: int = 0

        # Curriculum state
        self.curriculum_stage: int = 0
        self.curriculum_performance: List[float] = []
        self.skill_rating: float = 0.5  # ELO-style rating

        # Multi-turn state
        self.dialogue: Optional[MultiTurnDialogue] = None
        self.pending_clarifications: List[str] = []

        # Agent profile (persistent across episodes)
        self.agent_profile: Optional[AgentSkillProfile] = None

        # Context retrieval challenge state
        self.revealed_context_fragments: List[str] = []
        self.context_retrieval_turns: int = 0

        # Active model adapter (set via reset(model=...) for auto-play mode)
        self.active_adapter = None

        logger.info(f"Initialized HallucinationEnvironment (session={self.session_id})")

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        enable_multi_turn: bool = False,
        enable_context_retrieval: bool = False,
        model: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> HallucinationObservation:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility
            episode_id: Custom episode ID
            difficulty: Starting difficulty level
            enable_multi_turn: Enable multi-turn clarification
            enable_context_retrieval: Enable context retrieval challenges
            model: Model provider to use for auto-play mode.
                   Supported: "openai", "anthropic", "huggingface", "ollama", "generic".
                   When set, the environment calls the model automatically on each step
                   so you only need to call reset() + step() in a loop.
            model_config: Optional dict passed to create_adapter(). Keys:
                   model_name, api_key, api_base, temperature, max_tokens, etc.

        Returns:
            Initial observation
        """
        import random
        if seed is not None:
            random.seed(seed)

        # ── Model adapter setup ───────────────────────────────────────────────
        # When model= is supplied, the environment auto-generates answers by
        # calling the adapter inside step(), so callers just loop reset/step.
        if model is not None:
            try:
                import sys, os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from model_adapters import create_adapter
                cfg = model_config or {}
                self.active_adapter = create_adapter(model, **cfg)
                logger.info(f"Active adapter: {model} ({self.active_adapter.__class__.__name__})")
            except Exception as e:
                logger.warning(f"Could not create adapter for '{model}': {e}. Manual action mode.")
                self.active_adapter = None
        else:
            self.active_adapter = None

        # Generate episode ID
        self.episode_id = episode_id or f"ep_{uuid.uuid4().hex[:8]}"
        self.episode_start_time = time.time()
        self.last_step_time = time.time()

        # Reset counters
        self.step_count = 0
        self.total_hallucinations = 0
        self.total_correct = 0
        self.total_partial = 0
        self.reward_history = []
        self.confidence_history = []
        self.hallucination_history = []
        self.current_streak = 0

        # Reset multi-turn state
        self.dialogue = MultiTurnDialogue() if enable_multi_turn else None
        self.pending_clarifications = []

        # Reset context retrieval state
        self.revealed_context_fragments = []
        self.context_retrieval_turns = 0

        # Determine starting difficulty
        if difficulty:
            try:
                start_difficulty = DifficultyLevel(difficulty.lower())
            except ValueError:
                start_difficulty = self.config.initial_difficulty
        elif self.config.adaptive_difficulty and self.agent_profile:
            # Use agent's skill level
            start_difficulty = self.agent_profile.difficulty_ceiling
        else:
            start_difficulty = self.config.initial_difficulty

        # Load questions for this episode
        mix_difficulties = self.config.curriculum_enabled and start_difficulty == DifficultyLevel.INTERMEDIATE
        self.episode_examples = self.dataset_loader.start_new_episode(
            num_questions=self.config.max_questions_per_episode,
            difficulty=start_difficulty if not mix_difficulties else None,
            mix_difficulties=mix_difficulties
        )

        if not self.episode_examples:
            logger.error("No examples loaded for episode")
            return self._create_error_observation("No questions available")

        self.current_example = self.episode_examples[0]
        self.episode_phase = EpisodePhase.ACTIVE

        logger.info(f"Reset episode {self.episode_id} with {len(self.episode_examples)} questions")

        return self._create_observation(
            question=self.current_example.question,
            context=self._get_context_for_observation(self.current_example),
            feedback="Episode started. Answer using only the provided context.",
            metadata={"phase": self.episode_phase.value}
        )

    def step(
        self,
        action: Optional[HallucinationAction] = None,
        timeout_s: Optional[float] = None,
        **kwargs
    ) -> HallucinationObservation:
        """
        Process the AI's action and return the next observation.

        Auto-play mode: if reset(model=...) was called, action can be None —
        the environment calls the active adapter to generate an answer
        automatically using the current question and context.

        Manual mode: pass a HallucinationAction with answer, confidence, and
        source_quote filled in (the normal RL training loop).

        Handles:
        - Standard Q&A steps
        - Multi-turn clarifications
        - Context retrieval challenges
        """
        current_time = time.time()
        step_duration = current_time - (self.last_step_time or current_time)
        self.last_step_time = current_time

        # ── Auto-play: generate action via active adapter ─────────────────────
        if action is None or (not action.answer and self.active_adapter is not None):
            if self.current_example is not None and self.active_adapter is not None:
                try:
                    resp = self.active_adapter.generate_response(
                        question=self.current_example.question,
                        context=self.current_example.context,
                        require_citation=True,
                        require_confidence=True,
                    )
                    action = HallucinationAction(
                        answer=resp.answer,
                        confidence=resp.confidence,
                        source_quote=resp.source_quote or "",
                        reasoning=resp.reasoning or "",
                    )
                    logger.debug(f"Auto-play answer: {resp.answer[:80]}...")
                except Exception as e:
                    logger.warning(f"Adapter generate_response failed: {e}")
                    action = HallucinationAction(answer="", confidence=0.5)
            elif action is None:
                action = HallucinationAction(answer="", confidence=0.5)

        # Handle different episode phases
        if self.episode_phase == EpisodePhase.MULTI_TURN_CLARIFICATION:
            return self._handle_clarification_step(action)
        elif self.episode_phase == EpisodePhase.CONTEXT_RETRIEVAL:
            return self._handle_context_retrieval_step(action)

        # Standard Q&A step
        if self.current_example is None:
            return self._end_episode()

        # Validate action
        if not action.answer and not action.requires_clarification:
            return self._create_error_observation("No answer provided")

        # Handle clarification request
        if action.requires_clarification and self.dialogue:
            return self._handle_clarification_request(action)

        # Process the answer
        return self._process_answer(action, step_duration)

    def state(self) -> HallucinationState:
        """Return comprehensive state of the environment."""
        # Calculate derived metrics
        accuracy = self.total_correct / max(1, self.step_count)
        hallucination_rate = self.total_hallucinations / max(1, self.step_count)
        avg_confidence = sum(self.confidence_history) / max(1, len(self.confidence_history))

        # Calculate calibration error
        calibration_error = 0.0
        if self.confidence_history and self.reward_history:
            calibration_error = sum(
                abs(c - r) for c, r in zip(self.confidence_history, self.reward_history)
            ) / len(self.confidence_history)

        # Build episode statistics
        episode_stats = EpisodeStatistics(
            episode_id=self.episode_id or "",
            total_questions=len(self.episode_examples),
            questions_answered=self.step_count,
            correct_answers=self.total_correct,
            hallucinated_answers=self.total_hallucinations,
            partially_correct=self.total_partial,
            average_confidence=avg_confidence,
            average_reward=sum(self.reward_history) / max(1, len(self.reward_history)),
            calibration_error=calibration_error,
            reward_history=self.reward_history.copy(),
        )

        return HallucinationState(
            episode_id=self.episode_id,
            session_id=self.session_id,
            step_count=self.step_count,
            max_questions=self.config.max_questions_per_episode,
            total_hallucinations=self.total_hallucinations,
            hallucination_rate=hallucination_rate,
            total_correct=self.total_correct,
            total_partial=self.total_partial,
            accuracy=accuracy,
            average_reward=sum(self.reward_history) / max(1, len(self.reward_history)),
            average_confidence=avg_confidence,
            calibration_error=calibration_error,
            current_difficulty=self._get_current_difficulty(),
            curriculum_stage=self.curriculum_stage,
            skill_rating=self.skill_rating,
            current_streak=self.current_streak,
            best_streak=self.best_streak,
            episode_stats=episode_stats,
            agent_profile=self.agent_profile,
            config={
                "multi_turn_enabled": self.dialogue is not None,
                "context_retrieval_enabled": self.config.enable_multi_turn,
                "adaptive_difficulty": self.config.adaptive_difficulty,
            },
            episode_start_time=self.episode_start_time,
            last_step_time=self.last_step_time,
            metadata={
                "phase": self.episode_phase.value,
                "version": self.VERSION,
            }
        )

    def close(self) -> None:
        """Clean up resources and save agent profile."""
        if self.agent_profile:
            self._update_agent_profile()
        logger.info(f"Closed environment (session={self.session_id})")

    def _process_answer(
        self,
        action: HallucinationAction,
        step_duration: float
    ) -> HallucinationObservation:
        """Process a standard answer and compute rewards."""

        # Get ground truth
        ground_truth = self.current_example.answer
        context = self.current_example.context

        # Calculate reward using advanced grader
        difficulty_str = self.current_example.difficulty.value if self.current_example else "intermediate"
        prev_performance = self.skill_rating

        reward, info = calculate_reward(
            answer=action.answer,
            confidence=action.confidence,
            source_quote=action.source_quote,
            context=context,
            ground_truth=ground_truth,
            difficulty_level=difficulty_str,
            previous_performance=prev_performance,
            reward_weights=self.config.reward_weights
        )

        # Extract metrics from info
        is_hallucination = info.get("is_hallucination", False)
        hallucination_type_str = info.get("hallucination_type", "none")
        hallucination_severity_str = info.get("hallucination_severity", "NONE")
        correctness = info.get("correctness", 0.0)
        grounding_score = info.get("grounding", 0.0)
        calibration_score = info.get("calibration", 0.0)

        # Map hallucination type
        try:
            hallucination_type = HallucinationType(hallucination_type_str)
        except ValueError:
            hallucination_type = HallucinationType.NONE

        # Map severity
        try:
            severity = HallucinationSeverity[hallucination_severity_str]
        except KeyError:
            severity = HallucinationSeverity.NONE

        # Update statistics
        if is_hallucination:
            self.total_hallucinations += 1
            self.current_streak = 0
        elif correctness > 0.7:
            self.total_correct += 1
            self.current_streak += 1
            self.best_streak = max(self.best_streak, self.current_streak)
        else:
            self.total_partial += 1
            self.current_streak = 0

        # Track history
        self.reward_history.append(reward)
        self.confidence_history.append(action.confidence)
        self.hallucination_history.append(is_hallucination)

        # Update skill rating (ELO-style)
        expected_score = 1 / (1 + 10 ** ((0.5 - self.skill_rating) * 4))
        actual_score = 1.0 if correctness > 0.7 else (0.5 if correctness > 0.4 else 0.0)
        self.skill_rating += 0.05 * (actual_score - expected_score)
        self.skill_rating = max(0.0, min(1.0, self.skill_rating))

        # Generate feedback
        feedback = generate_feedback(
            answer=action.answer,
            ground_truth=ground_truth,
            is_hallucination=is_hallucination,
            hallucination_type=hallucination_type,
            hallucination_severity=severity,
            grounding_score=grounding_score,
            correctness=correctness,
            calibration_score=calibration_score,
            total_reward=reward
        )

        # Move to next question
        self.step_count += 1
        done = self.step_count >= self.config.max_questions_per_episode

        if not done:
            self.current_example = self.dataset_loader.get_example_for_step(self.step_count)
        else:
            self.current_example = None
            self.episode_phase = EpisodePhase.COMPLETION

        # Build observation
        observation = self._create_observation(
            question=self.current_example.question if self.current_example else "",
            context=self._get_context_for_observation(self.current_example) if self.current_example else "",
            ground_truth=ground_truth if done else "",  # Only reveal at end
            feedback=feedback,
            reward=reward,
            is_hallucination=is_hallucination,
            hallucination_type=hallucination_type,
            hallucination_severity=severity,
            grounding_score=grounding_score,
            done=done,
            metadata={
                "step": self.step_count,
                "correctness": correctness,
                "calibration": calibration_score,
                "hallucination_score": info.get("hallucination_score", 0.0),
                "reward_breakdown": self._extract_reward_breakdown(info),
                "semantic_analysis": info.get("semantic_analysis", {}),
                "citation_analysis": info.get("citation_analysis", {}),
            }
        )

        # Update dialogue history if enabled
        if self.dialogue:
            self.dialogue.turn_number += 1
            self.dialogue.conversation_history.append({
                "question": observation.question,
                "answer": action.answer,
                "feedback": feedback
            })

        return observation

    def _handle_clarification_request(
        self,
        action: HallucinationAction
    ) -> HallucinationObservation:
        """Handle a request for clarification."""
        if not self.dialogue:
            return self._create_error_observation("Multi-turn not enabled")

        # Add clarification questions to pending list
        self.pending_clarifications.extend(action.clarification_questions)
        self.dialogue.unresolved_queries.extend(action.clarification_questions)

        # Provide clarifications (simulated)
        clarifications = []
        for q in action.clarification_questions:
            # Simple keyword-based clarification
            clarification = self._generate_clarification(q, self.current_example)
            clarifications.append(clarification)
            if q in self.dialogue.unresolved_queries:
                self.dialogue.unresolved_queries.remove(q)

        # Switch to active phase
        self.episode_phase = EpisodePhase.ACTIVE

        return self._create_observation(
            question=self.current_example.question if self.current_example else "",
            context=self.current_example.context if self.current_example else "",
            feedback=f"Clarifications provided: {'; '.join(clarifications)}",
            metadata={
                "clarifications": clarifications,
                "phase": self.episode_phase.value
            }
        )

    def _handle_clarification_step(
        self,
        action: HallucinationAction
    ) -> HallucinationObservation:
        """Handle a step during multi-turn clarification."""
        # Process clarification and return to main question
        self.episode_phase = EpisodePhase.ACTIVE
        return self._process_answer(action, 0.0)

    def _handle_context_retrieval_step(
        self,
        action: HallucinationAction
    ) -> HallucinationObservation:
        """Handle context retrieval challenge."""
        # Reveal more context based on action
        full_context = self.current_example.context if self.current_example else ""
        context_fragments = self._split_context_into_fragments(full_context)

        # Reveal additional fragments
        new_revealed = min(
            len(self.revealed_context_fragments) + 1,
            len(context_fragments)
        )
        self.revealed_context_fragments = context_fragments[:new_revealed]

        revealed_context = " ".join(self.revealed_context_fragments)
        self.context_retrieval_turns += 1

        # Check if enough context revealed or max turns reached
        if self.context_retrieval_turns >= self.config.max_turns_per_question or \
           new_revealed >= len(context_fragments):
            self.episode_phase = EpisodePhase.ACTIVE
            # Update current example with full context
            if self.current_example:
                self.current_example.metadata["revealed_context"] = revealed_context
        else:
            # Stay in retrieval phase
            pass

        return self._create_observation(
            question=self.current_example.question if self.current_example else "",
            context=revealed_context,
            feedback=f"Context revealed: {new_revealed}/{len(context_fragments)} fragments",
            metadata={
                "fragments_revealed": new_revealed,
                "total_fragments": len(context_fragments),
                "phase": self.episode_phase.value
            }
        )

    def _create_observation(
        self,
        question: str = "",
        context: str = "",
        ground_truth: str = "",
        feedback: str = "",
        reward: Optional[float] = None,
        done: bool = False,
        is_hallucination: bool = False,
        hallucination_type: HallucinationType = HallucinationType.NONE,
        hallucination_severity: HallucinationSeverity = HallucinationSeverity.NONE,
        grounding_score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> HallucinationObservation:
        """Create a comprehensive observation."""
        accuracy_so_far = self.total_correct / max(1, self.step_count) if self.step_count > 0 else 0.0

        return HallucinationObservation(
            question=question,
            context=context,
            ground_truth=ground_truth,
            question_id=self.current_example.id if self.current_example else "",
            source_dataset=self.current_example.source if self.current_example else "",
            done=done,
            reward=reward,
            feedback=feedback,
            is_hallucination=is_hallucination,
            hallucination_type=hallucination_type,
            hallucination_severity=hallucination_severity,
            grounding_score=grounding_score,
            accuracy_so_far=accuracy_so_far,
            attempts_remaining=max(0, self.config.max_questions_per_episode - self.step_count),
            current_streak=self.current_streak,
            best_streak=self.best_streak,
            difficulty_level=self._get_current_difficulty(),
            curriculum_progress=self.step_count / max(1, self.config.max_questions_per_episode),
            skill_rating=self.skill_rating,
            dialogue=self.dialogue,
            metadata=metadata or {}
        )

    def _create_error_observation(self, error_message: str) -> HallucinationObservation:
        """Create an error observation."""
        return HallucinationObservation(
            done=True,
            reward=0.0,
            question="",
            context="",
            feedback=f"Error: {error_message}",
            is_hallucination=False,
            grounding_score=0.0,
            accuracy_so_far=0.0,
            attempts_remaining=0,
            metadata={"error": error_message}
        )

    def _end_episode(self) -> HallucinationObservation:
        """End the current episode."""
        self.episode_phase = EpisodePhase.COMPLETION

        # Update curriculum
        self._update_curriculum()

        return HallucinationObservation(
            done=True,
            reward=sum(self.reward_history) / max(1, len(self.reward_history)),
            question="",
            context="",
            feedback=self._generate_episode_summary(),
            is_hallucination=False,
            grounding_score=0.0,
            accuracy_so_far=self.total_correct / max(1, self.step_count),
            attempts_remaining=0,
            metadata={
                "episode_complete": True,
                "final_reward": sum(self.reward_history) / max(1, len(self.reward_history)),
                "total_hallucinations": self.total_hallucinations,
                "total_correct": self.total_correct,
            }
        )

    def _get_context_for_observation(self, example: Optional[QAExample]) -> str:
        """Get context, potentially with partial revelation for challenges."""
        if not example:
            return ""

        # Check if context retrieval is enabled
        if self.config.enable_multi_turn and self.revealed_context_fragments:
            return " ".join(self.revealed_context_fragments)

        return example.context

    def _get_current_difficulty(self) -> DifficultyLevel:
        """Determine current difficulty based on performance."""
        if not self.config.adaptive_difficulty:
            return self.config.initial_difficulty

        # Calculate recent performance
        recent_rewards = self.reward_history[-5:] if len(self.reward_history) >= 5 else self.reward_history
        avg_recent_reward = sum(recent_rewards) / max(1, len(recent_rewards))

        if avg_recent_reward > self.config.difficulty_threshold_increase:
            if self.current_example and self.current_example.difficulty != DifficultyLevel.EXPERT:
                return DifficultyLevel.ADVANCED
        elif avg_recent_reward < self.config.difficulty_threshold_decrease:
            return DifficultyLevel.INTERMEDIATE

        return self.config.initial_difficulty

    def _update_curriculum(self) -> None:
        """Update curriculum stage based on episode performance."""
        if not self.config.curriculum_enabled:
            return

        episode_reward = sum(self.reward_history) / max(1, len(self.reward_history))
        self.curriculum_performance.append(episode_reward)

        # Check for curriculum advancement
        if len(self.curriculum_performance) >= self.config.min_steps_per_curriculum_stage:
            recent_avg = sum(self.curriculum_performance[-self.config.min_steps_per_curriculum_stage:]) / \
                         self.config.min_steps_per_curriculum_stage

            if recent_avg > 0.7:
                self.curriculum_stage += 1
                self.curriculum_performance = []
                logger.info(f"Advanced to curriculum stage {self.curriculum_stage}")

    def _update_agent_profile(self) -> None:
        """Update the agent's long-term skill profile."""
        if not self.agent_profile:
            self.agent_profile = AgentSkillProfile()

        # Update metrics
        total_steps = self.agent_profile.total_steps + self.step_count
        weight = self.step_count / max(1, total_steps)

        self.agent_profile.overall_accuracy = (
            (1 - weight) * self.agent_profile.overall_accuracy +
            weight * (self.total_correct / max(1, self.step_count))
        )
        self.agent_profile.grounding_skill = (
            (1 - weight) * self.agent_profile.grounding_skill +
            weight * sum(self.reward_history) / max(1, len(self.reward_history))
        )
        self.agent_profile.hallucination_rate = (
            (1 - weight) * self.agent_profile.hallucination_rate +
            weight * (self.total_hallucinations / max(1, self.step_count))
        )
        self.agent_profile.total_episodes += 1
        self.agent_profile.total_steps = total_steps

        # Update difficulty ceiling
        if self.agent_profile.overall_accuracy > 0.8:
            self.agent_profile.difficulty_ceiling = DifficultyLevel.EXPERT
        elif self.agent_profile.overall_accuracy > 0.6:
            self.agent_profile.difficulty_ceiling = DifficultyLevel.ADVANCED
        elif self.agent_profile.overall_accuracy > 0.4:
            self.agent_profile.difficulty_ceiling = DifficultyLevel.INTERMEDIATE
        else:
            self.agent_profile.difficulty_ceiling = DifficultyLevel.BEGINNER

    def _generate_episode_summary(self) -> str:
        """Generate a summary of the completed episode."""
        total_reward = sum(self.reward_history) / max(1, len(self.reward_history))
        accuracy = self.total_correct / max(1, self.step_count)

        summary_parts = [
            f"Episode completed!",
            f"Total reward: {total_reward:.2f}",
            f"Accuracy: {accuracy:.1%}",
            f"Hallucinations: {self.total_hallucinations}/{self.step_count}",
            f"Best streak: {self.best_streak}",
        ]

        if total_reward > 0.8:
            summary_parts.append("Performance: OUTSTANDING!")
        elif total_reward > 0.6:
            summary_parts.append("Performance: Good")
        elif total_reward > 0.4:
            summary_parts.append("Performance: Needs improvement")
        else:
            summary_parts.append("Performance: Poor - review and recalibrate")

        return " ".join(summary_parts)

    def _extract_reward_breakdown(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract reward breakdown from grader info."""
        components = info.get("components", {})
        return {
            "factual_correctness": info.get("correctness", 0.0),
            "source_grounding": info.get("grounding", 0.0),
            "citation_accuracy": info.get("citation_analysis", {}).get("best_match_score", 0.0),
            "confidence_calibration": info.get("calibration", 0.0),
            "semantic_consistency": info.get("semantic_consistency", 0.0),
            "hallucination_penalty": info.get("hallucination_penalty", 0.0),
            "total": info.get("total_reward", 0.0),
            "difficulty_adjustment": info.get("difficulty_multiplier", 1.0),
            "consistency_bonus": info.get("consistency_bonus", 0.0),
        }

    def _split_context_into_fragments(self, context: str, num_fragments: int = 5) -> List[str]:
        """Split context into fragments for retrieval challenges."""
        if not context:
            return []

        sentences = context.split('.')
        fragments = []
        chunk_size = max(1, len(sentences) // num_fragments)

        for i in range(0, len(sentences), chunk_size):
            fragment = '.'.join(sentences[i:i + chunk_size]).strip()
            if fragment:
                fragments.append(fragment + '.')

        return fragments or [context]

    def _generate_clarification(self, question: str, example: Optional[QAExample]) -> str:
        """Generate a clarification response."""
        if not example:
            return "No context available for clarification."

        # Simple keyword-based clarification
        context_lower = example.context.lower()
        question_lower = question.lower()

        # Extract key terms from question
        key_terms = [w for w in question_lower.split() if len(w) > 3 and w not in {'what', 'when', 'where', 'who', 'why', 'how', 'does', 'have', 'has', 'with', 'from'}]

        clarifications = []
        for term in key_terms[:3]:
            if term in context_lower:
                clarifications.append(f"Context mentions '{term}'")

        return "; ".join(clarifications) if clarifications else "Review the provided context for relevant information."
