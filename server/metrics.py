"""Professional-grade metrics and visualization for HallucinationGuard-Env.

This module provides:
- Real-time metrics tracking
- Training curve visualization
- Hallucination heatmaps
- Comprehensive logging
- Export capabilities for analysis
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import os

logger = logging.getLogger(__name__)


@dataclass
class StepMetrics:
    """Metrics for a single step."""
    step: int
    episode_id: str
    reward: float
    correctness: float
    grounding: float
    calibration: float
    hallucination_score: float
    is_hallucination: bool
    confidence: float
    difficulty: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class EpisodeMetrics:
    """Metrics for a complete episode."""
    episode_id: str
    total_steps: int
    average_reward: float
    total_hallucinations: int
    hallucination_rate: float
    accuracy: float
    average_confidence: float
    calibration_error: float
    best_streak: int
    final_skill_rating: float
    difficulty_distribution: Dict[str, int] = field(default_factory=dict)
    reward_history: List[float] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0


@dataclass
class TrainingSession:
    """Complete training session metrics."""
    session_id: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_episodes: int = 0
    total_steps: int = 0
    episode_metrics: List[EpisodeMetrics] = field(default_factory=list)
    step_metrics: List[StepMetrics] = field(default_factory=list)

    # Aggregated metrics
    overall_accuracy: float = 0.0
    overall_hallucination_rate: float = 0.0
    average_reward: float = 0.0
    skill_rating_progress: List[float] = field(default_factory=list)

    # Trend analysis
    reward_trend: str = "stable"  # improving, stable, declining
    hallucination_trend: str = "stable"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "overall_accuracy": self.overall_accuracy,
            "overall_hallucination_rate": self.overall_hallucination_rate,
            "average_reward": self.average_reward,
            "skill_rating_progress": self.skill_rating_progress,
            "reward_trend": self.reward_trend,
            "hallucination_trend": self.hallucination_trend,
        }


class MetricsTracker:
    """
    Professional-grade metrics tracker for RL training.

    Features:
    - Real-time metric collection
    - Trend analysis
    - Visualization data generation
    - Export to multiple formats
    - Session persistence
    """

    def __init__(self, log_dir: Optional[str] = None, session_id: Optional[str] = None):
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = Path(log_dir) if log_dir else Path(__file__).parent / "metrics_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.current_session = TrainingSession(session_id=self.session_id)
        self.current_episode_data: List[StepMetrics] = []

        # Rolling windows for trend analysis
        self.reward_window: List[float] = []
        self.hallucination_window: List[bool] = []
        self.window_size = 10

        # Real-time aggregates
        self.running_reward_sum = 0.0
        self.running_reward_count = 0
        self.running_hallucination_count = 0
        self.running_step_count = 0

        logger.info(f"Initialized MetricsTracker (session={self.session_id})")

    def log_step(self, step_data: Dict[str, Any]) -> StepMetrics:
        """Log a single step."""
        step_metrics = StepMetrics(
            step=step_data.get("step", 0),
            episode_id=step_data.get("episode_id", ""),
            reward=step_data.get("reward", 0.0),
            correctness=step_data.get("correctness", 0.0),
            grounding=step_data.get("grounding", 0.0),
            calibration=step_data.get("calibration", 0.0),
            hallucination_score=step_data.get("hallucination_score", 0.0),
            is_hallucination=step_data.get("is_hallucination", False),
            confidence=step_data.get("confidence", 0.5),
            difficulty=step_data.get("difficulty", "intermediate"),
        )

        self.current_episode_data.append(step_metrics)
        self.current_session.step_metrics.append(step_metrics)

        # Update running aggregates
        self.running_reward_sum += step_metrics.reward
        self.running_reward_count += 1
        self.running_step_count += 1

        if step_metrics.is_hallucination:
            self.running_hallucination_count += 1

        # Update rolling windows
        self.reward_window.append(step_metrics.reward)
        self.hallucination_window.append(step_metrics.is_hallucination)

        if len(self.reward_window) > self.window_size:
            self.reward_window.pop(0)
            self.hallucination_window.pop(0)

        return step_metrics

    def end_episode(self, episode_data: Dict[str, Any]) -> EpisodeMetrics:
        """Mark the end of an episode and compute episode metrics."""
        episode_metrics = EpisodeMetrics(
            episode_id=episode_data.get("episode_id", ""),
            total_steps=episode_data.get("total_steps", len(self.current_episode_data)),
            average_reward=episode_data.get("average_reward", 0.0),
            total_hallucinations=episode_data.get("total_hallucinations", 0),
            hallucination_rate=episode_data.get("hallucination_rate", 0.0),
            accuracy=episode_data.get("accuracy", 0.0),
            average_confidence=episode_data.get("average_confidence", 0.5),
            calibration_error=episode_data.get("calibration_error", 0.0),
            best_streak=episode_data.get("best_streak", 0),
            final_skill_rating=episode_data.get("skill_rating", 0.5),
            reward_history=[s.reward for s in self.current_episode_data],
            start_time=episode_data.get("start_time", 0.0),
            end_time=episode_data.get("end_time", time.time()),
        )
        episode_metrics.duration = episode_metrics.end_time - episode_metrics.start_time

        self.current_session.episode_metrics.append(episode_metrics)
        self.current_session.total_episodes += 1
        self.current_session.total_steps += episode_metrics.total_steps

        # Update session aggregates
        self._update_session_aggregates()

        # Update trends
        self._update_trends()

        # Reset episode data
        self.current_episode_data = []

        logger.info(f"Episode {episode_metrics.episode_id} completed: reward={episode_metrics.average_reward:.3f}, "
                    f"hallucination_rate={episode_metrics.hallucination_rate:.3f}")

        return episode_metrics

    def _update_session_aggregates(self) -> None:
        """Update session-level aggregated metrics."""
        if not self.current_session.episode_metrics:
            return

        # Overall accuracy
        total_correct = sum(ep.accuracy * ep.total_steps for ep in self.current_session.episode_metrics)
        self.current_session.overall_accuracy = total_correct / max(1, self.current_session.total_steps)

        # Overall hallucination rate
        total_hallucinations = sum(ep.total_hallucinations for ep in self.current_session.episode_metrics)
        self.current_session.overall_hallucination_rate = total_hallucinations / max(1, self.current_session.total_steps)

        # Average reward
        total_reward = sum(ep.average_reward * ep.total_steps for ep in self.current_session.episode_metrics)
        self.current_session.average_reward = total_reward / max(1, self.current_session.total_steps)

        # Skill rating progress
        self.current_session.skill_rating_progress = [
            ep.final_skill_rating for ep in self.current_session.episode_metrics
        ]

    def _update_trends(self) -> None:
        """Analyze and update trend indicators."""
        if len(self.reward_window) < 3:
            return

        # Reward trend
        recent_avg = sum(self.reward_window[-5:]) / min(5, len(self.reward_window))
        older_avg = sum(self.reward_window[:-5]) / max(1, len(self.reward_window) - 5) if len(self.reward_window) > 5 else recent_avg

        if recent_avg > older_avg + 0.05:
            self.current_session.reward_trend = "improving"
        elif recent_avg < older_avg - 0.05:
            self.current_session.reward_trend = "declining"
        else:
            self.current_session.reward_trend = "stable"

        # Hallucination trend
        if len(self.hallucination_window) >= 5:
            recent_hallucination_rate = sum(self.hallucination_window[-5:]) / 5
            older_hallucination_rate = sum(self.hallucination_window[:-5]) / max(1, len(self.hallucination_window) - 5)

            if recent_hallucination_rate < older_hallucination_rate - 0.1:
                self.current_session.hallucination_trend = "improving"
            elif recent_hallucination_rate > older_hallucination_rate + 0.1:
                self.current_session.hallucination_trend = "worsening"
            else:
                self.current_session.hallucination_trend = "stable"

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics."""
        return {
            "session_id": self.current_session.session_id,
            "episodes_completed": self.current_session.total_episodes,
            "total_steps": self.current_session.total_steps,
            "overall_accuracy": self.current_session.overall_accuracy,
            "overall_hallucination_rate": self.current_session.overall_hallucination_rate,
            "average_reward": self.current_session.average_reward,
            "reward_trend": self.current_session.reward_trend,
            "hallucination_trend": self.current_session.hallucination_trend,
            "recent_reward_avg": sum(self.reward_window) / max(1, len(self.reward_window)),
            "recent_hallucination_rate": sum(self.hallucination_window) / max(1, len(self.hallucination_window)),
        }

    def get_training_curve_data(self) -> Dict[str, List[Any]]:
        """Get data for plotting training curves."""
        episode_rewards = [ep.average_reward for ep in self.current_session.episode_metrics]
        hallucination_rates = [ep.hallucination_rate for ep in self.current_session.episode_metrics]
        accuracies = [ep.accuracy for ep in self.current_session.episode_metrics]
        skill_ratings = self.current_session.skill_rating_progress

        # Calculate moving averages
        def moving_average(data: List[float], window: int = 5) -> List[float]:
            if len(data) < window:
                return data
            return [sum(data[i:i+window]) / window for i in range(len(data) - window + 1)]

        return {
            "episodes": list(range(1, len(episode_rewards) + 1)),
            "rewards": episode_rewards,
            "rewards_smooth": moving_average(episode_rewards),
            "hallucination_rates": hallucination_rates,
            "hallucination_rates_smooth": moving_average(hallucination_rates),
            "accuracies": accuracies,
            "skill_ratings": skill_ratings,
        }

    def get_hallucination_heatmap_data(self) -> Dict[str, Any]:
        """Get data for hallucination heatmap visualization."""
        # Group by difficulty and hallucination type
        heatmap_data = {}

        for step in self.current_session.step_metrics:
            difficulty = step.difficulty
            if difficulty not in heatmap_data:
                heatmap_data[difficulty] = {
                    "total": 0,
                    "hallucinations": 0,
                    "by_type": {}
                }

            heatmap_data[difficulty]["total"] += 1
            if step.is_hallucination:
                heatmap_data[difficulty]["hallucinations"] += 1

        # Calculate rates
        for difficulty in heatmap_data:
            total = heatmap_data[difficulty]["total"]
            hallucs = heatmap_data[difficulty]["hallucinations"]
            heatmap_data[difficulty]["rate"] = hallucs / max(1, total)

        return heatmap_data

    def get_reward_breakdown_analysis(self) -> Dict[str, Any]:
        """Get analysis of reward components."""
        if not self.current_session.step_metrics:
            return {}

        # Collect component values
        components = {
            "correctness": [],
            "grounding": [],
            "calibration": [],
            "hallucination_score": [],
        }

        for step in self.current_session.step_metrics:
            components["correctness"].append(step.correctness)
            components["grounding"].append(step.grounding)
            components["calibration"].append(step.calibration)
            components["hallucination_score"].append(step.hallucination_score)

        # Calculate statistics
        analysis = {}
        for name, values in components.items():
            if values:
                analysis[name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": self._calculate_std(values),
                }

        return analysis

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """Export session data to JSON."""
        if filepath is None:
            filepath = str(self.log_dir / f"{self.session_id}_metrics.json")

        data = {
            "session": self.current_session.to_dict(),
            "episode_metrics": [
                {
                    "episode_id": ep.episode_id,
                    "total_steps": ep.total_steps,
                    "average_reward": ep.average_reward,
                    "hallucination_rate": ep.hallucination_rate,
                    "accuracy": ep.accuracy,
                    "duration": ep.duration,
                }
                for ep in self.current_session.episode_metrics
            ],
            "training_curves": self.get_training_curve_data(),
            "heatmap_data": self.get_hallucination_heatmap_data(),
            "reward_analysis": self.get_reward_breakdown_analysis(),
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported metrics to {filepath}")
        return filepath

    def export_to_csv(self, filepath: Optional[str] = None) -> str:
        """Export step-level metrics to CSV."""
        if filepath is None:
            filepath = str(self.log_dir / f"{self.session_id}_steps.csv")

        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write("step,episode_id,reward,correctness,grounding,calibration,hallucination_score,is_hallucination,confidence,difficulty,timestamp\n")

            # Data
            for step in self.current_session.step_metrics:
                f.write(f"{step.step},{step.episode_id},{step.reward},{step.correctness},{step.grounding},"
                        f"{step.calibration},{step.hallucination_score},{int(step.is_hallucination)},"
                        f"{step.confidence},{step.difficulty},{step.timestamp}\n")

        logger.info(f"Exported CSV to {filepath}")
        return filepath

    def generate_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        metrics = self.get_real_time_metrics()

        report = f"""
╔══════════════════════════════════════════════════════════╗
║       HallucinationGuard-Env Training Summary            ║
╠══════════════════════════════════════════════════════════╣

Session: {self.current_session.session_id}
Episodes Completed: {metrics['episodes_completed']}
Total Steps: {metrics['total_steps']}

────────────────────────────────────────────────────────────
PERFORMANCE METRICS
────────────────────────────────────────────────────────────
Overall Accuracy: {metrics['overall_accuracy']:.1%}
Average Reward: {metrics['average_reward']:.3f}
Hallucination Rate: {metrics['overall_hallucination_rate']:.1%}

────────────────────────────────────────────────────────────
TREND ANALYSIS
────────────────────────────────────────────────────────────
Reward Trend: {metrics['reward_trend'].upper()}
Hallucination Trend: {metrics['hallucination_trend'].upper()}
Recent Reward Avg: {metrics['recent_reward_avg']:.3f}
Recent Hallucination Rate: {metrics['recent_hallucination_rate']:.1%}

────────────────────────────────────────────────────────────
INTERPRETATION
────────────────────────────────────────────────────────────
"""

        # Add interpretation
        if metrics['reward_trend'] == "improving":
            report += "✓ Model performance is IMPROVING over time\n"
        elif metrics['reward_trend'] == "declining":
            report += "⚠ Model performance is DECLINING - consider adjusting training\n"
        else:
            report += "→ Model performance is STABLE\n"

        if metrics['hallucination_trend'] == "improving":
            report += "✓ Hallucination rate is DECREASING\n"
        elif metrics['hallucination_trend'] == "worsening":
            report += "⚠ Hallucination rate is INCREASING - review training data\n"
        else:
            report += "→ Hallucination rate is STABLE\n"

        if metrics['overall_accuracy'] > 0.8:
            report += "\n★ EXCELLENT: Model is performing at expert level\n"
        elif metrics['overall_accuracy'] > 0.6:
            report += "\n✓ GOOD: Model is performing competently\n"
        elif metrics['overall_accuracy'] > 0.4:
            report += "\n→ MODERATE: Model needs more training\n"
        else:
            report += "\n⚠ POOR: Model requires significant training adjustment\n"

        report += "\n╚══════════════════════════════════════════════════════════╝\n"

        return report

    def close(self) -> None:
        """Close the session and export final metrics."""
        self.current_session.end_time = time.time()

        # Auto-export
        self.export_to_json()
        self.export_to_csv()

        # Print summary
        print(self.generate_summary_report())

        logger.info(f"Closed MetricsTracker session {self.session_id}")


class VisualizationDataGenerator:
    """Generate data for external visualization tools."""

    def __init__(self, tracker: MetricsTracker):
        self.tracker = tracker

    def get_plotly_training_curves(self) -> Dict[str, Any]:
        """Get data formatted for Plotly charts."""
        curve_data = self.tracker.get_training_curve_data()

        return {
            "data": [
                {
                    "name": "Reward",
                    "type": "scatter",
                    "x": curve_data["episodes"],
                    "y": curve_data["rewards"],
                    "mode": "lines+markers",
                    "yaxis": "y1",
                },
                {
                    "name": "Reward (smoothed)",
                    "type": "scatter",
                    "x": curve_data["episodes"][:len(curve_data["rewards_smooth"])],
                    "y": curve_data["rewards_smooth"],
                    "mode": "lines",
                    "line": {"dash": "dash"},
                },
                {
                    "name": "Hallucination Rate",
                    "type": "scatter",
                    "x": curve_data["episodes"],
                    "y": curve_data["hallucination_rates"],
                    "mode": "lines+markers",
                    "yaxis": "y2",
                },
                {
                    "name": "Accuracy",
                    "type": "scatter",
                    "x": curve_data["episodes"],
                    "y": curve_data["accuracies"],
                    "mode": "lines+markers",
                    "yaxis": "y1",
                },
            ],
            "layout": {
                "title": "Training Curves",
                "xaxis": {"title": "Episode"},
                "yaxis": {"title": "Reward / Accuracy"},
                "yaxis2": {
                    "title": "Hallucination Rate",
                    "overlaying": "y",
                    "side": "right",
                },
            }
        }

    def get_hallucination_type_distribution(self) -> Dict[str, Any]:
        """Get hallucination type distribution for pie chart."""
        type_counts = {}

        for step in self.tracker.current_session.step_metrics:
            if step.is_hallucination:
                # In a full implementation, track specific types
                type_key = "hallucination"
                type_counts[type_key] = type_counts.get(type_key, 0) + 1

        return {
            "labels": list(type_counts.keys()),
            "values": list(type_counts.values()),
        }

    def get_difficulty_performance_comparison(self) -> Dict[str, Any]:
        """Get performance comparison across difficulties."""
        heatmap_data = self.tracker.get_hallucination_heatmap_data()

        difficulties = list(heatmap_data.keys())
        rates = [heatmap_data[d]["rate"] for d in difficulties]
        totals = [heatmap_data[d]["total"] for d in difficulties]

        return {
            "difficulties": difficulties,
            "hallucination_rates": rates,
            "sample_sizes": totals,
        }


# Global tracker instance for easy access
_global_tracker: Optional[MetricsTracker] = None


def get_tracker(session_id: Optional[str] = None) -> MetricsTracker:
    """Get or create the global metrics tracker."""
    global _global_tracker
    if _global_tracker is None or (_global_tracker and session_id and session_id != _global_tracker.session_id):
        _global_tracker = MetricsTracker(session_id=session_id)
    return _global_tracker


def log_step(step_data: Dict[str, Any]) -> StepMetrics:
    """Log a step to the global tracker."""
    return get_tracker().log_step(step_data)


def end_episode(episode_data: Dict[str, Any]) -> EpisodeMetrics:
    """End an episode in the global tracker."""
    return get_tracker().end_episode(episode_data)


def get_metrics() -> Dict[str, Any]:
    """Get current metrics from the global tracker."""
    return get_tracker().get_real_time_metrics()
