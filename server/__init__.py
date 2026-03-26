"""Server module for HallucinationGuard-Env."""

import sys
import os

# Add server directory to path for relative imports
_server_dir = os.path.dirname(os.path.abspath(__file__))
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

# Now import from same directory (works for both local and HF Spaces)
from environment import HallucinationEnvironment
from grader import (
    calculate_reward,
    check_factual_accuracy_advanced,
    check_quote_in_context_advanced,
    detect_hallucination_advanced,
    generate_feedback,
)
from dataset_loader import DatasetLoader, QAExample

__all__ = [
    "HallucinationEnvironment",
    "calculate_reward",
    "check_factual_accuracy_advanced",
    "check_quote_in_context_advanced",
    "detect_hallucination_advanced",
    "generate_feedback",
    "DatasetLoader",
    "QAExample",
]
