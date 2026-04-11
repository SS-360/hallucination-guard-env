"""HTTP/WebSocket client for HallucinationGuard-Env."""

import requests
from typing import Optional, Dict, Any

from models import HallucinationAction, HallucinationObservation, HallucinationState


class HallucinationClient:
    """Client for interacting with the HallucinationGuard environment."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """Check if the server is healthy."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def reset(self) -> HallucinationObservation:
        """Reset the environment and get initial observation."""
        response = self.session.post(f"{self.base_url}/reset")
        response.raise_for_status()
        data = response.json()
        self._session_id = data.get("session_id")
        return HallucinationObservation(**data)

    def step(self, action: HallucinationAction) -> HallucinationObservation:
        """Take a step in the environment."""
        action_dict = {
            "answer": action.answer,
            "confidence": action.confidence,
            "source_quote": action.source_quote,
            "metadata": action.metadata
        }
        if getattr(self, '_session_id', None):
            action_dict["session_id"] = self._session_id
        response = self.session.post(
            f"{self.base_url}/step",
            json=action_dict
        )
        response.raise_for_status()
        data = response.json()
        return HallucinationObservation(**data)

    def get_state(self) -> HallucinationState:
        """Get the current environment state."""
        response = self.session.get(f"{self.base_url}/state")
        response.raise_for_status()
        data = response.json()
        return HallucinationState(**data)

    def close(self) -> None:
        """Close the client session."""
        self.session.close()


# Example usage
if __name__ == "__main__":
    client = HallucinationClient()

    # Check health
    print("Health:", client.health_check())

    # Reset environment
    obs = client.reset()
    print(f"\nQuestion: {obs.question}")
    print(f"Context: {obs.context[:200]}...")

    # Take a step with a sample action
    action = HallucinationAction(
        answer="This is a test answer",
        confidence=0.8,
        source_quote="test quote"
    )
    obs = client.step(action)
    print(f"\nReward: {obs.reward}")
    print(f"Feedback: {obs.feedback}")
    print(f"Is Hallucination: {obs.is_hallucination}")

    client.close()
