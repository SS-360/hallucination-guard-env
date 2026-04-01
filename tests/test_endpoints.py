"""Tests for FastAPI endpoints."""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBatchEvaluation:
    """Tests for batch evaluation endpoint."""

    def test_batch_evaluate_single_item(self):
        """Batch evaluation should work with single item."""
        from server.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        response = client.post("/batch/evaluate", json={
            "items": [{
                "question": "What is 2+2?",
                "context": "2+2 equals 4.",
                "answer": "4",
                "confidence": 0.9,
                "source_quote": "2+2 equals 4",
                "ground_truth": "4"
            }],
            "task_id": "task_1_factual_grounding"
        })

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "summary" in data
        assert data["total_items"] == 1

    def test_batch_evaluate_multiple_items(self):
        """Batch evaluation should work with multiple items."""
        from server.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        response = client.post("/batch/evaluate", json={
            "items": [
                {
                    "question": "What is 2+2?",
                    "context": "2+2 equals 4.",
                    "answer": "4",
                    "confidence": 0.9,
                    "ground_truth": "4"
                },
                {
                    "question": "Capital of France?",
                    "context": "The capital of France is Paris.",
                    "answer": "Paris",
                    "confidence": 0.8,
                    "ground_truth": "Paris"
                }
            ],
            "task_id": "task_1_factual_grounding"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["total_items"] == 2
        assert len(data["results"]) == 2

    def test_batch_evaluate_empty_items(self):
        """Batch evaluation should reject empty items."""
        from server.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        response = client.post("/batch/evaluate", json={
            "items": [],
            "task_id": "task_1_factual_grounding"
        })

        assert response.status_code == 422  # Validation error

    def test_batch_evaluate_invalid_task(self):
        """Batch evaluation should reject invalid task_id."""
        from server.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        response = client.post("/batch/evaluate", json={
            "items": [{"question": "test", "answer": "test"}],
            "task_id": "invalid_task"
        })

        assert response.status_code == 404


class TestMetricsEndpoints:
    """Tests for metrics endpoints."""

    def test_health_endpoint(self):
        """Health endpoint should return 200."""
        from server.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_metadata_endpoint(self):
        """Metadata endpoint should return environment info."""
        from server.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/metadata")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_tasks_endpoint(self):
        """Tasks endpoint should return 3 tasks."""
        from server.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/tasks")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["tasks"]) == 3

    def test_schema_endpoint(self):
        """Schema endpoint should return action schema."""
        from server.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/schema")

        assert response.status_code == 200
        data = response.json()
        assert "action" in data
        assert "observation" in data
        assert "state" in data


class TestLeaderboardEndpoints:
    """Tests for leaderboard endpoints."""

    def test_leaderboard_empty(self):
        """Empty leaderboard should return empty list."""
        from server.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/leaderboard")

        assert response.status_code == 200

    def test_leaderboard_submit(self):
        """Leaderboard submit should work."""
        from server.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post("/leaderboard/submit", json={
            "model_name": "test_model_endpoint",
            "avg_reward": 0.75,
            "avg_accuracy": 0.80,
            "hallucination_rate": 0.15,
            "total_episodes": 10,
            "total_steps": 50
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "submitted"

    def test_leaderboard_viz(self):
        """Leaderboard visualization should return chart data."""
        from server.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/leaderboard/viz")

        assert response.status_code == 200
        data = response.json()
        assert "bar_chart" in data
        assert "scatter_plot" in data
        assert "performance_tiers" in data


class TestOpenEnvEndpoints:
    """Tests for OpenEnv required endpoints."""

    def test_baseline_endpoint(self):
        """Baseline endpoint should run and return scores."""
        from server.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post("/baseline", json={
            "steps_per_task": 2,
            "seed": 42
        })

        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert "summary" in data
        assert data["summary"]["total_tasks"] == 3