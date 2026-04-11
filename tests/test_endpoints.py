"""Tests for the FastAPI server endpoints."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient


# Skip entire module if environment can't be created (e.g. missing datasets)
# This allows CI to run without HF dataset access
pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_SERVER_TESTS", "1") == "1",
    reason="Set SKIP_SERVER_TESTS=0 to run server tests (requires dataset access)"
)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        from server.app import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestTasksEndpoint:
    def test_tasks_returns_three_tasks(self):
        from server.app import app
        client = TestClient(app)
        resp = client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["tasks"]) == 3
        task_ids = [t["task_id"] for t in data["tasks"]]
        assert "task_1_factual_grounding" in task_ids
        assert "task_2_multi_hop_synthesis" in task_ids
        assert "task_3_adversarial_resistance" in task_ids

    def test_tasks_has_action_schema(self):
        from server.app import app
        client = TestClient(app)
        resp = client.get("/tasks")
        data = resp.json()
        assert "action_schema" in data
        assert "answer" in data["action_schema"]["properties"]


class TestGraderEndpoint:
    def test_grader_requires_task_id(self):
        from server.app import app
        client = TestClient(app)
        resp = client.post("/grader", json={"step_rewards": [0.5], "step_infos": [{}]})
        assert resp.status_code == 422

    def test_grader_invalid_task_id(self):
        from server.app import app
        client = TestClient(app)
        resp = client.post("/grader", json={
            "task_id": "nonexistent",
            "step_rewards": [0.5],
            "step_infos": [{}],
        })
        assert resp.status_code == 404

    def test_grader_returns_score(self):
        from server.app import app
        client = TestClient(app)
        resp = client.post("/grader", json={
            "task_id": "task_1_factual_grounding",
            "step_rewards": [0.7, 0.5, 0.3],
            "step_infos": [
                {"correctness": 0.7, "grounding": 0.6, "calibration": 0.8,
                 "hallucination_score": 0.1, "is_hallucination": False},
                {"correctness": 0.5, "grounding": 0.4, "calibration": 0.7,
                 "hallucination_score": 0.2, "is_hallucination": False},
                {"correctness": 0.3, "grounding": 0.3, "calibration": 0.6,
                 "hallucination_score": 0.5, "is_hallucination": True},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["score"] <= 1.0
        assert "breakdown" in data


class TestMetadataEndpoint:
    def test_metadata(self):
        from server.app import app
        client = TestClient(app)
        resp = client.get("/metadata")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "hallucination-guard-env"
        assert "version" in data