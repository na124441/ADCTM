"""
Security and Integrity Tests for ADCTM API Endpoints.
Guarantees C3 remediation:
1. Rejects injected TaskConfig parameters (e.g. safe_temperature, max_steps).
2. Rejects unauthorized / custom task names.
3. Accepts legitimate canonical tasks (easy, medium, hard) with optional seeds.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_reset_accepts_canonical_tasks(client):
    # Test easy, medium, hard via body and query params
    res_easy = client.post("/reset", json={"task_name": "easy"})
    assert res_easy.status_code == 200
    assert "temperatures" in res_easy.json()

    res_medium = client.post("/reset", json={"task_name": "medium", "seed": 999})
    assert res_medium.status_code == 200

    res_hard = client.post("/reset", params={"task_name": "hard"})
    assert res_hard.status_code == 200


def test_reset_rejects_injected_configuration(client):
    # Attacker tries to inject safe_temperature=9999 to game evaluation
    malicious_payload = {
        "task_name": "easy",
        "safe_temperature": 9999.0,
        "max_steps": 1,
    }
    response = client.post("/reset", json=malicious_payload)
    assert response.status_code == 422  # Extra fields forbidden


def test_reset_rejects_arbitrary_task_names(client):
    # Attacker tries to specify a non-existent or path traversal task
    bad_payload = {"task_name": "arbitrary_custom_eval"}
    response = client.post("/reset", json=bad_payload)
    assert response.status_code == 422

    bad_query = client.post("/reset", params={"task_name": "../../etc/passwd"})
    assert bad_query.status_code == 422
