import pytest

from grader.evaluator import evaluate_trajectory


def test_evaluate_trajectory_combines_metrics(monkeypatch, task_config):
    # safety=0.8, energy=0.25, jitter=0.1, target_error=0.05
    monkeypatch.setattr("grader.evaluator.compute_metrics", lambda observations, actions, config: (0.8, 0.25, 0.1, 0.05))
    score = evaluate_trajectory([], [], task_config)
    import math
    safety_factor = math.sqrt(0.8)
    expected_score = 0.4 * 0.8 + 0.3 * 0.95 + 0.2 * (0.75 * safety_factor) + 0.1 * (0.9 * safety_factor)
    assert score == pytest.approx(expected_score)


def test_zero_cooling_does_not_get_unearned_efficiency_bonus(monkeypatch, task_config):
    # Overheating scenario: safety = 0.0, zero cooling (energy = 0.0, jitter = 0.0)
    monkeypatch.setattr("grader.evaluator.compute_metrics", lambda observations, actions, config: (0.0, 0.0, 0.0, 0.5))
    details = evaluate_trajectory([], [], task_config, return_details=True)
    # Energy and jitter must be scaled to 0.0 because safety failed completely
    assert details["energy"] == 0.0
    assert details["jitter"] == 0.0
    assert details["score"] < 0.2


def test_evaluate_trajectory_clamps_score(monkeypatch, task_config):
    monkeypatch.setattr("grader.evaluator.compute_metrics", lambda observations, actions, config: (0.0, 5.0, 5.0, 2.0))
    assert evaluate_trajectory([], [], task_config) == 0.0
