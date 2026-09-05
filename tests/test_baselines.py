"""
Tests for ADCTM Gymnasium Environment Adapter and Classical Baselines.
Protects C1 remediation by enforcing behavioral contracts on all controllers and RL env.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.gym_env import ADCTMGymEnv
from baselines.classical import ZeroController, RuleBasedController, PIDController
from baselines.rl_agent import RLAgent
from tasks.task_config import TaskConfig


@pytest.fixture
def easy_env():
    return ADCTMGymEnv(task_name="easy")


def test_gym_env_space_contracts(easy_env):
    """Verifies observation and action space alignment and bounds."""
    obs, info = easy_env.reset(seed=123)
    assert obs.shape == (easy_env.num_zones * 3 + 4,)
    assert obs.dtype == np.float32
    assert "target_temperature" in info
    assert "safe_temperature" in info

    # Sample an action and step
    action = easy_env.action_space.sample()
    next_obs, reward, term, trunc, step_info = easy_env.step(action)
    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert not trunc
    assert "step" in step_info


def test_zero_controller_contract():
    controller = ZeroController()
    obs = {"temperatures": [65.0, 70.0, 75.0]}
    cfg = {"safe_temperature": 80.0, "target_temperature": 60.0}
    action = controller.act(obs, cfg)
    assert action == [0.0, 0.0, 0.0]


def test_rule_based_controller_safety_override():
    controller = RuleBasedController(high_margin=2.0)
    # Zone 1 safe, Zone 2 near threshold (78 >= 80 - 2), Zone 3 below target
    obs = {"temperatures": [65.0, 78.5, 55.0]}
    cfg = {"safe_temperature": 80.0, "target_temperature": 60.0}
    action = controller.act(obs, cfg)

    assert len(action) == 3
    assert action[1] == 1.0  # Must invoke maximum cooling
    assert action[2] < action[0]  # Cold zone cools less than warm zone


def test_pid_controller_anti_windup_and_bounds():
    pid = PIDController(kp=0.1, ki=0.05, kd=0.02)
    obs = {"temperatures": [75.0, 75.0, 75.0]}
    cfg = {"safe_temperature": 85.0, "target_temperature": 65.0}

    for _ in range(50):
        action = pid.act(obs, cfg)
        for a in action:
            assert 0.0 <= a <= 1.0  # Must respect physical limits

    # Integral state must be clamped
    assert np.all(np.abs(pid.integral) <= pid.integral_limit)


def test_rl_agent_fallback_behavior():
    agent = RLAgent(model_path="non_existent_model.zip")
    obs = {
        "temperatures": [65.0, 70.0, 75.0],
        "workloads": [0.5, 0.5, 0.5],
        "cooling": [0.3, 0.3, 0.3],
        "ambient_temp": 30.0,
    }
    cfg = {"safe_temperature": 80.0, "target_temperature": 60.0}
    action = agent.act(obs, cfg)
    assert len(action) == 3
    for a in action:
        assert 0.0 <= a <= 1.0


def test_multi_seed_evaluation_computes_statistics():
    from run_benchmark import evaluate_agent_on_task
    controller = RuleBasedController()
    seeds = [42, 101, 202]
    res = evaluate_agent_on_task(controller, "easy", seeds=seeds)
    assert "score_mean" in res
    assert "score_std" in res
    assert "safety_mean" in res
    assert 0.0 <= res["score_mean"] <= 1.0
    assert res["score_std"] >= 0.0


