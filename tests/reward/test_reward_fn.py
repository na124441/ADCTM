import pytest

from core.models import Action, Observation
from reward.reward_fn import compute_reward


def test_compute_reward_zero_penalty_on_safe_first_step(task_config):
    prev_obs = Observation(
        temperatures=[60.0, 61.0, 62.0],
        workloads=[0.2, 0.2, 0.2],
        cooling=[0.0, 0.0, 0.0],
        ambient_temp=22.0,
        time_step=0,
    )
    curr_obs = prev_obs.model_copy(update={"time_step": 1})
    reward = compute_reward(prev_obs, curr_obs, Action(cooling=[0.0, 0.0, 0.0]), task_config)
    assert reward.value < 0.0
    assert reward.value == pytest.approx(-0.034846938775510206)


def test_compute_reward_penalizes_violations_energy_and_jitter(task_config):
    prev_obs = Observation(
        temperatures=[60.0, 60.0, 60.0],
        workloads=[0.2, 0.2, 0.2],
        cooling=[0.1, 0.1, 0.1],
        ambient_temp=22.0,
        time_step=1,
    )
    curr_obs = Observation(
        temperatures=[86.0, 87.0, 84.0],
        workloads=[0.2, 0.2, 0.2],
        cooling=[0.5, 0.6, 0.7],
        ambient_temp=22.0,
        time_step=2,
    )
    reward = compute_reward(prev_obs, curr_obs, Action(cooling=[0.5, 0.6, 0.7]), task_config)
    assert reward.value < 0.0


def test_compute_reward_bypasses_jitter_near_safe_limit(task_config):
    prev_obs = Observation(
        temperatures=[80.0, 80.0, 80.0],
        workloads=[0.2, 0.2, 0.2],
        cooling=[0.0, 0.0, 0.0],
        ambient_temp=22.0,
        time_step=1,
    )
    curr_obs = Observation(
        temperatures=[84.5, 84.0, 83.0],
        workloads=[0.2, 0.2, 0.2],
        cooling=[1.0, 1.0, 1.0],
        ambient_temp=22.0,
        time_step=2,
    )
    reward = compute_reward(prev_obs, curr_obs, Action(cooling=[1.0, 1.0, 1.0]), task_config)
    assert reward.value == pytest.approx(-3.0933673469387757)


def test_compute_reward_jitter_bypass_is_per_zone(task_config):
    # task_config has safe_temperature=85.0, jitter_bypass_threshold=2.0 (threshold = 83.0)
    # Zone 0 is hot (84.0 >= 83.0), Zone 1 is cool (60.0 < 83.0), Zone 2 is cool (60.0 < 83.0)
    prev_obs = Observation(
        temperatures=[70.0, 60.0, 60.0],
        workloads=[0.2, 0.2, 0.2],
        cooling=[0.0, 0.0, 0.0],
        ambient_temp=22.0,
        time_step=1,
    )
    curr_obs = Observation(
        temperatures=[84.0, 60.0, 60.0],
        workloads=[0.2, 0.2, 0.2],
        cooling=[0.5, 0.5, 0.0],
        ambient_temp=22.0,
        time_step=2,
    )
    # Zone 0 jumps 0.0 -> 0.5 (exempt because temp 84.0 >= 83.0)
    # Zone 1 jumps 0.0 -> 0.5 (must pay jitter: 0.5)
    # Zone 2 jumps 0.0 -> 0.0 (jitter: 0.0)
    # Total jitter = 0.5 (with old bug, Zone 0 would zero out jitter for Zone 1 as well, yielding total jitter=0.0)
    reward = compute_reward(prev_obs, curr_obs, Action(cooling=[0.5, 0.5, 0.0]), task_config)
    
    # Calculate expected cost components:
    # temp_penalty: max(0, 84 - 85)^2 = 0
    # energy_cost: 0.5 + 0.5 = 1.0 (weight 1.0 -> 1.0)
    # jitter: 0.5 (weight 0.5 -> 0.25)
    # target tracking: compute_normalized_error for [84, 60, 60] with target=70
    # total cost must be greater by exactly 0.25 compared to a scenario where jitter is 0.0
    reward_no_zone1_jitter = compute_reward(
        prev_obs.model_copy(update={"cooling": [0.0, 0.5, 0.0]}),
        curr_obs,
        Action(cooling=[0.5, 0.5, 0.0]),
        task_config,
    )
    # Difference must be exactly lambda_3 * 0.5 = 0.5 * 0.5 = 0.25
    assert abs((reward_no_zone1_jitter.value - reward.value) - 0.25) < 1e-6
