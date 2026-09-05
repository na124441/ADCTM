import pytest
import numpy as np

from core.models import Action, Observation
from dynamics.thermal_model import apply_transition
from tasks.task_config import TaskConfig


def test_apply_transition_is_deterministic_without_volatility(task_config, observation):
    action = Action(cooling=[0.2, 0.4, 0.6])
    next_obs = apply_transition(observation, action, task_config, np.random.default_rng(42))
    assert next_obs.time_step == 1
    assert next_obs.cooling == [0.2, 0.4, 0.6]
    assert next_obs.workloads == pytest.approx(observation.workloads)


def test_apply_transition_clips_workloads(monkeypatch, task_config, observation):
    task_config = task_config.model_copy(update={"workload_volatility": 0.5})
    class MockRNG:
        def uniform(self, low, high, size):
            return [1.0, -1.0, 0.0]
    next_obs = apply_transition(observation, Action(cooling=[0.0, 0.0, 0.0]), task_config, MockRNG())
    assert next_obs.workloads == [1.0, 0.0, 0.3]


def test_apply_transition_never_drops_below_ambient(task_config):
    obs = Observation(
        temperatures=[22.0, 22.1, 22.2],
        workloads=[0.0, 0.0, 0.0],
        cooling=[0.0, 0.0, 0.0],
        ambient_temp=22.0,
        time_step=0,
    )
    next_obs = apply_transition(obs, Action(cooling=[1.0, 1.0, 1.0]), task_config, np.random.default_rng(42))
    assert min(next_obs.temperatures) >= 22.0


def test_apply_transition_applies_degradation_after_threshold(base_task_dict):
    base_task_dict.update({"degradation_step": 1, "degraded_zone": 1})
    config = TaskConfig.model_validate(base_task_dict)
    obs = Observation(
        temperatures=[45.0, 48.0, 50.0],
        workloads=[0.4, 0.5, 0.3],
        cooling=[0.0, 0.0, 0.0],
        ambient_temp=22.0,
        time_step=1,
    )
    cooled = apply_transition(obs, Action(cooling=[0.5, 0.5, 0.5]), config, np.random.default_rng(42))
    assert cooled.temperatures[1] > cooled.temperatures[0]


def test_apply_transition_enforces_upper_temperature_bound(task_config):
    # Under extreme workload and zero cooling, temperatures must not exceed MAX_PHYSICAL_TEMPERATURE (105.0°C)
    # delta_t = 7.5 * 1.0 - 0 + 0.1 * (25 - 100) = 7.5 - 7.5 = 0
    # To have delta_t > 0 with ambient=25: 7.5 * 1.0 > 0.1 * (T - 25) => T - 25 < 75 => T < 100
    # If initial T is 104.0 and ambient is 60.0: delta_t = 7.5 * 1.0 + 0.1 * (60 - 104) = 7.5 - 4.4 = +3.1 => T = 107.1 -> clamped to 105.0
    obs = Observation(
        temperatures=[104.0, 105.0, 110.0],
        workloads=[1.0, 1.0, 1.0],
        cooling=[0.0, 0.0, 0.0],
        ambient_temp=60.0,
        time_step=0,
    )
    next_obs = apply_transition(obs, Action(cooling=[0.0, 0.0, 0.0]), task_config, np.random.default_rng(42))
    assert max(next_obs.temperatures) <= 105.0
    assert next_obs.temperatures == [105.0, 105.0, 105.0]


def test_apply_transition_diffuses_heat_between_adjacent_zones(task_config):
    # Setup 3 zones with zero workload and zero cooling at ambient 20°C:
    # Zone 0 is hot (80°C), Zone 1 is cold (20°C), Zone 2 is cold (20°C)
    # Heat should conductively diffuse: Zone 0 -> Zone 1, raising Zone 1 above ambient.
    # Zone 2 is not adjacent to Zone 0, so it receives less heat than Zone 1.
    obs = Observation(
        temperatures=[80.0, 20.0, 20.0],
        workloads=[0.0, 0.0, 0.0],
        cooling=[0.0, 0.0, 0.0],
        ambient_temp=20.0,
        time_step=0,
    )
    next_obs = apply_transition(obs, Action(cooling=[0.0, 0.0, 0.0]), task_config, np.random.default_rng(42))
    
    t0, t1, t2 = next_obs.temperatures
    # Zone 0 cools due to ambient dissipation AND diffusion to Zone 1
    assert t0 < 80.0
    # Zone 1 heats up above ambient due to thermal diffusion from hot neighbor Zone 0
    assert t1 > 20.0
    # Zone 1 is hotter than Zone 2 because Zone 1 is directly adjacent to Zone 0
    assert t1 > t2



