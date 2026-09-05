import numpy as np

from config.constants import ALPHA, BETA, GAMMA, MAX_PHYSICAL_TEMPERATURE, KAPPA_DIFFUSION
from core.models import Action, Observation
from tasks.task_config import TaskConfig


def apply_transition(
    obs: Observation, 
    act: Action, 
    config: TaskConfig, 
    rng: np.random.Generator
) -> Observation:
    """
    Apply one simulation step and return the next observation.
    Uses localized deterministic random generation.
    """
    temperatures = np.array(obs.temperatures)
    workloads = np.array(obs.workloads)
    cooling = np.array(act.cooling)
    ambient_temp = obs.ambient_temp

    workloads = workloads + rng.uniform(
        -config.workload_volatility,
        config.workload_volatility,
        size=len(workloads),
    )
    workloads = np.clip(workloads, 0.0, 1.0)

    cooling_effect = np.full(len(workloads), BETA)
    if config.degradation_step is not None and config.degraded_zone is not None:
        if obs.time_step >= config.degradation_step:
            cooling_effect[config.degraded_zone] = BETA * 0.5

    # Discrete 1D Laplacian thermal diffusion between adjacent rack zones (Neumann boundary conditions)
    diffusion = np.zeros_like(temperatures)
    if len(temperatures) > 1:
        diffusion[0] = temperatures[1] - temperatures[0]
        diffusion[-1] = temperatures[-2] - temperatures[-1]
        if len(temperatures) > 2:
            diffusion[1:-1] = temperatures[:-2] + temperatures[2:] - 2.0 * temperatures[1:-1]
        diffusion = KAPPA_DIFFUSION * diffusion

    delta_t = (
        ALPHA * workloads 
        - cooling_effect * cooling 
        + GAMMA * (ambient_temp - temperatures)
        + diffusion
    )
    
    # Floor at ambient temperature and clamp at physical silicon thermal ceiling
    next_temperatures = np.clip(temperatures + delta_t, ambient_temp, MAX_PHYSICAL_TEMPERATURE)

    return Observation(
        temperatures=next_temperatures.tolist(),
        workloads=workloads.tolist(),
        cooling=cooling.tolist(),
        ambient_temp=ambient_temp,
        time_step=obs.time_step + 1,
    )
