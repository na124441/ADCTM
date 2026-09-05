"""
Standard Gymnasium Environment wrapper for ADCTM.
Exposes standard Box observation and action spaces for RL algorithms (PPO, SAC, TD3)
and wraps the in-process SimulationSession physics cleanly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from core.paths import TASKS_DIR
from core.simulator import SimulationSession
from tasks.task_config import TaskConfig
from grader.evaluator import evaluate_trajectory


class ADCTMGymEnv(gym.Env):
    """
    Gymnasium interface for ADCTM simulation.

    Observation vector layout (size = num_zones * 3 + 4):
      - Normalized Temperatures: (T - target_temp) / 50.0  [num_zones]
      - Workloads: [0, 1]                                  [num_zones]
      - Previous Cooling: [0, 1]                           [num_zones]
      - Ambient Temperature normalized: (ambient - 20) / 40
      - Step progress: time_step / max_steps
      - Safe temperature offset: (safe_temp - target_temp) / 50.0
      - Target temperature normalized: target_temp / 100.0

    Action vector layout (size = num_zones):
      - Continuous cooling commands in [0.0, 1.0] per zone.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        task_name: str = "easy",
        config_override: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.task_name = task_name

        if config_override is not None:
            self.base_config_dict = config_override
        else:
            task_file = TASKS_DIR / f"{task_name.replace('.json', '')}.json"
            if not task_file.exists():
                raise FileNotFoundError(f"Task configuration not found: {task_file}")
            with task_file.open(encoding="utf-8") as handle:
                self.base_config_dict = json.load(handle)

        self.config = TaskConfig.model_validate(self.base_config_dict)
        self.num_zones = self.config.num_zones

        # Action: Continuous cooling level per zone in [0.0, 1.0]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_zones,),
            dtype=np.float32,
        )

        # Observation dimension: 3 per-zone series + 4 scalar context features
        obs_dim = self.num_zones * 3 + 4
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.session: Optional[SimulationSession] = None

    def _get_obs(self) -> np.ndarray:
        obs = self.session.observation
        cfg = self.session.config

        # 1. Per-zone normalized relative temperatures
        norm_temps = [(t - cfg.target_temperature) / 50.0 for t in obs.temperatures]
        # 2. Per-zone workloads in [0, 1]
        workloads = list(obs.workloads)
        # 3. Per-zone previous cooling in [0, 1]
        prev_cooling = list(obs.cooling)
        # 4. Context scalars
        ambient_norm = (obs.ambient_temp - 20.0) / 40.0
        step_progress = obs.time_step / max(1, cfg.max_steps)
        safe_margin = (cfg.safe_temperature - cfg.target_temperature) / 50.0
        target_norm = cfg.target_temperature / 100.0

        raw_vec = norm_temps + workloads + prev_cooling + [
            ambient_norm,
            step_progress,
            safe_margin,
            target_norm,
        ]
        return np.array(raw_vec, dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        cfg_dict = dict(self.base_config_dict)
        if seed is not None:
            cfg_dict["seed"] = seed
        if options and "task_name" in options:
            task_file = TASKS_DIR / f"{options['task_name'].replace('.json', '')}.json"
            with task_file.open(encoding="utf-8") as handle:
                cfg_dict = json.load(handle)
            if seed is not None:
                cfg_dict["seed"] = seed

        self.config = TaskConfig.model_validate(cfg_dict)
        self.num_zones = self.config.num_zones
        self.session = SimulationSession(self.config)

        obs = self._get_obs()
        info = {
            "step": 0,
            "temperatures": list(self.session.observation.temperatures),
            "target_temperature": self.config.target_temperature,
            "safe_temperature": self.config.safe_temperature,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.session is None:
            raise RuntimeError("Environment step called before reset.")

        # Ensure action is clipped and formatted for SimulationSession
        clamped_action = [float(np.clip(a, 0.0, 1.0)) for a in action]
        step_result = self.session.step({"cooling": clamped_action})

        obs = self._get_obs()
        reward = float(step_result["reward"]["value"])
        terminated = bool(step_result["done"])
        truncated = False

        info = {
            "step": self.session.step_counter,
            "raw_observation": step_result["observation"],
            "done": terminated,
        }

        # Include trajectory evaluation score if episode terminated
        if terminated:
            score_details = evaluate_trajectory(
                self.session.history_obs,
                self.session.history_actions,
                self.config,
                return_details=True,
            )
            info["evaluation"] = score_details

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.session:
            temps = [f"{t:.1f}°C" for t in self.session.observation.temperatures]
            print(f"[Step {self.session.step_counter}] Temps: {temps}")
