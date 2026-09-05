"""
RL Agent baseline wrapper for inference and benchmarking.
Loads trained Stable-Baselines3 PPO model and interfaces with ADCTM tasks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from baselines.classical import BaseController


class RLAgent(BaseController):
    """
    PPO Reinforcement Learning Policy agent.
    Interfaces with trained Stable-Baselines3 weights or falls back to PID if uninitialized.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path and Path(model_path).exists():
            try:
                from stable_baselines3 import PPO

                self.model = PPO.load(model_path)
            except Exception as e:
                print(f"[RLAgent] Warning loading model from {model_path}: {e}")
                self.model = None

    def reset(self) -> None:
        pass

    def act_from_obs_vec(self, obs_vec: np.ndarray) -> List[float]:
        if self.model is not None:
            action, _ = self.model.predict(obs_vec, deterministic=True)
            return [float(np.clip(a, 0.0, 1.0)) for a in action]
        # Fallback default
        return [0.3]

    def act(self, obs_dict: Dict[str, Any], config_dict: Dict[str, Any]) -> List[float]:
        num_zones = len(obs_dict["temperatures"])
        if self.model is None:
            return [0.3] * num_zones

        target_temp = float(config_dict["target_temperature"])
        safe_temp = float(config_dict["safe_temperature"])
        max_steps = float(config_dict.get("max_steps", 20))

        norm_temps = [(t - target_temp) / 50.0 for t in obs_dict["temperatures"]]
        workloads = list(obs_dict["workloads"])
        prev_cooling = list(obs_dict["cooling"])
        ambient_norm = (obs_dict["ambient_temp"] - 20.0) / 40.0
        step_progress = obs_dict.get("time_step", 0) / max(1.0, max_steps)
        safe_margin = (safe_temp - target_temp) / 50.0
        target_norm = target_temp / 100.0

        vec = np.array(
            norm_temps
            + workloads
            + prev_cooling
            + [ambient_norm, step_progress, safe_margin, target_norm],
            dtype=np.float32,
        )

        action, _ = self.model.predict(vec, deterministic=True)
        return [float(np.clip(a, 0.0, 1.0)) for a in action]
