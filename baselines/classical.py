"""
Classical Baseline Controllers for ADCTM.
Includes:
- ZeroPolicy: Passive/do-nothing lower anchor
- RuleBasedController: Threshold-driven heuristic controller
- PIDController: Multi-zone PID controller with anti-windup clamping
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np


class BaseController:
    """Abstract interface for all baseline controllers."""

    def reset(self) -> None:
        pass

    def act(self, obs_dict: Dict[str, Any], config_dict: Dict[str, Any]) -> List[float]:
        raise NotImplementedError


class ZeroController(BaseController):
    """
    Lower anchor baseline: Applies zero cooling across all zones.
    Useful for demonstrating passive thermal dynamics.
    """

    def act(self, obs_dict: Dict[str, Any], config_dict: Dict[str, Any]) -> List[float]:
        num_zones = len(obs_dict["temperatures"])
        return [0.0] * num_zones


class RuleBasedController(BaseController):
    """
    Heuristic, threshold-based reactive controller.
    Ramps up cooling aggressively as temperature approaches safe limits,
    reduces cooling when below target to conserve energy.
    """

    def __init__(self, high_margin: float = 3.0, low_margin: float = 2.0):
        self.high_margin = high_margin
        self.low_margin = low_margin

    def act(self, obs_dict: Dict[str, Any], config_dict: Dict[str, Any]) -> List[float]:
        temps = obs_dict["temperatures"]
        safe_temp = config_dict["safe_temperature"]
        target_temp = config_dict["target_temperature"]

        actions = []
        for t in temps:
            # Urgent zone safety override: near or exceeding safe temp
            if t >= safe_temp - self.high_margin:
                actions.append(1.0)
            elif t >= safe_temp - (self.high_margin * 2):
                actions.append(0.75)
            elif t > target_temp:
                # Proportional intermediate band
                span = max(1.0, (safe_temp - self.high_margin * 2) - target_temp)
                ratio = (t - target_temp) / span
                actions.append(float(np.clip(0.3 + 0.4 * ratio, 0.2, 0.7)))
            elif t <= target_temp - self.low_margin:
                # Well below target: drop cooling to minimum to save energy
                actions.append(0.05)
            else:
                # Steady-state hover around target
                actions.append(0.25)

        return actions


class PIDController(BaseController):
    """
    Decoupled per-zone PID controller with anti-windup clamping.
    Regulates zone temperature toward target_temperature while maintaining stability.
    """

    def __init__(
        self,
        kp: float = 0.08,
        ki: float = 0.015,
        kd: float = 0.04,
        integral_limit: float = 10.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit

        self.integral: Optional[np.ndarray] = None
        self.prev_error: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.integral = None
        self.prev_error = None

    def act(self, obs_dict: Dict[str, Any], config_dict: Dict[str, Any]) -> List[float]:
        temps = np.array(obs_dict["temperatures"], dtype=float)
        target = float(config_dict["target_temperature"])
        safe = float(config_dict["safe_temperature"])

        # Error: positive when zone is hotter than target
        error = temps - target

        if self.integral is None:
            self.integral = np.zeros_like(error)
            self.prev_error = np.copy(error)

        # Update integral term with anti-windup saturation
        self.integral += error
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        # Derivative term
        derivative = error - self.prev_error
        self.prev_error = np.copy(error)

        # Nominal PID output centered around baseline cooling (0.3)
        base_cooling = 0.3
        u = base_cooling + self.kp * error + self.ki * self.integral + self.kd * derivative

        # Proactive safety override if any zone gets dangerously close to safe threshold
        danger_mask = temps >= (safe - 2.0)
        u = np.where(danger_mask, np.maximum(u, 0.9), u)

        # Mechanical clamping to [0.0, 1.0]
        clamped = np.clip(u, 0.0, 1.0)
        return clamped.tolist()
