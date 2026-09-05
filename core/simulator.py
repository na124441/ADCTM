"""
Core physics simulation orchestration logic module.
Encapsulates environment lifecycles internally managing transitions mapped continuously 
by discrete time steps tracking iteration bounds safely.
"""
from __future__ import annotations

import json
import numpy as np
from typing import Any, Dict, Optional

from core.models import Action, Observation
from core.paths import TASKS_DIR
from dynamics.thermal_model import apply_transition
from grader.evaluator import evaluate_trajectory
from reward.reward_fn import compute_reward
from tasks.task_config import TaskConfig


class SimulationSession:
    """
    A unified wrapper containing the full environment state logic. 
    Maintains all observation iterations, physical constraints, and termination rules locally.
    """
    def __init__(self, config: TaskConfig):
        """
        Initialization assigning the rigid task environment structure globally.
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.observation = Observation(
            temperatures=config.initial_temperatures,
            workloads=config.initial_workloads,
            cooling=[0.0] * config.num_zones,
            ambient_temp=config.ambient_temperature,
            time_step=0,
            target_temperature=config.target_temperature,
            safe_temperature=config.safe_temperature,
        )
        self.step_counter = 0
        self.done = False
        self.is_canonical = False
        self.task_name: Optional[str] = None
        self.history_obs = [self.observation.model_dump()]
        self.history_actions = []

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the full internal session snapshot mapped functionally.
        Used extensively for API consumption payloads seamlessly.
        """
        return {
            "config": self.config.model_dump(),
            "observation": self.observation.model_dump(),
            "step_counter": self.step_counter,
            "done": self.done,
            "seed": self.config.seed,
            "rng_state": self.rng.bit_generator.state,
        }

    @classmethod
    def from_dict(cls, task_config: Dict[str, Any]) -> "SimulationSession":
        """
        Alternate constructor passing raw dictionary definitions towards TaskConfig validations natively.
        """
        return cls(TaskConfig.model_validate(task_config))

    @classmethod
    def from_task_name(cls, task_name: str = "easy", seed: Optional[int] = None) -> "SimulationSession":
        """
        Syntactic constructor loading authorized benchmark tasks from the canonical tasks directory.
        """
        clean_name = task_name.replace(".json", "").strip().lower()
        if clean_name not in {"easy", "medium", "hard"}:
            raise ValueError(f"Task '{task_name}' is not an authorized canonical task.")
        task_file = TASKS_DIR / f"{clean_name}.json"
        with task_file.open(encoding="utf-8") as handle:
            cfg_dict = json.load(handle)
        if seed is not None:
            cfg_dict["seed"] = seed
        session = cls.from_dict(cfg_dict)
        session.task_name = clean_name
        session.is_canonical = True
        return session

    def step(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main transition function pushing physics simulations forward.
        Resolves input actions, manages state loops calculating reward variables incrementally.
        
        Args:
            action_dict: Target cooling commands map.
        
        Returns:
            Dict mapping next physical observations, resulting rewards structures and termination statuses dynamically.
        """
        # Short-circuit logic check blocking attempts to actuate against terminated scenarios securely.
        if self.done:
            raise ValueError("Episode already finished. Reset before stepping again.")

        # Cast raw inputs securely parsing Pydantic typing definitions strictly mapping parameters locally constraint wise.
        action = Action.model_validate(action_dict)
        if len(action.cooling) != self.config.num_zones:
            raise ValueError(
                f"Action length ({len(action.cooling)}) does not match num_zones ({self.config.num_zones})"
            )

        # Retain history
        prev_obs = self.observation
        
        # Extrapolate physical heat generation formula mathematically.
        new_obs = apply_transition(prev_obs, action, self.config, self.rng)
        
        # Calculate grade scoring mechanics for current action iteration mapped sequentially.
        reward_obj = compute_reward(
            prev_obs=prev_obs,
            curr_obs=new_obs,
            act=action,
            config=self.config,
        )

        # Mutate internal state
        self.observation = new_obs
        self.step_counter += 1
        self.history_obs.append(new_obs.model_dump())
        self.history_actions.append(action_dict)
        
        # Compare iteration lengths defining termination boundaries securely
        self.done = self.step_counter >= self.config.max_steps

        # Transmit standard OpenAI integration output format dictionary natively
        return {
            "observation": new_obs.model_dump(),
            "reward": reward_obj.model_dump(),
            "done": self.done,
            "info": {"step": self.step_counter},
        }

    def get_score(self) -> Dict[str, Any]:
        """
        Calculates and returns the trajectory score breakdown for the active session.
        """
        if len(self.history_actions) == 0:
            return {
                "total": 0.0,
                "score": 0.0,
                "metrics": {"safety": 0.0, "precision": 0.0, "efficiency": 0.0, "smoothness": 0.0},
                "status": "no_steps_executed"
            }
        details = evaluate_trajectory(self.history_obs, self.history_actions, self.config, return_details=True)
        return {
            "total": details["score"],
            "score": details["score"],
            "metrics": {
                "safety": details["safety"],
                "precision": details["target"],
                "efficiency": details["energy"],
                "smoothness": details["jitter"]
            }
        }

    def model_dump(self) -> Dict[str, Any]:
        """
        Utility mapping redirect abstracting class instance states safely.
        """
        return self.get_state()
