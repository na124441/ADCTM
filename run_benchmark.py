"""
Comprehensive Multi-Seed Benchmark Evaluator for ADCTM.
Evaluates:
  1. Zero Policy (lower anchor)
  2. Rule-Based Controller
  3. PID Controller
  4. LLM Agent Fallback (Proportional controller matching inference.py)
  5. PPO Reinforcement Learning Policy
Across:
  - easy, medium, hard
Across multiple distinct random seeds.
Outputs genuine statistical performance metrics (mean +/- std) ready for README.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from baselines.classical import ZeroController, RuleBasedController, PIDController
from baselines.rl_agent import RLAgent
from core.simulator import SimulationSession
from core.paths import TASKS_DIR
from grader.evaluator import evaluate_trajectory
from tasks.task_config import TaskConfig


class ProportionalFallbackController:
    """Proportional policy directly matching inference.py predict_action logic."""

    def act(self, obs_dict: Dict[str, Any], config_dict: Dict[str, Any]) -> List[float]:
        temps = obs_dict.get("temperatures", [])
        target = config_dict.get("target_temperature", 35.0)
        return [float(np.clip((t - target) / 20.0, 0.0, 1.0)) for t in temps]


def evaluate_agent_on_task(
    agent: Any,
    task_name: str,
    seeds: List[int],
) -> Dict[str, Any]:
    task_file = TASKS_DIR / f"{task_name}.json"
    with task_file.open(encoding="utf-8") as f:
        base_cfg = json.load(f)

    scores = []
    safeties = []
    targets = []
    energies = []
    jitters = []

    for s in seeds:
        cfg_dict = dict(base_cfg)
        cfg_dict["seed"] = s
        cfg = TaskConfig.model_validate(cfg_dict)

        session = SimulationSession(cfg)
        if hasattr(agent, "reset"):
            agent.reset()

        while not session.done:
            obs = session.observation.model_dump()
            action = agent.act(obs, cfg.model_dump())
            session.step({"cooling": action})

        eval_res = evaluate_trajectory(
            session.history_obs,
            session.history_actions,
            session.config,
            return_details=True,
        )

        scores.append(eval_res["score"])
        safeties.append(eval_res["safety"])
        targets.append(eval_res["target"])
        energies.append(eval_res["energy"])
        jitters.append(eval_res["jitter"])

    return {
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "safety_mean": float(np.mean(safeties)),
        "target_mean": float(np.mean(targets)),
        "energy_mean": float(np.mean(energies)),
        "jitter_mean": float(np.mean(jitters)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run ADCTM Baseline Benchmark")
    parser.add_argument("--episodes", type=int, default=5, help="Number of seeds per task")
    args = parser.parse_args()

    seeds = [100 + i * 17 for i in range(args.episodes)]
    tasks = ["easy", "medium", "hard"]

    print("=" * 80)
    print(" ADCTM EMPIRICAL BASELINE BENCHMARK")
    print(f" Episodes per task: {args.episodes} | Evaluation Seeds: {seeds}")
    print("=" * 80)

    # Initialize all 5 distinct controller paradigms
    agents = {
        "Zero (Passive)": ZeroController(),
        "Rule-Based": RuleBasedController(),
        "PID Controller": PIDController(),
        "LLM Agent (Eval)": ProportionalFallbackController(),
    }

    # Add PPO agents per task
    results = {}

    for task_name in tasks:
        results[task_name] = {}
        ppo_model_path = Path("models") / f"ppo_{task_name}.zip"
        ppo_agent = RLAgent(str(ppo_model_path) if ppo_model_path.exists() else None)

        task_agents = dict(agents)
        task_agents["RL (PPO)"] = ppo_agent

        print(f"\n--- Running Task: {task_name.upper()} ---")
        for name, agent in task_agents.items():
            metrics = evaluate_agent_on_task(agent, task_name, seeds)
            results[task_name][name] = metrics
            print(
                f"{name:<20} | Score: {metrics['score_mean']:.4f} +/- {metrics['score_std']:.4f} "
                f"| Safety: {metrics['safety_mean']:.3f} | Energy: {metrics['energy_mean']:.3f}"
            )

    # Compute aggregate scores across all tasks
    print("\n" + "=" * 80)
    print(" OVERALL BENCHMARK RESULTS (Average Across Easy, Medium, Hard)")
    print("=" * 80)
    print(
        f"{'Approach':<20} | {'Overall Score':<15} | {'Safety':<8} | {'Precision':<10} | {'Efficiency':<10} | {'Smoothness':<10}"
    )
    print("-" * 80)

    overall_table = []
    agent_names = list(agents.keys()) + ["RL (PPO)"]
    for name in agent_names:
        avg_score = np.mean([results[t][name]["score_mean"] for t in tasks])
        avg_safety = np.mean([results[t][name]["safety_mean"] for t in tasks])
        avg_target = np.mean([results[t][name]["target_mean"] for t in tasks])
        avg_energy = np.mean([results[t][name]["energy_mean"] for t in tasks])
        avg_jitter = np.mean([results[t][name]["jitter_mean"] for t in tasks])

        print(
            f"{name:<20} | {avg_score:.4f}{'':<9} | {avg_safety:.3f}{'':<3} | {avg_target:.3f}{'':<5} | {avg_energy:.3f}{'':<5} | {avg_jitter:.3f}"
        )
        overall_table.append(
            {
                "name": name,
                "score": avg_score,
                "safety": avg_safety,
                "target": avg_target,
                "energy": avg_energy,
                "jitter": avg_jitter,
            }
        )

    # Save output json
    out_file = Path("benchmark_results.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump({"tasks": results, "overall": overall_table}, f, indent=2)
    print(f"\nSaved benchmark results to {out_file.resolve()}")


if __name__ == "__main__":
    main()
