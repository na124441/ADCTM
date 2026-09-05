"""
RL Training Script for ADCTM.
Trains continuous PPO agents using Stable-Baselines3 on the ADCTMGymEnv environment.
Saves model checkpoints to models/ppo_<task>.zip
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from core.gym_env import ADCTMGymEnv


def train_task_ppo(
    task_name: str = "easy",
    total_timesteps: int = 40_000,
    seed: int = 42,
    output_dir: str = "models",
) -> Path:
    print(f"\n[RL Training] Starting PPO training for task: {task_name.upper()}")
    print(f"Total timesteps: {total_timesteps:,} | Seed: {seed}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_file = out_path / f"ppo_{task_name}"

    def _env_creator():
        return ADCTMGymEnv(task_name=task_name)

    # 4 parallel vectorized environments for fast rollout collection
    train_env = make_vec_env(_env_creator, n_envs=4, seed=seed)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model.save(str(save_file))
    print(f"[RL Training] Model successfully saved to: {save_file}.zip")

    # Evaluate on a distinct evaluation environment
    eval_env = ADCTMGymEnv(task_name=task_name)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=5, deterministic=True
    )
    print(f"[RL Training] Evaluation over 5 episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

    return save_file.with_suffix(".zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on ADCTM tasks")
    parser.add_argument(
        "--task",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard", "all"],
        help="Task to train on",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=30_000,
        help="Number of training timesteps per task",
    )
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    for t in tasks:
        train_task_ppo(task_name=t, total_timesteps=args.timesteps)
