#!/usr/bin/env python3

import os
import json
import time
import requests
import sys
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

from grader.evaluator import evaluate_trajectory
from tasks.task_config import TaskConfig
from models import Observation
from inference.prompt import build_prompt as detailed_build_prompt

load_dotenv()

# Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_URL = "http://localhost:7860"
TEMPERATURE = 0.0


def parse_action(content: str, num_zones: int) -> List[float]:
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError
        data = json.loads(content[start:end])
        cooling = data.get("cooling", [])
        if len(cooling) != num_zones:
            raise ValueError
        return [max(0.0, min(1.0, float(v))) for v in cooling]
    except Exception:
        return [0.3] * num_zones


def call_llm(prompt: str) -> str:
    # Use HF_TOKEN if available, otherwise "none"
    api_key_for_client = HF_TOKEN if HF_TOKEN and HF_TOKEN != "none" else "none"
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key_for_client)

    messages = [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages, # type: ignore
                temperature=TEMPERATURE,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(2 ** attempt)
    return ""


def log_start(task_name: str, env: str, model: str) -> None:
    print(f"[START] task={task_name} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def run_task(task_name: str) -> float:
    log_start(task_name=task_name, env="ADCTM", model=MODEL_NAME)

    steps = 0
    score = 0.0
    rewards = []
    success = False
    
    try:
        # Reset env
        try:
            resp = requests.post(f"{ENV_URL}/reset", json={"task_name": task_name}, timeout=30)
            if resp.status_code != 200:
                resp = requests.post(f"{ENV_URL}/reset", params={"task_name": task_name}, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            error_msg = type(e).__name__ + ": " + str(e).replace('\n', ' ')
            print(f"[DEBUG] run_task exception on /reset: {error_msg}", file=sys.stderr, flush=True)
            log_end(success=False, steps=0, score=0.0, rewards=[])
            return 0.0

        initial_obs_dict = resp.json()

        current_observation = Observation(**initial_obs_dict)

        # Load config
        task_config_path = os.path.join(os.path.dirname(__file__), "tasks", f"{task_name}.json")
        if not os.path.exists(task_config_path):
            task_config_path = os.path.join(os.getcwd(), "tasks", f"{task_name}.json")

        with open(task_config_path) as f:
            config = TaskConfig(**json.load(f))

        all_obs = [initial_obs_dict]
        all_actions = []

        done = False

        while not done:
            steps += 1
            error_msg = None

            try:
                prompt = detailed_build_prompt(current_observation, config.model_dump())
                raw = call_llm(prompt)
                action_vals = parse_action(raw, len(current_observation.temperatures)) if raw else [0.3]*len(current_observation.temperatures)
            except Exception as e:
                # Provide a better error message if it's an API error
                error_msg = type(e).__name__ + ": " + str(e).replace('\n', ' ')
                action_vals = [0.3]*len(current_observation.temperatures)

            all_actions.append({"cooling": action_vals})

            try:
                step_resp = requests.post(
                    f"{ENV_URL}/step",
                    json={"cooling": action_vals},
                    timeout=30,
                )
                step_resp.raise_for_status()
                step_data = step_resp.json()
            except Exception as e:
                if not error_msg:
                    error_msg = type(e).__name__ + ": " + str(e).replace('\n', ' ')
                step_data = {"observation": initial_obs_dict, "reward": {"value": 0.0}, "done": True}

            obs_dict_from_step = step_data.get("observation", initial_obs_dict)
            current_observation = Observation(**obs_dict_from_step)
            all_obs.append(obs_dict_from_step)

            # Ensure reward is extracted correctly depending on environment
            reward_val = step_data.get("reward")
            if isinstance(reward_val, dict):
                reward = float(reward_val.get("value", 0.0))
            else:
                try:
                    reward = float(reward_val) if reward_val is not None else 0.0
                except (ValueError, TypeError):
                    reward = 0.0
            
            done = step_data.get("done", True)

            rewards.append(reward)

            action_str = f"cooling({action_vals})".replace(" ", "")

            log_step(step=steps, action=action_str, reward=reward, done=done, error=error_msg)

        # Score calculation
        try:
            result = evaluate_trajectory(all_obs, all_actions, config, return_details=True)
            score = float(result.get("score", 0.0))
            score = min(max(score, 0.0), 1.0)
        except Exception:
            score = 0.0

        success = score >= 0.4  # threshold

    except Exception as e:
        error_msg = type(e).__name__ + ": " + str(e).replace('\n', ' ')
        print(f"[DEBUG] run_task exception: {error_msg}", file=sys.stderr, flush=True)
    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)

    return score


def main():
    for task_name in ["easy", "medium", "hard"]:
        try:
            run_task(task_name)
        except Exception:
            pass


if __name__ == "__main__":
    main()