"""
Core API Server Environment for OpenEnv Multi-Zone Cooling.
This module defines the RESTful endpoints (using FastAPI) that expose 
the simulation environment. It acts as the HTTP interface for agents to reset
and step through the thermal management simulation.
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from core.models import Observation, ResetPayload
from core.paths import TASKS_DIR
from core.simulator import SimulationSession

from grader.evaluator import evaluate_trajectory

class PrettyJSONResponse(JSONResponse):
    """
    Custom response class to automatically pretty-print all JSON outputs.
    This ensures judges evaluating via browser or curl see human-readable
    data without requiring a secondary UI layer or breaking strict OpenEnv spec.
    """
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=4,
            separators=(", ", ": "),
        ).encode("utf-8")

# Initialize the FastAPI application instance
app = FastAPI(
    title="Multi-Zone Cooling OpenEnv", 
    version="0.1.0",
    default_response_class=PrettyJSONResponse
)

# Global variables to retain the active simulation session state across HTTP calls
CURRENT_SESSION: Optional[SimulationSession] = None
# A thread lock is required to prevent race conditions if multiple concurrent 
# HTTP requests attempt to modify the environment state simultaneously.
env_lock = threading.Lock()


@app.get("/dashboard")
@app.get("/ui")
def get_web_dashboard():
    """
    Serves the interactive ADCTM Command Center Web Application.
    """
    web_dir = Path(__file__).resolve().parent.parent / "ui" / "web"
    main_file = web_dir / "main_index.html"
    index_file = web_dir / "index.html"
    
    if main_file.exists():
        return FileResponse(main_file)
    elif index_file.exists():
        return FileResponse(index_file)
    return {"message": "Welcome to the OpenEnv Multi-Zone Cooling API!"}


@app.get("/")
def read_root():
    """
    Simple health-check and root endpoint.
    Retrieving this endpoint verifies that the FastAPI server is running.
    """
    return {"message": "Welcome to the OpenEnv Multi-Zone Cooling API!", "dashboard_url": "/dashboard"}


@app.get("/command_center")
def get_command_center():
    """
    Serves the command_center.html view.
    """
    cmd_file = Path(__file__).resolve().parent.parent / "ui" / "web" / "command_center.html"
    if cmd_file.exists():
        return FileResponse(cmd_file)
    return {"message": "command_center.html not found."}


def _ensure_initialized() -> None:
    """
    Helper function to verify the active simulation session exists.
    Raises an HTTP 400 exception if an agent attempts to step before calling /reset.
    """
    if CURRENT_SESSION is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")


@app.post("/reset")
def reset(
    payload: Optional[Dict[str, Any]] = Body(default=None),
    task_name: Optional[str] = Query(default=None),
    seed: Optional[int] = Query(default=None),
) -> Dict[str, Any]:
    """
    Endpoint to load an authorized benchmark task config and formally instantiate/reset the simulation session.
    
    Security Contract:
        Only canonical benchmark tiers ('easy', 'medium', 'hard') and an optional RNG seed are accepted.
        Arbitrary configuration overrides are strictly forbidden to prevent evaluation gaming.
    
    Args:
        payload: Optional body containing `{"task_name": "<tier>", "seed": <int>}`.
        task_name: Optional task name query parameter.
        seed: Optional RNG seed query parameter.
    Returns:
        Observation: The initial state observation.
    """
    global CURRENT_SESSION

    try:
        # Resolve task_name and seed with strict ResetPayload validation
        parsed_payload = {}
        if payload is not None:
            if not isinstance(payload, dict):
                raise HTTPException(status_code=422, detail="Reset payload must be a JSON object.")
            # Validate through ResetPayload (enforces extra='forbid' to block parameter injection)
            validated = ResetPayload.model_validate(payload)
            parsed_payload["task_name"] = validated.task_name
            parsed_payload["seed"] = validated.seed

        # Query parameters take priority if provided
        final_task = task_name if task_name is not None else parsed_payload.get("task_name", "easy")
        final_seed = seed if seed is not None else parsed_payload.get("seed", None)

        # Validate task through ResetPayload to guarantee canonical name
        validated_task = ResetPayload(task_name=final_task, seed=final_seed)
        session = SimulationSession.from_task_name(validated_task.task_name, seed=validated_task.seed)

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Task configuration file not found.")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json()))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    with env_lock:
        CURRENT_SESSION = session

    return session.observation.model_dump()


@app.post("/step")
def step(action_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Advances the simulation by one physical tick. 
    Accepts the agent's action and computes the resulting environment dynamics.
    
    Args:
        action_dict (Dict): The proposed cooling allocations.
        
    Returns:
        Dict: Contains the updated observation, the step reward, and the 'done' termination flag.
    """
    # Verify environment has been loaded
    _ensure_initialized()

    # Block other requests so physics simulation executes deterministically
    with env_lock:
        try:
            return CURRENT_SESSION.step(action_dict)
        except ValidationError as exc:
            # Action schema payload verification fails
            raise HTTPException(status_code=422, detail=exc.errors())
        except ValueError as exc:
            # Out of bounds or physics violation error
            raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state")
def get_full_state() -> Dict[str, Any]:
    """
    Debugging endpoint exposing the entire internal data structure of the Simulation Session.
    Useful for diagnostic or trajectory recording functions testing outside of normal flow.
    """
    _ensure_initialized()
    with env_lock:
        return CURRENT_SESSION.model_dump()


@app.get("/score")
def get_score() -> Dict[str, Any]:
    """
    Computes and returns the evaluation score for the active simulation session.
    """
    _ensure_initialized()
    with env_lock:
        return CURRENT_SESSION.get_score()


@app.post("/simulate")
def simulate(task_name: str = "easy", cooling_level: float = 0.4) -> Dict[str, Any]:
    """
    Runs a full simulation from start to finish using a fixed cooling policy.
    Returns the final trajectory grade and performance metrics.
    """
    try:
        # Directly call the reset function
        initial_observation = reset(task_name=task_name)
        session_config = initial_observation.model_dump() # Convert Observation to dict for consistency
        
        # We need the num_zones from the session config to create actions.
        # Since /reset returns an Observation, we need to infer num_zones from it.
        # Or, ideally, we would have a way to get the config directly.
        # For now, let's assume we can get it from the initial observation.
        # This is a bit of a hack, but necessary if /reset only returns Observation.
        # A better approach would be to fetch the config from /state after reset,
        # but that would require another API call.
        # Let's re-instantiate a dummy session to get the config for num_zones.
        # This is not ideal, but it avoids making another API call to /state.
        # A more robust solution would be to modify the /reset endpoint to return
        # more comprehensive session details or have a /config endpoint.
        temp_session = SimulationSession.from_task_name(task_name)
        num_zones = temp_session.config.num_zones

    except HTTPException as exc:
        raise HTTPException(status_code=400, detail=f"Error resetting environment: {exc.detail}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error loading task configuration: {str(exc)}")

    observations = [session_config] # Initial observation from reset
    actions = []
    total_reward = 0.0
    done = False
    
    # Loop until the simulation is done
    while not done:
        # Simple policy: apply fixed cooling level to all zones
        action = {"cooling": [cooling_level] * num_zones}
        
        try:
            # Directly call the step function
            step_result = step(action)
        except HTTPException as exc:
            raise HTTPException(status_code=400, detail=f"Error stepping environment: {exc.detail}")

        observations.append(step_result["observation"])
        actions.append(action)
        total_reward += step_result["reward"]["value"]
        done = step_result["done"]

    # Compute final grade
    # Note: evaluate_trajectory expects a TaskConfig object, but we only have the initial observation.
    # We need to pass the actual config to evaluate_trajectory.
    # Since we instantiated a temp_session above, we can use its config.
    score = evaluate_trajectory(observations, actions, temp_session.config)

    return {
        "task": task_name,
        "steps": len(actions),
        "total_reward": total_reward,
        "score": score,
        "status": "completed"
    }


if __name__ == "__main__":
    import uvicorn
    # Execute backend host interface when run locally directly
    uvicorn.run(app, host="0.0.0.0", port=8000)
