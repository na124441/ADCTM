"""
Core API Server Environment for OpenEnv Multi-Zone Cooling.
This module defines the RESTful endpoints (using FastAPI) that expose 
the simulation environment. It acts as the HTTP interface for agents to reset
and step through the thermal management simulation.
"""

import json
import threading
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from core.models import Observation
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


@app.get("/")
def read_root():
    """
    Simple health-check and root endpoint.
    Retrieving this endpoint verifies that the FastAPI server is running.
    """
    return {"message": "Welcome to the OpenEnv Multi-Zone Cooling API!"}


def _ensure_initialized() -> None:
    """
    Helper function to verify the active simulation session exists.
    Raises an HTTP 400 exception if an agent attempts to step before calling /reset.
    """
    if CURRENT_SESSION is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")


@app.post("/reset", response_model=Observation)
def reset(
    payload: Optional[Dict[str, Any]] = Body(default=None),
    task_name: Optional[str] = Query(default=None),
) -> Observation:
    """
    Endpoint to load a task config and formally instantiate/reset the simulation session.
    
    Args:
        payload: Optional task configuration overrides or a body containing
                 `{"task_name": "<tier>"}`.
        task_name: Optional task name query parameter for compatibility with
                   external validators and inference clients.
    Returns:
        Observation: The initial state observation.
    """
    global CURRENT_SESSION

    try:
        config_payload = payload or None
        selected_task = task_name

        if isinstance(config_payload, dict) and "task_name" in config_payload:
            selected_task = str(config_payload["task_name"])
            config_payload = {
                key: value
                for key, value in config_payload.items()
                if key != "task_name"
            } or None

        if config_payload is None:
            session = SimulationSession.from_task_name(selected_task or "easy")
        else:
            # Load the custom dictionary parsed by FastAPI via Pydantic payload models.
            session = SimulationSession.from_dict(config_payload)
    except FileNotFoundError:
        # File parsing error
        raise HTTPException(status_code=500, detail="Default task config 'tasks/easy.json' not found.")
    except ValidationError as exc:
        # Pydantic configuration validation issue
        raise HTTPException(status_code=422, detail=exc.errors())
    except ValueError as exc:
        # Custom logic mismatch error
        raise HTTPException(status_code=422, detail=str(exc))

    # Safely lock and overwrite the global session instance locally
    with env_lock:
        CURRENT_SESSION = session

    return session.observation


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
