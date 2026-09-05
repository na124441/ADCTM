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

# Global session registry to retain active simulation sessions across HTTP calls with multi-tenant isolation
ACTIVE_SESSIONS: Dict[str, SimulationSession] = {}
SESSION_LOCKS: Dict[str, threading.Lock] = {}
REGISTRY_LOCK = threading.Lock()
DEFAULT_SESSION_ID = "default"

# Backward compatibility alias
CURRENT_SESSION: Optional[SimulationSession] = None
env_lock = threading.Lock()


def _get_or_create_lock(session_id: str) -> threading.Lock:
    with REGISTRY_LOCK:
        if session_id not in SESSION_LOCKS:
            SESSION_LOCKS[session_id] = threading.Lock()
        return SESSION_LOCKS[session_id]


def _resolve_session_id(
    session_id_query: Optional[str] = None,
    session_id_header: Optional[str] = None,
    payload_session_id: Optional[str] = None,
) -> str:
    # Priority: Header -> Query -> Payload -> DEFAULT_SESSION_ID
    if session_id_header is not None and not hasattr(session_id_header, "default") and session_id_header.strip():
        return session_id_header.strip()
    if session_id_query is not None and not hasattr(session_id_query, "default") and session_id_query.strip():
        return session_id_query.strip()
    if payload_session_id is not None and not hasattr(payload_session_id, "default") and payload_session_id.strip():
        return payload_session_id.strip()
    return DEFAULT_SESSION_ID


def _get_session(session_id: str) -> SimulationSession:
    """
    Helper function to verify the active simulation session exists for given session_id.
    Raises an HTTP 400 exception if an agent attempts to step before calling /reset.
    """
    with REGISTRY_LOCK:
        session = ACTIVE_SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=400, 
            detail=f"Environment not initialized for session '{session_id}'. Call /reset first."
        )
    return session


def _ensure_initialized() -> None:
    """
    Legacy helper function for backward compatibility.
    """
    _get_session(DEFAULT_SESSION_ID)


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


@app.post("/reset")
def reset(
    payload: Optional[Dict[str, Any]] = Body(default=None),
    task_name: Optional[str] = Query(default=None),
    seed: Optional[int] = Query(default=None),
    session_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """
    Endpoint to load an authorized benchmark task config and formally instantiate/reset the simulation session.
    
    Security Contract:
        Only canonical benchmark tiers ('easy', 'medium', 'hard') and an optional RNG seed are accepted.
        Arbitrary configuration overrides are strictly forbidden to prevent evaluation gaming.
    
    Multi-Tenant Support:
        Supports independent concurrent sessions via optional `session_id`. Defaults to 'default'.
    """
    global CURRENT_SESSION

    try:
        # Resolve task_name, seed, and session_id with strict ResetPayload validation
        parsed_payload = {}
        if payload is not None and not hasattr(payload, "default"):
            if not isinstance(payload, dict):
                raise HTTPException(status_code=422, detail="Reset payload must be a JSON object.")
            validated = ResetPayload.model_validate(payload)
            parsed_payload["task_name"] = validated.task_name
            parsed_payload["seed"] = validated.seed
            parsed_payload["session_id"] = validated.session_id

        # Query parameters take priority if provided
        final_task = task_name if (task_name is not None and not hasattr(task_name, "default")) else parsed_payload.get("task_name", "easy")
        final_seed = seed if (seed is not None and not hasattr(seed, "default")) else parsed_payload.get("seed", None)
        active_session_id = _resolve_session_id(
            session_id_query=session_id, 
            payload_session_id=parsed_payload.get("session_id")
        )

        # Validate task through ResetPayload to guarantee canonical name
        validated_task = ResetPayload(task_name=final_task, seed=final_seed, session_id=active_session_id)
        session = SimulationSession.from_task_name(validated_task.task_name, seed=validated_task.seed)

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Task configuration file not found.")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json()))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    s_lock = _get_or_create_lock(active_session_id)
    with s_lock:
        with REGISTRY_LOCK:
            ACTIVE_SESSIONS[active_session_id] = session
            if active_session_id == DEFAULT_SESSION_ID:
                CURRENT_SESSION = session

    obs_dict = session.observation.model_dump()
    return obs_dict


@app.post("/step")
def step(
    action_dict: Dict[str, Any],
    session_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """
    Advances the simulation by one physical tick. 
    Accepts the agent's action and computes the resulting environment dynamics.
    """
    active_session_id = _resolve_session_id(session_id_query=session_id)
    session = _get_session(active_session_id)
    s_lock = _get_or_create_lock(active_session_id)

    with s_lock:
        try:
            return session.step(action_dict)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state")
def get_full_state(
    session_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """
    Debugging endpoint exposing the entire internal data structure of the Simulation Session.
    """
    active_session_id = _resolve_session_id(session_id_query=session_id)
    session = _get_session(active_session_id)
    s_lock = _get_or_create_lock(active_session_id)
    with s_lock:
        return session.model_dump()


@app.get("/score")
def get_score(
    session_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """
    Computes and returns the evaluation score for the active simulation session.
    Requires at least one simulation step to have been executed.
    """
    active_session_id = _resolve_session_id(session_id_query=session_id)
    session = _get_session(active_session_id)
    s_lock = _get_or_create_lock(active_session_id)
    with s_lock:
        if len(session.history_actions) == 0:
            raise HTTPException(
                status_code=400,
                detail="Cannot score an unexecuted session. Take at least one step before requesting /score.",
            )
        return session.get_score()


@app.post("/simulate")
def simulate(
    task_name: str = "easy", 
    cooling_level: float = 0.4,
    session_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """
    Runs a full simulation from start to finish using a fixed cooling policy.
    Returns the final trajectory grade and performance metrics.
    """
    active_session_id = _resolve_session_id(session_id_query=session_id)

    try:
        initial_observation = reset(payload=None, task_name=task_name, session_id=active_session_id)
        obs_dict = initial_observation if isinstance(initial_observation, dict) else initial_observation.model_dump()
        session = _get_session(active_session_id)
        config = session.config
        num_zones = config.num_zones

    except HTTPException as exc:
        raise HTTPException(status_code=exc.status_code, detail=f"Error resetting environment: {exc.detail}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error loading task configuration: {str(exc)}")

    observations = [obs_dict]
    actions = []
    total_reward = 0.0
    done = False
    
    while not done:
        action = {"cooling": [cooling_level] * num_zones}
        try:
            step_result = step(action, session_id=active_session_id)
        except HTTPException as exc:
            raise HTTPException(status_code=exc.status_code, detail=f"Error stepping environment: {exc.detail}")

        observations.append(step_result["observation"])
        actions.append(action)
        total_reward += step_result["reward"]["value"]
        done = step_result["done"]

    score = evaluate_trajectory(observations, actions, config)

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
