import json
import logging
import re
from typing import Callable, List, Optional, Tuple

from core.models import Action

logger = logging.getLogger(__name__)


def parse_llm_response(response_text: str, expected_num_zones: int) -> Action:
    """
    Parse the model response into a validated cooling action.
    Robustly handles markdown code fences, embedded commentary, and JSON formatting.
    """
    if not response_text or not response_text.strip():
        raise ValueError("Empty or whitespace-only response received from model")

    text = response_text.strip()

    # Strip markdown code blocks if present (```json ... ``` or ``` ...)
    md_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    cleaned_text = md_match.group(1).strip() if md_match else text

    # Attempt 1: Direct JSON parse of cleaned text
    data = None
    try:
        data = json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Find outermost JSON object
    if not isinstance(data, dict):
        start = cleaned_text.find("{")
        end = cleaned_text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(cleaned_text[start:end])
            except json.JSONDecodeError:
                pass

    # Attempt 3: Regex match specifically for "cooling": [...]
    if not isinstance(data, dict) or "cooling" not in data:
        cooling_match = re.search(r'"cooling"\s*:\s*(\[[^\]]*\])', cleaned_text)
        if cooling_match:
            try:
                cooling_arr = json.loads(cooling_match.group(1))
                data = {"cooling": cooling_arr}
            except json.JSONDecodeError:
                pass

    if not isinstance(data, dict) or "cooling" not in data:
        snippet = response_text[:120].replace("\n", " ")
        raise ValueError(f"No valid JSON containing 'cooling' key found in model response: '{snippet}'")

    cooling_list = data.get("cooling")
    if not isinstance(cooling_list, list) or len(cooling_list) != expected_num_zones:
        raise ValueError(
            f"Expected 'cooling' to contain {expected_num_zones} entries, got {len(cooling_list) if isinstance(cooling_list, list) else type(cooling_list).__name__}"
        )

    try:
        clamped = [max(0.0, min(1.0, float(value))) for value in cooling_list]
    except (TypeError, ValueError) as exc:
        raise ValueError("Cooling values must be numeric") from exc

    return Action(cooling=clamped)


def parse_action_safe(
    content: str,
    num_zones: int,
    fallback_policy: Optional[Callable[..., List[float]]] = None,
    fallback_kwargs: Optional[dict] = None,
) -> Tuple[List[float], Optional[str]]:
    """
    Safely parses an LLM response into a cooling list, returning both the action
    and an optional diagnostic error message if parsing failed.
    Does not silently mask parse failures.
    """
    try:
        action = parse_llm_response(content, num_zones)
        return action.cooling, None
    except Exception as exc:
        error_msg = f"LLM parse failure: {type(exc).__name__}: {str(exc)}"
        logger.warning(error_msg)
        if fallback_policy is not None:
            kwargs = fallback_kwargs or {}
            fallback_action = fallback_policy(**kwargs)
            return fallback_action, error_msg
        return [0.3] * num_zones, error_msg

