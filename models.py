"""
Pydantic Schemas establishing rigid object interfaces.
Enforcing structural stability upon internal environment boundaries and
ensuring API JSON payload parsing compliance securely.
"""
from typing import List
from pydantic import BaseModel, Field, conlist, field_validator, model_validator


class Observation(BaseModel):
    """
    Represents the full observable state of the environment.
    Sent to the agent at every step simulating hardware sensors.
    """
    temperatures: List[float] = Field(..., min_length=1)  # Â°C per zone
    workloads: List[float] = Field(..., min_length=1)     # normalized workload index [0,1] per zone
    cooling: List[float] = Field(..., min_length=1)       # last applied cooling level tracking buffer
    ambient_temp: float = Field(..., description="Ambient temperature in Celsius")
    time_step: int = Field(..., ge=0)

    @model_validator(mode='after')
    def same_length(self) -> 'Observation':
        """
        Structural verification step checking geometric constraint alignment between parallel state arrays.
        """
        # Ensure all lists have the same number of zones
        n_temps = len(self.temperatures)
        if len(self.workloads) != n_temps or len(self.cooling) != n_temps:
            raise ValueError("All lists (temperatures, workloads, cooling) must have equal length.")
        return self


class Action(BaseModel):
    """
    Cooling action output produced by the agent.
    One float per zone, mechanically constrained to [0, 1].
    """
    cooling: List[float] = Field(..., min_length=1)

    @field_validator("cooling")
    @classmethod
    def clip_values(cls, v: List[float]) -> List[float]:
        """
        Prevents physics breaking values by safely clamping cooling actuation commands inside [0, 1] limits.
        """
        # Clamp values to valid range even if passed slightly off-range
        return [max(0.0, min(1.0, x)) for x in v]


class Reward(BaseModel):
    """
    Scalar reward computed after each step.
    Mainline reinforcement mechanism. Negative values reflect costs
    (temperature violations, energy consumption, action jitter).
    """
    value: float = Field(..., description="Scalar reward value")


class InfoDict(BaseModel):
    """
    Optional: Debugging or internal tracking info model context block payload.
    """
    step: int
