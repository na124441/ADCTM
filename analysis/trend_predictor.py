from typing import List
import numpy as np


def predict_thermal_future(
    temps: List[float], 
    history: List[List[float]], 
    safe_temp: float, 
    window: int = 5
) -> List[str]:
    """
    Predicts time-to-violation for each zone based on robust rolling linear regression.
    Uses OLS slope over the rolling history window to incorporate all intermediate points.
    """
    num_zones = len(temps)
    if len(history) < window:
        return ["Initializing..." for _ in temps]
        
    recent_history = np.array(history[-window:])  # shape: (window, num_zones)
    x = np.arange(window)
    x_mean = np.mean(x)
    x_var = np.sum((x - x_mean) ** 2)

    forecasts = []
    for z_idx in range(num_zones):
        current_temp = temps[z_idx]
        y = recent_history[:, z_idx]
        y_mean = np.mean(y)
        
        # Closed-form OLS slope (velocity in °C/step)
        velocity = float(np.sum((x - x_mean) * (y - y_mean)) / x_var)
        
        if velocity <= 0.05:  # Flat or cooling
            forecasts.append("Stable/Cooling")
        else:
            steps_to_critical = (safe_temp - current_temp) / velocity
            if steps_to_critical < 0:
                forecasts.append("CRITICAL")
            elif steps_to_critical < 1.0:
                forecasts.append("< 1 step ⚠️")
            elif steps_to_critical < 10.0:
                forecasts.append(f"~{steps_to_critical:.1f} steps")
            else:
                forecasts.append(f"~{int(np.ceil(steps_to_critical))} steps")

                
    return forecasts

