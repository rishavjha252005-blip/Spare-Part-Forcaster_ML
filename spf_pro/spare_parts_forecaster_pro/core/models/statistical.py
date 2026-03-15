# core/models/statistical.py
import numpy as np
from spare_parts_forecaster_pro.config.settings import SES_ALPHA, SBA_ALPHA


def _croston_components(demand: np.ndarray) -> tuple[float, float]:
    """Shared helper: (avg_nonzero_demand, avg_interval)."""
    non_zero = demand[demand > 0]
    if non_zero.size == 0:
        return 0.0, 1.0
    return float(np.mean(non_zero)), len(demand) / len(non_zero)


def ses_forecast(demand: np.ndarray, alpha: float = SES_ALPHA) -> float:
    demand = np.asarray(demand, dtype=float)
    if demand.size == 0:
        return 0.0
    level = demand[0]
    for actual in demand[1:]:
        level = alpha * actual + (1.0 - alpha) * level
    return float(level)


def croston_forecast(demand: np.ndarray) -> float:
    demand = np.asarray(demand, dtype=float)
    avg_demand, avg_interval = _croston_components(demand)
    return avg_demand / avg_interval


def sba_forecast(demand: np.ndarray, alpha: float = SBA_ALPHA) -> float:
    demand = np.asarray(demand, dtype=float)
    avg_demand, avg_interval = _croston_components(demand)
    return (avg_demand / avg_interval) * (1.0 - alpha / 2.0)
