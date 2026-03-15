# core/pipeline/simulation.py
import numpy as np
from spare_parts_forecaster_pro.config.settings import BOOTSTRAP_N_SIMULATIONS, BOOTSTRAP_LEAD_TIME
from spare_parts_forecaster_pro.core.types import SimulationResult


def bootstrap_simulation(
    demand: np.ndarray,
    n_simulations: int = BOOTSTRAP_N_SIMULATIONS,
    lead_time: int = BOOTSTRAP_LEAD_TIME,
) -> SimulationResult:
    demand = np.asarray(demand, dtype=float)
    matrix = np.random.choice(demand, size=(n_simulations, lead_time), replace=True)
    totals = matrix.sum(axis=1)
    return SimulationResult(
        samples=totals,
        mean=float(totals.mean()),
        std=float(totals.std()),
        percentile_90=float(np.percentile(totals, 90)),
        percentile_95=float(np.percentile(totals, 95)),
        percentile_99=float(np.percentile(totals, 99)),
    )
