# core/pipeline/inventory.py
import numpy as np
from spare_parts_forecaster_pro.config.settings import (
    SERVICE_LEVEL_Z, HOLDING_COST_RATE, STOCKOUT_COST_RATE
)
from spare_parts_forecaster_pro.core.types import InventoryPolicy, ABCXYZSegment, SimulationResult


def compute_inventory_policy(
    sim: SimulationResult,
    segment: ABCXYZSegment | None = None,
    unit_cost: float = 1.0,
    holding_cost_rate: float = HOLDING_COST_RATE,
    stockout_cost_rate: float = STOCKOUT_COST_RATE,
) -> InventoryPolicy:
    """
    Compute safety stock and reorder point using segment-aware z-score.
    Also estimates expected holding and stockout costs.
    """
    z = segment.service_level_z if segment else SERVICE_LEVEL_Z

    safety_stock = z * sim.std
    reorder_point = sim.mean + safety_stock

    # Expected holding cost = 0.5 × SS × unit_cost × holding_rate
    expected_holding = 0.5 * safety_stock * unit_cost * holding_cost_rate

    # Expected stockout risk = probability demand > reorder point
    samples = sim.samples
    stockout_prob = float((samples > reorder_point).mean())
    expected_stockout = stockout_prob * unit_cost * stockout_cost_rate

    return InventoryPolicy(
        safety_stock=safety_stock,
        reorder_point=reorder_point,
        service_level_z=z,
        expected_holding_cost=expected_holding,
        expected_stockout_risk=expected_stockout,
        segment=segment,
    )
