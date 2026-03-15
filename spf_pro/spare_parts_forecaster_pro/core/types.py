# core/types.py
from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np
import pandas as pd

DemandCategory = Literal["Smooth", "Erratic", "Intermittent", "Lumpy", "No Demand"]
ModelName = Literal["SES", "SBA", "Croston", "LightGBM", "Prophet", "Ensemble", "Unknown"]


@dataclass(frozen=True)
class DemandProfile:
    category: DemandCategory
    adi: float
    cv2: float
    zero_rate: float        # fraction of zero periods
    mean_nonzero: float     # mean of non-zero demand
    total_periods: int


@dataclass(frozen=True)
class ABCXYZSegment:
    abc: Literal["A", "B", "C"]   # value-based
    xyz: Literal["X", "Y", "Z"]   # variability-based
    label: str                     # e.g. "AX"
    service_level_z: float         # recommended z-score for this segment


@dataclass
class ForecastResult:
    model: str
    value: float                         # point forecast
    lower_80: float = 0.0               # 80% prediction interval lower
    upper_80: float = 0.0               # 80% prediction interval upper
    lower_95: float = 0.0               # 95% prediction interval lower
    upper_95: float = 0.0               # 95% prediction interval upper
    feature_importance: Optional[dict] = None   # SHAP values if available
    mase: Optional[float] = None        # out-of-sample MASE if backtested
    smape: Optional[float] = None       # out-of-sample sMAPE if backtested


@dataclass
class BacktestResult:
    model: str
    mase_scores: list[float]
    smape_scores: list[float]
    mean_mase: float
    mean_smape: float


@dataclass(frozen=True)
class SimulationResult:
    samples: np.ndarray
    mean: float
    std: float
    percentile_90: float
    percentile_95: float
    percentile_99: float


@dataclass(frozen=True)
class InventoryPolicy:
    safety_stock: float
    reorder_point: float
    service_level_z: float
    expected_holding_cost: float
    expected_stockout_risk: float
    segment: Optional[ABCXYZSegment] = None


@dataclass
class DriftReport:
    is_drifting: bool
    ks_statistic: float
    p_value: float
    baseline_mean: float
    recent_mean: float
    recommendation: str
