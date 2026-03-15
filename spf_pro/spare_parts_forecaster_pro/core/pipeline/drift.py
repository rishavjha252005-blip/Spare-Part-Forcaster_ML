# core/pipeline/drift.py
import numpy as np
from scipy import stats
from spare_parts_forecaster_pro.config.settings import DRIFT_WINDOW, DRIFT_KS_ALPHA
from spare_parts_forecaster_pro.core.types import DriftReport


def detect_drift(
    demand: np.ndarray,
    window: int = DRIFT_WINDOW,
    alpha: float = DRIFT_KS_ALPHA,
) -> DriftReport:
    """
    Kolmogorov-Smirnov test between the baseline (first 50% of data)
    and the most recent `window` periods.
    """
    demand = np.asarray(demand, dtype=float)
    if len(demand) < window * 2:
        return DriftReport(
            is_drifting=False, ks_statistic=0.0, p_value=1.0,
            baseline_mean=float(np.mean(demand)),
            recent_mean=float(np.mean(demand[-window:])),
            recommendation="Insufficient data for drift detection.",
        )

    half = len(demand) // 2
    baseline = demand[:half]
    recent = demand[-window:]

    ks_stat, p_value = stats.ks_2samp(baseline, recent)
    is_drifting = p_value < alpha

    baseline_mean = float(np.mean(baseline))
    recent_mean = float(np.mean(recent))
    pct_change = ((recent_mean - baseline_mean) / baseline_mean * 100) if baseline_mean != 0 else 0.0

    if is_drifting:
        direction = "increased" if recent_mean > baseline_mean else "decreased"
        recommendation = (
            f"Demand has significantly {direction} by {abs(pct_change):.1f}%. "
            "Consider retraining models with recent data."
        )
    else:
        recommendation = "Demand pattern is stable. No retraining needed."

    return DriftReport(
        is_drifting=is_drifting,
        ks_statistic=round(ks_stat, 4),
        p_value=round(p_value, 4),
        baseline_mean=round(baseline_mean, 3),
        recent_mean=round(recent_mean, 3),
        recommendation=recommendation,
    )
