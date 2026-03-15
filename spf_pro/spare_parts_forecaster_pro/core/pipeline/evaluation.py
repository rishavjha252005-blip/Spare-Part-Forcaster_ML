# core/pipeline/evaluation.py
import numpy as np
from spare_parts_forecaster_pro.config.settings import BACKTEST_N_SPLITS, BACKTEST_MIN_TRAIN
from spare_parts_forecaster_pro.core.types import BacktestResult


def mase(actual: np.ndarray, forecast: np.ndarray, naive_mae: float | None = None) -> float:
    """Mean Absolute Scaled Error — robust for intermittent demand (handles zeros)."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if naive_mae is None or naive_mae == 0:
        diffs = np.abs(np.diff(actual))
        naive_mae = float(np.mean(diffs)) if len(diffs) > 0 else 1.0
    if naive_mae == 0:
        return 0.0
    return float(np.mean(np.abs(actual - forecast)) / naive_mae)


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Symmetric MAPE — avoids division by zero issues."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = (np.abs(actual) + np.abs(forecast)) / 2.0
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(actual[mask] - forecast[mask]) / denom[mask]) * 100)


def walk_forward_backtest(
    demand: np.ndarray,
    forecast_fn,          # callable(train_array) -> float
    n_splits: int = BACKTEST_N_SPLITS,
    min_train: int = BACKTEST_MIN_TRAIN,
    model_name: str = "model",
) -> BacktestResult:
    """
    Walk-forward cross-validation.  Each fold trains on [0:split] and
    tests on [split:split+1].  Never shuffles — strictly chronological.
    """
    demand = np.asarray(demand, dtype=float)
    n = len(demand)

    if n < min_train + n_splits:
        return BacktestResult(
            model=model_name,
            mase_scores=[],
            smape_scores=[],
            mean_mase=float("nan"),
            mean_smape=float("nan"),
        )

    step = max(1, (n - min_train) // n_splits)
    split_points = [min_train + i * step for i in range(n_splits) if min_train + i * step < n]

    mase_scores, smape_scores = [], []
    for split in split_points:
        train = demand[:split]
        actual = demand[split : split + 1]
        try:
            pred = forecast_fn(train)
            naive_mae = float(np.mean(np.abs(np.diff(train)))) if len(train) > 1 else 1.0
            mase_scores.append(mase(actual, np.array([pred]), naive_mae))
            smape_scores.append(smape(actual, np.array([pred])))
        except Exception:
            pass

    return BacktestResult(
        model=model_name,
        mase_scores=mase_scores,
        smape_scores=smape_scores,
        mean_mase=float(np.mean(mase_scores)) if mase_scores else float("nan"),
        mean_smape=float(np.mean(smape_scores)) if smape_scores else float("nan"),
    )
