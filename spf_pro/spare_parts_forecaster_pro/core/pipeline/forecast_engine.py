# core/pipeline/forecast_engine.py
import numpy as np
import pandas as pd
from typing import Optional

from spare_parts_forecaster_pro.core.types import ForecastResult, BacktestResult
from spare_parts_forecaster_pro.core.models.statistical import ses_forecast, croston_forecast, sba_forecast
from spare_parts_forecaster_pro.core.models.lgbm_model import lgbm_forecast
from spare_parts_forecaster_pro.core.models.prophet_model import prophet_forecast
from spare_parts_forecaster_pro.core.pipeline.evaluation import walk_forward_backtest
from spare_parts_forecaster_pro.config.settings import MIN_NONZERO_FOR_ML


def run_all_forecasts(
    demand: np.ndarray,
    dates: Optional[pd.Series] = None,
    run_backtest: bool = True,
) -> tuple[list[ForecastResult], list[BacktestResult], ForecastResult]:
    """
    Run every available model, backtest each, and return:
    - individual_results  list[ForecastResult]
    - backtest_results    list[BacktestResult]
    - best_result         ForecastResult  (lowest mean MASE wins)
    """
    demand = np.asarray(demand, dtype=float)
    non_zero_count = int((demand > 0).sum())
    results: list[ForecastResult] = []
    backtests: list[BacktestResult] = []

    # ── Statistical models ────────────────────────────────────────────────────
    stat_models = [
        ("SES",     ses_forecast),
        ("Croston", croston_forecast),
        ("SBA",     sba_forecast),
    ]
    for name, fn in stat_models:
        point = fn(demand)
        std = float(np.std(demand))
        fr = ForecastResult(
            model=name, value=max(0.0, point),
            lower_80=max(0.0, point - 1.28 * std),
            upper_80=max(0.0, point + 1.28 * std),
            lower_95=max(0.0, point - 1.96 * std),
            upper_95=max(0.0, point + 1.96 * std),
        )
        if run_backtest:
            bt = walk_forward_backtest(demand, fn, model_name=name)
            fr.mase  = bt.mean_mase
            fr.smape = bt.mean_smape
            backtests.append(bt)
        results.append(fr)

    # ── LightGBM ──────────────────────────────────────────────────────────────
    if non_zero_count >= MIN_NONZERO_FOR_ML:
        point, lower, upper, importance = lgbm_forecast(demand, dates)
        if point > 0 or lower > 0 or upper > 0:
            fr = ForecastResult(
                model="LightGBM", value=point,
                lower_80=lower + (upper - lower) * 0.15,
                upper_80=upper - (upper - lower) * 0.15,
                lower_95=lower, upper_95=upper,
                feature_importance=importance,
            )
            if run_backtest:
                bt = walk_forward_backtest(
                    demand,
                    lambda d: lgbm_forecast(d)[0],
                    model_name="LightGBM",
                )
                fr.mase  = bt.mean_mase
                fr.smape = bt.mean_smape
                backtests.append(bt)
            results.append(fr)

    # ── Prophet ───────────────────────────────────────────────────────────────
    if non_zero_count >= MIN_NONZERO_FOR_ML:
        point, lower, upper = prophet_forecast(demand, dates)
        if point > 0 or lower > 0 or upper > 0:
            fr = ForecastResult(
                model="Prophet", value=point,
                lower_80=lower + (upper - lower) * 0.15,
                upper_80=upper - (upper - lower) * 0.15,
                lower_95=lower, upper_95=upper,
            )
            if run_backtest:
                bt = walk_forward_backtest(
                    demand,
                    lambda d: prophet_forecast(d)[0],
                    model_name="Prophet",
                )
                fr.mase  = bt.mean_mase
                fr.smape = bt.mean_smape
                backtests.append(bt)
            results.append(fr)

    # ── Ensemble (simple average of all available point forecasts) ────────────
    if len(results) >= 2:
        vals = [r.value for r in results]
        lowers = [r.lower_95 for r in results]
        uppers = [r.upper_95 for r in results]
        ens_mase  = float(np.nanmean([r.mase  for r in results if r.mase  is not None])) if any(r.mase  is not None for r in results) else None
        ens_smape = float(np.nanmean([r.smape for r in results if r.smape is not None])) if any(r.smape is not None for r in results) else None
        ens = ForecastResult(
            model="Ensemble",
            value=float(np.mean(vals)),
            lower_80=float(np.mean(lowers)) + (float(np.mean(uppers)) - float(np.mean(lowers))) * 0.15,
            upper_80=float(np.mean(uppers)) - (float(np.mean(uppers)) - float(np.mean(lowers))) * 0.15,
            lower_95=float(np.mean(lowers)),
            upper_95=float(np.mean(uppers)),
            mase=ens_mase, smape=ens_smape,
        )
        results.append(ens)

    if not results:
        fallback = ForecastResult(model="Unknown", value=0.0)
        return [fallback], [], fallback

    # ── Best model = lowest MASE (or first if no backtests) ──────────────────
    scored = [r for r in results if r.mase is not None and not np.isnan(r.mase)]
    best = min(scored, key=lambda r: r.mase) if scored else results[0]

    return results, backtests, best
