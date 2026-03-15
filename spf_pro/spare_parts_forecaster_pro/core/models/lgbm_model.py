# core/models/lgbm_model.py
import numpy as np
import pandas as pd
from typing import Optional

from spare_parts_forecaster_pro.config.settings import LGBM_PARAMS, MIN_NONZERO_FOR_ML
from spare_parts_forecaster_pro.core.models.features import build_features, get_feature_columns


def lgbm_forecast(
    demand: np.ndarray,
    dates: Optional[pd.Series] = None,
    return_importance: bool = True,
) -> tuple[float, float, float, Optional[dict]]:
    """
    Train LightGBM on (t-1 … t-lag) → t and predict the next period.

    Returns
    -------
    point_forecast, lower_95, upper_95, feature_importance_dict
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return 0.0, 0.0, 0.0, None

    demand = np.asarray(demand, dtype=float)
    non_zero = demand[demand > 0]
    if len(non_zero) < MIN_NONZERO_FOR_ML:
        return 0.0, 0.0, 0.0, None

    df = build_features(demand, dates)
    if len(df) < 10:
        return 0.0, 0.0, 0.0, None

    feat_cols = get_feature_columns(df)
    X, y = df[feat_cols].values, df["demand"].values

    # ── Train on all available data (single-step ahead) ────────────────────────
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X, y)

    # Predict next period using last row's features
    last_features = X[-1:].copy()
    point = float(max(0.0, model.predict(last_features)[0]))

    # Quantile models for prediction intervals
    lower_model = lgb.LGBMRegressor(**{**LGBM_PARAMS, "objective": "quantile", "alpha": 0.025})
    upper_model = lgb.LGBMRegressor(**{**LGBM_PARAMS, "objective": "quantile", "alpha": 0.975})
    lower_model.fit(X, y)
    upper_model.fit(X, y)
    lower = float(max(0.0, lower_model.predict(last_features)[0]))
    upper = float(max(0.0, upper_model.predict(last_features)[0]))

    # ── SHAP feature importance ───────────────────────────────────────────────
    importance = None
    if return_importance:
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X[-1:])
            importance = {
                feat_cols[i]: float(abs(shap_vals[0][i]))
                for i in range(len(feat_cols))
            }
            # Sort descending and keep top 10
            importance = dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            )
        except Exception:
            pass

    return point, lower, upper, importance
