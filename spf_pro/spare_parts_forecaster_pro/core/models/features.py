# core/models/features.py
import numpy as np
import pandas as pd
from spare_parts_forecaster_pro.config.settings import LAG_PERIODS, ROLLING_WINDOWS


def build_features(demand: np.ndarray, dates: pd.Series | None = None) -> pd.DataFrame:
    """
    Build a rich feature matrix for ML models.
    Each row = one time period. Target column = 'demand'.
    """
    df = pd.DataFrame({"demand": demand.astype(float)})

    # ── Lag features ─────────────────────────────────────────────────────────
    for lag in LAG_PERIODS:
        df[f"lag_{lag}"] = df["demand"].shift(lag)

    # ── Rolling statistics ────────────────────────────────────────────────────
    for w in ROLLING_WINDOWS:
        df[f"roll_mean_{w}"] = df["demand"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = df["demand"].shift(1).rolling(w).std().fillna(0)
        df[f"roll_max_{w}"]  = df["demand"].shift(1).rolling(w).max()
        df[f"roll_min_{w}"]  = df["demand"].shift(1).rolling(w).min()

    # ── Intermittency features ────────────────────────────────────────────────
    df["is_zero"] = (df["demand"] == 0).astype(int)
    df["days_since_demand"] = (
        df["demand"].shift(1)
        .eq(0)
        .groupby((df["demand"].shift(1) != 0).cumsum())
        .cumcount()
    )
    df["zero_streak"] = (
        df["demand"].shift(1)
        .eq(0)
        .groupby((df["demand"].shift(1) != 0).cumsum())
        .transform("sum")
    )

    # ── Calendar features (if dates provided) ────────────────────────────────
    if dates is not None:
        d = pd.to_datetime(dates).reset_index(drop=True)
        df["month"]       = d.dt.month
        df["quarter"]     = d.dt.quarter
        df["dayofweek"]   = d.dt.dayofweek
        df["weekofyear"]  = d.dt.isocalendar().week.astype(int)
        df["is_month_end"] = d.dt.is_month_end.astype(int)

    # ── Cumulative / trend ────────────────────────────────────────────────────
    df["cum_mean"] = df["demand"].expanding().mean().shift(1)
    df["period"]   = np.arange(len(df))

    return df.dropna()


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "demand"]
