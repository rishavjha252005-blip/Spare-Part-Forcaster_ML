# core/models/prophet_model.py
import numpy as np
import pandas as pd
from typing import Optional


def prophet_forecast(
    demand: np.ndarray,
    dates: Optional[pd.Series] = None,
) -> tuple[float, float, float]:
    """
    Fit Prophet and return (point, lower_95, upper_95) for the next period.
    Falls back to (0,0,0) gracefully if Prophet is not installed or series too short.
    """
    try:
        from prophet import Prophet
        import logging
        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    except ImportError:
        return 0.0, 0.0, 0.0

    demand = np.asarray(demand, dtype=float)
    if len(demand) < 10:
        return 0.0, 0.0, 0.0

    # Build Prophet dataframe
    if dates is not None:
        ds = pd.to_datetime(dates).reset_index(drop=True)
    else:
        ds = pd.date_range(end=pd.Timestamp.today(), periods=len(demand), freq="W")

    prophet_df = pd.DataFrame({"ds": ds, "y": demand})

    try:
        m = Prophet(
            interval_width=0.95,
            daily_seasonality=False,
            weekly_seasonality=len(demand) >= 14,
            yearly_seasonality=len(demand) >= 52,
            uncertainty_samples=200,
        )
        m.fit(prophet_df)

        # Forecast one period ahead
        freq = pd.infer_freq(pd.to_datetime(ds)) or "W"
        future = m.make_future_dataframe(periods=1, freq=freq)
        forecast = m.predict(future)

        last = forecast.iloc[-1]
        point = float(max(0.0, last["yhat"]))
        lower = float(max(0.0, last["yhat_lower"]))
        upper = float(max(0.0, last["yhat_upper"]))
        return point, lower, upper
    except Exception:
        return 0.0, 0.0, 0.0
