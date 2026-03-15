# ui/sidebar.py
import streamlit as st


def render_sidebar() -> dict:
    """Render the settings sidebar and return user-configured parameters."""
    st.sidebar.markdown("## ⚙ Settings")

    st.sidebar.markdown("### Simulation")
    n_sim = st.sidebar.slider("Bootstrap simulations", 200, 5000, 1000, step=100)
    lead_time = st.sidebar.slider("Lead time (periods)", 1, 30, 7)

    st.sidebar.markdown("### Inventory costs")
    unit_cost = st.sidebar.number_input("Unit cost ($)", min_value=0.1, value=100.0, step=10.0)
    holding_rate = st.sidebar.slider("Holding cost rate (%/year)", 5, 50, 25) / 100
    stockout_rate = st.sidebar.slider("Stockout cost multiplier", 1.0, 5.0, 2.0, step=0.5)

    st.sidebar.markdown("### Backtesting")
    run_backtest = st.sidebar.checkbox("Run walk-forward backtest", value=True)
    n_splits = st.sidebar.slider("Backtest folds", 2, 6, 3)

    st.sidebar.markdown("### Drift detection")
    drift_window = st.sidebar.slider("Recent window (periods)", 6, 30, 12)

    st.sidebar.markdown("---")
    st.sidebar.caption("Spare Parts Forecaster Pro · ML Edition")

    return dict(
        n_sim=n_sim,
        lead_time=lead_time,
        unit_cost=unit_cost,
        holding_rate=holding_rate,
        stockout_rate=stockout_rate,
        run_backtest=run_backtest,
        n_splits=n_splits,
        drift_window=drift_window,
    )
