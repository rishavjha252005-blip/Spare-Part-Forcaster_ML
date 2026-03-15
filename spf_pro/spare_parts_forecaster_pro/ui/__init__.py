# Lazy imports — heavy packages (plotly, streamlit) are only pulled in
# when the functions are actually called, not at package load time.
# This prevents ModuleNotFoundError from crashing before app.py can
# display a helpful "please run pip install" message.

def __getattr__(name):
    if name in (
        "render_kpi_row", "render_demand_chart", "render_demand_profile",
        "render_forecast_comparison", "render_backtest_leaderboard",
        "render_shap", "render_simulation", "render_inventory_policy",
        "render_abcxyz_heatmap", "render_drift_report",
    ):
        from .components import (
            render_kpi_row, render_demand_chart, render_demand_profile,
            render_forecast_comparison, render_backtest_leaderboard,
            render_shap, render_simulation, render_inventory_policy,
            render_abcxyz_heatmap, render_drift_report,
        )
        return locals()[name]
    if name == "render_sidebar":
        from .sidebar import render_sidebar
        return render_sidebar
    if name == "inject_styles":
        from .styles import inject_styles
        return inject_styles
    if name == "generate_sample_csv":
        from .sample_data import generate_sample_csv
        return generate_sample_csv
    raise AttributeError(f"module 'spare_parts_forecaster_pro.ui' has no attribute {name!r}")
