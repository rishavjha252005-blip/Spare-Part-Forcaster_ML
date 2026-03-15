# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Spare Parts Forecaster Pro — ML Edition
# Run from the spf_pro/ folder:  python -m streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────
import sys
import importlib.util
import streamlit as st

# ── Step 1: dependency check ──────────────────────────────────────────────────
# This runs before ANY heavy import.  If a package is missing, Streamlit shows
# a clear install command instead of a raw traceback.
_REQUIRED = {
    "plotly":      "plotly>=5.20.0",
    "lightgbm":    "lightgbm>=4.3.0",
    "scipy":       "scipy>=1.12.0",
    "sklearn":     "scikit-learn>=1.4.0",
    "shap":        "shap>=0.44.0",
    "prophet":     "prophet>=1.1.5",
    "statsmodels": "statsmodels>=0.14.0",
}
_missing = [pkg for mod, pkg in _REQUIRED.items()
            if importlib.util.find_spec(mod) is None]

if _missing:
    st.set_page_config(page_title="Install required packages", page_icon="⚙")
    st.error("### Missing packages — please install them and restart the app")
    st.code("pip install " + " ".join(_missing), language="bash")
    st.info("Or install everything at once:\n\n```bash\npip install -r requirements.txt\n```")
    st.stop()

# ── Step 2: page config (first Streamlit call after the check) ────────────────
st.set_page_config(
    page_title="Spare Parts Forecaster",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Step 3: all heavy imports — only reached when packages are present ─────────
import io
import numpy as np
import pandas as pd

from spare_parts_forecaster_pro.core.pipeline.classification  import classify_demand
from spare_parts_forecaster_pro.core.pipeline.forecast_engine import run_all_forecasts
from spare_parts_forecaster_pro.core.pipeline.simulation      import bootstrap_simulation
from spare_parts_forecaster_pro.core.pipeline.segmentation    import get_segment
from spare_parts_forecaster_pro.core.pipeline.inventory       import compute_inventory_policy
from spare_parts_forecaster_pro.core.pipeline.drift           import detect_drift

from spare_parts_forecaster_pro.ui.styles      import inject_styles
from spare_parts_forecaster_pro.ui.sidebar     import render_sidebar
from spare_parts_forecaster_pro.ui.sample_data import generate_sample_csv
from spare_parts_forecaster_pro.ui.components  import (
    render_demand_chart, render_demand_profile,
    render_forecast_comparison, render_backtest_leaderboard,
    render_shap, render_simulation, render_inventory_policy,
    render_abcxyz_heatmap, render_drift_report, render_kpi_row,
)

# ── Styles & header ───────────────────────────────────────────────────────────
inject_styles()

st.markdown(
    """<div class="app-header">
         <div class="app-logo">⚙</div>
         <div>
           <div class="app-title">Spare Parts Forecaster</div>
           <div class="app-subtitle">
             ML-powered demand forecasting · LightGBM · Prophet · Ensemble · Drift detection
           </div>
         </div>
       </div>""",
    unsafe_allow_html=True,
)

# ── Sidebar settings ──────────────────────────────────────────────────────────
cfg = render_sidebar()

# ── Data loading ──────────────────────────────────────────────────────────────
col_up, col_demo = st.columns([3, 1])
with col_up:
    uploaded = st.file_uploader(
        "Upload your spare-parts CSV  (columns: date, part_id, demand)",
        type="csv",
    )
with col_demo:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if st.button("📦 Load demo data"):
        st.session_state["demo_csv"] = generate_sample_csv()
        st.rerun()

if uploaded:
    raw_bytes = uploaded.read()
elif "demo_csv" in st.session_state:
    raw_bytes = st.session_state["demo_csv"]
else:
    st.info("Upload a CSV or click **Load demo data** to get started.")
    st.stop()

df = pd.read_csv(io.BytesIO(raw_bytes))
df["date"] = pd.to_datetime(df["date"])

# ── Part selector ─────────────────────────────────────────────────────────────
parts   = sorted(df["part_id"].unique())
part_id = st.selectbox("Select spare part", parts)
part_data = df[df["part_id"] == part_id].sort_values("date").reset_index(drop=True)
demand    = part_data["demand"].values.astype(float)
dates     = part_data["date"]

# ── Pre-compute for the current part (used across all tabs) ───────────────────
all_demand_totals = df.groupby("part_id")["demand"].sum().to_dict()
profile = classify_demand(demand)
segment = get_segment(
    total_demand=float(demand.sum()),
    cv2=profile.cv2,
    all_demands=list(all_demand_totals.values()),
)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
(
    tab_data, tab_classify, tab_forecast,
    tab_compare, tab_simulate, tab_inventory, tab_drift,
) = st.tabs([
    "📊 Data", "🔍 Classification", "🤖 Forecast",
    "📈 Model comparison", "🎲 Simulation", "📦 Inventory", "⚡ Drift",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 · DATA
# ════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    render_kpi_row([
        ("Parts in dataset", str(len(parts)),        "unique SKUs"),
        ("Total periods",    str(len(part_data)),     f"for {part_id}"),
        ("Total demand",     f"{int(demand.sum())}",  "units"),
        ("Date range",
         f"{dates.min().strftime('%b %Y')} – {dates.max().strftime('%b %Y')}",
         ""),
        ("Missing values",   str(part_data.isnull().sum().sum()), ""),
    ])

    st.markdown('<div class="section-title">Demand history</div>', unsafe_allow_html=True)
    render_demand_chart(dates, part_data["demand"])

    st.markdown('<div class="section-title">Raw data</div>', unsafe_allow_html=True)
    st.dataframe(
        part_data.style.background_gradient(subset=["demand"], cmap="YlOrRd"),
        use_container_width=True, height=280,
    )

    st.markdown('<div class="section-title">Portfolio overview</div>', unsafe_allow_html=True)
    summary_rows = []
    for pid in parts:
        d = df[df["part_id"] == pid]["demand"].values.astype(float)
        p = classify_demand(d)
        summary_rows.append({
            "Part ID":     pid,
            "Category":    p.category,
            "ADI":         round(p.adi, 2),
            "CV²":         round(p.cv2, 2),
            "Zero rate":   f"{p.zero_rate*100:.1f}%",
            "Mean demand": round(p.mean_nonzero, 2),
            "Total demand":int(d.sum()),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 · CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════
with tab_classify:
    st.markdown('<div class="section-title">Demand profile</div>', unsafe_allow_html=True)
    render_demand_profile(profile)

    st.markdown('<div class="section-title">ABC-XYZ segmentation</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""<div class="kpi-card" style="text-align:left;padding:16px 20px">
                  <div class="kpi-label">Segment for {part_id}</div>
                  <div style="font-size:48px;font-weight:700;color:#C75B39;">
                    {segment.label}</div>
                  <div style="margin-top:8px;font-size:13px;color:#8A8A9A;">
                    <b style="color:#E8E8E8;">A/B/C</b> = demand value rank &nbsp;·&nbsp;
                    <b style="color:#E8E8E8;">X/Y/Z</b> = demand variability<br>
                    Recommended service level z =
                    <b style="color:#C75B39;">{segment.service_level_z:.2f}</b>
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        if len(parts) > 1:
            hm_rows = []
            for pid in parts:
                d = df[df["part_id"] == pid]["demand"].values.astype(float)
                p = classify_demand(d)
                s = get_segment(float(d.sum()), p.cv2, list(all_demand_totals.values()))
                hm_rows.append({"part_id": pid, "abc": s.abc, "xyz": s.xyz,
                                 "total_demand": int(d.sum())})
            render_abcxyz_heatmap(pd.DataFrame(hm_rows))
        else:
            st.info("Upload data with multiple parts to see the portfolio heatmap.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 · FORECAST
# ════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.markdown('<div class="section-title">Run forecast engine</div>', unsafe_allow_html=True)

    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_btn = st.button("🚀 Run all models", use_container_width=True)
    with col_info:
        st.caption(
            "Runs SES, Croston, SBA, LightGBM, Prophet, and Ensemble. "
            "Walk-forward backtesting scores each model on MASE. "
            "The model with the lowest MASE is highlighted as best."
        )

    if run_btn:
        with st.spinner("Training models and running backtest…"):
            all_results, bt_results, best = run_all_forecasts(
                demand, dates=dates, run_backtest=cfg["run_backtest"]
            )
        st.session_state["forecast_results"] = all_results
        st.session_state["backtest_results"]  = bt_results
        st.session_state["best_forecast"]     = best

    if "forecast_results" in st.session_state:
        best = st.session_state["best_forecast"]

        st.markdown('<div class="section-title">Best model result</div>',
                    unsafe_allow_html=True)
        render_kpi_row([
            ("Best model",     best.model,                                  "lowest MASE"),
            ("Point forecast", f"{best.value:.3f}",                         "units / period"),
            ("Lower 95%",      f"{best.lower_95:.3f}",                      ""),
            ("Upper 95%",      f"{best.upper_95:.3f}",                      ""),
            ("MASE",           f"{best.mase:.3f}" if best.mase else "—",    "lower = better"),
        ])

        st.markdown('<div class="section-title">Forecast overlay</div>',
                    unsafe_allow_html=True)
        render_forecast_comparison(
            st.session_state["forecast_results"],
            best.model, dates, part_data["demand"],
        )

        lgbm_result = next(
            (r for r in st.session_state["forecast_results"]
             if r.model == "LightGBM" and r.feature_importance),
            None,
        )
        if lgbm_result:
            st.markdown(
                '<div class="section-title">SHAP feature importance (LightGBM)</div>',
                unsafe_allow_html=True,
            )
            render_shap(lgbm_result.feature_importance)
    else:
        st.info("Click **Run all models** to start.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 · MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab_compare:
    if "forecast_results" not in st.session_state:
        st.info("Run the forecast engine first (Forecast tab).")
    else:
        st.markdown('<div class="section-title">Backtest leaderboard</div>',
                    unsafe_allow_html=True)
        render_backtest_leaderboard(
            st.session_state["forecast_results"],
            st.session_state["best_forecast"].model,
        )

        st.markdown('<div class="section-title">All model results</div>',
                    unsafe_allow_html=True)
        rows = []
        for r in st.session_state["forecast_results"]:
            rows.append({
                "Model":     r.model,
                "Forecast":  round(r.value, 3),
                "Lower 95%": round(r.lower_95, 3),
                "Upper 95%": round(r.upper_95, 3),
                "MASE":      round(r.mase,  4) if r.mase  is not None else "—",
                "sMAPE %":   round(r.smape, 2) if r.smape is not None else "—",
            })
        result_df = pd.DataFrame(rows)
        st.dataframe(result_df, use_container_width=True)

        st.download_button(
            "⬇ Download results CSV",
            result_df.to_csv(index=False).encode(),
            file_name=f"{part_id}_forecast_comparison.csv",
            mime="text/csv",
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 · SIMULATION
# ════════════════════════════════════════════════════════════════════════════
with tab_simulate:
    st.markdown('<div class="section-title">Bootstrap risk simulation</div>',
                unsafe_allow_html=True)
    col_sim, col_desc = st.columns([1, 3])
    with col_sim:
        sim_btn = st.button("🎲 Run simulation", use_container_width=True)
    with col_desc:
        st.caption(
            f"Draws {cfg['n_sim']:,} lead-time samples "
            f"(lead time = {cfg['lead_time']} periods) "
            "with replacement from the demand history."
        )

    if sim_btn:
        with st.spinner("Simulating…"):
            sim = bootstrap_simulation(
                demand,
                n_simulations=cfg["n_sim"],
                lead_time=cfg["lead_time"],
            )
            policy = compute_inventory_policy(
                sim, segment=segment,
                unit_cost=cfg["unit_cost"],
                holding_cost_rate=cfg["holding_rate"],
                stockout_cost_rate=cfg["stockout_rate"],
            )
        st.session_state["simulation"] = sim
        st.session_state["inv_policy"] = policy

    if "simulation" in st.session_state:
        render_simulation(
            st.session_state["simulation"],
            st.session_state["inv_policy"],
        )
    else:
        st.info("Click **Run simulation** to estimate lead-time demand distribution.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 6 · INVENTORY
# ════════════════════════════════════════════════════════════════════════════
with tab_inventory:
    if "inv_policy" not in st.session_state:
        st.info("Run the simulation first (Simulation tab).")
    else:
        st.markdown('<div class="section-title">Inventory policy</div>',
                    unsafe_allow_html=True)
        render_inventory_policy(st.session_state["inv_policy"])

        st.markdown('<div class="section-title">Service level sensitivity</div>',
                    unsafe_allow_html=True)
        sim = st.session_state["simulation"]

        z_vals = [1.04, 1.28, 1.65, 1.88, 2.05, 2.33]
        sens_rows = []
        for z in z_vals:
            ss  = z * sim.std
            rop = sim.mean + ss
            hc  = 0.5 * ss * cfg["unit_cost"] * cfg["holding_rate"]
            sp  = float((sim.samples > rop).mean()) * 100
            sens_rows.append({
                "z-score":         z,
                "Service level":   f"{(1 - sp/100)*100:.1f}%",
                "Safety stock":    round(ss, 2),
                "Reorder point":   round(rop, 2),
                "Holding cost $":  round(hc, 2),
                "Stockout risk %": round(sp, 2),
            })
        st.dataframe(pd.DataFrame(sens_rows), use_container_width=True)

        policy = st.session_state["inv_policy"]
        st.download_button(
            "⬇ Download inventory policy",
            pd.DataFrame({
                "part_id":               [part_id],
                "segment":               [policy.segment.label if policy.segment else "—"],
                "safety_stock":          [round(policy.safety_stock, 3)],
                "reorder_point":         [round(policy.reorder_point, 3)],
                "service_level_z":       [policy.service_level_z],
                "expected_holding_cost": [round(policy.expected_holding_cost, 3)],
                "stockout_risk_pct":     [round(policy.expected_stockout_risk * 100, 2)],
            }).to_csv(index=False).encode(),
            file_name=f"{part_id}_inventory_policy.csv",
            mime="text/csv",
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 7 · DRIFT
# ════════════════════════════════════════════════════════════════════════════
with tab_drift:
    st.markdown('<div class="section-title">Demand drift detection</div>',
                unsafe_allow_html=True)

    col_dr, col_desc = st.columns([1, 3])
    with col_dr:
        drift_btn = st.button("⚡ Detect drift", use_container_width=True)
    with col_desc:
        st.caption(
            f"Kolmogorov-Smirnov test compares baseline demand distribution "
            f"against the most recent {cfg['drift_window']} periods. "
            "p < 0.05 flags a significant shift and recommends retraining."
        )

    if drift_btn:
        drift = detect_drift(demand, window=cfg["drift_window"])
        st.session_state["drift"] = drift

    if "drift" in st.session_state:
        render_drift_report(
            st.session_state["drift"],
            dates, part_data["demand"],
            window=cfg["drift_window"],
        )

        if len(parts) > 1:
            st.markdown('<div class="section-title">Portfolio drift scan</div>',
                        unsafe_allow_html=True)
            scan_rows = []
            for pid in parts:
                d  = df[df["part_id"] == pid].sort_values("date")["demand"].values.astype(float)
                dr = detect_drift(d, window=cfg["drift_window"])
                scan_rows.append({
                    "Part ID":    pid,
                    "Drifting":   "⚠ YES" if dr.is_drifting else "✓ no",
                    "p-value":    round(dr.p_value, 4),
                    "KS stat":    round(dr.ks_statistic, 4),
                    "Baseline μ": round(dr.baseline_mean, 2),
                    "Recent μ":   round(dr.recent_mean, 2),
                })
            scan_df = pd.DataFrame(scan_rows)
            st.dataframe(
                scan_df.style.apply(
                    lambda col: [
                        "color: #E63946" if v == "⚠ YES" else "color: #2A9D8F"
                        for v in col
                    ] if col.name == "Drifting" else [""] * len(col),
                    axis=0,
                ),
                use_container_width=True,
            )
    else:
        st.info("Click **Detect drift** to run the KS test.")