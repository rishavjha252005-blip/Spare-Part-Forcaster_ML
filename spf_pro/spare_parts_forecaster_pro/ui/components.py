# ui/components.py
# ─────────────────────────────────────────────────────────────────────────────
# All Streamlit/Plotly rendering helpers.
# Pure presentation — zero business logic.
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from spare_parts_forecaster_pro.core.types import (
    DemandProfile, ForecastResult, BacktestResult,
    SimulationResult, InventoryPolicy, DriftReport, ABCXYZSegment,
)

# ── Design tokens ─────────────────────────────────────────────────────────────
C_ACCENT = "#C75B39"
C_PURPLE = "#7B6FD0"
C_TEAL = "#2A9D8F"
C_AMBER = "#E9C46A"
C_RED = "#E63946"
C_TEXT = "#E8E8E8"
C_MUTED = "#8A8A9A"
C_GRID = "rgba(255,255,255,0.06)"

# Plotly does NOT accept 8-digit hex (#rrggbbaa).
# All semi-transparent fills use explicit rgba() strings.
_FILL = {
    "accent": "rgba(199,91,57,0.10)",
    "purple": "rgba(123,111,208,0.12)",
    "teal": "rgba(42,157,143,0.12)",
    "amber": "rgba(233,196,106,0.12)",
    "red": "rgba(230,57,70,0.12)",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color=C_TEXT, size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor=C_GRID, zeroline=False),
    yaxis=dict(gridcolor=C_GRID, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)


def _fig(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=14, color=C_TEXT)),
    )
    return fig


# ── KPI row ───────────────────────────────────────────────────────────────────
def render_kpi_row(kpis: list) -> None:
    """kpis = list of (label, value, hint)"""
    cols = st.columns(len(kpis))
    for col, (label, value, hint) in zip(cols, kpis):
        col.markdown(
            f"""<div class="kpi-card">
                                  <div class="kpi-label">{label}</div>
                                  <div class="kpi-value">{value}</div>
                                  <div class="kpi-hint">{hint}</div>
                                </div>""",
            unsafe_allow_html=True,
        )


# ── Demand history chart ──────────────────────────────────────────────────────
def render_demand_chart(dates: pd.Series, demand: pd.Series,
                        title: str = "Demand history") -> None:
    fig = _fig(title)
    fig.add_trace(go.Scatter(
        x=dates, y=demand,
        mode="lines+markers",
        line=dict(color=C_ACCENT, width=2),
        marker=dict(size=5, color=C_ACCENT),
        name="Demand",
        fill="tozeroy",
        fillcolor=_FILL["accent"],
    ))
    fig.update_layout(height=280)
    st.plotly_chart(fig, use_container_width=True)


# ── Demand profile panel ──────────────────────────────────────────────────────
def render_demand_profile(profile: DemandProfile) -> None:
    _CAT_COLOR = {
        "Smooth": C_TEAL,
        "Erratic": C_AMBER,
        "Intermittent": C_PURPLE,
        "Lumpy": C_RED,
        "No Demand": C_MUTED,
    }
    color = _CAT_COLOR.get(profile.category, C_MUTED)

    st.markdown(
        f"""<div style="border-left:4px solid {color};padding:10px 16px;
                                        background:rgba(255,255,255,0.04);border-radius:0 8px 8px 0;
                                        margin-bottom:12px;">
                              <span style="font-size:11px;color:{C_MUTED};text-transform:uppercase;
                                           letter-spacing:1px;">Demand category</span>
                              <div style="font-size:24px;font-weight:700;color:{color};margin:2px 0;">
                                {profile.category}</div>
                            </div>""",
        unsafe_allow_html=True,
    )
    render_kpi_row([
        ("ADI", f"{profile.adi:.2f}", "inter-demand interval"),
        ("CV\u00b2", f"{profile.cv2:.2f}", "variability index"),
        ("Zero rate", f"{profile.zero_rate * 100:.1f}%", "fraction of zeros"),
        ("Mean demand", f"{profile.mean_nonzero:.2f}", "when non-zero"),
        ("Periods", str(profile.total_periods), "total observations"),
    ])

    # Syntetos-Boylan quadrant — each region has a pre-defined rgba fill
    _QUADRANTS = [
        (0, 1.32, 0, 0.49, "Smooth", C_TEAL, _FILL["teal"]),
        (0, 1.32, 0.49, 3, "Erratic", C_AMBER, _FILL["amber"]),
        (1.32, 5, 0, 0.49, "Intermittent", C_PURPLE, _FILL["purple"]),
        (1.32, 5, 0.49, 3, "Lumpy", C_RED, _FILL["red"]),
    ]

    fig = _fig("Syntetos-Boylan matrix")
    for (x0, x1, y0, y1, label, col, fill) in _QUADRANTS:
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
            fillcolor=fill,
            line_width=0,
        )
        fig.add_annotation(
            x=(x0 + x1) / 2, y=(y0 + y1) / 2,
            text=label, showarrow=False,
            font=dict(color=col, size=11),
        )

    fig.add_hline(y=0.49, line=dict(color=C_MUTED, width=1, dash="dot"))
    fig.add_vline(x=1.32, line=dict(color=C_MUTED, width=1, dash="dot"))
    fig.add_trace(go.Scatter(
        x=[min(profile.adi, 4.9)], y=[min(profile.cv2, 2.9)],
        mode="markers",
        marker=dict(size=14, color=color, symbol="star",
                    line=dict(color="white", width=1)),
        name="This part",
    ))
    fig.update_layout(
        height=300,
        xaxis=dict(range=[0, 5], title="ADI", gridcolor=C_GRID, zeroline=False),
        yaxis=dict(range=[0, 3], title="CV\u00b2", gridcolor=C_GRID, zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Forecast comparison ───────────────────────────────────────────────────────
def render_forecast_comparison(
        results: list,
        best_model: str,
        dates: pd.Series,
        demand: pd.Series,
) -> None:
    # Bar chart — all models side by side
    fig = _fig("Point forecast — all models")
    bar_colors = [C_ACCENT if r.model == best_model else C_PURPLE for r in results]
    fig.add_trace(go.Bar(
        x=[r.model for r in results],
        y=[r.value for r in results],
        marker_color=bar_colors,
        text=[f"{r.value:.2f}" for r in results],
        textposition="outside",
        name="Forecast",
    ))
    fig.add_trace(go.Scatter(
        x=[r.model for r in results], y=[r.upper_95 for r in results],
        mode="markers",
        marker=dict(symbol="triangle-up", color=C_TEAL, size=8),
        name="Upper 95%",
    ))
    fig.add_trace(go.Scatter(
        x=[r.model for r in results], y=[r.lower_95 for r in results],
        mode="markers",
        marker=dict(symbol="triangle-down", color=C_RED, size=8),
        name="Lower 95%",
    ))
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

    # History + best-model forecast overlay
    best = next((r for r in results if r.model == best_model), results[0])
    fig2 = _fig(f"History + {best.model} forecast")
    fig2.add_trace(go.Scatter(
        x=dates, y=demand, mode="lines", name="Actual",
        line=dict(color=C_MUTED, width=1.5),
    ))

    try:
        freq = pd.infer_freq(pd.to_datetime(dates)) or "W"
        next_date = pd.to_datetime(dates.iloc[-1]) + pd.tseries.frequencies.to_offset(freq)
    except Exception:
        next_date = len(dates)

    if best.upper_95 > 0:
        fig2.add_trace(go.Scatter(
            x=[dates.iloc[-1], next_date, next_date, dates.iloc[-1]],
            y=[demand.iloc[-1], best.upper_95, best.lower_95, demand.iloc[-1]],
            fill="toself",
            fillcolor=_FILL["accent"],
            line_width=0,
            name="95% interval",
        ))
    fig2.add_trace(go.Scatter(
        x=[dates.iloc[-1], next_date],
        y=[demand.iloc[-1], best.value],
        mode="lines+markers",
        line=dict(color=C_ACCENT, width=2.5, dash="dot"),
        marker=dict(size=9, color=C_ACCENT),
        name=f"{best.model} forecast",
    ))
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)


# ── Backtest leaderboard ──────────────────────────────────────────────────────
def render_backtest_leaderboard(results: list, best_model: str) -> None:
    scored = [r for r in results
              if r.mase is not None and not np.isnan(r.mase)]
    if not scored:
        st.info("No backtest results available.")
        return

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["MASE (lower = better)", "sMAPE % (lower = better)"],
    )
    mase_colors = [C_ACCENT if r.model == best_model else C_PURPLE for r in scored]
    smape_colors = [C_ACCENT if r.model == best_model else C_TEAL for r in scored]

    fig.add_trace(go.Bar(
        y=[r.model for r in scored], x=[r.mase for r in scored],
        orientation="h", marker_color=mase_colors,
        text=[f"{r.mase:.3f}" for r in scored], textposition="outside",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        y=[r.model for r in scored], x=[r.smape for r in scored],
        orientation="h", marker_color=smape_colors,
        text=[f"{r.smape:.1f}%" for r in scored], textposition="outside",
    ), row=1, col=2)

    fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    best_mase = next(
        (r.mase for r in scored if r.model == best_model), None
    )
    if best_mase is not None:
        st.success(
            f"\U0001f3c6 **Best model:** {best_model}  —  MASE = {best_mase:.3f}"
        )


# ── SHAP feature importance ───────────────────────────────────────────────────
def render_shap(importance: dict) -> None:
    if not importance:
        return
    fig = _fig("Feature importance (SHAP)")
    names = list(importance.keys())[::-1]
    vals = list(importance.values())[::-1]
    max_v = max(vals) if vals else 1.0
    bar_colors = [
        f"rgba(199,91,57,{0.35 + 0.65 * (v / max_v):.2f})" for v in vals
    ]
    fig.add_trace(go.Bar(
        y=names, x=vals, orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(height=max(280, len(names) * 28 + 80))
    st.plotly_chart(fig, use_container_width=True)


# ── Bootstrap simulation ──────────────────────────────────────────────────────
def render_simulation(sim: SimulationResult, policy: InventoryPolicy) -> None:
    fig = _fig("Lead-time demand distribution (bootstrap)")
    fig.add_trace(go.Histogram(
        x=sim.samples, nbinsx=40,
        marker_color=C_PURPLE, opacity=0.75,
        name="Simulated demand",
    ))
    for val, label, color in [
        (sim.mean, "Mean", C_ACCENT),
        (policy.reorder_point, "Reorder pt", C_RED),
        (sim.percentile_95, "95th pct", C_AMBER),
    ]:
        fig.add_vline(
            x=val,
            line=dict(color=color, width=1.5, dash="dash"),
            annotation_text=label,
            annotation_font=dict(color=color, size=11),
        )
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

    render_kpi_row([
        ("Sim mean", f"{sim.mean:.1f}", "expected LT demand"),
        ("Std dev", f"{sim.std:.1f}", "variability"),
        ("90th pct", f"{sim.percentile_90:.1f}", ""),
        ("95th pct", f"{sim.percentile_95:.1f}", ""),
        ("99th pct", f"{sim.percentile_99:.1f}", ""),
    ])


# ── Inventory policy ──────────────────────────────────────────────────────────
def render_inventory_policy(policy: InventoryPolicy) -> None:
    seg = policy.segment
    render_kpi_row([
        ("Safety stock", f"{policy.safety_stock:.2f}", "units"),
        ("Reorder point", f"{policy.reorder_point:.2f}", "units"),
        ("Service level", f"{policy.service_level_z:.2f}\u03c3", "z-score"),
        ("Holding cost", f"${policy.expected_holding_cost:.2f}", "est. / year"),
        ("Stockout risk", f"{policy.expected_stockout_risk * 100:.1f}%", "probability"),
    ])

    if seg:
        st.markdown(
            f"""<div style="margin-top:12px;padding:10px 16px;
                                            background:rgba(255,255,255,0.04);border-radius:8px;">
                                  <span style="font-size:11px;color:{C_MUTED};text-transform:uppercase;
                                               letter-spacing:1px;">ABC-XYZ segment</span>
                                  <span style="font-size:28px;font-weight:700;color:{C_ACCENT};
                                               margin-left:12px;">{seg.label}</span>
                                  <span style="font-size:13px;color:{C_MUTED};margin-left:8px;">
                                    ({seg.abc} value &middot; {seg.xyz} variability)</span>
                                </div>""",
            unsafe_allow_html=True,
        )

    # Cost gauge
    fig = _fig("Cost trade-off: holding vs stockout risk")
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=policy.expected_stockout_risk * 100,
        title=dict(text="Stockout probability %",
                   font=dict(size=13, color=C_TEXT)),
        gauge=dict(
            axis=dict(range=[0, 20], tickcolor=C_TEXT),
            bar=dict(color=C_RED),
            steps=[
                dict(range=[0, 5], color="rgba(42,157,143,0.25)"),
                dict(range=[5, 10], color="rgba(233,196,106,0.25)"),
                dict(range=[10, 20], color="rgba(230,57,70,0.25)"),
            ],
        ),
        number=dict(suffix="%", font=dict(color=C_TEXT)),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=260)
    st.plotly_chart(fig, use_container_width=True)


# ── ABC-XYZ portfolio heatmap ─────────────────────────────────────────────────
def render_abcxyz_heatmap(parts_df: pd.DataFrame) -> None:
    """parts_df columns: part_id, abc, xyz, total_demand"""
    matrix = {a: {x: [] for x in ["X", "Y", "Z"]} for a in ["A", "B", "C"]}
    for _, row in parts_df.iterrows():
        matrix[row["abc"]][row["xyz"]].append(row["part_id"])

    counts = [
        [len(matrix[a][x]) for x in ["X", "Y", "Z"]]
        for a in ["A", "B", "C"]
    ]
    fig = go.Figure(go.Heatmap(
        z=counts,
        x=["X (stable)", "Y (variable)", "Z (erratic)"],
        y=["A (high value)", "B (mid value)", "C (low value)"],
        colorscale=[
            [0.0, "rgba(199,91,57,0.05)"],
            [1.0, C_ACCENT],
        ],
        text=[[f"{v} parts" for v in row] for row in counts],
        texttemplate="%{text}",
        showscale=False,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="ABC-XYZ portfolio map",
                   font=dict(size=14, color=C_TEXT)),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Drift report ──────────────────────────────────────────────────────────────
def render_drift_report(
        drift: DriftReport,
        dates: pd.Series,
        demand: pd.Series,
        window: int = 12,
) -> None:
    color = C_RED if drift.is_drifting else C_TEAL
    status = "\u26a0 DRIFT DETECTED" if drift.is_drifting else "\u2713 STABLE"

    st.markdown(
        f"""<div style="border:1.5px solid {color};border-radius:10px;
                                        padding:14px 18px;margin-bottom:16px;">
                              <div style="font-size:11px;color:{C_MUTED};text-transform:uppercase;
                                          letter-spacing:1px;">Demand drift status</div>
                              <div style="font-size:22px;font-weight:700;color:{color};margin:4px 0;">
                                {status}</div>
                              <div style="font-size:13px;color:{C_TEXT};">
                                {drift.recommendation}</div>
                            </div>""",
        unsafe_allow_html=True,
    )
    render_kpi_row([
        ("KS statistic", f"{drift.ks_statistic:.4f}", "test statistic"),
        ("p-value", f"{drift.p_value:.4f}", "< 0.05 = drift"),
        ("Baseline mean", f"{drift.baseline_mean:.2f}", "historical avg"),
        ("Recent mean", f"{drift.recent_mean:.2f}", f"last {window} periods"),
    ])

    # Rolling mean — coloured by regime
    half = len(demand) // 2
    roll = pd.Series(demand.values, dtype=float).rolling(max(3, window // 2)).mean()
    recent_color = C_RED if drift.is_drifting else C_AMBER

    fig = _fig("Demand mean — baseline vs recent")
    fig.add_trace(go.Scatter(
        x=dates[:half], y=roll[:half],
        mode="lines", line=dict(color=C_TEAL, width=2), name="Baseline",
    ))
    fig.add_trace(go.Scatter(
        x=dates[half:], y=roll[half:],
        mode="lines", line=dict(color=recent_color, width=2), name="Recent",
    ))
    fig.add_vline(
        x=dates.iloc[half],
        line=dict(color=C_MUTED, dash="dot"),
        annotation_text="Split point",
        annotation_font=dict(color=C_MUTED),
    )
    fig.update_layout(height=280)
    st.plotly_chart(fig, use_container_width=True)