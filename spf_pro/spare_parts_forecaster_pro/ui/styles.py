# ui/styles.py
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Root & page ───────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0F1117 !important;
    color: #E8E8E8;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: #16213E !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* ── Header ────────────────────────────────────────────────────────── */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px 0 8px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 24px;
}
.app-logo {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #C75B39, #7B6FD0);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
}
.app-title {
    font-size: 22px; font-weight: 700;
    background: linear-gradient(90deg, #C75B39, #7B6FD0);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.app-subtitle { font-size: 13px; color: #8A8A9A; margin-top: -2px; }

/* ── KPI cards ─────────────────────────────────────────────────────── */
.kpi-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
    transition: background 0.2s;
}
.kpi-card:hover { background: rgba(255,255,255,0.07); }
.kpi-label {
    font-size: 11px; color: #8A8A9A;
    text-transform: uppercase; letter-spacing: 0.8px;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 22px; font-weight: 700; color: #E8E8E8;
    font-family: 'DM Mono', monospace;
}
.kpi-hint { font-size: 11px; color: #6A6A7A; margin-top: 2px; }

/* ── Section headers ───────────────────────────────────────────────── */
.section-title {
    font-size: 16px; font-weight: 600; color: #E8E8E8;
    margin: 20px 0 10px;
    display: flex; align-items: center; gap: 8px;
}
.section-title::after {
    content: ""; flex: 1;
    height: 1px; background: rgba(255,255,255,0.08);
}

/* ── Tabs ──────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid rgba(255,255,255,0.1) !important;
    gap: 4px;
}
[data-testid="stTabs"] [role="tab"] {
    color: #8A8A9A !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 8px 16px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: color 0.2s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #C75B39 !important;
    border-bottom: 2px solid #C75B39 !important;
    background: rgba(199,91,57,0.08) !important;
}

/* ── Streamlit widgets ─────────────────────────────────────────────── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stCheckbox"] label {
    color: #8A8A9A !important;
    font-size: 13px !important;
}
.stButton > button {
    background: #C75B39 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── File uploader ─────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed rgba(199,91,57,0.4) !important;
    border-radius: 10px !important;
}

/* ── Success / warning / info ──────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 13px !important;
}
</style>
"""


def inject_styles() -> None:
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
