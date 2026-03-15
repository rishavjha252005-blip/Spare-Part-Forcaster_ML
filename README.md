# Spare Parts Forecaster Pro

> ML-powered demand forecasting and inventory optimisation for spare parts.

---

## Overview

**Spare Parts Forecaster Pro** is a Streamlit-based end-to-end forecasting system that automates inventory decisions for spare parts. It classifies demand patterns, runs and backtests multiple statistical and ML models, simulates lead-time demand distributions, and produces segment-aware safety stock and reorder point recommendations — all from a single CSV upload.

---

## Quick start

```bash
# 1. Clone or extract the project
cd spf_pro

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python -m streamlit run app.py
```

Then open `http://localhost:8501` in your browser. Click **Load demo data** to explore immediately without uploading a file.

> **Note:** Run all commands from the `spf_pro/` folder — the one that contains `app.py`.

---

## Input format

Upload a CSV file with exactly three columns:

| Column | Type | Description |
|---|---|---|
| `date` | date string | Period date (e.g. `2023-01-01`) |
| `part_id` | string | Unique spare part identifier |
| `demand` | integer / float | Demand quantity for that period |

**Example:**

```csv
date,part_id,demand
2022-01-01,PUMP-001,0
2022-01-08,PUMP-001,5
2022-01-15,PUMP-001,0
2022-01-22,BEAR-104,12
```

---

## Pipeline

The system processes each part through five sequential stages:

```
CSV upload
    │
    ▼
Stage 1 — Demand classification
    ADI + CV² → Syntetos-Boylan matrix → Smooth / Erratic / Intermittent / Lumpy
    │
    ▼
Stage 2 — Model selection & backtesting
    SES · Croston · SBA · LightGBM · Prophet · Ensemble
    Walk-forward cross-validation → MASE + sMAPE → best model selected
    │
    ▼
Stage 3 — Forecast output
    Point forecast · 80% & 95% prediction intervals · SHAP feature importance
    │
    ▼
Stage 4 — Bootstrap simulation (n = 1,000)
    Lead-time demand distribution · mean · std · 90th / 95th / 99th percentiles
    │
    ▼
Stage 5 — Inventory policy
    ABC-XYZ segmentation · safety stock · reorder point · holding & stockout costs
    + KS-test drift detection → retrain recommendation
```

---

## Features

| Feature | Description |
|---|---|
| Demand classification | Syntetos-Boylan matrix using ADI and CV² |
| Statistical models | Simple Exponential Smoothing, Croston, Syntetos-Boylan Approximation |
| ML models | LightGBM (lag + rolling features), Prophet (trend + seasonality) |
| Ensemble | Average of all available model forecasts |
| Backtesting | Walk-forward cross-validation with MASE and sMAPE scoring |
| Prediction intervals | 80% and 95% quantile regression intervals |
| SHAP explainability | Feature importance for LightGBM predictions |
| Bootstrap simulation | 1,000-trial Monte Carlo lead-time demand distribution |
| ABC-XYZ segmentation | 9-cell value × variability matrix with differentiated service levels |
| Inventory policy | Safety stock, reorder point, holding cost, and stockout risk estimates |
| Sensitivity analysis | Service level trade-off table across z-scores |
| Drift detection | Kolmogorov-Smirnov test comparing baseline vs recent demand |
| Portfolio scan | Drift detection across all parts simultaneously |
| Downloads | Forecast comparison CSV, inventory policy CSV |
| Demo data | Built-in 5-part, 60-period synthetic dataset |

---

## Application tabs

| Tab | What it shows |
|---|---|
| **Data** | Demand history chart, raw table, portfolio summary |
| **Classification** | Syntetos-Boylan quadrant plot, ABC-XYZ segment badge, portfolio heatmap |
| **Forecast** | All model forecasts, history overlay with confidence bands, SHAP chart |
| **Model comparison** | MASE / sMAPE leaderboard, full results table, CSV download |
| **Simulation** | Bootstrap histogram with mean, reorder point, and percentile markers |
| **Inventory** | Safety stock / reorder point / cost gauge, sensitivity table, CSV download |
| **Drift** | KS-test status, rolling mean split chart, full portfolio drift scan |

---

## Project structure

```
spf_pro/
├── app.py                                        ← Streamlit entry point (7 tabs)
├── requirements.txt
├── README.md
└── spare_parts_forecaster_pro/
    ├── config/
    │   └── settings.py                           ← all constants and hyper-parameters
    ├── core/
    │   ├── types.py                              ← typed dataclasses (pipeline contracts)
    │   ├── models/
    │   │   ├── statistical.py                    ← SES, Croston, SBA
    │   │   ├── features.py                       ← lag, rolling, calendar feature builder
    │   │   ├── lgbm_model.py                     ← LightGBM + SHAP
    │   │   └── prophet_model.py                  ← Prophet
    │   └── pipeline/
    │       ├── classification.py                 ← Stage 1: ADI / CV² classification
    │       ├── evaluation.py                     ← MASE, sMAPE, walk-forward CV
    │       ├── forecast_engine.py                ← Stage 2–3: model dispatch + backtest
    │       ├── simulation.py                     ← Stage 4: bootstrap simulation
    │       ├── segmentation.py                   ← ABC-XYZ with 9 service-level z-scores
    │       ├── inventory.py                      ← Stage 5: safety stock + cost estimates
    │       └── drift.py                          ← KS-test drift detection
    └── ui/
        ├── styles.py                             ← custom CSS (dark theme, DM Sans)
        ├── sidebar.py                            ← sidebar settings panel
        ├── components.py                         ← all Plotly charts and metric panels
        └── sample_data.py                        ← demo dataset generator
```

---

## Configuration

All thresholds and hyper-parameters are centralised in `spare_parts_forecaster_pro/config/settings.py`. No hunting through multiple files to change a value.

| Setting | Default | Description |
|---|---|---|
| `ADI_THRESHOLD` | 1.32 | Syntetos-Boylan ADI boundary |
| `CV2_THRESHOLD` | 0.49 | Syntetos-Boylan CV² boundary |
| `SES_ALPHA` | 0.3 | SES smoothing factor |
| `SBA_ALPHA` | 0.1 | SBA bias-correction factor |
| `BOOTSTRAP_N_SIMULATIONS` | 1,000 | Monte Carlo trials |
| `BOOTSTRAP_LEAD_TIME` | 7 | Lead-time periods |
| `SERVICE_LEVEL_Z` | 1.65 | Default z-score (≈ 95% service level) |
| `HOLDING_COST_RATE` | 0.25 | Annual holding cost as fraction of unit cost |
| `STOCKOUT_COST_RATE` | 2.0 | Stockout cost as multiple of unit cost |
| `BACKTEST_N_SPLITS` | 3 | Walk-forward CV folds |
| `DRIFT_WINDOW` | 12 | Periods used for recent demand window |
| `DRIFT_KS_ALPHA` | 0.05 | KS-test significance threshold |

The sidebar also exposes the most common settings at runtime without requiring a code change.

---

## Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
plotly>=5.20.0
scikit-learn>=1.4.0
lightgbm>=4.3.0
shap>=0.44.0
statsmodels>=0.14.0
scipy>=1.12.0
prophet>=1.1.5
```

> **Prophet on Windows** — if installation fails, try:
> ```bash
> pip install prophet --no-deps
> pip install pystan cmdstanpy holidays convertdate lunarcalendar
> ```

---

## Metrics reference

| Metric | Formula | Notes |
|---|---|---|
| **ADI** | Total periods ÷ non-zero periods | Higher = more intermittent |
| **CV²** | (std / mean)² of non-zero demand | Higher = more variable |
| **MASE** | MAE ÷ naïve in-sample MAE | Scale-free; handles zero demand correctly |
| **sMAPE** | Mean of 2\|A−F\| ÷ (\|A\|+\|F\|) × 100 | Symmetric; avoids division by zero |
| **Safety stock** | z × σ\_LTD | σ\_LTD from bootstrap simulation |
| **Reorder point** | μ\_LTD + safety stock | μ\_LTD from bootstrap simulation |

---

## Extending the system

**Add a new forecasting model:**
1. Create a function `my_model(demand: np.ndarray) -> float` in `core/models/`.
2. Register it in `_MODEL_REGISTRY` inside `core/pipeline/forecast_engine.py`.
3. That's it — backtesting, ensemble, and UI pick it up automatically.

**Change a threshold or parameter:**
Edit `spare_parts_forecaster_pro/config/settings.py` only. No other file needs touching.

---

## License

For academic and internal prototype use.
