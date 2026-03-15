# config/settings.py
# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for every threshold and hyper-parameter.
# ─────────────────────────────────────────────────────────────────────────────

# ── Demand classification (Syntetos-Boylan matrix) ────────────────────────────
ADI_THRESHOLD: float = 1.32
CV2_THRESHOLD: float = 0.49

# ── Statistical model parameters ─────────────────────────────────────────────
SES_ALPHA: float = 0.3
SBA_ALPHA: float = 0.1

# ── ML model parameters ───────────────────────────────────────────────────────
LGBM_PARAMS: dict = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
}

# Lag and rolling window sizes for feature engineering
LAG_PERIODS: list[int] = [1, 2, 3, 4, 7, 14]
ROLLING_WINDOWS: list[int] = [3, 7, 14]

# Minimum non-zero observations required to train ML models
MIN_NONZERO_FOR_ML: int = 8

# ── Bootstrap / risk simulation ───────────────────────────────────────────────
BOOTSTRAP_N_SIMULATIONS: int = 1_000
BOOTSTRAP_LEAD_TIME: int = 7

# ── Inventory policy ──────────────────────────────────────────────────────────
SERVICE_LEVEL_Z: float = 1.65   # ≈ 95 % cycle service level
HOLDING_COST_RATE: float = 0.25  # 25% of unit cost per year
STOCKOUT_COST_RATE: float = 2.0  # 2× unit cost per stockout event

# ── Backtesting / evaluation ──────────────────────────────────────────────────
BACKTEST_N_SPLITS: int = 3
BACKTEST_MIN_TRAIN: int = 12   # minimum periods in each training fold

# ── Drift detection ───────────────────────────────────────────────────────────
DRIFT_WINDOW: int = 12          # periods to compare against baseline
DRIFT_KS_ALPHA: float = 0.05    # significance level for KS test
