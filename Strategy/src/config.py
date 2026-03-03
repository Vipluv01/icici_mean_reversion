"""
config.py  —  Central configuration for ICICI Bank Futures Mean Reversion Strategy.
ALL parameters live here. No hardcoded values anywhere else in the codebase.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"


# ── Data ───────────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    ticker_x: str = "ICICIBANK.NS"     # Primary: ICICI Bank
    ticker_y: str = "^NSEBANK"         # Pair:    Nifty Bank Index
    start_date: str = "2015-01-01"
    end_date:   str = "2024-12-31"
    # Angel One API tokens (set via .env)
    angel_token_x:  str = os.getenv("ANGEL_TOKEN_ICICI",     "1660")
    angel_token_y:  str = os.getenv("ANGEL_TOKEN_BANKNIFTY", "26009")
    angel_exchange: str = "NSE"
    angel_interval: str = "ONE_DAY"
    raw_file_x:     str = "icici_daily.csv"
    raw_file_y:     str = "banknifty_daily.csv"
    processed_file: str = "processed_pair.csv"


# ── Statistical Tests ──────────────────────────────────────────────────────
@dataclass
class StatsConfig:
    alpha: float = 0.05               # Significance level for all tests
    adf_maxlags: int = 10             # ADF max lags (AIC selection)
    johansen_maxlags: int = 1
    rolling_corr_window: int = 60     # Trading days


# ── Kalman Filter ──────────────────────────────────────────────────────────
@dataclass
class KalmanConfig:
    observation_covariance: float = 1.0
    transition_covariance:  float = 0.01   # Lower = slower adaptation
    initial_state_mean: list = field(default_factory=lambda: [1.0, 0.0])


# ── Strategy ───────────────────────────────────────────────────────────────
@dataclass
class StrategyConfig:
    zscore_window: int   = 60     # Rolling window for z-score computation
    entry_zscore:  float = 2.0    # Enter when |z| > this
    exit_zscore:   float = 0.5    # Exit when |z| < this (mean reversion)
    stop_zscore:   float = 3.5    # Hard stop if |z| blows out past this
    transaction_cost_bps: float = 10.0   # Round-trip, bps
    slippage_bps:         float = 5.0    # One-way, bps
    position_fraction:    float = 0.95   # Fraction of equity per trade


# ── Backtest ───────────────────────────────────────────────────────────────
@dataclass
class BacktestConfig:
    initial_capital:  float = 1_000_000.0   # INR 10 lakh
    train_ratio:      float = 0.70
    test_ratio:       float = 0.30
    risk_free_rate:   float = 0.065          # India 10Y yield approx
    trading_days:     int   = 252
    monte_carlo_sims: int   = 1000
    random_seed:      int   = 42


# ── Validation ─────────────────────────────────────────────────────────────
@dataclass
class ValidationConfig:
    sensitivity_range: float = 0.20
    sensitivity_params: list = field(default_factory=lambda: [
        "entry_zscore", "exit_zscore", "zscore_window", "stop_zscore"
    ])
    oos_sharpe_min_ratio:   float = 0.80
    mc_drawdown_threshold:  float = 0.20


# ── Minimum Performance Thresholds (auto-fail if not met) ──────────────────
@dataclass
class ThresholdConfig:
    min_sharpe:           float = 1.3
    max_drawdown:         float = 0.15
    min_calmar:           float = 0.8
    min_trades:           int   = 50
    min_win_rate:         float = 0.50
    max_single_trade_pct: float = 0.25
    cointegration_pvalue: float = 0.05
    spread_adf_pvalue:    float = 0.05


# ── Single import point ─────────────────────────────────────────────────────
data_cfg       = DataConfig()
stats_cfg      = StatsConfig()
kalman_cfg     = KalmanConfig()
strategy_cfg   = StrategyConfig()
backtest_cfg   = BacktestConfig()
validation_cfg = ValidationConfig()
threshold_cfg  = ThresholdConfig()
