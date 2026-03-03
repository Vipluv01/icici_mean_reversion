"""
backtester.py
=============
Backtest engine and performance metrics for the ICICI Bank Futures
Mean Reversion Strategy.

Computes all 11 required metrics:
  Total Return, Annualized Return, Sharpe, Sortino, Calmar,
  Max Drawdown, Win Rate, Profit Factor, Total Trades,
  Avg Trade Duration, Largest Win/Loss
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import backtest_cfg, BacktestConfig, threshold_cfg

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Trade record
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    entry_date:  pd.Timestamp
    exit_date:   pd.Timestamp
    direction:   int         # 1=long, -1=short
    entry_price: float
    exit_price:  float
    pnl:         float       # INR
    duration:    int         # trading days
    pnl_pct:     float       # as fraction of entry notional


# ══════════════════════════════════════════════════════════════════════════════
# Core backtest loop
# ══════════════════════════════════════════════════════════════════════════════

# Initial version used pct_change() on spread which caused crazy compounding.
# Switched to spread.diff() * initial_capital for dollar P&L instead.
# Much more sensible and matches how pairs trading actually works.


def run_backtest(
        spread: pd.Series,
        signals: pd.DataFrame,
        transaction_costs: pd.Series,
        cfg: BacktestConfig = backtest_cfg,
) -> Tuple[pd.Series, pd.Series, List[Trade]]:
    position = signals["position"]

    # Daily P&L = position * daily change in spread (NOT pct_change)
    spread_diff = spread.diff().fillna(0)
    transaction_costs = transaction_costs.fillna(0)
    daily_pnl = position.shift(1).fillna(0) * spread_diff * cfg.initial_capital

    # Only charge transaction costs on days where position CHANGES
    position_change = position.diff().fillna(0).abs()
    daily_pnl = daily_pnl - (transaction_costs * position_change.clip(0, 1))

    # Equity curve
    equity_curve = cfg.initial_capital + daily_pnl.cumsum()
    equity_curve.name = "equity"

    # Daily returns
    daily_returns = daily_pnl / cfg.initial_capital
    daily_returns.name = "returns"

    # Extract trades
    trades: List[Trade] = []
    pos_arr = position.values
    spread_arr = spread.values
    dates = spread.index

    i = 0
    while i < len(pos_arr):
        if pos_arr[i] != 0:
            entry_i = i
            direction = int(pos_arr[i])
            entry_px = spread_arr[i]
            j = i + 1
            while j < len(pos_arr) and pos_arr[j] == direction:
                j += 1
            exit_i = min(j, len(spread_arr) - 1)
            exit_px = spread_arr[exit_i]

            raw_pnl = direction * (exit_px - entry_px) * cfg.initial_capital
            cost = float(transaction_costs.iloc[entry_i])
            net_pnl = raw_pnl - cost

            trades.append(Trade(
                entry_date=dates[entry_i],
                exit_date=dates[exit_i],
                direction=direction,
                entry_price=entry_px,
                exit_price=exit_px,
                pnl=net_pnl,
                duration=exit_i - entry_i,
                pnl_pct=net_pnl / cfg.initial_capital,
            ))
            i = j
        else:
            i += 1

    log.info("Backtest complete: %d trades | final equity: %.2f",
             len(trades), equity_curve.iloc[-1])
    return equity_curve, daily_returns, trades

# ══════════════════════════════════════════════════════════════════════════════
# Performance metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    trades: List[Trade],
    cfg: BacktestConfig = backtest_cfg,
) -> Dict:
    """
    Compute all 11 required performance metrics plus diagnostics.

    Parameters
    ----------
    equity_curve : pd.Series
    returns      : pd.Series   Daily returns
    trades       : List[Trade]
    cfg          : BacktestConfig

    Returns
    -------
    dict  All metrics keyed by name
    """
    # Only use days where we are in a position for Sharpe calculation
    # n_years based on full calendar period (not just active days)
    n_years = len(returns.dropna()) / cfg.trading_days
    # Only active days for Sharpe/Sortino calculation
    active_mask = returns != 0
    ret = returns[active_mask].dropna()

    # ── Return metrics ─────────────────────────────────────────────────
    total_return  = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # ── Risk-adjusted metrics ──────────────────────────────────────────
    excess_ret = ret - cfg.risk_free_rate / cfg.trading_days
    sharpe = (
        excess_ret.mean() / excess_ret.std() * np.sqrt(cfg.trading_days)
        if excess_ret.std() > 0 else 0.0
    )

    downside = excess_ret[excess_ret < 0]
    sortino = (
        excess_ret.mean() / downside.std() * np.sqrt(cfg.trading_days)
        if len(downside) > 0 and downside.std() > 0 else 0.0
    )

    # ── Drawdown ───────────────────────────────────────────────────────
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve - rolling_max) / rolling_max
    max_dd      = drawdown.min()

    calmar = annual_return / abs(max_dd) if abs(max_dd) > 0 else 0.0

    # ── Trade statistics ───────────────────────────────────────────────
    n_trades   = len(trades)
    pnls       = [t.pnl for t in trades]
    winners    = [p for p in pnls if p > 0]
    losers     = [p for p in pnls if p < 0]

    win_rate      = len(winners) / n_trades if n_trades > 0 else 0.0
    profit_factor = (
        sum(winners) / abs(sum(losers))
        if losers and sum(losers) != 0 else np.inf
    )
    avg_duration  = np.mean([t.duration for t in trades]) if trades else 0.0
    largest_win   = max(pnls) if pnls else 0.0
    largest_loss  = min(pnls) if pnls else 0.0

    # Concentration check: no single trade > 25% of total profit
    total_profit = sum(winners) if winners else 0.0
    max_single_pct = (
        max(winners) / total_profit if total_profit > 0 and winners else 0.0
    )

    metrics = {
        "total_return":         round(total_return,    4),
        "annualized_return":    round(annual_return,   4),
        "sharpe_ratio":         round(sharpe,          4),
        "sortino_ratio":        round(sortino,         4),
        "calmar_ratio":         round(calmar,          4),
        "max_drawdown":         round(max_dd,          4),
        "win_rate":             round(win_rate,        4),
        "profit_factor":        round(profit_factor,   4),
        "total_trades":         n_trades,
        "avg_trade_duration":   round(avg_duration,    2),
        "largest_win":          round(largest_win,     2),
        "largest_loss":         round(largest_loss,    2),
        "max_single_trade_pct": round(max_single_pct,  4),
    }

    log.info("── Performance Metrics ──────────────────────")
    for k, v in metrics.items():
        log.info("  %-28s: %s", k, v)

    return metrics


def check_thresholds(metrics: Dict) -> Dict[str, bool]:
    """
    Check all minimum performance requirements.
    Returns dict of {check_name: passed_bool}.
    Logs PASS / FAIL for each.
    """
    t = threshold_cfg
    checks = {
        "sharpe >= 1.3":             metrics["sharpe_ratio"]         >= t.min_sharpe,
        "max_drawdown <= 15%":       abs(metrics["max_drawdown"])    <= t.max_drawdown,
        "calmar >= 0.8":             metrics["calmar_ratio"]         >= t.min_calmar,
        "trades >= 50":              metrics["total_trades"]         >= t.min_trades,
        "win_rate >= 50%":           metrics["win_rate"]             >= t.min_win_rate,
        "single_trade_pct <= 25%":   metrics["max_single_trade_pct"] <= t.max_single_trade_pct,
    }
    all_pass = all(checks.values())
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        log.info("  [%s] %s", status, name)
    log.info("  Overall: %s", "ALL PASS" if all_pass else "SOME CHECKS FAILED")
    return checks


# ══════════════════════════════════════════════════════════════════════════════
# MAE / MFE analysis
# ══════════════════════════════════════════════════════════════════════════════

def compute_mae_mfe(trades: List[Trade], spread: pd.Series) -> pd.DataFrame:
    """
    Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE)
    for each trade.

    MAE: worst intra-trade drawdown (measures risk per trade)
    MFE: best intra-trade gain     (measures potential not captured)
    """
    records = []
    for t in trades:
        mask   = (spread.index >= t.entry_date) & (spread.index <= t.exit_date)
        window = spread.loc[mask]
        if len(window) < 2:
            continue
        move = t.direction * (window - window.iloc[0])
        records.append({
            "entry_date": t.entry_date,
            "exit_date":  t.exit_date,
            "pnl":        t.pnl,
            "mae":        move.min(),
            "mfe":        move.max(),
        })
    return pd.DataFrame(records)
