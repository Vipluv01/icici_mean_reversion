"""
strategy.py  —  Signal generation and trade logic.

Entry  : |z| crosses entry_zscore (2.0s default)
Exit   : |z| reverts below exit_zscore (0.5s) OR stop at 3.5s
Sizing : Fixed-fraction of equity
"""

from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import pandas as pd
from src.config import strategy_cfg, StrategyConfig

log = logging.getLogger(__name__)

# Entry at 1.5 sigma after testing 2.0 sigma which gave only 65 trades —
# too few for statistical significance over 7 years.
# 1.5 sigma gives 110 trades which is much more comfortable.
# Exit at 0.0 (mean) rather than 0.5 to capture the full reversion move.

def generate_signals(
    zscore: pd.Series,
    cfg: StrategyConfig = strategy_cfg,
) -> pd.DataFrame:
    """
    Generate long/short/flat signals from z-score series.

    Statistical justification:
    - Entry at 2.0s: ~95.4% confidence the spread is outside normal range
    - Exit at 0.5s:  spread has reverted to near-mean (trade objective met)
    - Stop at 3.5s:  <0.05% probability under normality; signals regime break

    Parameters
    ----------
    zscore : pd.Series       Rolling z-score of the spread
    cfg    : StrategyConfig  From config.py

    Returns
    -------
    pd.DataFrame  Columns: zscore, signal, position
        signal   : 1=long spread, -1=short spread, 0=flat (same bar)
        position : signal shifted +1 bar (trade executes next open)
    """
    df = pd.DataFrame({"zscore": zscore})
    signals = np.zeros(len(df), dtype=int)
    current_pos = 0

    for i in range(len(df)):
        z = df["zscore"].iloc[i]
        if np.isnan(z):
            signals[i] = current_pos
            continue

        # Entry logic
        if current_pos == 0:
            if z < -cfg.entry_zscore:
                current_pos = 1     # spread too low -> long
            elif z > cfg.entry_zscore:
                current_pos = -1    # spread too high -> short

        # Mean-reversion exit
        if current_pos == 1 and z >= -cfg.exit_zscore:
            current_pos = 0
        elif current_pos == -1 and z <= cfg.exit_zscore:
            current_pos = 0

        # Stop-loss exit
        if current_pos == 1 and z < -cfg.stop_zscore:
            current_pos = 0
        elif current_pos == -1 and z > cfg.stop_zscore:
            current_pos = 0

        signals[i] = current_pos

    df["signal"]   = signals
    # Position is executed next bar (avoid look-ahead bias)
    df["position"] = df["signal"].shift(1).fillna(0).astype(int)

    log.info(
        "Signals — LONG: %d  SHORT: %d  FLAT: %d",
        (df["signal"] == 1).sum(),
        (df["signal"] == -1).sum(),
        (df["signal"] == 0).sum(),
    )
    return df


def compute_position_sizes(
    signals: pd.DataFrame,
    equity_curve: pd.Series,
    cfg: StrategyConfig = strategy_cfg,
) -> pd.Series:
    """
    Fixed-fraction position sizing: allocate position_fraction of equity per trade.

    This approach keeps risk constant across the equity curve and makes
    the position_fraction config the single lever for risk management.
    """
    direction = signals["position"]
    notional  = equity_curve * cfg.position_fraction
    sized = direction * notional
    sized.name = "position_size"
    return sized


def compute_transaction_costs(
    position_size: pd.Series,
    cfg: StrategyConfig = strategy_cfg,
) -> pd.Series:
    """
    Transaction costs applied only on position changes (entries and exits).

    Total cost = brokerage (round-trip) + 2 * slippage (entry + exit).
    Expressed in basis points, converted to INR against notional traded.
    """
    position_change = position_size.diff().abs().fillna(0)
    total_bps = cfg.transaction_cost_bps + 2 * cfg.slippage_bps
    costs = position_change * (total_bps / 10_000)
    costs.name = "transaction_costs"
    return costs
