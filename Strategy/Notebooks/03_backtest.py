# %% [markdown]
# # Notebook 03: Backtest
# ICICI Bank Futures Mean Reversion Strategy | Report Sections 3 & 4
#
# Covers: Strategy logic, all 11 performance metrics,
# equity curve, drawdown, monthly heatmap, trade distribution, MAE/MFE.

# %% Setup
import sys; sys.path.append("..")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec, seaborn as sns
import calendar

from src.config import strategy_cfg, backtest_cfg, threshold_cfg, RESULTS_DIR
from src.data_loader import load_processed
from src.stats import compute_zscore
from src.strategy import generate_signals, compute_position_sizes, compute_transaction_costs
from src.backtester import run_backtest, compute_metrics, check_thresholds, compute_mae_mfe

np.random.seed(backtest_cfg.random_seed)
sns.set_theme(style="darkgrid"); plt.rcParams["figure.dpi"] = 120
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% Load stats results from notebook 02
import pickle
results_02 = pd.read_pickle(RESULTS_DIR / "02_stats_results.pkl")
spread     = results_02["spread"]
zscore     = results_02["zscore"]
print(f"Spread loaded: {len(spread)} rows")

# %% [markdown]
# ## 3.1  Strategy Logic — Signal Generation

# %%
print("Strategy Parameters:")
print(f"  Entry z-score  : +/- {strategy_cfg.entry_zscore}")
print(f"  Exit  z-score  : +/- {strategy_cfg.exit_zscore}")
print(f"  Stop  z-score  : +/- {strategy_cfg.stop_zscore}")
print(f"  Z-score window : {strategy_cfg.zscore_window} days")
print(f"  Transaction costs: {strategy_cfg.transaction_cost_bps} bps (round-trip)")
print(f"  Slippage       : {strategy_cfg.slippage_bps} bps (one-way)")

signals = generate_signals(zscore, cfg=strategy_cfg)
print(f"\nSignals generated: {len(signals)} bars")
print(f"  Long  signals: {(signals['signal']==1).sum()}")
print(f"  Short signals: {(signals['signal']==-1).sum()}")
print(f"  Flat  signals: {(signals['signal']==0).sum()}")

# %% [markdown]
# ## 3.2  Position Sizing & Transaction Costs

# %%
equity_init = pd.Series(backtest_cfg.initial_capital, index=spread.index)
position_sizes   = compute_position_sizes(signals, equity_init, cfg=strategy_cfg)
transaction_costs = compute_transaction_costs(position_sizes, cfg=strategy_cfg)

print(f"Initial capital: INR {backtest_cfg.initial_capital:,.0f}")
print(f"Position fraction: {strategy_cfg.position_fraction:.0%}")
print(f"Total transaction costs: INR {transaction_costs.sum():,.2f}")

# %% [markdown]
# ## 3.3  Run Backtest

# %%
equity_curve, daily_returns, trades = run_backtest(
    spread, signals, transaction_costs, cfg=backtest_cfg
)

print(f"\nBacktest complete:")
print(f"  Period    : {equity_curve.index[0].date()} to {equity_curve.index[-1].date()}")
print(f"  Start NAV : INR {equity_curve.iloc[0]:>14,.2f}")
print(f"  End NAV   : INR {equity_curve.iloc[-1]:>14,.2f}")
print(f"  Trades    : {len(trades)}")

# %% [markdown]
# ## 3.4  Performance Metrics (All 11 Required)

# %%
metrics = compute_metrics(equity_curve, daily_returns, trades, cfg=backtest_cfg)

print("\n" + "="*55)
print("PERFORMANCE METRICS")
print("="*55)
rows = [
    ("Total Return",          f"{metrics['total_return']:.2%}"),
    ("Annualized Return",     f"{metrics['annualized_return']:.2%}"),
    ("Sharpe Ratio",          f"{metrics['sharpe_ratio']:.4f}  (min: {threshold_cfg.min_sharpe})"),
    ("Sortino Ratio",         f"{metrics['sortino_ratio']:.4f}"),
    ("Calmar Ratio",          f"{metrics['calmar_ratio']:.4f}  (min: {threshold_cfg.min_calmar})"),
    ("Maximum Drawdown",      f"{metrics['max_drawdown']:.2%}  (max: {threshold_cfg.max_drawdown:.0%})"),
    ("Win Rate",              f"{metrics['win_rate']:.2%}  (min: {threshold_cfg.min_win_rate:.0%})"),
    ("Profit Factor",         f"{metrics['profit_factor']:.4f}"),
    ("Total Trades",          f"{metrics['total_trades']}  (min: {threshold_cfg.min_trades})"),
    ("Avg Trade Duration",    f"{metrics['avg_trade_duration']:.1f} days"),
    ("Largest Win",           f"INR {metrics['largest_win']:,.2f}"),
    ("Largest Loss",          f"INR {metrics['largest_loss']:,.2f}"),
    ("Max Single Trade %",    f"{metrics['max_single_trade_pct']:.2%}  (max: {threshold_cfg.max_single_trade_pct:.0%})"),
]
for k, v in rows:
    print(f"  {k:<26} : {v}")

print("\nThreshold Checks:")
checks = check_thresholds(metrics)
all_pass = all(checks.values())
print(f"\n{'ALL CHECKS PASSED' if all_pass else 'WARNING: SOME CHECKS FAILED'}")

# %% [markdown]
# ## 3.5  Required Visual 1 — Equity Curve

# %%
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(equity_curve.index, equity_curve.values, color="#4299e1", lw=1.5, label="Strategy NAV")
ax.axhline(backtest_cfg.initial_capital, color="gray", ls="--", lw=1, label="Initial Capital")
ax.fill_between(equity_curve.index, backtest_cfg.initial_capital, equity_curve.values,
                where=(equity_curve.values > backtest_cfg.initial_capital),
                alpha=0.15, color="#68d391")
ax.fill_between(equity_curve.index, backtest_cfg.initial_capital, equity_curve.values,
                where=(equity_curve.values < backtest_cfg.initial_capital),
                alpha=0.15, color="#fc8181")
ax.set_title("Equity Curve — ICICI Bank Futures Mean Reversion Strategy")
ax.set_ylabel("Portfolio Value (INR)"); ax.legend()
plt.tight_layout(); plt.savefig(RESULTS_DIR / "03_equity_curve.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 3.6  Required Visual 2 — Drawdown Curve

# %%
rolling_max = equity_curve.cummax()
drawdown    = (equity_curve - rolling_max) / rolling_max

fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(drawdown.index, drawdown.values, 0, color="#fc8181", alpha=0.7, label="Drawdown")
ax.plot(drawdown.index, drawdown.values, color="#e53e3e", lw=0.8, alpha=0.9)
ax.axhline(-threshold_cfg.max_drawdown, color="orange", ls="--", lw=1.5,
           label=f"Max allowed ({threshold_cfg.max_drawdown:.0%})")
ax.set_title("Portfolio Drawdown"); ax.set_ylabel("Drawdown (%)"); ax.legend()
plt.tight_layout(); plt.savefig(RESULTS_DIR / "03_drawdown.png", bbox_inches="tight"); plt.show()

print(f"Maximum Drawdown: {drawdown.min():.2%}")

# %% [markdown]
# ## 3.7  Required Visual 3 — Monthly Returns Heatmap

# %%
monthly_ret = daily_returns.resample("ME").apply(lambda r: (1 + r).prod() - 1)
monthly_df  = monthly_ret.to_frame("return")
monthly_df["year"]  = monthly_df.index.year
monthly_df["month"] = monthly_df.index.month
pivot = monthly_df.pivot(index="year", columns="month", values="return")
pivot.columns = [calendar.month_abbr[m] for m in pivot.columns]

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(pivot * 100, annot=True, fmt=".1f", center=0,
            cmap="RdYlGn", ax=ax, linewidths=0.5,
            cbar_kws={"label": "Monthly Return (%)"})
ax.set_title("Monthly Returns Heatmap  (%)")
ax.set_xlabel("Month"); ax.set_ylabel("Year")
plt.tight_layout(); plt.savefig(RESULTS_DIR / "03_monthly_heatmap.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 3.8  Required Visual 4 — Trade Return Distribution

# %%
pnls = [t.pnl for t in trades]
pnl_pcts = [t.pnl_pct * 100 for t in trades]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(pnls, bins=40, color="#4299e1", alpha=0.7, edgecolor="white")
axes[0].axvline(0, color="red", ls="--", lw=1.5)
axes[0].axvline(np.mean(pnls), color="#68d391", ls="--", lw=1.5, label=f"Mean = {np.mean(pnls):.1f}")
axes[0].set_title("Trade P&L Distribution (INR)"); axes[0].set_xlabel("P&L (INR)"); axes[0].legend()

axes[1].hist(pnl_pcts, bins=40, color="#9f7aea", alpha=0.7, edgecolor="white")
axes[1].axvline(0, color="red", ls="--", lw=1.5)
axes[1].set_title("Trade Return Distribution (%)"); axes[1].set_xlabel("Return (%)")

plt.tight_layout(); plt.savefig(RESULTS_DIR / "03_trade_dist.png", bbox_inches="tight"); plt.show()
print(f"Trade P&L: mean={np.mean(pnls):.2f}  median={np.median(pnls):.2f}  std={np.std(pnls):.2f}")

# %% [markdown]
# ## 3.9  Required Visual 5 — MAE / MFE Analysis

# %%
mae_mfe = compute_mae_mfe(trades, spread)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
scatter_kw = dict(alpha=0.5, s=20, edgecolors="none")

colors = ["#68d391" if p > 0 else "#fc8181" for p in mae_mfe["pnl"]]
axes[0].scatter(mae_mfe["mae"], mae_mfe["pnl"], c=colors, **scatter_kw)
axes[0].axhline(0, color="gray", ls="--", lw=1); axes[0].axvline(0, color="gray", ls="--", lw=1)
axes[0].set_title("MAE vs Trade P&L")
axes[0].set_xlabel("Max Adverse Excursion"); axes[0].set_ylabel("P&L")

axes[1].scatter(mae_mfe["mfe"], mae_mfe["pnl"], c=colors, **scatter_kw)
axes[1].axhline(0, color="gray", ls="--", lw=1)
axes[1].set_title("MFE vs Trade P&L")
axes[1].set_xlabel("Max Favorable Excursion"); axes[1].set_ylabel("P&L")

plt.tight_layout(); plt.savefig(RESULTS_DIR / "03_mae_mfe.png", bbox_inches="tight"); plt.show()

# %% Save backtest results for validation notebook
import pickle
backtest_results = {
    "equity_curve":   equity_curve,
    "daily_returns":  daily_returns,
    "trades":         trades,
    "metrics":        metrics,
    "signals":        signals,
    "drawdown":       drawdown,
    "spread":         spread,
    "zscore":         zscore,
}
pd.to_pickle(backtest_results, RESULTS_DIR / "03_backtest_results.pkl")
print("\nBacktest results saved to results/03_backtest_results.pkl")
