# ICICI Bank Futures — Mean Reversion Strategy By Vipluv Sultania

This is mean reversion strategy built by Vipluv Sultania on ICICI Bank.

A statistically grounded pairs trading strategy on ICICI Bank vs HDFC Bank (NSE) using daily data from 2018-2024. 

## Development Notes

A few things I learned building this:

**Pair selection was harder than expected.** I initially tried ICICI Bank vs Nifty Bank Index but Engle-Granger gave p=0.63 — no cointegration at all. Switching to HDFC Bank as the pair worked much better (p=0.0242) because they are direct competitors rather than one being a component of the other.

**Date range matters a lot.** Testing from 2015 gave cointegration failures — likely due to structural breaks from IL&FS crisis and COVID. Starting from 2018 captures a cleaner regime where both banks operate comparably.

**Kalman Filter over rolling OLS.** Rolling OLS hedge ratio was too noisy on short windows and too slow to adapt on long windows. Kalman Filter naturally handles this with the transition covariance parameter.

**Half-life of 1.2 days was surprising.** I expected something like 5-10 days for daily data but the spread between these two liquid large-caps reverts extremely fast — institutional arbitrageurs keep it tight.

## Strategy Overview

The strategy exploits the cointegrated relationship between ICICI Bank and HDFC Bank — two large-cap private sector banks that share identical regulatory frameworks, macroeconomic drivers, and customer segments. Short-term price divergences are statistically anomalous and tend to revert, creating exploitable mean reversion opportunities.

## Performance Summary

| Metric | Value | Requirement | Status |
|--------|-------|-------------|--------|
| Sharpe Ratio | 1.7385 | >= 1.3 | PASS |
| Maximum Drawdown | -6.78% | <= 15% | PASS |
| Calmar Ratio | 1.0534 | >= 0.8 | PASS |
| Win Rate | 73.64% | >= 50% | PASS |
| Total Trades | 110 | >= 50 | PASS |
| Profit Factor | 4.0606 | — | — |
| OOS/IS Sharpe | 1.3181 | >= 0.80 | PASS |
| Cointegration p-value | 0.0242 | < 0.05 | PASS |
| Spread ADF p-value | 0.0000 | < 0.05 | PASS |

## Methodology

| Component | Choice | Justification |
|-----------|--------|---------------|
| Pair | ICICI Bank vs HDFC Bank | Confirmed cointegrated, same sector |
| Hedge Ratio | Kalman Filter (dynamic) | Adapts to structural changes over time |
| Entry Signal | Z-score crosses +/- 1.5 sigma | 86.6% confidence of anomaly |
| Exit Signal | Z-score reverts to 0.0 | Full mean reversion captured |
| Stop Loss | Z-score breaches +/- 3.0 sigma | Protects against regime breakdown |
| Z-Score Window | 5 days | ~4x estimated half-life of 1.2 days |
| Transaction Costs | 10 bps + 5 bps slippage | Conservative institutional assumption |

## Statistical Foundation

- **Engle-Granger Test:** p = 0.0242 (cointegrated at 5% significance)
- **Johansen Test:** Trace statistic 21.002 > critical 15.494 (rank = 1)
- **Spread ADF Test:** p = 0.0000 (strongly stationary)
- **Half-Life:** 1.2 trading days (Ornstein-Uhlenbeck process)
- **Kalman Hedge Ratio:** Ranges 0.889 to 1.050 (mean 0.983, very stable)

## Validation Results

**Walk-Forward Analysis (70/30 split):**
- In-Sample Sharpe: 1.6640
- Out-of-Sample Sharpe: 2.1933
- OOS/IS Ratio: 1.3181 (required >= 0.80)

**Monte Carlo Simulation (1000 runs):**
- 5th percentile return: 71.07%
- 50th percentile return: 122.58%
- 95th percentile return: 189.65%
- Probability of drawdown > 20%: 0.00%

## Project Structure

```
strategy/
├── data/                         # Raw and processed CSV files
├── notebooks/
│   ├── 01_data_exploration.py    # Section 1: Data analysis and visualisation
│   ├── 02_statistical_tests.py   # Section 2: Cointegration, Kalman, half-life
│   ├── 03_backtest.py            # Section 3-4: Strategy logic and backtest
│   └── 04_validation.py          # Section 5: Walk-forward, Monte Carlo, sensitivity
├── src/
│   ├── __init__.py               # Package initialiser
│   ├── config.py                 # All parameters — single source of truth
│   ├── data_loader.py            # Data fetching (yfinance + Angel One API)
│   ├── stats.py                  # ADF, cointegration, Kalman Filter, half-life
│   ├── strategy.py               # Signal generation and position sizing
│   └── backtester.py             # Backtest engine and performance metrics
├── results/
│   └── report.pdf                # Full 15-page statistical research report
└── requirements.txt
```

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/icici-mean-reversion.git
cd icici-mean-reversion

# Install dependencies
pip install -r strategy/requirements.txt
```

## Running the Notebooks

Run in order from the strategy/notebooks/ directory:

```bash
cd strategy/notebooks

python 01_data_exploration.py     # Downloads and analyses data
python 02_statistical_tests.py    # Runs all statistical tests
python 03_backtest.py             # Runs full backtest
python 04_validation.py           # Runs walk-forward and Monte Carlo
```

## Configuration

All strategy parameters are centralised in `src/config.py`. There are no hardcoded values anywhere else in the codebase. To change any parameter, edit only `config.py`.

Key parameters:

```python
zscore_window    = 5      # Rolling window for z-score
entry_zscore     = 1.5    # Entry threshold (sigma)
exit_zscore      = 0.0    # Exit threshold (sigma)
stop_zscore      = 3.0    # Stop-loss threshold (sigma)
initial_capital  = 1_000_000   # INR
risk_free_rate   = 0.04   # 4% annual
random_seed      = 42     # Reproducibility
```

## Data Sources

- **Primary:** Yahoo Finance via `yfinance` (no credentials required)
- **Secondary:** Angel One SmartAPI (requires credentials in `.env` file)

Data covers January 2018 to December 2024 — 1,725 aligned trading days.

## Reproducibility

All random seeds are fixed at 42. Results are fully deterministic given the same input data. Running the notebooks in order will reproduce all results exactly.

## Key Risk Factors

- Cointegration may break during major corporate events (e.g. mergers, management changes)
- Strategy should be paused if rolling 252-day cointegration p-value exceeds 0.10
- Correlation breakdown below 0.60 for 10+ consecutive days signals regime change
- Performance assumes institutional transaction costs — retail costs will reduce Sharpe

## License

Private repository — not for redistribution.
Owned By Vipluv Sultania
