
# ICICI Bank Futures — Mean Reversion Strategy
### Round 1: Quantitative Developer Assessment | trade360.ai

## Setup
```bash
pip install -r requirements.txt
```

## Run Notebooks in order
```
01_data_exploration -> 02_statistical_tests -> 03_backtest -> 04_validation
```

## Strategy Summary
- Instruments: ICICI Bank + Nifty Bank (NSE)
- Data: Daily OHLCV, 2015-2024
- Hedge Ratio: Kalman Filter (dynamic)
- Entry: Z-score crosses +/- 2.0 sigma
- Exit: Z-score reverts to +/- 0.5 sigma
- Stop: Z-score breaches +/- 3.5 sigma

## Config
All parameters in src/config.py — no hardcoded values anywhere.

## Reproducibility
Random seed fixed at 42. Results are deterministic.
```

3. Save it

---

Now your full file count is **12 files** across the repo:
```
icici-mean-reversion/
├── README.md
├── strategy/
│   ├── requirements.txt
│   ├── data/          (.gitkeep)
│   ├── results/       (.gitkeep)
│   ├── src/           (6 files)
│   └── notebooks/     (4 files)
```

Commit in GitHub Desktop:
```
Add README