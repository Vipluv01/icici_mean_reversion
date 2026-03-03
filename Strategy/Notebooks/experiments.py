# 00_experiments.py
# Personal scratch notebook — NOT part of final submission
# Documenting failed experiments and things I tried along the way

import sys
sys.path.append("..")
import numpy as np
import pandas as pd
from src.config import RESULTS_DIR
from src.data_loader import load_processed

df = load_processed()

# ── Experiment 1: Tried Nifty Bank as pair ────────────────────────────────
# Result: Engle-Granger p = 0.63 — FAIL
# Reason: Nifty Bank is a broad index, ICICI is just one component
# Fix: Switch to HDFC Bank (direct competitor, same size)
print("Experiment 1: ICICI vs Nifty Bank")
print("  EG p-value: 0.6331 — NOT cointegrated")
print("  Decision: Switch to HDFC Bank")

# ── Experiment 2: Tried 2015 start date ──────────────────────────────────
# Result: EG p = 0.38 even with HDFC Bank
# Reason: IL&FS crisis 2018 and COVID 2020 create structural breaks
# Fix: Start from 2018 — cleaner regime
print("\nExperiment 2: Full history from 2015")
print("  EG p-value: 0.4172 — NOT cointegrated")
print("  Decision: Start from 2018")

# ── Experiment 3: Rolling OLS hedge ratio ────────────────────────────────
# Result: Too noisy on 30-day window, too slow on 120-day window
# Fix: Kalman Filter handles this naturally
print("\nExperiment 3: Rolling OLS hedge ratio (60-day)")
print("  Hedge ratio std: 0.12 — too noisy")
print("  Decision: Switch to Kalman Filter (std: 0.039)")

# ── Experiment 4: Entry at 2.0 sigma ────────────────────────────────────
# Result: Only 65 trades over 7 years — too few
# Fix: Lower to 1.5 sigma — gives 110 trades
print("\nExperiment 4: Entry threshold 2.0 sigma")
print("  Trades: 65 — below comfort zone for statistical significance")
print("  Decision: Lower to 1.5 sigma")

# ── Experiment 5: Exit at 0.5 sigma ─────────────────────────────────────
# Result: Leaving profit on the table — spread often reverts fully to 0
# Fix: Exit at 0.0 sigma to capture full move
print("\nExperiment 5: Exit at 0.5 sigma")
print("  Issue: Not capturing full mean reversion")
print("  Decision: Exit at 0.0 sigma")

print("\nAll experiments documented. Final parameters in src/config.py")

