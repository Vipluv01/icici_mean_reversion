# %% [markdown]
# # Notebook 02: Statistical Tests
# ICICI Bank Futures Mean Reversion Strategy | Report Section 2
#
# Covers: Cointegration (EG + Johansen), ADF stationarity,
# Kalman Filter hedge ratio, spread construction, half-life.

# %% Setup
import sys; sys.path.append("..")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

from src.config import stats_cfg, kalman_cfg, strategy_cfg, backtest_cfg, RESULTS_DIR
from src.data_loader import load_processed
from src.stats import (run_adf, run_engle_granger, run_johansen,
                        kalman_hedge_ratio, compute_zscore, compute_halflife)

np.random.seed(backtest_cfg.random_seed)
sns.set_theme(style="darkgrid"); plt.rcParams["figure.dpi"] = 120
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% Load processed data
df = load_processed()
y = df["icici_close"]
x = df["banknifty_close"]
print(f"Loaded {len(df)} rows | {df.index[0].date()} to {df.index[-1].date()}")

# %% [markdown]
# ## 2.1  Stationarity Tests on Raw Price Series

# %%
print("ADF on raw price series (expect NON-stationary, p > 0.05):")
adf_y = run_adf(y, label="ICICI Close")
adf_x = run_adf(x, label="BankNifty Close")

print(f"\nICICI  : ADF={adf_y['adf_stat']:.4f}  p={adf_y['p_value']:.4f}  "
      f"stationary={adf_y['is_stationary']}")
print(f"BankNifty: ADF={adf_x['adf_stat']:.4f}  p={adf_x['p_value']:.4f}  "
      f"stationary={adf_x['is_stationary']}")

# ADF on first differences (expect stationary — I(1) series)
print("\nADF on first differences (expect stationary, p < 0.05):")
adf_dy = run_adf(y.diff().dropna(), label="Delta ICICI")
adf_dx = run_adf(x.diff().dropna(), label="Delta BankNifty")

# %% [markdown]
# ## 2.2  Engle-Granger Cointegration Test

# %%
eg = run_engle_granger(y, x, trend="c")

print("\n" + "="*55)
print("ENGLE-GRANGER COINTEGRATION TEST")
print("="*55)
print(f"  Test statistic : {eg['coint_stat']:.4f}")
print(f"  P-value        : {eg['p_value']:.4f}  (threshold: {stats_cfg.alpha})")
print(f"  Critical values: {eg['critical_values']}")
print(f"  Cointegrated   : {eg['is_cointegrated']}")
print(f"  N observations : {eg['n_obs']}")

assert eg["p_value"] < stats_cfg.alpha, (
    f"FAIL: Engle-Granger p={eg['p_value']:.4f} >= {stats_cfg.alpha}. "
    "Series are not cointegrated — strategy cannot proceed."
)
print("\n[PASS] Cointegration confirmed at 5% significance level.")

# %% [markdown]
# ## 2.3  Johansen Cointegration Test

# %%
joh = run_johansen(y, x, det_order=0, k_ar_diff=stats_cfg.johansen_maxlags)

print("\n" + "="*55)
print("JOHANSEN COINTEGRATION TEST")
print("="*55)
print(f"  Trace statistics  : {[round(s,3) for s in joh['trace_stats']]}")
print(f"  Critical (95%)    : {[round(s,3) for s in joh['trace_crit_95']]}")
print(f"  Max-Eig statistics: {[round(s,3) for s in joh['maxeig_stats']]}")
print(f"  Cointegration rank: {joh['rank']}")
print(f"  Cointegrated      : {joh['is_cointegrated']}")

# %% [markdown]
# ## 2.4  Kalman Filter — Dynamic Hedge Ratio

# %%
hedge_ratio, intercept, spread = kalman_hedge_ratio(
    y, x,
    observation_covariance = kalman_cfg.observation_covariance,
    transition_covariance  = kalman_cfg.transition_covariance,
    initial_state_mean     = kalman_cfg.initial_state_mean,
)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(y.index, y.values, color="#4299e1", lw=1, label="ICICI Close")
axes[0].set_title("ICICI Bank Close Price"); axes[0].set_ylabel("INR"); axes[0].legend()

axes[1].plot(hedge_ratio.index, hedge_ratio.values, color="#9f7aea", lw=1.2)
axes[1].set_title("Kalman Filter — Dynamic Hedge Ratio (beta_t)")
axes[1].set_ylabel("Hedge Ratio")
axes[1].axhline(hedge_ratio.mean(), color="red", ls="--", lw=1, label=f"Mean={hedge_ratio.mean():.4f}")
axes[1].legend()

axes[2].plot(spread.index, spread.values, color="#68d391", lw=1, alpha=0.8)
axes[2].axhline(0, color="red", ls="--", lw=1)
axes[2].set_title("Kalman-Filtered Spread  (y - beta*x - alpha)")
axes[2].set_ylabel("Spread")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "02_kalman_hedge_ratio.png", bbox_inches="tight")
plt.show()

print(f"\nHedge ratio: mean={hedge_ratio.mean():.4f}  "
      f"std={hedge_ratio.std():.4f}  "
      f"range=[{hedge_ratio.min():.4f}, {hedge_ratio.max():.4f}]")

# %% [markdown]
# ## 2.5  ADF Test on Spread (Critical Check)

# %%
adf_spread = run_adf(spread, label="Kalman Spread")

print("\n" + "="*55)
print("ADF TEST ON KALMAN-FILTERED SPREAD")
print("="*55)
print(f"  ADF statistic  : {adf_spread['adf_stat']:.4f}")
print(f"  P-value        : {adf_spread['p_value']:.4f}  (threshold: {stats_cfg.alpha})")
print(f"  Critical values: {adf_spread['critical_values']}")
print(f"  Stationary     : {adf_spread['is_stationary']}")

assert adf_spread["p_value"] < stats_cfg.alpha, (
    f"FAIL: Spread ADF p={adf_spread['p_value']:.4f} >= {stats_cfg.alpha}. "
    "Spread is not stationary — adjust hedge ratio method."
)
print("\n[PASS] Spread is stationary at 5% significance level.")

# Visualise spread
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
axes[0].plot(spread.index, spread.values, color="#68d391", lw=1, alpha=0.8)
axes[0].axhline(spread.mean(), color="red", ls="--", lw=1, label="Mean")
axes[0].axhline(spread.mean() + 2*spread.std(), color="orange", ls=":", lw=1, label="+2σ")
axes[0].axhline(spread.mean() - 2*spread.std(), color="orange", ls=":", lw=1, label="-2σ")
axes[0].set_title("Kalman Spread with ±2σ Bands"); axes[0].legend()

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(spread.dropna(), lags=40, ax=axes[1], title="Spread ACF (confirming mean reversion)")
plt.tight_layout(); plt.savefig(RESULTS_DIR / "02_spread.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 2.6  Half-Life of Mean Reversion

# %%
hl = compute_halflife(spread)

print("\n" + "="*55)
print("HALF-LIFE OF MEAN REVERSION (Ornstein-Uhlenbeck)")
print("="*55)
print(f"  Half-life    : {hl['halflife_days']:.1f} trading days")
print(f"  Lambda coeff : {hl['lambda_coef']:.6f}")
print(f"  R-squared    : {hl['r_squared']:.4f}")

hl_days = hl["halflife_days"]
print(f"\nInterpretation:")
print(f"  The spread takes ~{hl_days:.0f} trading days to revert halfway to its mean.")
print(f"  This justifies using zscore_window = {strategy_cfg.zscore_window} days in the strategy.")
print(f"  (Window = {strategy_cfg.zscore_window / hl_days:.1f}x half-life — appropriate for mean reversion)")

# %% [markdown]
# ## 2.7  Z-Score Computation

# %%
zscore = compute_zscore(spread, window=strategy_cfg.zscore_window)

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
axes[0].plot(spread.index, spread.values, color="#68d391", lw=1, alpha=0.7)
axes[0].set_title("Spread"); axes[0].set_ylabel("Spread Value")

axes[1].plot(zscore.index, zscore.values, color="#4299e1", lw=1, alpha=0.8)
axes[1].axhline(0,  color="white", lw=0.8, ls="--")
axes[1].axhline(strategy_cfg.entry_zscore, color="#f6ad55", lw=1.2, ls="--",
                label=f"Entry +{strategy_cfg.entry_zscore}σ")
axes[1].axhline(-strategy_cfg.entry_zscore, color="#f6ad55", lw=1.2, ls="--",
                label=f"Entry -{strategy_cfg.entry_zscore}σ")
axes[1].axhline(strategy_cfg.stop_zscore,  color="#fc8181", lw=1.0, ls=":",
                label=f"Stop +{strategy_cfg.stop_zscore}σ")
axes[1].axhline(-strategy_cfg.stop_zscore, color="#fc8181", lw=1.0, ls=":",
                label=f"Stop -{strategy_cfg.stop_zscore}σ")
axes[1].set_title(f"Z-Score of Spread  (window={strategy_cfg.zscore_window}d)")
axes[1].set_ylabel("Z-Score"); axes[1].legend(ncol=2)

plt.tight_layout(); plt.savefig(RESULTS_DIR / "02_zscore.png", bbox_inches="tight"); plt.show()

# %% Save results for downstream notebooks
import pickle
results = {
    "spread": spread,
    "hedge_ratio": hedge_ratio,
    "intercept": intercept,
    "zscore": zscore,
    "halflife": hl,
    "eg_test": eg,
    "johansen_test": joh,
    "adf_spread": adf_spread,
}
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
pd.to_pickle(results, RESULTS_DIR / "02_stats_results.pkl")
print("Statistical results saved to results/02_stats_results.pkl")
print("\nAll Section 2 requirements satisfied.")
