# %% [markdown]
# # Notebook 01: Data Exploration
# ICICI Bank Futures Mean Reversion Strategy | Report Section 1

# %% Setup
import sys; sys.path.append("..")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from src.config import stats_cfg, backtest_cfg, RESULTS_DIR
from src.data_loader import load_pair_data, save_processed

np.random.seed(backtest_cfg.random_seed)
sns.set_theme(style="darkgrid"); plt.rcParams["figure.dpi"] = 120
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% Load data
df_x, df_y = load_pair_data(use_cache=True, source="yfinance")
print(f"ICICI: {len(df_x)} days  |  Nifty Bank: {len(df_y)} days")
print(df_x["close"].describe().round(2))

# %% Price series plots
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(df_x.index, df_x["close"], color="#4299e1", lw=1.2)
axes[0].set_title("ICICI Bank — Daily Close 2015-2024"); axes[0].set_ylabel("INR")
axes[1].plot(df_y.index, df_y["close"], color="#f6ad55", lw=1.2)
axes[1].set_title("Nifty Bank Index — Daily Close 2015-2024"); axes[1].set_ylabel("Index")
plt.tight_layout(); plt.savefig(RESULTS_DIR / "01_price_series.png", bbox_inches="tight"); plt.show()

# %% Returns distribution
log_ret_x = np.log(df_x["close"] / df_x["close"].shift(1)).dropna()
log_ret_y = np.log(df_y["close"] / df_y["close"].shift(1)).dropna()

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for row, (ret, label, color) in enumerate([
        (log_ret_x, "ICICI", "#4299e1"), (log_ret_y, "BankNifty", "#f6ad55")]):
    axes[row,0].hist(ret, bins=80, density=True, alpha=0.65, color=color)
    xr = np.linspace(ret.min(), ret.max(), 200)
    axes[row,0].plot(xr, stats.norm.pdf(xr, ret.mean(), ret.std()), "r--", label="Normal")
    axes[row,0].set_title(f"{label} Returns"); axes[row,0].legend()
    (osm, osr), (s, i, _) = stats.probplot(ret)
    axes[row,1].plot(osm, osr, ".", ms=2, color=color, alpha=0.5)
    axes[row,1].plot(osm, s*np.array(osm)+i, "r--"); axes[row,1].set_title(f"{label} Q-Q")
    rv = ret.rolling(30).std() * 252**0.5
    axes[row,2].plot(rv.index, rv, color=color, lw=1); axes[row,2].set_title(f"{label} 30d Vol")
plt.tight_layout(); plt.savefig(RESULTS_DIR / "01_returns_dist.png", bbox_inches="tight"); plt.show()

for ret, label in [(log_ret_x, "ICICI Bank"), (log_ret_y, "Nifty Bank")]:
    jb, jp = stats.jarque_bera(ret)
    print(f"{label}: ann_ret={ret.mean()*252:.2%}  ann_vol={ret.std()*252**.5:.2%}  "
          f"skew={stats.skew(ret):.3f}  kurt={stats.kurtosis(ret):.3f}  JB-p={jp:.4f}")

# %% Rolling correlation
common = log_ret_x.index.intersection(log_ret_y.index)
roll_corr = log_ret_x.loc[common].rolling(stats_cfg.rolling_corr_window).corr(log_ret_y.loc[common])

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(roll_corr.index, roll_corr, color="#68d391", lw=1.0)
ax.axhline(roll_corr.mean(), color="red", ls="--", lw=1.5, label=f"Mean={roll_corr.mean():.3f}")
ax.set_title(f"{stats_cfg.rolling_corr_window}d Rolling Correlation"); ax.legend()
plt.tight_layout(); plt.savefig(RESULTS_DIR / "01_rolling_corr.png", bbox_inches="tight"); plt.show()
print(f"Corr: mean={roll_corr.mean():.4f} min={roll_corr.min():.4f} max={roll_corr.max():.4f}")

# %% Missing data and outliers
for df, label in [(df_x, "ICICI"), (df_y, "Nifty Bank")]:
    z = (df["close"] - df["close"].mean()) / df["close"].std()
    print(f"{label}: missing={df.close.isna().sum()}  outliers(|z|>5)={(z.abs()>5).sum()}")

# %% Save processed
processed = pd.DataFrame({
    "icici_close": df_x["close"], "banknifty_close": df_y["close"],
    "icici_ret": log_ret_x, "banknifty_ret": log_ret_y
}).dropna()
save_processed(processed)
print(f"Saved {len(processed)} rows")
