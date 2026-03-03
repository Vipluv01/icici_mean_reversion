# Notebook 04: Validation and Robustness
# Walk-forward (70/30), Monte Carlo (1000+), Sensitivity, Regime Analysis

import sys; sys.path.append("..")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from dataclasses import replace

from src.config import strategy_cfg, backtest_cfg, validation_cfg, threshold_cfg, kalman_cfg, RESULTS_DIR
from src.stats import kalman_hedge_ratio, compute_zscore
from src.strategy import generate_signals, compute_position_sizes, compute_transaction_costs
from src.backtester import run_backtest, compute_metrics
from src.data_loader import load_processed

np.random.seed(backtest_cfg.random_seed)
sns.set_theme(style="darkgrid")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

bt = pd.read_pickle(RESULTS_DIR / "03_backtest_results.pkl")
df = load_processed()
y, x = df["icici_close"], df["banknifty_close"]

def pipeline(y_s, x_s, **kw):
    cfg = replace(strategy_cfg, **kw)
    hr, ic, sp = kalman_hedge_ratio(y_s, x_s, kalman_cfg.observation_covariance, kalman_cfg.transition_covariance)
    zs = compute_zscore(sp, cfg.zscore_window)
    sg = generate_signals(zs, cfg)
    eq0 = pd.Series(backtest_cfg.initial_capital, index=sp.index)
    ps = compute_position_sizes(sg, eq0, cfg)
    tc = compute_transaction_costs(ps, cfg)
    eq, ret, trd = run_backtest(sp, sg, tc, backtest_cfg)
    return compute_metrics(eq, ret, trd, backtest_cfg) if trd else None

# Walk-forward split
n = len(y); split = int(n * backtest_cfg.train_ratio)
m_is  = pipeline(y.iloc[:split], x.iloc[:split])
m_oos = pipeline(y.iloc[split:], x.iloc[split:])

print("WALK-FORWARD RESULTS")
if m_is and m_oos:
    for k in ["sharpe_ratio","annualized_return","max_drawdown","calmar_ratio","win_rate","total_trades"]:
        print(f"  {k:<26} IS={m_is[k]:.4f}  OOS={m_oos[k]:.4f}")
    ratio = m_oos["sharpe_ratio"] / m_is["sharpe_ratio"]
    print(f"  OOS/IS Sharpe: {ratio:.4f}  Pass={ratio >= validation_cfg.oos_sharpe_min_ratio}")

# Monte Carlo
pnl_pcts = np.array([t.pnl_pct for t in bt["trades"]])
sim_finals, sim_dds = [], []
for s in range(backtest_cfg.monte_carlo_sims):
    rng = np.random.default_rng(s)
    sampled = rng.choice(pnl_pcts, len(bt["trades"]), replace=True)
    eq = backtest_cfg.initial_capital * np.cumprod(1 + sampled)
    sim_finals.append((eq[-1]/backtest_cfg.initial_capital)-1)
    rm = np.maximum.accumulate(eq)
    sim_dds.append(abs(((eq-rm)/rm).min()))

sim_finals, sim_dds = np.array(sim_finals), np.array(sim_dds)
p5,p50,p95 = np.percentile(sim_finals,[5,50,95])
prob_dd = (sim_dds > 0.20).mean()
print(f"MC: p5={p5:.2%} p50={p50:.2%} p95={p95:.2%}  P(DD>20%)={prob_dd:.2%}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(sim_finals*100, bins=50, color="#4299e1", alpha=0.7)
for p,c,l in [(p5,"red","5th"),(p50,"green","50th"),(p95,"orange","95th")]:
    axes[0].axvline(p*100, color=c, ls="--", lw=1.5, label=f"{l}: {p:.1%}")
axes[0].set_title("MC: Return Distribution"); axes[0].legend()
axes[1].hist(sim_dds*100, bins=50, color="#f6ad55", alpha=0.7)
axes[1].axvline(20, color="red", ls="--", lw=1.5, label="20% threshold")
axes[1].set_title("MC: Max Drawdown Distribution"); axes[1].legend()
plt.tight_layout(); plt.savefig(RESULTS_DIR/"04_monte_carlo.png", bbox_inches="tight")

# Sensitivity
entries=[round(strategy_cfg.entry_zscore*(1+d),2) for d in [-0.2,-0.1,0,0.1,0.2]]
windows=[int(strategy_cfg.zscore_window*(1+d)) for d in [-0.2,-0.1,0,0.1,0.2]]
grid=np.full((len(entries),len(windows)),np.nan)
for i,e in enumerate(entries):
    for j,w in enumerate(windows):
        m=pipeline(y,x,entry_zscore=e,zscore_window=w)
        if m: grid[i,j]=m["sharpe_ratio"]

fig,ax=plt.subplots(figsize=(10,6))
sns.heatmap(grid,annot=True,fmt=".2f",xticklabels=[f"{w}d" for w in windows],
    yticklabels=[f"{e}s" for e in entries],cmap="RdYlGn",center=threshold_cfg.min_sharpe,ax=ax,linewidths=0.5)
ax.set_title("Sharpe Sensitivity: Entry Z-Score vs Window")
plt.tight_layout(); plt.savefig(RESULTS_DIR/"04_sensitivity.png", bbox_inches="tight")

# Regime analysis
log_ret=df["icici_ret"].dropna()
rv=log_ret.rolling(60).std()*252**.5; vt=rv.median()
dr=bt["daily_returns"]; common=dr.index.intersection(rv.index)
for label,mask in [("High Vol",rv.loc[common]>=vt),("Low Vol",rv.loc[common]<vt)]:
    r=dr.loc[common][mask]
    sr=(r.mean()*252)/(r.std()*252**.5) if r.std()>0 else 0
    print(f"Regime [{label}]: days={len(r)}  ann_ret={r.mean()*252:.2%}  sharpe={sr:.3f}")

pd.to_pickle({"is":m_is,"oos":m_oos,"p5":p5,"p50":p50,"p95":p95,"prob_dd":prob_dd},
             RESULTS_DIR/"04_validation_results.pkl")
print("All validation complete. Results saved.")