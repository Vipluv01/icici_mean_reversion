"""
stats.py
========
Statistical toolkit for the ICICI Bank Futures Mean Reversion Strategy.

Implements:
  1. Augmented Dickey-Fuller (stationarity)
  2. Engle-Granger cointegration test
  3. Johansen cointegration test
  4. Kalman Filter dynamic hedge ratio estimation
  5. Spread construction
  6. Half-life of mean reversion (Ornstein-Uhlenbeck)
  7. Z-score computation

All functions are pure (no side-effects on global state).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Augmented Dickey-Fuller Test
# ══════════════════════════════════════════════════════════════════════════════

def run_adf(
    series: pd.Series,
    maxlags: Optional[int] = None,
    regression: str = "c",
    label: str = "",
) -> dict:
    """
    Augmented Dickey-Fuller test for stationarity.

    H0: Unit root (non-stationary)
    H1: Stationary

    Parameters
    ----------
    series     : pd.Series   Time series to test
    maxlags    : int or None  Max lags (None = auto by AIC)
    regression : str          'c' = constant, 'ct' = constant+trend
    label      : str          Name for logging

    Returns
    -------
    dict with keys: adf_stat, p_value, n_lags, is_stationary, critical_values
    """
    result = adfuller(series.dropna(), maxlag=maxlags,
                      regression=regression, autolag="AIC")
    adf_stat, p_value, n_lags, _, crit_vals = result[0], result[1], result[2], result[3], result[4]
    is_stationary = p_value < 0.05

    log.info(
        "ADF Test [%s]: stat=%.4f  p=%.4f  lags=%d  stationary=%s",
        label, adf_stat, p_value, n_lags, is_stationary,
    )
    return {
        "adf_stat":        adf_stat,
        "p_value":         p_value,
        "n_lags":          n_lags,
        "is_stationary":   is_stationary,
        "critical_values": crit_vals,
        "label":           label,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Engle-Granger Cointegration Test
# ══════════════════════════════════════════════════════════════════════════════

def run_engle_granger(
    y: pd.Series,
    x: pd.Series,
    trend: str = "c",
) -> dict:
    """
    Engle-Granger two-step cointegration test.

    Tests whether y and x are cointegrated.
    H0: No cointegration
    H1: Cointegrated

    Parameters
    ----------
    y, x  : pd.Series   Price series (must be aligned)
    trend : str          'c' = constant, 'ct' = trend

    Returns
    -------
    dict with keys: coint_stat, p_value, is_cointegrated, critical_values
    """
    # Align
    idx = y.index.intersection(x.index)
    y_, x_ = y.loc[idx].dropna(), x.loc[idx].dropna()
    idx2 = y_.index.intersection(x_.index)
    y_, x_ = y_.loc[idx2], x_.loc[idx2]

    stat, p_value, crit = coint(y_, x_, trend=trend)
    is_cointegrated = p_value < 0.05

    log.info(
        "Engle-Granger: stat=%.4f  p=%.4f  cointegrated=%s",
        stat, p_value, is_cointegrated,
    )
    return {
        "coint_stat":       stat,
        "p_value":          p_value,
        "is_cointegrated":  is_cointegrated,
        "critical_values":  crit,
        "n_obs":            len(y_),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Johansen Cointegration Test
# ══════════════════════════════════════════════════════════════════════════════

def run_johansen(
    y: pd.Series,
    x: pd.Series,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> dict:
    """
    Johansen cointegration test (trace and max-eigenvalue statistics).

    Parameters
    ----------
    y, x       : pd.Series  Price series
    det_order  : int         -1=no deterministic, 0=constant, 1=trend
    k_ar_diff  : int         Number of lagged differences

    Returns
    -------
    dict with trace statistic, max-eigenvalue statistic, critical values,
         and cointegration rank.
    """
    idx = y.index.intersection(x.index)
    data = pd.concat([y.loc[idx], x.loc[idx]], axis=1).dropna()

    result = coint_johansen(data.values, det_order=det_order, k_ar_diff=k_ar_diff)

    # Trace test: H0 = r <= i  vs H1 = r > i
    trace_stat    = result.lr1          # shape (2,)
    trace_crit    = result.cvt          # shape (2, 3) -> 90%, 95%, 99%
    max_eig_stat  = result.lr2
    max_eig_crit  = result.cvm

    # Cointegration rank from trace test at 95%
    rank = int(np.sum(trace_stat > trace_crit[:, 1]))

    log.info(
        "Johansen: trace_stats=%s  crit_95=%s  rank=%d",
        np.round(trace_stat, 3), np.round(trace_crit[:, 1], 3), rank,
    )
    return {
        "trace_stats":    trace_stat.tolist(),
        "trace_crit_95":  trace_crit[:, 1].tolist(),
        "maxeig_stats":   max_eig_stat.tolist(),
        "maxeig_crit_95": max_eig_crit[:, 1].tolist(),
        "rank":           rank,
        "is_cointegrated": rank >= 1,
        "eigenvectors":   result.evec.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Kalman Filter  — Dynamic Hedge Ratio
# ══════════════════════════════════════════════════════════════════════════════

def kalman_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    observation_covariance: float = 1.0,
    transition_covariance: float = 0.01,
    initial_state_mean: Optional[list] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Estimate a time-varying hedge ratio using a Kalman Filter.

    State vector: [beta, alpha]  where  y_t = beta_t * x_t + alpha_t + eps_t

    The Kalman Filter allows beta and alpha to evolve as a random walk,
    making it superior to static OLS for non-stationary hedge ratios.

    Parameters
    ----------
    y, x                     : pd.Series  Aligned price series
    observation_covariance   : float       Measurement noise variance
    transition_covariance    : float       State transition noise (adapt speed)
    initial_state_mean       : list        Initial [beta, alpha]

    Returns
    -------
    hedge_ratio : pd.Series   Time-varying beta_t
    intercept   : pd.Series   Time-varying alpha_t
    spread      : pd.Series   y_t - beta_t * x_t - alpha_t
    """
    if initial_state_mean is None:
        initial_state_mean = [1.0, 0.0]

    idx   = y.index.intersection(x.index)
    y_arr = y.loc[idx].values.astype(float)
    x_arr = x.loc[idx].values.astype(float)
    n     = len(y_arr)

    # State: [beta, alpha]
    state_dim = 2
    obs_dim   = 1

    # Initialise
    beta         = np.zeros(n)
    alpha        = np.zeros(n)
    P            = np.eye(state_dim)                     # state covariance
    Q            = transition_covariance * np.eye(state_dim)  # process noise
    R            = np.array([[observation_covariance]])  # obs noise
    state        = np.array(initial_state_mean, dtype=float)

    for t in range(n):
        # Observation matrix: [x_t, 1]
        H = np.array([[x_arr[t], 1.0]])

        # --- Predict ---
        # (state transition is identity: state does not change by itself)
        state_pred = state
        P_pred     = P + Q

        # --- Update ---
        innovation = y_arr[t] - H @ state_pred            # scalar
        S          = H @ P_pred @ H.T + R                  # innovation cov
        K          = P_pred @ H.T @ np.linalg.inv(S)      # Kalman gain
        state      = state_pred + K.flatten() * innovation.item()
        P          = (np.eye(state_dim) - K @ H) @ P_pred

        beta[t]  = state[0]
        alpha[t] = state[1]

    spread = y_arr - beta * x_arr - alpha

    hedge_ratio = pd.Series(beta,  index=idx, name="hedge_ratio")
    intercept   = pd.Series(alpha, index=idx, name="intercept")
    spread_s    = pd.Series(spread, index=idx, name="spread")

    log.info(
        "Kalman Filter: beta range [%.4f, %.4f]  final beta=%.4f",
        beta.min(), beta.max(), beta[-1],
    )
    return hedge_ratio, intercept, spread_s


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Spread Z-score
# ══════════════════════════════════════════════════════════════════════════════

def compute_zscore(
    spread: pd.Series,
    window: int,
) -> pd.Series:
    """
    Compute rolling z-score of the spread.

    z_t = (spread_t - rolling_mean_t) / rolling_std_t

    Parameters
    ----------
    spread : pd.Series  Raw spread series
    window : int        Rolling window in trading days

    Returns
    -------
    pd.Series  Z-score series (NaN for first `window` observations)
    """
    roll_mean = spread.rolling(window=window, min_periods=window).mean()
    roll_std  = spread.rolling(window=window, min_periods=window).std()
    zscore = (spread - roll_mean) / roll_std
    zscore.name = "zscore"
    return zscore


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Half-Life of Mean Reversion
# ══════════════════════════════════════════════════════════════════════════════

def compute_halflife(spread: pd.Series) -> dict:
    """
    Estimate the half-life of mean reversion via the Ornstein-Uhlenbeck process.

    Regresses:  delta_spread_t = lambda * spread_{t-1} + epsilon_t
    Half-life  = -ln(2) / lambda

    A half-life of N days implies the spread takes ~N days to revert
    halfway to its mean, guiding the zscore_window parameter choice.

    Parameters
    ----------
    spread : pd.Series  Stationary spread series

    Returns
    -------
    dict with keys: halflife_days, lambda_coef, r_squared
    """
    spread_clean = spread.dropna()
    delta = spread_clean.diff().dropna()
    lag   = spread_clean.shift(1).dropna()

    # Align
    idx   = delta.index.intersection(lag.index)
    delta = delta.loc[idx].values
    lag   = lag.loc[idx].values

    # OLS: delta = lambda * lag + eps
    X = np.column_stack([lag, np.ones(len(lag))])
    coefs, residuals, _, _ = np.linalg.lstsq(X, delta, rcond=None)
    lam = coefs[0]

    if lam >= 0:
        log.warning("Half-life: lambda=%.4f >= 0 — spread may not be mean-reverting.", lam)
        halflife = np.nan
    else:
        halflife = -np.log(2) / lam

    # R-squared
    y_pred = X @ coefs
    ss_res = np.sum((delta - y_pred) ** 2)
    ss_tot = np.sum((delta - delta.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    log.info(
        "Half-life: %.1f days  lambda=%.6f  R2=%.4f",
        halflife, lam, r2,
    )
    return {
        "halflife_days": halflife,
        "lambda_coef":   lam,
        "r_squared":     r2,
    }
