"""
data_loader.py
==============
Fetches, cleans and saves daily OHLCV data for the ICICI Bank Futures
mean reversion strategy.

Primary source  : yfinance (no credentials needed, good for 2015-2024)
Secondary source: Angel One SmartAPI (requires credentials in .env)

Usage
-----
    from src.data_loader import load_pair_data
    df = load_pair_data()
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import pyotp
    PYOTP_AVAILABLE = True
except ImportError:
    PYOTP_AVAILABLE = False

from src.config import DATA_DIR, data_cfg

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  yfinance loader  (primary, no credentials needed)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_yfinance(
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker : str   Yahoo Finance ticker (e.g. 'ICICIBANK.NS')
    start  : str   Start date 'YYYY-MM-DD'
    end    : str   End date   'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame  Columns: open, high, low, close, volume  (index: Date)
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    log.info("Downloading %s from Yahoo Finance (%s to %s)...", ticker, start, end)
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker symbol.")

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df = df.dropna(subset=["close"])

    log.info("  -> %d rows fetched for %s", len(df), ticker)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Angel One API loader  (secondary, requires credentials)
# ══════════════════════════════════════════════════════════════════════════════

_ANGEL_BASE_URL  = "https://apiconnect.angelone.in"
_ANGEL_LOGIN_URL = f"{_ANGEL_BASE_URL}/rest/auth/angelbroking/user/v1/loginByPassword"
_ANGEL_HIST_URL  = f"{_ANGEL_BASE_URL}/rest/secure/angelbroking/historical/v1/getCandleData"


@dataclass
class _AngelSession:
    client_code: str
    pin:         str
    api_key:     str
    totp_secret: str
    local_ip:    str = "127.0.0.1"
    public_ip:   str = "127.0.0.1"
    mac:         str = "00:00:00:00:00:00"
    jwt_token:   Optional[str] = None


def _angel_headers(session: _AngelSession) -> Dict[str, str]:
    h: Dict[str, str] = {
        "Content-Type":      "application/json",
        "Accept":            "application/json",
        "User-Agent":        "Mozilla/5.0",
        "X-UserType":        "USER",
        "X-SourceID":        "WEB",
        "X-ClientLocalIP":   session.local_ip,
        "X-ClientPublicIP":  session.public_ip,
        "X-MACAddress":      session.mac,
        "X-PrivateKey":      session.api_key,
    }
    if session.jwt_token:
        h["Authorization"] = f"Bearer {session.jwt_token}"
    return h


def _angel_login(session: _AngelSession) -> None:
    if not PYOTP_AVAILABLE:
        raise ImportError("pyotp not installed. Run: pip install pyotp")
    totp = pyotp.TOTP(session.totp_secret).now()
    payload = {
        "clientcode": session.client_code,
        "password":   session.pin,
        "totp":       totp,
        "state":      "live",
    }
    r = requests.post(_ANGEL_LOGIN_URL, json=payload,
                      headers=_angel_headers(session), timeout=10)
    data = r.json()
    if not data.get("status"):
        raise RuntimeError(f"Angel One login failed: {data}")
    session.jwt_token = data["data"]["jwtToken"]
    log.info("Angel One login successful.")


def _angel_fetch_chunk(
    session: _AngelSession,
    symbol_token: str,
    exchange: str,
    interval: str,
    from_dt: datetime,
    to_dt: datetime,
    max_retries: int = 5,
) -> List[List[Any]]:
    for attempt in range(1, max_retries + 1):
        try:
            if not session.jwt_token:
                _angel_login(session)
            payload = {
                "exchange":    exchange,
                "symboltoken": symbol_token,
                "interval":    interval,
                "fromdate":    from_dt.strftime("%Y-%m-%d %H:%M"),
                "todate":      to_dt.strftime("%Y-%m-%d %H:%M"),
            }
            r = requests.post(
                _ANGEL_HIST_URL,
                json=payload,
                headers=_angel_headers(session),
                timeout=10,
            )
            data = r.json()
            if data.get("status"):
                return data.get("data", {}).get("candles", []) or []
            raise RuntimeError(data.get("message", "Unknown error"))
        except Exception as exc:
            log.warning("Chunk failed (attempt %d): %s", attempt, exc)
            session.jwt_token = None
            time.sleep(3 * attempt)
    log.error("Skipping chunk %s -> %s permanently.", from_dt.date(), to_dt.date())
    return []


def fetch_angel_one(
    symbol_token: str,
    exchange: str,
    interval: str,
    start: str,
    end: str,
    chunk_days: int = 30,
    sleep_between: int = 5,
) -> pd.DataFrame:
    """
    Download daily OHLCV data from Angel One SmartAPI.

    Credentials are read from environment variables:
        ANGEL_CLIENT_CODE, ANGEL_PIN, ANGEL_API_KEY, ANGEL_TOTP_SECRET

    Parameters
    ----------
    symbol_token : str  Angel One symbol token
    exchange     : str  Exchange code (e.g. 'NSE')
    interval     : str  Data interval (e.g. 'ONE_DAY')
    start, end   : str  Date range 'YYYY-MM-DD'
    chunk_days   : int  Days per API chunk (default 30)
    sleep_between: int  Seconds to sleep between chunks

    Returns
    -------
    pd.DataFrame  Columns: open, high, low, close, volume  (index: date)
    """
    session = _AngelSession(
        client_code  = os.environ["ANGEL_CLIENT_CODE"],
        pin          = os.environ["ANGEL_PIN"],
        api_key      = os.environ["ANGEL_API_KEY"],
        totp_secret  = os.environ["ANGEL_TOTP_SECRET"],
    )

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(hour=9,  minute=15)
    end_dt   = datetime.strptime(end,   "%Y-%m-%d").replace(hour=15, minute=30)
    delta    = timedelta(days=chunk_days)
    all_rows: Dict[str, List[Any]] = {}
    cursor   = start_dt

    while cursor < end_dt:
        chunk_end = min(cursor + delta, end_dt)
        rows = _angel_fetch_chunk(session, symbol_token, exchange, interval,
                                   cursor, chunk_end)
        for row in rows:
            all_rows[row[0]] = row
        cursor = chunk_end
        time.sleep(sleep_between)

    if not all_rows:
        raise ValueError("No data fetched from Angel One.")

    ordered = [all_rows[k] for k in sorted(all_rows)]
    df = pd.DataFrame(ordered,
                      columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    log.info("Angel One: %d candles fetched.", len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Data cleaning
# ══════════════════════════════════════════════════════════════════════════════

def clean_ohlcv(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    Clean OHLCV data:
      - Forward-fill up to 3 consecutive missing days
      - Drop rows where close is zero or negative
      - Detect and log outliers (|z-score| > 5 on log-returns)
      - Sort by date

    Parameters
    ----------
    df    : pd.DataFrame  Raw OHLCV (index = DatetimeIndex, col 'close' required)
    label : str           Name for logging

    Returns
    -------
    pd.DataFrame  Cleaned OHLCV
    """
    original_len = len(df)
    df = df.sort_index()

    # Remove zero / negative prices
    df = df[df["close"] > 0]

    # Forward-fill short gaps (weekends / holidays already removed by exchange,
    # but API may have isolated NaNs)
    df = df.ffill(limit=3)
    df = df.dropna(subset=["close"])

    # Outlier detection on log-returns
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    z_scores = (log_ret - log_ret.mean()) / log_ret.std()
    outliers = z_scores[z_scores.abs() > 5]
    if not outliers.empty:
        log.warning("%s: %d outlier return(s) detected (|z|>5) at: %s",
                    label, len(outliers), outliers.index.tolist())

    log.info("%s: %d rows before cleaning -> %d after.", label, original_len, len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Master loader
# ══════════════════════════════════════════════════════════════════════════════

def load_pair_data(
    use_cache: bool = True,
    source: str = "yfinance",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and return aligned (X, Y) daily close price series.

    X = ICICI Bank  (ticker_x in config)
    Y = Nifty Bank  (ticker_y in config)

    Parameters
    ----------
    use_cache : bool   If True, load from CSV if it exists (skip download)
    source    : str    'yfinance' (default) or 'angel_one'

    Returns
    -------
    (df_x, df_y) : Tuple of cleaned, aligned DataFrames
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path_x = DATA_DIR / data_cfg.raw_file_x
    path_y = DATA_DIR / data_cfg.raw_file_y

    # ── Load from cache if available ──────────────────────────────────────
    if use_cache and path_x.exists() and path_y.exists():
        log.info("Loading from cache: %s, %s", path_x.name, path_y.name)
        df_x = pd.read_csv(path_x, index_col="date", parse_dates=True)
        df_y = pd.read_csv(path_y, index_col="date", parse_dates=True)
    else:
        # ── Download ──────────────────────────────────────────────────────
        if source == "yfinance":
            df_x = fetch_yfinance(data_cfg.ticker_x,
                                   data_cfg.start_date, data_cfg.end_date)
            df_y = fetch_yfinance(data_cfg.ticker_y,
                                   data_cfg.start_date, data_cfg.end_date)
        elif source == "angel_one":
            df_x = fetch_angel_one(
                data_cfg.angel_token_x, data_cfg.angel_exchange,
                data_cfg.angel_interval, data_cfg.start_date, data_cfg.end_date,
            )
            df_y = fetch_angel_one(
                data_cfg.angel_token_y, data_cfg.angel_exchange,
                data_cfg.angel_interval, data_cfg.start_date, data_cfg.end_date,
            )
        else:
            raise ValueError(f"Unknown source '{source}'. Use 'yfinance' or 'angel_one'.")

        # ── Clean & save ──────────────────────────────────────────────────
        df_x = clean_ohlcv(df_x, label="ICICI")
        df_y = clean_ohlcv(df_y, label="HDFCBank")
        df_x.to_csv(path_x)
        df_y.to_csv(path_y)
        log.info("Data saved to %s and %s", path_x, path_y)

    # ── Align on common trading dates ────────────────────────────────────
    common_idx = df_x.index.intersection(df_y.index)
    df_x = df_x.loc[common_idx]
    df_y = df_y.loc[common_idx]

    log.info("Aligned pair: %d common trading days (%s to %s).",
             len(common_idx), common_idx[0].date(), common_idx[-1].date())
    return df_x, df_y


def save_processed(df: pd.DataFrame) -> None:
    """Save the merged processed DataFrame to data/processed_pair.csv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / data_cfg.processed_file
    df.to_csv(path)
    log.info("Processed pair saved to %s", path)


def load_processed() -> pd.DataFrame:
    """Load the merged processed DataFrame from data/processed_pair.csv."""
    path = DATA_DIR / data_cfg.processed_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run load_pair_data() first."
        )
    return pd.read_csv(path, index_col="date", parse_dates=True)
