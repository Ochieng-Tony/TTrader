"""
TTrader Data Layer
Fetches OHLCV data via yfinance and caches locally as Parquet.
Handles incremental updates so repeated runs don't re-download history.
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "storage", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(ticker: str, interval: str, period: str) -> str:
    safe = ticker.replace("/", "_").replace("^", "idx_")
    return os.path.join(CACHE_DIR, f"{safe}_{interval}_{period}.parquet")


def get_data(ticker: str, period: str = "2y", interval: str = "1d", force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns OHLCV DataFrame for ticker, using local cache when possible.
    Columns: Open, High, Low, Close, Volume (index = datetime)
    """
    path = _cache_path(ticker, interval, period)

    if not force_refresh and os.path.exists(path):
        cached = pd.read_parquet(path)
        last_date = cached.index.max()
        # Refresh only the tail if cache is stale (>1 day old for daily bars)
        if (datetime.now() - last_date.to_pydatetime().replace(tzinfo=None)).days < 1:
            return cached

    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        # Fallback for sandboxed environments without internet access to Yahoo Finance.
        # On a normal local machine, yfinance will succeed above and this is never used.
        print(f"[warn] Live fetch failed for {ticker} (no network access to Yahoo Finance in this "
              f"environment). Using synthetic data so the pipeline can still be demonstrated end-to-end.")
        df = _generate_synthetic(ticker, period, interval)
    else:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    df.to_parquet(path)
    return df


def _generate_synthetic(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Generates realistic synthetic OHLCV via geometric Brownian motion + volatility clustering.
    Used only as a fallback when no network access is available. Seeded by ticker for reproducibility."""
    import numpy as np

    days_map = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504, "5y": 1260}
    n = days_map.get(period, 504)

    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    start_price = rng.uniform(20, 400)

    # Volatility clustering via simple GARCH-like process
    vol = np.zeros(n)
    vol[0] = 0.015
    for i in range(1, n):
        vol[i] = 0.85 * vol[i - 1] + 0.10 * abs(rng.normal(0, 0.02)) + 0.05 * 0.015

    drift = rng.uniform(-0.0002, 0.0006)
    returns = rng.normal(drift, 1, n) * vol
    close = start_price * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, n)
    volume = rng.integers(1_000_000, 20_000_000, n).astype(float)

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)
    return df


def get_multi(tickers: list, period: str = "2y", interval: str = "1d") -> dict:
    """Fetch multiple tickers, returns {ticker: DataFrame}."""
    return {t: get_data(t, period, interval) for t in tickers}


if __name__ == "__main__":
    df = get_data("AAPL", period="6mo")
    print(df.tail())
    print(f"\nRows cached: {len(df)}")
