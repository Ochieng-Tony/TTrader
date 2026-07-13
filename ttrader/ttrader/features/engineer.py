"""
TTrader Feature Engineering
Builds a feature matrix from OHLCV data: technical indicators, momentum,
volatility regime, and the forward-return label used for supervised training.
"""
import numpy as np
import pandas as pd
import ta


def build_features(df: pd.DataFrame, horizon: int = 5, label_threshold: float = 0.01) -> pd.DataFrame:
    """
    df: OHLCV DataFrame (Open, High, Low, Close, Volume)
    horizon: bars ahead to predict
    label_threshold: return magnitude required to call it Up/Down vs Flat

    Returns DataFrame of features + 'label' column (2=Up, 1=Flat, 0=Down)
    and 'fwd_return' (raw forward return, for backtesting/eval).
    """
    out = pd.DataFrame(index=df.index)
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    # --- Momentum ---
    out["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    out["stoch_k"] = ta.momentum.StochasticOscillator(high, low, close).stoch()
    out["roc_10"] = ta.momentum.ROCIndicator(close, window=10).roc()

    # --- Trend ---
    macd = ta.trend.MACD(close)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()
    out["adx_14"] = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    out["sma_20_ratio"] = close / ta.trend.SMAIndicator(close, window=20).sma_indicator() - 1
    out["sma_50_ratio"] = close / ta.trend.SMAIndicator(close, window=50).sma_indicator() - 1
    out["ema_12_26_diff"] = (
        ta.trend.EMAIndicator(close, window=12).ema_indicator()
        - ta.trend.EMAIndicator(close, window=26).ema_indicator()
    ) / close

    # --- Volatility / regime ---
    bb = ta.volatility.BollingerBands(close, window=20)
    out["bb_pct"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    out["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range() / close
    out["volatility_20"] = close.pct_change().rolling(20).std()
    out["volatility_regime"] = out["volatility_20"] / out["volatility_20"].rolling(60).mean()

    # --- Volume ---
    out["volume_z"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()
    out["obv_slope"] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume().diff(5)

    # --- Lagged returns ---
    for lag in [1, 3, 5, 10]:
        out[f"ret_{lag}"] = close.pct_change(lag)

    # --- Label: forward return over horizon bars ---
    fwd_return = close.shift(-horizon) / close - 1
    out["fwd_return"] = fwd_return
    out["label"] = np.where(
        fwd_return > label_threshold, 2,
        np.where(fwd_return < -label_threshold, 0, 1)
    )

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


FEATURE_COLUMNS = [
    "rsi_14", "stoch_k", "roc_10",
    "macd", "macd_signal", "macd_diff", "adx_14",
    "sma_20_ratio", "sma_50_ratio", "ema_12_26_diff",
    "bb_pct", "atr_14", "volatility_20", "volatility_regime",
    "volume_z", "obv_slope",
    "ret_1", "ret_3", "ret_5", "ret_10",
]

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.loader import get_data
    df = get_data("AAPL", period="1y")
    feats = build_features(df)
    print(feats[FEATURE_COLUMNS + ["label", "fwd_return"]].tail())
    print(f"\nFeature rows: {len(feats)}  Label distribution:\n{feats['label'].value_counts()}")
