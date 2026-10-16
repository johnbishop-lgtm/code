"""Technical indicators. Every function uses only data up to and including each bar (no lookahead)."""
import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    return mid - k * sd, mid, mid + k * sd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of an OHLCV frame with every indicator the strategies use."""
    df = df.copy()
    c = df["close"]
    df["sma5"] = sma(c, 5)
    df["sma20"] = sma(c, 20)
    df["sma50"] = sma(c, 50)
    df["sma200"] = sma(c, 200)
    df["rsi2"] = rsi(c, 2)
    df["rsi14"] = rsi(c, 14)
    df["atr14"] = atr(df, 14)
    df["bb_lower"], _, df["bb_upper"] = bollinger(c, 20, 2.0)
    df["high20"] = df["high"].rolling(20).max().shift(1)  # prior 20-day high, excludes today
    df["low10"] = df["low"].rolling(10).min().shift(1)
    df["vol_avg20"] = df["volume"].rolling(20).mean()
    return df
