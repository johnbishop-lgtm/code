"""Daily OHLCV data: Yahoo Finance (free, no key), local CSV, or a synthetic generator for offline demos/tests."""
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["open", "high", "low", "close", "volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[COLS].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def from_yahoo(symbol: str, years: int = 10) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(symbol, period=f"{years}y", interval="1d", auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"no data returned for {symbol} (network blocked or bad symbol?)")
    return _normalize(raw)


def from_csv(path: Path) -> pd.DataFrame:
    return _normalize(pd.read_csv(path, index_col=0, parse_dates=True))


def synthetic(symbol: str, years: int = 10, seed: int | None = None) -> pd.DataFrame:
    """Random-walk prices with drift, volatility clustering and short-term mean reversion.

    Only for exercising the code offline. Results on synthetic data say nothing about real markets.
    """
    rng = np.random.default_rng(seed if seed is not None else zlib.crc32(symbol.encode()))
    keep = 252 * years
    n = max(keep, 252 * 20)  # always simulate 20 years so shorter requests are a slice of the same series
    vol = np.empty(n)
    vol[0] = 0.012
    for i in range(1, n):
        vol[i] = 0.0002 + 0.9 * vol[i - 1] + 0.08 * abs(rng.normal(0, 0.012))
    rets = np.empty(n)
    rets[0] = 0
    for i in range(1, n):
        rets[i] = 0.0004 - 0.08 * rets[i - 1] + vol[i] * rng.standard_t(5) / np.sqrt(5 / 3)
    close = 100 * np.exp(np.cumsum(rets))
    gap = rng.normal(0, 0.003, n)
    open_ = close / np.exp(rets) * np.exp(gap)
    span = np.abs(rng.normal(0, vol, n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    volume = rng.lognormal(15, 0.4, n)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
    return df.iloc[-keep:]


def load(symbols, source: str = "yahoo", years: int = 10, csv_dir: str = "data") -> dict[str, pd.DataFrame]:
    out = {}
    for s in symbols:
        if source == "synthetic":
            out[s] = synthetic(s, years)
        elif source == "csv":
            out[s] = from_csv(Path(csv_dir) / f"{s}.csv")
        else:
            out[s] = from_yahoo(s, years)
    return out
