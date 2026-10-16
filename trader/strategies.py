"""Rule-based strategies.

Each strategy looks at a fully-formed bar (today's close) and answers two questions:
  * entry(row) -> should we buy at the next open?
  * exit(row, pos) -> should we sell at the next open?
plus how wide the protective stop should be.

The mean-reversion strategies are the "high win rate" family: they buy short-term
dips inside a long-term uptrend and take small, frequent profits. Their weakness is
the occasional large loss, which is why every position also carries an ATR stop.
The breakout strategy is the opposite profile (lower win rate, bigger winners) and
is included so the portfolio isn't purely betting on mean reversion.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class Position:
    symbol: str
    strategy: str
    qty: int
    entry_price: float
    entry_date: pd.Timestamp
    stop: float
    bars_held: int = 0
    highest_close: float = 0.0


class Strategy:
    name = "base"
    stop_atr = 3.0
    trail = False  # trailing stops ratchet up with the highest close
    max_bars = 30

    def entry(self, r: pd.Series) -> bool:
        raise NotImplementedError

    def exit(self, r: pd.Series, pos: Position) -> bool:
        return pos.bars_held >= self.max_bars

    def describe(self) -> str:
        return self.__doc__.strip().splitlines()[0]


class RSI2MeanReversion(Strategy):
    """Connors-style RSI(2): buy deep 2-day oversold dips above the 200-day average, exit on a bounce above the 5-day average."""

    name = "rsi2"
    stop_atr = 2.5
    max_bars = 10

    def entry(self, r):
        return r.close > r.sma200 and r.rsi2 < 10

    def exit(self, r, pos):
        return r.close > r.sma5 or super().exit(r, pos)


class BollingerReversion(Strategy):
    """Buy a close below the lower Bollinger band in an uptrend, exit back at the 20-day mean."""

    name = "bollinger"
    stop_atr = 2.5
    max_bars = 15

    def entry(self, r):
        return r.close > r.sma200 and r.close < r.bb_lower

    def exit(self, r, pos):
        return r.close > r.sma20 or super().exit(r, pos)


class TrendBreakout(Strategy):
    """Buy a 20-day high breakout when the 50-day is above the 200-day, ride it with a 3 ATR trailing stop."""

    name = "breakout"
    stop_atr = 3.0
    trail = True
    max_bars = 60

    def entry(self, r):
        return (
            r.sma50 > r.sma200
            and r.close > r.high20
            and r.volume > r.vol_avg20
            and r.rsi14 < 80
        )

    def exit(self, r, pos):
        return r.close < r.sma50 or super().exit(r, pos)


ALL = {s.name: s for s in (RSI2MeanReversion(), BollingerReversion(), TrendBreakout())}


def get(names) -> list[Strategy]:
    unknown = set(names) - ALL.keys()
    if unknown:
        raise ValueError(f"unknown strategies {sorted(unknown)}; choose from {sorted(ALL)}")
    return [ALL[n] for n in names]


def ready(r) -> bool:
    """True once every indicator has enough history."""
    return not any(pd.isna(getattr(r, c)) for c in ("sma200", "rsi2", "atr14", "bb_lower", "high20", "vol_avg20"))
