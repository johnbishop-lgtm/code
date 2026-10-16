"""Portfolio backtester.

Signals are computed on a bar's close and filled at the NEXT bar's open, so there is no
lookahead. Stops are checked against each bar's low and fill at the stop (or the open,
if the market gaps through it). Slippage is charged on every fill.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import strategies as strat
from .indicators import add_indicators
from .risk import RiskConfig, position_size, trading_halted
from .strategies import Position


@dataclass
class Trade:
    symbol: str
    strategy: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry: float
    exit: float
    qty: int
    reason: str

    @property
    def pnl(self):
        return (self.exit - self.entry) * self.qty

    @property
    def ret(self):
        return self.exit / self.entry - 1


@dataclass
class Result:
    equity: pd.Series
    trades: list[Trade]
    benchmark: pd.Series
    stats: dict = field(default_factory=dict)


def _stats(equity: pd.Series, trades: list[Trade], benchmark: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    dd = equity / equity.cummax() - 1
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl <= 0]
    bench_years = years
    return {
        "start": equity.index[0].date(),
        "end": equity.index[-1].date(),
        "final_equity": equity.iloc[-1],
        "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
        "cagr": (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1,
        "sharpe": rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0,
        "max_drawdown": dd.min(),
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "avg_win": np.mean(wins) if wins else 0.0,
        "avg_loss": np.mean(losses) if losses else 0.0,
        "profit_factor": sum(wins) / -sum(losses) if losses and sum(losses) < 0 else float("inf"),
        "expectancy_per_trade": np.mean([t.pnl for t in trades]) if trades else 0.0,
        "exposure": None,
        "benchmark_return": benchmark.iloc[-1] / benchmark.iloc[0] - 1,
        "benchmark_cagr": (benchmark.iloc[-1] / benchmark.iloc[0]) ** (1 / bench_years) - 1,
        "benchmark_max_drawdown": (benchmark / benchmark.cummax() - 1).min(),
    }


def per_strategy(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([{"strategy": t.strategy, "pnl": t.pnl, "ret": t.ret, "win": t.pnl > 0} for t in trades])
    g = df.groupby("strategy")
    out = pd.DataFrame({
        "trades": g.size(),
        "win_rate": g["win"].mean(),
        "avg_return": g["ret"].mean(),
        "total_pnl": g["pnl"].sum(),
        "profit_factor": g["pnl"].apply(
            lambda s: s[s > 0].sum() / -s[s <= 0].sum() if (s <= 0).any() and s[s <= 0].sum() < 0 else float("inf")),
    })
    return out


def run(data: dict[str, pd.DataFrame], strategy_names, capital: float = 100_000,
        risk: RiskConfig | None = None, slippage_bps: float = 5.0) -> Result:
    risk = risk or RiskConfig()
    strats = strat.get(strategy_names)
    frames = {s: add_indicators(df) for s, df in data.items()}
    rows = {s: {r.Index: r for r in f.itertuples()} for s, f in frames.items()}
    dates = sorted(set().union(*(f.index for f in frames.values())))
    slip = slippage_bps / 10_000

    cash = capital
    positions: dict[str, Position] = {}
    pending_entries: dict[str, tuple[strat.Strategy, int, float]] = {}  # sym -> (strategy, qty, atr)
    pending_exits: dict[str, str] = {}  # sym -> reason
    trades: list[Trade] = []
    equity_curve = {}
    last_close = {}
    peak = capital
    prev_equity = capital
    exposure_days = 0

    def close_position(sym, price, date, reason):
        nonlocal cash
        pos = positions.pop(sym)
        cash += pos.qty * price
        trades.append(Trade(sym, pos.strategy, pos.entry_date, date, pos.entry_price, price, pos.qty, reason))

    for d in dates:
        bars = {s: rs[d] for s, rs in rows.items() if d in rs}

        # 1) fills at the open
        for sym, reason in list(pending_exits.items()):
            if sym in bars and sym in positions:
                close_position(sym, bars[sym].open * (1 - slip), d, reason)
                del pending_exits[sym]
        for sym, (s, qty, a) in list(pending_entries.items()):
            if sym not in bars:
                continue
            del pending_entries[sym]
            fill = bars[sym].open * (1 + slip)
            stop = fill - s.stop_atr * a
            qty = min(qty, int(cash / fill))
            if qty <= 0:
                continue
            cash -= qty * fill
            positions[sym] = Position(sym, s.name, qty, fill, d, stop, 0, fill)

        # 2) intraday stops
        for sym, pos in list(positions.items()):
            if sym in bars and bars[sym].low <= pos.stop:
                price = min(bars[sym].open, pos.stop) * (1 - slip)
                close_position(sym, price, d, "stop")
                pending_exits.pop(sym, None)

        # 3) end of day: update positions and look for exits
        for sym, r in bars.items():
            last_close[sym] = r.close
        for sym, pos in positions.items():
            if sym not in bars:
                continue
            r = bars[sym]
            pos.bars_held += 1
            pos.highest_close = max(pos.highest_close, r.close)
            s = strat.ALL[pos.strategy]
            if s.trail and not np.isnan(r.atr14):
                pos.stop = max(pos.stop, pos.highest_close - s.stop_atr * r.atr14)
            if s.exit(r, pos):
                pending_exits[sym] = "signal" if pos.bars_held < s.max_bars else "time"

        equity = cash + sum(p.qty * last_close[p.symbol] for p in positions.values())
        equity_curve[d] = equity
        exposure_days += bool(positions)
        peak = max(peak, equity)

        # 4) new entry signals, filled tomorrow
        if not trading_halted(equity, peak, prev_equity, risk):
            open_slots = risk.max_positions - len(positions) - len(pending_entries) + len(pending_exits)
            reserved = sum(q * bars[s].close for s, (_, q, _) in pending_entries.items() if s in bars)
            candidates = []
            for sym, r in bars.items():
                if sym in positions or sym in pending_entries or not strat.ready(r):
                    continue
                for s in strats:
                    if s.entry(r):
                        candidates.append((r.rsi2, sym, s, r))
                        break
            candidates.sort(key=lambda c: c[0])  # most oversold first
            for _, sym, s, r in candidates[:max(open_slots, 0)]:
                stop = r.close - s.stop_atr * r.atr14
                qty = position_size(equity, cash - reserved, r.close, stop, risk)
                if qty > 0:
                    pending_entries[sym] = (s, qty, r.atr14)
                    reserved += qty * r.close
        prev_equity = equity

    # mark open positions to the last close
    for sym in list(positions):
        close_position(sym, last_close[sym], dates[-1], "end of test")
    equity = pd.Series(equity_curve)

    norm = [f["close"] / f["close"].iloc[0] for f in frames.values()]
    benchmark = pd.concat(norm, axis=1).ffill().mean(axis=1).reindex(equity.index).ffill() * capital

    stats = _stats(equity, trades, benchmark)
    stats["exposure"] = exposure_days / len(equity)
    return Result(equity, trades, benchmark, stats)


def walk_forward(data, strategy_names, splits: int = 4, **kw) -> pd.DataFrame:
    """Run the same fixed rules on consecutive out-of-sample windows to see if results are stable over time."""
    idx = sorted(set().union(*(df.index for df in data.values())))
    warmup = 220  # bars needed before indicators are valid
    usable = idx[warmup:]
    chunk = len(usable) // splits
    rows = []
    for i in range(splits):
        start = usable[i * chunk]
        end = usable[(i + 1) * chunk - 1] if i < splits - 1 else usable[-1]
        lookback_start = idx[idx.index(start) - warmup]
        window = {s: df.loc[lookback_start:end] for s, df in data.items()}
        res = run(window, strategy_names, **kw)
        eq = res.equity.loc[start:]
        bench = res.benchmark.loc[start:]
        in_window = [t for t in res.trades if t.entry_date >= start]
        wins = sum(t.pnl > 0 for t in in_window)
        rows.append({
            "window": f"{start.date()} → {end.date()}",
            "return": eq.iloc[-1] / eq.iloc[0] - 1,
            "benchmark": bench.iloc[-1] / bench.iloc[0] - 1,
            "max_dd": (eq / eq.cummax() - 1).min(),
            "trades": len(in_window),
            "win_rate": wins / len(in_window) if in_window else 0.0,
        })
    return pd.DataFrame(rows)
