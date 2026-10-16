"""Live/paper trading loop. Run once per trading day after the close (e.g. 4:15pm ET)."""
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from . import ai, backtest, data
from . import strategies as strat
from .broker import AlpacaBroker, Broker, PaperBroker
from .indicators import add_indicators
from .risk import RiskConfig, position_size, trading_halted
from .strategies import Position

STATE = Path("state/engine.json")
JOURNAL = Path("state/journal.jsonl")
MIN_AI_CONFIDENCE = 0.5


def _load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"positions": {}, "peak_equity": 0.0, "day_start": {}, "closed": []}


def _save_state(st: dict):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2, default=str))


def _log(event: dict):
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **event}, default=str) + "\n")


def _symbol_track_record(df: pd.DataFrame, strategy: str) -> dict:
    """How this one strategy has done on this one symbol historically."""
    res = backtest.run({"X": df}, [strategy])
    s = res.stats
    return {"trades": s["trades"], "win_rate": round(s["win_rate"], 3),
            "profit_factor": round(s["profit_factor"], 2) if s["profit_factor"] != float("inf") else "inf",
            "avg_trade_pnl_per_100k": round(s["expectancy_per_trade"], 2)}


def make_broker(kind: str, live: bool = False) -> Broker:
    return AlpacaBroker(live=live) if kind == "alpaca" else PaperBroker()


def run_once(broker: Broker, symbols, strategy_names, risk: RiskConfig | None = None,
             source: str = "yahoo", use_ai: bool = True, ai_news: bool = True,
             dry_run: bool = False, out=print) -> dict:
    risk = risk or RiskConfig()
    strats = strat.get(strategy_names)
    st = _load_state()
    raw = data.load(symbols, source=source, years=3)
    frames = {s: add_indicators(df) for s, df in raw.items()}
    last = {s: f.iloc[-1] for s, f in frames.items()}
    bar_date = max(f.index[-1] for f in frames.values()).date()
    prices = {s: r.close for s, r in last.items()}
    actions = []

    # Paper account: simulate resting stops against the newest bar.
    if isinstance(broker, PaperBroker):
        lows, opens = {}, {}
        for sym, meta in st["positions"].items():
            if sym in last and pd.Timestamp(meta["entry_date"]).date() < bar_date:
                lows[sym], opens[sym] = last[sym].low, last[sym].open
        for sym, px in broker.check_stops(lows, opens):
            actions.append(f"STOPPED OUT {sym} @ {px:.2f}")

    holdings = broker.holdings()
    # Positions that disappeared at the broker were stopped out (or closed by hand).
    for sym in list(st["positions"]):
        if sym not in holdings:
            meta = st["positions"].pop(sym)
            st["closed"].append({**meta, "symbol": sym, "exit_date": bar_date, "reason": "stop/closed at broker"})
            _log({"event": "closed", "symbol": sym, "reason": "stop/closed at broker"})
            actions.append(f"{sym} no longer held (stop filled or closed manually)")
    unmanaged = sorted(set(holdings) - set(st["positions"]))

    # Manage open positions: trail stops, check exit rules.
    for sym, meta in list(st["positions"].items()):
        if sym not in frames:
            continue
        f = frames[sym]
        since = f.loc[f.index > pd.Timestamp(meta["entry_date"])]
        s = strat.ALL[meta["strategy"]]
        pos = Position(sym, s.name, holdings[sym], meta["entry_price"], pd.Timestamp(meta["entry_date"]),
                       meta["stop"], len(since), max([meta["entry_price"], *since["close"].tolist()]))
        r = last[sym]
        if s.trail:
            new_stop = max(pos.stop, pos.highest_close - s.stop_atr * r.atr14)
            if new_stop > pos.stop + 0.01:
                actions.append(f"RAISE STOP {sym} {pos.stop:.2f} -> {new_stop:.2f}")
                if not dry_run:
                    broker.update_stop(sym, pos.qty, new_stop)
                    meta["stop"] = new_stop
        if s.exit(r, pos):
            why = "time limit" if pos.bars_held >= s.max_bars else "exit signal"
            actions.append(f"SELL {pos.qty} {sym} ({s.name}: {why})")
            if not dry_run:
                broker.sell(sym, pos.qty, r.close)
                st["positions"].pop(sym)
                pnl = (r.close - pos.entry_price) * pos.qty
                st["closed"].append({**meta, "symbol": sym, "exit_date": bar_date, "exit_price": r.close,
                                     "pnl": pnl, "reason": why})
                _log({"event": "sell", "symbol": sym, "qty": pos.qty, "price": r.close, "pnl": pnl, "reason": why})

    # Risk gates.
    equity = broker.equity(prices)
    st["peak_equity"] = max(st["peak_equity"], equity)
    today = str(date.today())
    if st["day_start"].get("date") != today:
        st["day_start"] = {"date": today, "equity": equity}
    halted = trading_halted(equity, st["peak_equity"], st["day_start"]["equity"], risk)

    # New entries.
    candidates = []
    if halted:
        actions.append(f"NO NEW TRADES: {halted}")
    else:
        held = set(broker.holdings()) | set(st["positions"])
        slots = risk.max_positions - len(held)
        for sym, r in last.items():
            if sym in held or not strat.ready(r):
                continue
            for s in strats:
                if s.entry(r):
                    candidates.append({"symbol": sym, "strategy": s.name, "rule": s.describe(),
                                       "close": round(r.close, 2), "stop": round(r.close - s.stop_atr * r.atr14, 2),
                                       "rsi2": round(r.rsi2, 1), "rsi14": round(r.rsi14, 1),
                                       "pct_above_sma200": round((r.close / r.sma200 - 1) * 100, 1),
                                       "atr_pct": round(r.atr14 / r.close * 100, 2)})
                    break
        candidates.sort(key=lambda c: c["rsi2"])
        candidates = candidates[:max(slots, 0)]

    decisions = {}
    if candidates:
        for c in candidates:
            c["backtest_on_this_symbol"] = _symbol_track_record(raw[c["symbol"]], c["strategy"])
        if use_ai:
            if not ai.available():
                actions.append("AI review requested but ANTHROPIC_API_KEY is not set: skipping new entries (fail closed). "
                               "Use --no-ai to trade on rules alone.")
                candidates = []
            else:
                portfolio = {"equity": round(equity, 2), "cash": round(broker.cash(), 2),
                             "positions": {s: {"strategy": m["strategy"], "entry_date": m["entry_date"]}
                                           for s, m in st["positions"].items()}}
                try:
                    rv = ai.review(candidates, portfolio, use_news=ai_news)
                    decisions = rv["decisions"]
                    if rv["market_note"]:
                        actions.append(f"AI market note: {rv['market_note']}")
                except Exception as e:  # fail closed: no AI verdict means no new trades
                    actions.append(f"AI review failed ({e}); skipping new entries")
                    candidates = []

    cash = broker.cash()
    for c in candidates:
        sym = c["symbol"]
        scale = 1.0
        if use_ai:
            d = decisions.get(sym)
            if not d or not d["approve"] or d["confidence"] < MIN_AI_CONFIDENCE:
                reason = d["reason"] if d else "no decision returned"
                actions.append(f"AI REJECTED {sym} ({c['strategy']}): {reason}")
                _log({"event": "ai_reject", "symbol": sym, "candidate": c, "decision": d})
                continue
            scale = d["confidence"]
        qty = position_size(equity, cash, c["close"], c["stop"], risk, scale=scale)
        if qty <= 0:
            actions.append(f"SKIP {sym}: size rounds to 0 shares")
            continue
        note = f" — AI {scale:.0%}: {decisions[sym]['reason']}" if use_ai else ""
        actions.append(f"BUY {qty} {sym} @ ~{c['close']} stop {c['stop']} ({c['strategy']}){note}")
        if not dry_run:
            broker.buy(sym, qty, c["close"], c["stop"])
            cash -= qty * c["close"]
            st["positions"][sym] = {"strategy": c["strategy"], "entry_date": str(bar_date),
                                    "entry_price": c["close"], "stop": c["stop"]}
            _log({"event": "buy", "symbol": sym, "qty": qty, "candidate": c, "ai": decisions.get(sym)})

    if not dry_run:
        _save_state(st)
    summary = {"bar_date": bar_date, "equity": equity, "actions": actions, "unmanaged": unmanaged}
    out(f"Data through {bar_date} | equity ${equity:,.2f} | peak ${st['peak_equity']:,.2f}"
        + (" | DRY RUN" if dry_run else ""))
    for a in actions or ["No trades today."]:
        out("  " + a)
    if unmanaged:
        out(f"  (ignoring positions not opened by this program: {', '.join(unmanaged)})")
    return summary
