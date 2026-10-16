"""Command line: python -m trader {backtest,scan,trade,status}"""
import argparse
import os
import sys
from pathlib import Path

from . import backtest, data, engine
from . import strategies as strat
from .risk import RiskConfig

UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP",
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "V", "COST"]


def _load_dotenv():
    p = Path(".env")
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def _risk(a) -> RiskConfig:
    return RiskConfig(risk_per_trade=a.risk, max_positions=a.max_positions)


def cmd_backtest(a):
    d = data.load(a.symbols, source=a.source, years=a.years)
    r = backtest.run(d, a.strategies, capital=a.capital, risk=_risk(a))
    s = r.stats
    print(f"\nBacktest {s['start']} → {s['end']} | {len(a.symbols)} symbols | strategies: {', '.join(a.strategies)}"
          f" | data: {a.source}")
    if a.source == "synthetic":
        print("  !! synthetic random data: these numbers only show the code runs, not that it makes money")
    rows = [
        ("Total return", f"{s['total_return']:+.1%}", f"{s['benchmark_return']:+.1%}"),
        ("CAGR", f"{s['cagr']:+.1%}", f"{s['benchmark_cagr']:+.1%}"),
        ("Max drawdown", f"{s['max_drawdown']:.1%}", f"{s['benchmark_max_drawdown']:.1%}"),
        ("Sharpe", f"{s['sharpe']:.2f}", ""),
        ("Trades", f"{s['trades']}", ""),
        ("Win rate", f"{s['win_rate']:.1%}", ""),
        ("Profit factor", f"{s['profit_factor']:.2f}", ""),
        ("Avg win / avg loss", f"${s['avg_win']:,.0f} / ${s['avg_loss']:,.0f}", ""),
        ("Time in market", f"{s['exposure']:.0%}", "100%"),
    ]
    print(f"\n  {'':22}{'Strategy':>18}{'Buy & hold':>14}")
    for name, v, b in rows:
        print(f"  {name:22}{v:>18}{b:>14}")
    print("\nBy strategy:")
    print(backtest.per_strategy(r.trades).to_string(float_format=lambda x: f"{x:,.3f}"))
    if a.walk_forward:
        print("\nWalk-forward (same rules on separate time windows — look for consistency):")
        wf = backtest.walk_forward(d, a.strategies, capital=a.capital, risk=_risk(a))
        print(wf.to_string(index=False, float_format=lambda x: f"{x:+.1%}"))
    out = Path("reports")
    out.mkdir(exist_ok=True)
    r.equity.rename("equity").to_frame().assign(benchmark=r.benchmark).to_csv(out / "equity.csv")
    import pandas as pd
    pd.DataFrame([{**t.__dict__, "pnl": t.pnl, "ret": t.ret} for t in r.trades]).to_csv(out / "trades.csv", index=False)
    print(f"\nSaved {out}/equity.csv and {out}/trades.csv")


def cmd_scan(a):
    engine.run_once(engine.PaperBroker(path="state/scan_scratch.json"), a.symbols, a.strategies,
                    risk=_risk(a), source=a.source, use_ai=False, dry_run=True)


def cmd_trade(a):
    if a.live:
        if os.environ.get("ALPACA_LIVE_CONFIRM") != "I accept the risk of losing real money":
            sys.exit("Live trading is locked. Set ALPACA_LIVE_CONFIRM='I accept the risk of losing real money' "
                     "after you have paper traded this for a while.")
        if a.broker != "alpaca":
            sys.exit("--live requires --broker alpaca")
    broker = engine.make_broker(a.broker, live=a.live)
    engine.run_once(broker, a.symbols, a.strategies, risk=_risk(a), source=a.source,
                    use_ai=not a.no_ai, ai_news=not a.no_news, dry_run=a.dry_run)


def cmd_status(a):
    broker = engine.make_broker(a.broker, live=a.live)
    st = engine._load_state()
    holdings = broker.holdings()
    prices = {}
    if holdings and a.broker == "paper":
        prices = {s: df["close"].iloc[-1] for s, df in data.load(holdings, source=a.source, years=1).items()}
    print(f"Equity ${broker.equity(prices):,.2f} | cash ${broker.cash():,.2f}")
    for sym, q in holdings.items():
        m = st["positions"].get(sym, {})
        print(f"  {sym:6} {q:>6} sh  {m.get('strategy', 'manual'):10} entry {m.get('entry_price', 0):>9.2f}"
              f"  stop {m.get('stop', 0):>9.2f}  since {m.get('entry_date', '?')}")
    closed = st["closed"]
    pnls = [c["pnl"] for c in closed if "pnl" in c]
    if pnls:
        print(f"Closed trades with recorded P&L: {len(pnls)} | win rate {sum(p > 0 for p in pnls) / len(pnls):.0%}"
              f" | total ${sum(pnls):,.2f}")


def main(argv=None):
    _load_dotenv()
    p = argparse.ArgumentParser(prog="trader", description="AI-assisted swing trading (paper by default)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--symbols", nargs="+", default=UNIVERSE)
        sp.add_argument("--strategies", nargs="+", default=["rsi2", "bollinger", "breakout"], choices=sorted(strat.ALL))
        sp.add_argument("--source", default="yahoo", choices=["yahoo", "csv", "synthetic"])
        sp.add_argument("--risk", type=float, default=0.01, help="fraction of equity risked per trade")
        sp.add_argument("--max-positions", type=int, default=6)

    b = sub.add_parser("backtest", help="test the strategies on historical data")
    common(b)
    b.add_argument("--years", type=int, default=10)
    b.add_argument("--capital", type=float, default=100_000)
    b.add_argument("--walk-forward", action="store_true")
    b.set_defaults(fn=cmd_backtest)

    s = sub.add_parser("scan", help="show today's signals without trading")
    common(s)
    s.set_defaults(fn=cmd_scan)

    t = sub.add_parser("trade", help="run the daily trading cycle (after the close)")
    common(t)
    t.add_argument("--broker", default="paper", choices=["paper", "alpaca"])
    t.add_argument("--live", action="store_true", help="real money (Alpaca only, requires ALPACA_LIVE_CONFIRM)")
    t.add_argument("--no-ai", action="store_true", help="trade on rules alone, skip Claude's review")
    t.add_argument("--no-news", action="store_true", help="AI review without web search")
    t.add_argument("--dry-run", action="store_true", help="print what would happen, place no orders")
    t.set_defaults(fn=cmd_trade)

    st = sub.add_parser("status", help="show account, positions and realized results")
    st.add_argument("--broker", default="paper", choices=["paper", "alpaca"])
    st.add_argument("--live", action="store_true")
    st.add_argument("--source", default="yahoo", choices=["yahoo", "csv", "synthetic"])
    st.set_defaults(fn=cmd_status)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
