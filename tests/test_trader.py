import numpy as np
import pandas as pd
import pytest

from trader import backtest, data, engine
from trader.broker import PaperBroker
from trader.indicators import add_indicators, rsi
from trader.risk import RiskConfig, position_size, trading_halted


@pytest.fixture(autouse=True)
def tmp_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "STATE", tmp_path / "state/engine.json")
    monkeypatch.setattr(engine, "JOURNAL", tmp_path / "state/journal.jsonl")


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 50, dtype=float))
    assert rsi(up, 2).iloc[-1] == 100
    down = pd.Series(np.arange(50, 1, -1, dtype=float))
    assert rsi(down, 2).iloc[-1] < 1
    r = rsi(data.synthetic("X", 2)["close"], 14).dropna()
    assert r.between(0, 100).all()


def test_indicators_have_no_lookahead():
    df = data.synthetic("X", 3)
    full = add_indicators(df)
    cut = add_indicators(df.iloc[:400])
    cols = ["sma200", "rsi2", "atr14", "bb_lower", "high20"]
    pd.testing.assert_frame_equal(full[cols].iloc[:400], cut[cols])


def test_position_size_respects_all_caps():
    cfg = RiskConfig(risk_per_trade=0.01, max_position_pct=0.2)
    # risk cap: $1,000 risk / $5 per share = 200 shares, but 20% cap = $20k / $100 = 200
    assert position_size(100_000, 100_000, 100, 95, cfg) == 200
    # tight stop would allow 1000 shares; position cap limits to 200
    assert position_size(100_000, 100_000, 100, 99, cfg) == 200
    # cash cap
    assert position_size(100_000, 5_000, 100, 95, cfg) == 50
    # AI confidence scales down, never up
    assert position_size(100_000, 100_000, 100, 90, cfg, scale=0.5) == 50
    assert position_size(100_000, 100_000, 100, 90, cfg, scale=3) == 100
    # invalid stop
    assert position_size(100_000, 100_000, 100, 100, cfg) == 0


def test_kill_switches():
    cfg = RiskConfig()
    assert trading_halted(84_000, 100_000, 84_000, cfg)
    assert trading_halted(96_000, 100_000, 99_500, cfg)
    assert trading_halted(99_000, 100_000, 99_500, cfg) is None


def test_backtest_runs_and_accounts_consistently():
    d = data.load(["A", "B", "C"], source="synthetic", years=5)
    r = backtest.run(d, ["rsi2", "bollinger", "breakout"], capital=50_000)
    assert r.stats["trades"] > 20
    # every trade's P&L must add up to the final equity
    assert abs(50_000 + sum(t.pnl for t in r.trades) - r.equity.iloc[-1]) < 1
    assert all(t.entry_date <= t.exit_date for t in r.trades)
    assert 0 <= r.stats["win_rate"] <= 1
    assert r.stats["max_drawdown"] <= 0


def test_walk_forward_windows():
    d = data.load(["A", "B"], source="synthetic", years=6)
    wf = backtest.walk_forward(d, ["rsi2"], splits=3)
    assert len(wf) == 3


def test_paper_broker_round_trip():
    b = PaperBroker(path="acct.json", capital=10_000, slippage_bps=0)
    b.buy("X", 10, 100, 90)
    assert b.cash() == 9_000 and b.holdings() == {"X": 10}
    assert b.check_stops({"X": 95}, {"X": 99}) == []
    assert b.check_stops({"X": 85}, {"X": 88}) == [("X", 88)]  # gapped below stop: fill at open
    assert b.holdings() == {} and b.cash() == 9_880
    # state persists
    assert PaperBroker(path="acct.json").cash() == 9_880


def test_engine_fails_closed_without_ai_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    syms = [f"S{i}" for i in range(15)]
    b = PaperBroker(path="acct.json")
    out = engine.run_once(b, syms, ["rsi2", "bollinger", "breakout"], source="synthetic", use_ai=True, out=lambda *_: None)
    assert b.holdings() == {}
    if any("AI review requested" in a for a in out["actions"]):
        assert not any(a.startswith("BUY") for a in out["actions"])


def test_engine_ai_veto_and_scaling(monkeypatch):
    syms = [f"S{i}" for i in range(15)]
    seen = {}

    def fake_review(cands, portfolio, use_news=True, model=None):
        seen["cands"] = cands
        first = cands[0]["symbol"]
        return {"market_note": "", "decisions": {
            c["symbol"]: {"symbol": c["symbol"], "approve": c["symbol"] != first, "confidence": 0.6, "reason": "test"}
            for c in cands}}

    monkeypatch.setattr(engine.ai, "available", lambda: True)
    monkeypatch.setattr(engine.ai, "review", fake_review)
    b = PaperBroker(path="acct.json")
    engine.run_once(b, syms, ["rsi2", "bollinger", "breakout"], source="synthetic", out=lambda *_: None)
    if not seen:
        pytest.skip("no signals on this synthetic day")
    assert seen["cands"][0]["symbol"] not in b.holdings()
    assert set(b.holdings()) == {c["symbol"] for c in seen["cands"][1:]}
