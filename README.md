# AI Swing Trader

A stock trading bot that combines rules that have historically been right more often than not with an AI reviewer (Claude) that checks each trade before it's placed. It runs on **paper money by default**.

## How it decides

1. **Strategies propose trades** (daily bars, holding for days to weeks, long only):
   | Strategy | Idea | Typical profile |
   |---|---|---|
   | `rsi2` | Buy a 2-day RSI below 10 while price is above its 200-day average, sell when it closes back above the 5-day average | High win rate, small wins |
   | `bollinger` | Buy a close below the lower Bollinger band in an uptrend, sell back at the 20-day mean | High win rate, small wins |
   | `breakout` | Buy a 20-day high when the 50-day average is above the 200-day, ride it with a trailing stop | Lower win rate, bigger wins |
2. **Claude reviews each candidate.** It sees the setup, how that strategy has performed on that symbol, your portfolio, and recent news and upcoming earnings from a web search. It approves or rejects the trade and gives a confidence score. A lower confidence makes the position smaller. Claude can only veto or shrink trades. It cannot add trades or go past the risk limits. If the review fails, no new trades are placed.
3. **Risk rules size every trade.** Each trade risks 1% of equity down to its stop. A single position is capped at 20% of equity, and you hold at most 6 positions. Each position has a stop order at the broker. New entries stop after a 3% down day or when equity is more than 15% below its peak.

## Read this before using real money

- **No strategy has a guaranteed high win rate, and win rate alone is misleading.** Mean-reversion strategies often win 60–75% of trades, but their losses are bigger than their wins. A 70% win rate can still lose money. Look at profit factor, drawdown and the comparison against buy-and-hold.
- Backtests show the **rules only**. The AI review can't be backtested honestly, because the model already knows what happened next. You find out whether the AI helps by paper trading.
- On many periods these strategies return **less than simply holding SPY**, with lower drawdowns. That's a common, legitimate trade-off, but you should know if it applies to you.
- Paper trade for at least a few months. Compare your results with the backtest before you unlock live trading. Only trade money you can afford to lose. None of this is financial advice.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your keys
```

## Use

```bash
# 1. Test the rules on 10 years of real data (Yahoo Finance, no key needed)
python -m trader backtest --walk-forward
python -m trader backtest --symbols SPY QQQ IWM --strategies rsi2

# 2. See today's signals, with no orders placed
python -m trader scan

# 3. Daily trading cycle: run once each trading day AFTER the close (~4:15pm ET)
python -m trader trade                      # local paper account, with AI review
python -m trader trade --broker alpaca      # Alpaca paper account
python -m trader trade --dry-run            # show what it would do
python -m trader trade --no-ai              # rules only
python -m trader trade --broker alpaca --live   # real money (needs ALPACA_LIVE_CONFIRM)

# 4. Positions and results
python -m trader status [--broker alpaca]
```

To schedule the daily run, use cron, for example (16:15 ET on weekdays):
```
15 16 * * 1-5  cd /path/to/code && python -m trader trade --broker alpaca >> state/cron.log 2>&1
```

Use `--source synthetic` to try any command offline with random data. Those numbers only show that the code runs.

## Files

- `trader/strategies.py` holds the entry and exit rules. Add your own by subclassing `Strategy`.
- `trader/backtest.py` is a portfolio backtester. It fills at the next bar's open, applies slippage and handles stop gaps.
- `trader/risk.py` sets position sizing and the kill switches. The backtest and live trading share it.
- `trader/ai.py` runs the Claude trade review. It uses structured output and optional web search.
- `trader/broker.py` contains the local paper broker and the Alpaca broker (paper or live).
- `trader/engine.py` runs the daily cycle.
- `state/` holds the account, positions and a decision journal (`journal.jsonl`). It is git-ignored.

Tests: `python -m pytest`
