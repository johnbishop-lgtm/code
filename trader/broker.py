"""Brokers. PaperBroker is a local simulator (default). AlpacaBroker talks to Alpaca's paper or live API."""
import json
import os
from datetime import datetime
from pathlib import Path


class Broker:
    def equity(self, prices: dict[str, float]) -> float: ...
    def cash(self) -> float: ...
    def holdings(self) -> dict[str, int]: ...
    def buy(self, symbol: str, qty: int, price: float, stop: float) -> None: ...
    def sell(self, symbol: str, qty: int, price: float) -> None: ...
    def update_stop(self, symbol: str, qty: int, stop: float) -> None: ...


class PaperBroker(Broker):
    """Simulated account stored in a JSON file. Fills at the given price plus slippage."""

    def __init__(self, path: str = "state/paper_account.json", capital: float = 100_000, slippage_bps: float = 5):
        self.path = Path(path)
        self.slip = slippage_bps / 10_000
        if self.path.exists():
            self.acct = json.loads(self.path.read_text())
        else:
            self.acct = {"cash": capital, "holdings": {}, "stops": {}, "fills": []}
            self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.acct, indent=2))

    def _fill(self, side, symbol, qty, price):
        self.acct["fills"].append({"time": datetime.now().isoformat(timespec="seconds"),
                                   "side": side, "symbol": symbol, "qty": qty, "price": round(price, 4)})

    def equity(self, prices):
        return self.acct["cash"] + sum(q * prices[s] for s, q in self.acct["holdings"].items())

    def cash(self):
        return self.acct["cash"]

    def holdings(self):
        return dict(self.acct["holdings"])

    def buy(self, symbol, qty, price, stop):
        fill = price * (1 + self.slip)
        if qty * fill > self.acct["cash"]:
            raise RuntimeError(f"insufficient cash for {qty} {symbol}")
        self.acct["cash"] -= qty * fill
        self.acct["holdings"][symbol] = self.acct["holdings"].get(symbol, 0) + qty
        self.acct["stops"][symbol] = stop
        self._fill("buy", symbol, qty, fill)
        self._save()

    def sell(self, symbol, qty, price):
        fill = price * (1 - self.slip)
        self.acct["cash"] += qty * fill
        left = self.acct["holdings"].get(symbol, 0) - qty
        if left > 0:
            self.acct["holdings"][symbol] = left
        else:
            self.acct["holdings"].pop(symbol, None)
            self.acct["stops"].pop(symbol, None)
        self._fill("sell", symbol, qty, fill)
        self._save()

    def update_stop(self, symbol, qty, stop):
        self.acct["stops"][symbol] = stop
        self._save()

    def check_stops(self, lows: dict[str, float], opens: dict[str, float]) -> list[tuple[str, float]]:
        """Simulate resting stop orders against the latest bar. Returns (symbol, fill price) for each stop hit."""
        hit = []
        for sym, stop in list(self.acct["stops"].items()):
            if sym in lows and lows[sym] <= stop:
                price = min(opens[sym], stop)
                qty = self.acct["holdings"].get(sym, 0)
                if qty:
                    self.sell(sym, qty, price)
                    hit.append((sym, price))
        return hit


class AlpacaBroker(Broker):
    """Alpaca account. Uses the paper endpoint unless live trading is explicitly enabled.

    Orders are market DAY orders: placed after the close they execute at the next open,
    matching the backtest. Each position gets a resting GTC stop order at the broker,
    so stops are enforced even if this program isn't running.
    """

    def __init__(self, live: bool = False):
        from alpaca.trading.client import TradingClient

        key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("set ALPACA_API_KEY and ALPACA_SECRET_KEY (see .env.example)")
        self.client = TradingClient(key, secret, paper=not live)
        self.live = live

    def equity(self, prices=None):
        return float(self.client.get_account().equity)

    def cash(self):
        return float(self.client.get_account().cash)

    def holdings(self):
        return {p.symbol: int(float(p.qty)) for p in self.client.get_all_positions()}

    def _cancel_stops(self, symbol):
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))
        for o in orders:
            if o.side.value == "sell":
                self.client.cancel_order_by_id(o.id)

    def _place_stop(self, symbol, qty, stop):
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import StopOrderRequest

        self.client.submit_order(StopOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL,
                                                  time_in_force=TimeInForce.GTC, stop_price=round(stop, 2)))

    def buy(self, symbol, qty, price, stop):
        from alpaca.trading.enums import OrderSide, OrderClass, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, StopLossRequest

        # OTO order: the stop is attached and becomes active once the buy fills.
        # GTC so the attached stop leg doesn't expire at the end of the day.
        self.client.submit_order(MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
            order_class=OrderClass.OTO, stop_loss=StopLossRequest(stop_price=round(stop, 2))))

    def sell(self, symbol, qty, price):
        self._cancel_stops(symbol)
        self.client.close_position(symbol)

    def update_stop(self, symbol, qty, stop):
        self._cancel_stops(symbol)
        self._place_stop(symbol, qty, stop)
