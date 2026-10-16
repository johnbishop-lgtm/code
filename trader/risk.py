"""Risk management. Shared by the backtester and the live engine so both size trades identically.

The AI layer can veto a trade or shrink it, but never make it bigger than these limits allow.
"""
from dataclasses import dataclass


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01      # lose at most 1% of equity if the stop is hit
    max_position_pct: float = 0.20    # no single position larger than 20% of equity
    max_positions: int = 6
    max_daily_loss_pct: float = 0.03  # stop opening trades after a 3% down day
    max_drawdown_pct: float = 0.15    # kill switch: no new trades below 85% of peak equity


def position_size(equity: float, cash: float, price: float, stop: float,
                  cfg: RiskConfig, scale: float = 1.0) -> int:
    """Shares to buy so that hitting `stop` loses ~risk_per_trade of equity, capped by position size and cash."""
    per_share_risk = price - stop
    if price <= 0 or per_share_risk <= 0:
        return 0
    scale = max(0.0, min(scale, 1.0))
    by_risk = equity * cfg.risk_per_trade * scale / per_share_risk
    by_cap = equity * cfg.max_position_pct / price
    by_cash = cash / price
    return max(0, int(min(by_risk, by_cap, by_cash)))


def trading_halted(equity: float, peak_equity: float, day_start_equity: float,
                   cfg: RiskConfig) -> str | None:
    """Return a reason string if new entries must be blocked, else None."""
    if peak_equity > 0 and equity < peak_equity * (1 - cfg.max_drawdown_pct):
        return f"drawdown kill switch: equity {equity:,.0f} is more than {cfg.max_drawdown_pct:.0%} below peak {peak_equity:,.0f}"
    if day_start_equity > 0 and equity < day_start_equity * (1 - cfg.max_daily_loss_pct):
        return f"daily loss limit hit ({cfg.max_daily_loss_pct:.0%})"
    return None
