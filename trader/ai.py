"""AI trade review with Claude.

The rules generate candidate trades; Claude acts as a second opinion. For each candidate
it sees the technical picture, how that strategy has performed on that symbol in the
backtest, the current portfolio, and (optionally) fresh news via web search. It returns
approve/reject and a confidence in [0, 1] that scales the position size DOWN.

Claude can only veto or shrink trades. It can never pick symbols the rules didn't
propose, or size above the risk limits.
"""
import json
import os

SYSTEM = """You are the risk reviewer for a systematic swing-trading program that holds US stocks/ETFs for days to weeks.
Rule-based strategies propose long entries; your job is to filter out the bad ones.

Reject or lower confidence when:
- earnings, an FDA decision, a merger vote or another binary event is due before the expected exit
- the dip is caused by news that changes the company's fundamentals (fraud, guidance cut, lost major customer, delisting risk), since mean reversion assumes the drop is noise
- the strategy's backtest on this symbol is weak (profit factor below ~1.1 or very few trades)
- the portfolio would become concentrated in one sector or highly correlated names
Approve with high confidence when the setup is clean and nothing above applies.

Be concise and concrete in reasons. Do not invent news: if you have not searched or found nothing, say so."""

SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "approve": {"type": "boolean"},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["symbol", "approve", "confidence", "reason"],
                "additionalProperties": False,
            },
        },
        "market_note": {"type": "string"},
    },
    "required": ["decisions", "market_note"],
    "additionalProperties": False,
}


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def review(candidates: list[dict], portfolio: dict, use_news: bool = True,
           model: str | None = None) -> dict:
    """Ask Claude to review candidate trades. Returns {"decisions": {...by symbol}, "market_note": str}."""
    import anthropic

    client = anthropic.Anthropic()
    prompt = (
        "Candidate trades for the next market open:\n"
        f"{json.dumps(candidates, indent=2, default=str)}\n\n"
        f"Current portfolio:\n{json.dumps(portfolio, indent=2, default=str)}\n\n"
        + ("Search the web for recent news and upcoming earnings dates for each candidate before deciding.\n"
           if use_news else "Web search is off; judge from the data given.\n")
        + "Return one decision per candidate."
    )
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 2 * len(candidates) + 2}] if use_news else []
    messages = [{"role": "user", "content": prompt}]

    for _ in range(5):  # continue if the server pauses a long web-search turn
        with client.beta.messages.stream(
            model=model or os.environ.get("TRADER_AI_MODEL", "claude-opus-5-5"),
            max_tokens=32000,
            system=SYSTEM,
            messages=messages,
            tools=tools,
            output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            resp = stream.get_final_message()
        if resp.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": resp.content})

    if resp.stop_reason == "refusal":
        raise RuntimeError(f"AI review declined: {resp.stop_details}")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("AI review was cut off (max_tokens)")
    text = next(b.text for b in reversed(resp.content) if b.type == "text")
    out = json.loads(text)
    return {
        "decisions": {d["symbol"].upper(): d for d in out["decisions"]},
        "market_note": out.get("market_note", ""),
    }
