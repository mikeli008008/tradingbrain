"""
Daily agent run — the entrypoint GitHub Actions calls.

Uses the Claude Agent SDK to run a tool-calling loop. The agent:
  1. Reads yesterday's journal + learned rules
  2. Checks regime, scans screener, researches candidates
  3. Plans trades, submits through the risk-gated broker
  4. Writes today's journal and potentially new rules
"""
from __future__ import annotations
import asyncio
import os
import json
from datetime import date
from pathlib import Path

from anthropic import Anthropic
from tools import market_data, news_research, broker, journal
from learning import calibration, rule_ledger, shadow_journal

ROOT = Path(__file__).parent.parent
MODEL = "claude-opus-4-7"  # Full reasoning for judgment calls


# --- Tool registry: the schema Claude sees ---

TOOLS = [
    {
        "name": "get_market_regime",
        "description": "Check SPY regime. Returns MA50/MA200 status and whether SPY broke down on high volume. Required before any buy decision.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_minervini_scan",
        "description": "Fetch latest Minervini screener output. Returns Super / Perfect / Leader / Watchlist tiered candidates.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_quote",
        "description": "Current price, ATR, and % from 52w high for a ticker. Use to size position and place stop.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "research_ticker_news",
        "description": "Recent headlines for a ticker. Scan for catalysts, earnings, lawsuits.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "days": {"type": "integer", "default": 3},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "research_macro",
        "description": "Macro snapshot — SPY, QQQ, VIX, TLT, DXY day changes. Use once per session.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "earnings_on_deck",
        "description": "Flag which of your candidates report earnings in the next 7 days. Avoid new entries on these.",
        "input_schema": {
            "type": "object",
            "properties": {"tickers": {"type": "array", "items": {"type": "string"}}},
            "required": ["tickers"],
        },
    },
    {
        "name": "list_positions",
        "description": "Current open positions with P&L.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "place_trade",
        "description": "Place a long entry. Goes through the risk manager which may reject. Always include all fields.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "entry_price": {"type": "number"},
                "stop_price": {"type": "number", "description": "Must be 3-8% below entry"},
                "shares": {"type": "integer"},
                "signal_tier": {"type": "string", "enum": ["super", "perfect", "leader", "watchlist"]},
                "reason": {"type": "string", "description": "Plain-English rationale, 1-2 sentences"},
            },
            "required": ["ticker", "entry_price", "stop_price", "shares", "signal_tier", "reason"],
        },
    },
    {
        "name": "close_position",
        "description": "Exit a position. Use if regime changed or thesis broken.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["ticker", "reason"],
        },
    },
    {
        "name": "write_journal_entry",
        "description": "Persist today's plan, executed trades, and reflection. Call this at the end of the session.",
        "input_schema": {
            "type": "object",
            "properties": {
                "plan": {"type": "object"},
                "trades_executed": {"type": "array"},
                "reflection": {"type": "string"},
            },
            "required": ["reflection"],
        },
    },
    {
        "name": "append_learned_rule",
        "description": "Add a rule to your rulebook. Only for patterns, not one-offs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rule": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["rule", "reason"],
        },
    },
]


# --- Tool dispatch ---

def execute_tool(name: str, args: dict) -> dict:
    """Route a tool call to the actual implementation."""
    # Cache regime across calls in a single session
    regime = execute_tool._cached_regime

    try:
        if name == "get_market_regime":
            r = market_data.get_market_regime()
            execute_tool._cached_regime = r
            return r
        if name == "get_minervini_scan":
            return market_data.get_minervini_scan()
        if name == "get_quote":
            return market_data.get_quote(args["ticker"])
        if name == "research_ticker_news":
            return news_research.research_ticker_news(args["ticker"], args.get("days", 3))
        if name == "research_macro":
            return news_research.research_macro()
        if name == "earnings_on_deck":
            return news_research.earnings_on_deck(args["tickers"])
        if name == "list_positions":
            return broker.list_positions()
        if name == "place_trade":
            if regime is None:
                regime = market_data.get_market_regime()
                execute_tool._cached_regime = regime
            return broker.place_trade(
                ticker=args["ticker"],
                entry_price=args["entry_price"],
                stop_price=args["stop_price"],
                shares=args["shares"],
                signal_tier=args["signal_tier"],
                reason=args["reason"],
                market_regime=regime,
            )
        if name == "close_position":
            return broker.close_position(args["ticker"], args["reason"])
        if name == "write_journal_entry":
            journal.write_journal(date.today(), args)
            return {"written": True}
        if name == "append_learned_rule":
            return journal.append_learned_rule(args["rule"], args["reason"])
        return {"error": f"unknown tool {name}"}
    except Exception as e:
        return {"error": str(e), "hint": "Tool failed. Consider skipping this step."}

execute_tool._cached_regime = None


# --- Agent loop ---

def run_session(max_iterations: int = 40, dry_run: bool = False) -> dict:
    """Run one daily trading session."""
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    system = (ROOT / "prompts" / "system.md").read_text()

    # Seed the session with memory
    recent_shadows = shadow_journal.load_since(5)
    shadow_summary = (
        f"{len(recent_shadows)} shadow decisions in last 5 days — "
        f"{sum(1 for d in recent_shadows if d.action == 'buy')} would-buys, "
        f"{sum(1 for d in recent_shadows if d.action == 'skip')} skips."
    )

    morning_context = "\n\n".join([
        f"## Today is {date.today().isoformat()}",
        f"## Yesterday's journal\n{journal.summarize_yesterday()}",
        f"## Your evidence-backed rules (empirically validated)\n"
        f"{rule_ledger.render_active_rules_for_agent()}",
        f"## Your calibration\n{calibration.render_for_agent()}",
        f"## Your hand-written rules\n{journal.read_learned_rules()}",
        f"## Recent shadow activity\n{shadow_summary}",
        "## Task\nExecute your daily workflow: review → regime → scan → research → plan → execute → reflect. "
        "End with write_journal_entry. The evidence-backed rules above come from "
        "statistical analysis of your own past decisions — weight them heavily.",
    ])

    messages = [{"role": "user", "content": morning_context}]
    trace = []

    for i in range(max_iterations):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})
        trace.append({"iteration": i, "stop_reason": resp.stop_reason})

        if resp.stop_reason == "end_turn":
            break

        if resp.stop_reason == "tool_use":
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str)[:8000],
                    })
                    trace.append({"tool": block.name, "input": block.input, "result_preview": str(result)[:200]})
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    # Dump trace for post-mortem
    trace_path = ROOT / "state" / "traces" / f"{date.today().isoformat()}.json"
    trace_path.parent.mkdir(exist_ok=True)
    trace_path.write_text(json.dumps(trace, indent=2, default=str))

    return {"iterations": len(trace), "final_stop": resp.stop_reason}


if __name__ == "__main__":
    result = run_session(dry_run=os.getenv("DRY_RUN", "false").lower() == "true")
    print(json.dumps(result, indent=2))
