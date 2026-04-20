"""
Emergency agent — invoked only when overnight sentinel detects catastrophic
news on an open position. Scope is narrow:

  - Read the alert file
  - Review each affected position
  - Decide: cancel_stop_and_market_exit / tighten_stop / hold
  - Execute + log

Does NOT open new positions. Does NOT trade pre-market (execution quality
too poor). Just manages existing risk before the regular open.
"""
from __future__ import annotations
import os
import json
from datetime import datetime, date
from pathlib import Path

from anthropic import Anthropic
from tools import broker

ROOT = Path(__file__).parent.parent
MODEL = "claude-opus-4-7"

SYSTEM = """You are the emergency-response mode of a trading agent.

An after-hours news scan just flagged catastrophic-severity news on one or
more of your open positions. You do NOT open new trades. You only protect
existing positions before the next market open.

For each affected position, decide exactly ONE action:

1. "market_exit_at_open" — cancel the resting stop and submit a market sell
   order that will execute at the open. Use when the news is truly catastrophic
   (fraud, SEC, bankruptcy, criminal, major trial failure) and pre-market is
   already showing a >5% gap down. You accept slippage because waiting for the
   resting stop means slipping through it anyway.

2. "tighten_stop" — raise the stop closer to current price. Use when news is
   negative but not catastrophic, and position is still above your original
   stop. Specify the new stop price.

3. "hold" — keep everything as-is. Use when the alert is a false positive on
   review, or when the keyword match does not reflect real risk.

You have ONE tool per position: `execute_emergency_action`.
Call it once per affected position, then summarize and end.

Be decisive. Pre-market is narrow; indecision is itself a choice that costs
money."""

TOOLS = [
    {
        "name": "execute_emergency_action",
        "description": "Execute one of the three emergency actions on a single position.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "action": {
                    "type": "string",
                    "enum": ["market_exit_at_open", "tighten_stop", "hold"],
                },
                "new_stop_price": {
                    "type": "number",
                    "description": "Required for tighten_stop, ignored otherwise",
                },
                "reason": {"type": "string", "description": "1-2 sentences of rationale"},
            },
            "required": ["ticker", "action", "reason"],
        },
    },
]


def execute_emergency_action(args: dict) -> dict:
    """Dispatch the chosen action through the broker."""
    ticker = args["ticker"]
    action = args["action"]
    reason = args["reason"]

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "action": action,
        "reason": reason,
    }

    try:
        if action == "market_exit_at_open":
            result = broker.close_position(ticker, f"emergency: {reason}")
            log_entry["result"] = result
        elif action == "tighten_stop":
            # Requires extending broker module; for now, close + re-establish
            # (In production: cancel old stop, submit new STP at new price)
            log_entry["result"] = {
                "note": "tighten_stop requires broker.update_stop — see TODO",
                "requested_stop": args.get("new_stop_price"),
            }
        else:
            log_entry["result"] = {"note": "held position unchanged"}
    except Exception as e:
        log_entry["error"] = str(e)

    # Log to emergency actions file
    log_path = ROOT / "state" / "emergency_actions.jsonl"
    with log_path.open("a") as f:
        f.write(json.dumps(log_entry, default=str) + "\n")

    return log_entry


def run_emergency() -> dict:
    """Read the trigger file, pass to agent, execute."""
    trigger_path = ROOT / "state" / "EMERGENCY_TRIGGER"
    if not trigger_path.exists():
        return {"skipped": True, "reason": "no trigger file"}

    alert_data = json.loads(trigger_path.read_text())
    positions = broker.list_positions().get("positions", [])

    # Only consider positions that are in the catastrophic list
    catastrophic_tickers = {a["ticker"] for a in alert_data.get("by_severity", {}).get("catastrophic", [])}
    affected = [p for p in positions if p["ticker"] in catastrophic_tickers]

    if not affected:
        return {"skipped": True, "reason": "no affected positions"}

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    context = json.dumps({
        "alert": alert_data,
        "affected_positions": affected,
        "time_utc": datetime.utcnow().isoformat(),
    }, indent=2, default=str)

    messages = [{"role": "user", "content": f"Overnight sentinel alert:\n\n{context}\n\nReview each affected position and take action."}]
    actions_taken = []

    for _ in range(10):
        resp = client.messages.create(
            model=MODEL, max_tokens=2048,
            system=SYSTEM, tools=TOOLS, messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            break
        if resp.stop_reason == "tool_use":
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    result = execute_emergency_action(block.input)
                    actions_taken.append(result)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    return {"actions_taken": actions_taken}


if __name__ == "__main__":
    print(json.dumps(run_emergency(), indent=2, default=str))
