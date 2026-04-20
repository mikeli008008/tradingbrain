"""
Hourly shadow run — the learning engine's data generator.

Runs every hour during market hours. Does NOT trade. Just records
what the agent WOULD do if this were the moment to decide, with a
confidence estimate. Outcomes are graded later by outcome_grader.

Target: ~7 runs/day × 5 days = 35 decisions/week.
"""
from __future__ import annotations
import os
import json
from datetime import date, datetime
from pathlib import Path

from anthropic import Anthropic
from tools import market_data, news_research
from learning import shadow_journal, calibration, rule_ledger

ROOT = Path(__file__).parent.parent
# Use smaller/cheaper model for hourly — Haiku is plenty for scoring
MODEL = os.getenv("SHADOW_MODEL", "claude-haiku-4-5-20251001")


SHADOW_SYSTEM = """You are the shadow-decision mode of a Minervini trading agent.

You are NOT trading right now. You are recording decisions for learning.

For each candidate provided, output a JSON decision. You have full freedom
to be honest — this goes into the learning journal, not the brokerage.

For each decision include:
  - action: "buy" | "watch" | "skip"
  - confidence: 0.0-1.0 — how sure are you? BE HONEST. Overconfidence
    will be caught by the calibration tracker.
  - entry_price, stop_price, target_price (if action=buy)
  - thesis: 1-2 sentences — why
  - thesis_tags: pick from [
        "vcp_breakout", "earnings_catalyst", "rs_leader",
        "oversold_bounce", "sector_momentum", "avoid_earnings_risk",
        "regime_strong", "regime_weak", "near_pivot", "far_from_pivot",
        "fundamental_grade_a", "fundamental_grade_d",
        "catalyst_positive_news", "catalyst_negative_news",
        "high_volume_confirmation", "low_volume_warning"
    ]

Tags are how patterns get mined into rules. Use them consistently.

Output ONLY valid JSON in an array — one object per candidate.
"""


def run_shadow_session() -> dict:
    """Record shadow decisions for current scan."""
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Gather state
    regime = market_data.get_market_regime()
    scan = market_data.get_minervini_scan()

    candidates = (scan.get("super", []) + scan.get("perfect", []))[:8]
    if not candidates:
        return {"recorded": 0, "reason": "no candidates in scan"}

    # Fetch quotes + news snippets for each
    enriched = []
    for c in candidates:
        tkr = c.get("ticker") or c.get("symbol")
        if not tkr:
            continue
        quote = market_data.get_quote(tkr)
        news = news_research.research_ticker_news(tkr, days=2)
        enriched.append({
            "scan_data": c,
            "quote": quote,
            "recent_headlines": [h["headline"] for h in news.get("headlines", [])[:5]],
        })

    # Compose the context
    context = {
        "timestamp": datetime.utcnow().isoformat(timespec="minutes"),
        "market_regime": regime,
        "candidates": enriched,
        "calibration_feedback": calibration.render_for_agent(),
        "active_rules": rule_ledger.render_active_rules_for_agent(),
    }

    # Call the model
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SHADOW_SYSTEM,
        messages=[{
            "role": "user",
            "content": json.dumps(context, default=str, indent=2)
        }],
    )

    # Extract JSON from response
    text = "".join(b.text for b in resp.content if b.type == "text")
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]

    try:
        decisions = json.loads(text)
    except json.JSONDecodeError as e:
        # Save raw output for debugging
        (ROOT / "state" / "shadow_parse_errors.log").open("a").write(
            f"{datetime.utcnow().isoformat()}: {e}\n{text[:500]}\n\n"
        )
        return {"recorded": 0, "error": "failed to parse LLM JSON"}

    recorded = 0
    for d in decisions:
        try:
            shadow_journal.record({
                "ticker": d["ticker"],
                "action": d["action"],
                "confidence": float(d["confidence"]),
                "signal_tier": d.get("signal_tier", "perfect"),
                "entry_price": float(d.get("entry_price", 0) or 0),
                "stop_price": float(d["stop_price"]) if d.get("stop_price") else None,
                "target_price": float(d["target_price"]) if d.get("target_price") else None,
                "thesis": d.get("thesis", ""),
                "thesis_tags": d.get("thesis_tags", []),
                "market_regime": regime.get("regime", "unknown"),
            })
            recorded += 1
        except (KeyError, ValueError, TypeError):
            continue

    return {"recorded": recorded, "timestamp": context["timestamp"]}


if __name__ == "__main__":
    print(json.dumps(run_shadow_session(), indent=2))
