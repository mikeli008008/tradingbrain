"""
After-hours & pre-market news sentinel.

Runs twice in the overnight window:
  - 7pm ET  (23:00 UTC)  → after earnings releases hit
  - 8am ET  (12:00 UTC)  → pre-market, before open

For each open position AND each watchlist ticker:
  1. Fetch last 18h of news
  2. Score severity (keyword match + pre-market price)
  3. If any ticker scores HIGH → trigger the emergency agent run

The emergency agent can:
  - Cancel a resting stop + place a market order at open (if catastrophic)
  - Tighten stop (if mildly negative)
  - Do nothing (if noise or positive)

This closes the overnight gap risk hole without trying to trade pre-market
itself (retail pre-market execution is dangerous — spreads are huge).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Literal, Any
import json
import os

import yfinance as yf
from tools import news_research, broker

ALERTS_DIR = Path(__file__).parent.parent / "state" / "overnight_alerts"
ALERTS_DIR.mkdir(parents=True, exist_ok=True)


# Severity keyword map. Tuned conservative — false positives cost
# nothing but a closer look; false negatives cost money.
CATASTROPHIC_KEYWORDS = [
    "sec investigation", "fraud", "bankruptcy", "going concern",
    "delisting", "halt", "restatement", "criminal", "indicted",
    "ceo resigns", "cfo resigns", "accounting irregularit",
    "trial failed", "phase 3 fail", "fda rejection", "fda reject",
    "clinical hold", "data fabrication", "class action", "short report",
    "hindenburg", "muddy waters", "citron",
]

NEGATIVE_KEYWORDS = [
    "misses earnings", "misses revenue", "guidance cut", "guides below",
    "lowers outlook", "downgrade", "target cut", "lawsuit",
    "layoffs", "restructuring", "warning", "profit warning",
    "lost contract", "customer loss",
]

POSITIVE_KEYWORDS = [
    "beats", "beat estimates", "raises guidance", "raised guidance",
    "upgrade", "acquisition", "buyout", "approval", "fda approval",
    "contract win", "partnership", "breakthrough",
]


@dataclass
class OvernightAlert:
    ticker: str
    severity: Literal["catastrophic", "negative", "positive", "noise"]
    score: float
    headlines: list[dict]
    premarket_pct: float | None
    recommended_action: str
    timestamp: str


def _score_headlines(headlines: list[dict]) -> tuple[str, float, list[str]]:
    """Keyword-match severity score. Returns (severity, score, matches)."""
    if not headlines:
        return "noise", 0.0, []

    text_blob = " ".join(h.get("headline", "").lower() for h in headlines)
    matches = []

    catastrophic_hits = sum(1 for kw in CATASTROPHIC_KEYWORDS if kw in text_blob)
    negative_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_blob)
    positive_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_blob)

    if catastrophic_hits > 0:
        matches.extend([kw for kw in CATASTROPHIC_KEYWORDS if kw in text_blob])
        return "catastrophic", 1.0, matches
    if negative_hits >= 2:
        matches.extend([kw for kw in NEGATIVE_KEYWORDS if kw in text_blob])
        return "negative", 0.7, matches
    if negative_hits == 1:
        matches.extend([kw for kw in NEGATIVE_KEYWORDS if kw in text_blob])
        return "negative", 0.4, matches
    if positive_hits >= 1:
        matches.extend([kw for kw in POSITIVE_KEYWORDS if kw in text_blob])
        return "positive", 0.5, matches
    return "noise", 0.1, []


def _premarket_move(ticker: str) -> float | None:
    """Pre-market % change vs yesterday's close. Returns None if unavailable."""
    try:
        t = yf.Ticker(ticker)
        # 1m intervals for last 2 days captures pre-market quotes
        hist = t.history(period="2d", interval="1m", prepost=True)
        if hist.empty or len(hist) < 2:
            return None
        # Find last regular-session close
        info = t.info
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        if not prev_close:
            return None
        last_price = float(hist["Close"].iloc[-1])
        return round((last_price - prev_close) / prev_close * 100, 2)
    except Exception:
        return None


def _recommend_action(severity: str, premarket_pct: float | None, position: dict | None) -> str:
    """Concrete action advice for the agent / user."""
    if severity == "catastrophic":
        if position:
            return (
                "IMMEDIATE: Cancel resting stop, prepare market sell at open. "
                "Expect slippage past stop price. Do not wait for stop to trigger."
            )
        return "AVOID: Do not enter this name. Remove from watchlist."

    if severity == "negative":
        if position and premarket_pct is not None and premarket_pct < -3:
            return (
                f"Position at risk: pre-market {premarket_pct}%. Cancel stop, "
                "plan market exit at open (stop will slip)."
            )
        if position:
            return "Monitor: negative news but pre-market contained. Keep stop as-is."
        return "Skip today: wait for price action to digest news."

    if severity == "positive":
        if not position:
            return "Do not chase: wait for pullback to pivot or first consolidation."
        return "Position benefits: consider trailing stop up after strong opening."

    return "No action: routine noise."


def scan_tickers(tickers: list[str], position_map: dict[str, dict]) -> list[OvernightAlert]:
    """Scan a list of tickers, return only non-noise alerts."""
    alerts = []
    for tkr in tickers:
        news = news_research.research_ticker_news(tkr, days=1)
        headlines = news.get("headlines", [])
        severity, score, _ = _score_headlines(headlines)

        if severity == "noise":
            continue

        premarket = _premarket_move(tkr)
        position = position_map.get(tkr)
        action = _recommend_action(severity, premarket, position)

        alerts.append(OvernightAlert(
            ticker=tkr, severity=severity, score=score,
            headlines=headlines[:5],
            premarket_pct=premarket,
            recommended_action=action,
            timestamp=datetime.utcnow().isoformat(timespec="minutes"),
        ))
    return alerts


def run_sentinel() -> dict[str, Any]:
    """
    Main entry point. Scans open positions + watchlist.
    Writes alerts to disk. If any catastrophic, triggers emergency agent run.
    """
    # Open positions
    try:
        positions = broker.list_positions().get("positions", [])
    except Exception as e:
        positions = []
        print(f"Could not fetch positions: {e}")

    position_map = {p["ticker"]: p for p in positions}
    position_tickers = list(position_map.keys())

    # Watchlist — read from existing watchlist.txt convention of the screener
    # (skipped here for brevity; can add if you mirror that file locally)
    watchlist_tickers: list[str] = []
    watchlist_file = Path(__file__).parent.parent / "state" / "watchlist.txt"
    if watchlist_file.exists():
        watchlist_tickers = [
            line.split("|")[0].strip()
            for line in watchlist_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    all_tickers = list(set(position_tickers + watchlist_tickers))
    if not all_tickers:
        return {"scanned": 0, "alerts": []}

    alerts = scan_tickers(all_tickers, position_map)

    # Persist
    today = date.today().isoformat()
    path = ALERTS_DIR / f"{today}.jsonl"
    with path.open("a") as f:
        for a in alerts:
            f.write(json.dumps(asdict(a), default=str) + "\n")

    # Summarize for output
    by_severity = {"catastrophic": [], "negative": [], "positive": []}
    for a in alerts:
        if a.severity in by_severity:
            by_severity[a.severity].append({
                "ticker": a.ticker,
                "premarket_pct": a.premarket_pct,
                "action": a.recommended_action,
            })

    result = {
        "scanned": len(all_tickers),
        "alerts_total": len(alerts),
        "by_severity": by_severity,
        "emergency_triggered": len(by_severity["catastrophic"]) > 0,
    }

    # If any catastrophic, write a trigger file the emergency workflow picks up
    if result["emergency_triggered"]:
        trigger_path = Path(__file__).parent.parent / "state" / "EMERGENCY_TRIGGER"
        trigger_path.write_text(json.dumps(result, indent=2, default=str))

    return result


if __name__ == "__main__":
    print(json.dumps(run_sentinel(), indent=2, default=str))
