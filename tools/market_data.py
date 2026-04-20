"""
Market data tools. The agent calls these via the Claude Agent SDK.

Design rule: tool outputs are LLM-optimized — structured, small, and
include the next-action hint when something fails (sensor pattern).
"""
from __future__ import annotations
import json
import os
from datetime import datetime, date
from pathlib import Path
from typing import Any

import yfinance as yf
import pandas as pd
import requests

STATE_DIR = Path(__file__).parent.parent / "state"


# --- Minervini screener integration ---

def get_minervini_scan(screener_repo: str | None = None) -> dict[str, Any]:
    """
    Fetch the latest scan from mikeli008008/minervini-screener.

    Tries multiple paths in order:
      1. SCREENER_URL env var (explicit override)
      2. screener_repo argument
      3. Default locations: latest.json, reports/latest.json
      4. Local file at state/screener_latest.json (for testing)

    Returns tiered candidates: super, perfect, leader, watchlist.
    Returns empty tiers (not an error) if no source found — strategies
    handle empty input gracefully.
    """
    candidates_urls = []
    if screener_repo:
        candidates_urls.append(screener_repo)
    if env_url := os.getenv("SCREENER_URL"):
        candidates_urls.append(env_url)
    candidates_urls.extend([
        "https://mikeli008008.github.io/minervini-screener/latest.json",
        "https://mikeli008008.github.io/minervini-screener/reports/latest.json",
    ])

    data = None
    last_error = None
    for url in candidates_urls:
        try:
            r = requests.get(url, timeout=15)
            if r.ok:
                data = r.json()
                break
        except Exception as e:
            last_error = str(e)

    # Local fallback for offline testing
    if data is None:
        local = STATE_DIR / "screener_latest.json"
        if local.exists():
            data = json.loads(local.read_text())

    if data is None:
        return {
            "scan_date": None,
            "counts": {"super": 0, "perfect": 0, "leader": 0, "watchlist": 0, "other": 0},
            "super": [], "perfect": [], "leader": [], "watchlist": [],
            "warning": (
                "No screener output reachable. Set SCREENER_URL secret or place a "
                "screener_latest.json in state/. Strategy will skip this cycle."
            ),
            "tried_urls": candidates_urls,
            "last_error": last_error,
        }

    # Normalize the output into tiers
    rows = data.get("stocks", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        rows = []

    # Build tier lookup from watchlist_status (contains current_signal)
    tier_lookup = {}
    for w in data.get("watchlist_status", []):
        t = w.get("ticker")
        s = (w.get("current_signal") or "").lower()
        if t and s:
            tier_lookup[t] = s

    tiers = {"super": [], "perfect": [], "leader": [], "watchlist": [], "other": []}
    for row in rows:
        ticker = row.get("ticker", "")
        if ticker in tier_lookup:
            tier = tier_lookup[ticker]
        elif row.get("super_stock_candidate"):
            tier = "super"
        elif row.get("all_8_passed") and row.get("vcp", {}).get("near_pivot"):
            tier = "perfect"
        elif row.get("all_8_passed"):
            tier = "leader"
        else:
            tier = "other"
        if "super" in tier:
            tiers["super"].append(row)
        elif "perfect" in tier:
            tiers["perfect"].append(row)
        elif "leader" in tier:
            tiers["leader"].append(row)
        elif "watch" in tier:
            tiers["watchlist"].append(row)
        else:
            tiers["other"].append(row)

    return {
        "scan_date": data.get("date") if isinstance(data, dict) else None,
        "counts": {k: len(v) for k, v in tiers.items()},
        "super": tiers["super"][:10],
        "perfect": tiers["perfect"][:15],
        "leader": tiers["leader"][:20],
        "watchlist": tiers["watchlist"][:20],
    }


# --- SPY regime (for the harness gate) ---

def get_market_regime() -> dict[str, Any]:
    """Compute SPY regime for the risk manager's regime gate."""
    spy = yf.Ticker("SPY")
    hist = spy.history(period="1y")
    if hist.empty:
        return {"error": "Could not fetch SPY data"}

    close = hist["Close"]
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    last = close.iloc[-1]
    prev = close.iloc[-2]

    # "High vol breakdown" = today down >1% on above-average volume
    vol = hist["Volume"]
    avg_vol = vol.rolling(50).mean().iloc[-1]
    today_vol = vol.iloc[-1]
    daily_pct = (last - prev) / prev
    high_vol_breakdown = daily_pct < -0.01 and today_vol > avg_vol * 1.2

    return {
        "spy_price": round(float(last), 2),
        "spy_above_ma50": bool(last > ma50),
        "spy_above_ma200": bool(last > ma200),
        "spy_high_vol_breakdown": bool(high_vol_breakdown),
        "ma50": round(float(ma50), 2),
        "ma200": round(float(ma200), 2),
        "regime": (
            "strong" if last > ma50 > ma200 else
            "weak" if last < ma200 else
            "transitional"
        ),
    }


# --- Per-ticker quote for entry/stop validation ---

def get_quote(ticker: str) -> dict[str, Any]:
    """Current price + recent context for sizing and stop placement."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="3mo")
        if hist.empty:
            return {"error": f"No data for {ticker}", "hint": "Check ticker symbol."}

        last = float(hist["Close"].iloc[-1])
        high_52w = float(t.history(period="1y")["High"].max())
        # ATR for stop placement hint
        h, l, c = hist["High"], hist["Low"], hist["Close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])

        return {
            "ticker": ticker,
            "price": round(last, 2),
            "high_52w": round(high_52w, 2),
            "pct_from_high": round((last - high_52w) / high_52w * 100, 2),
            "atr_14": round(atr14, 2),
            "suggested_stop_dist_pct": round(atr14 / last * 1.5 * 100, 2),
            "hint": "Stop ~1.5× ATR below entry is a reasonable starting point.",
        }
    except Exception as e:
        return {"error": str(e)}
