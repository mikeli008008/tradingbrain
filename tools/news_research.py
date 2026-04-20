"""
News research layer — the '每天自我research news' part.

Two approaches combined:
1. Per-ticker catalyst check (earnings, upgrades, filings)
2. Macro/sector sentiment for regime-aware decisions

We return compressed, LLM-friendly output. The agent synthesizes,
not the tool.
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta
from typing import Any
import requests


def research_ticker_news(ticker: str, days: int = 3) -> dict[str, Any]:
    """
    Fetch recent news for a ticker. Uses Finnhub (free tier) if API key
    present, falls back to Yahoo Finance news otherwise.

    Returns headlines only — the agent reads and weighs them.
    """
    key = os.getenv("FINNHUB_API_KEY")
    if key:
        return _finnhub_news(ticker, days, key)
    return _yfinance_news(ticker)


def _finnhub_news(ticker: str, days: int, key: str) -> dict[str, Any]:
    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=days)
    url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={ticker}&from={from_date}&to={to_date}&token={key}"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        items = r.json() or []
    except Exception as e:
        return {"error": str(e), "ticker": ticker}

    # Compact to LLM-friendly form
    headlines = [
        {
            "date": datetime.fromtimestamp(it["datetime"]).strftime("%Y-%m-%d"),
            "headline": it["headline"],
            "source": it["source"],
            "url": it["url"],
        }
        for it in items[:15]
    ]
    return {"ticker": ticker, "headlines": headlines, "count": len(headlines)}


def _yfinance_news(ticker: str) -> dict[str, Any]:
    import yfinance as yf
    try:
        items = yf.Ticker(ticker).news or []
    except Exception as e:
        return {"error": str(e), "ticker": ticker}
    headlines = [
        {
            "date": datetime.fromtimestamp(it.get("providerPublishTime", 0)).strftime("%Y-%m-%d")
            if it.get("providerPublishTime") else "",
            "headline": it.get("title", ""),
            "source": it.get("publisher", ""),
            "url": it.get("link", ""),
        }
        for it in items[:15]
    ]
    return {"ticker": ticker, "headlines": headlines, "count": len(headlines)}


def research_macro() -> dict[str, Any]:
    """
    Daily macro scan — what the agent should know before trading.
    Returns compressed snapshot of Fed, rates, major indices, VIX.
    """
    import yfinance as yf

    tickers = {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "IWM": "Russell 2000 (small cap)",
        "^VIX": "VIX (fear index)",
        "TLT": "20y Treasuries",
        "DXY": "US Dollar",
    }
    snapshot = {}
    for sym, name in tickers.items():
        try:
            h = yf.Ticker(sym).history(period="5d")
            if h.empty:
                continue
            last = float(h["Close"].iloc[-1])
            prev = float(h["Close"].iloc[-2])
            chg = (last - prev) / prev * 100
            snapshot[sym] = {
                "name": name,
                "price": round(last, 2),
                "day_change_pct": round(chg, 2),
            }
        except Exception:
            pass
    return {"date": datetime.utcnow().isoformat(), "snapshot": snapshot}


def earnings_on_deck(tickers: list[str], days_ahead: int = 7) -> dict[str, Any]:
    """
    Flag tickers with earnings in the next N days — Minervini rule says
    avoid new entries right before earnings (event risk).
    """
    import yfinance as yf
    today = datetime.utcnow().date()
    cutoff = today + timedelta(days=days_ahead)
    flagged = []
    for t in tickers:
        try:
            cal = yf.Ticker(t).calendar
            if cal is None:
                continue
            # yfinance returns different shapes across versions
            earnings_date = None
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if isinstance(ed, list) and ed:
                    earnings_date = ed[0]
            if earnings_date and today <= earnings_date <= cutoff:
                flagged.append({"ticker": t, "earnings_date": str(earnings_date)})
        except Exception:
            pass
    return {"flagged": flagged, "window_days": days_ahead}
