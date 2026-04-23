"""
Broker adapter — Alpaca paper/live. Isolated so you can swap to IBKR later.

The agent never calls Alpaca directly; it calls these functions, which
route through the risk_manager gate.
"""
from __future__ import annotations
import os
from typing import Any, Literal
from dataclasses import asdict
from pathlib import Path
import json
from datetime import date

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest, GetOrdersRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from harness.risk_manager import (
    TradeIntent, AccountSnapshot, RiskConfig,
    approve_trade, log_decision,
)

STATE_DIR = Path(__file__).parent.parent / "state"


def _client() -> "TradingClient":
    if not ALPACA_AVAILABLE:
        raise RuntimeError("alpaca-py not installed")
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
    return TradingClient(key, secret, paper=paper)


def get_account_snapshot(market_regime: dict) -> AccountSnapshot:
    """Assemble the snapshot the risk manager needs."""
    c = _client()
    acct = c.get_account()
    positions = c.get_all_positions()

    # Count today's activity
    today = date.today()
    orders = c.get_orders(filter=GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        after=today.isoformat(),
    ))
    filled_today = [o for o in orders if o.filled_at and o.filled_at.date() == today]
    new_positions_today = len([o for o in filled_today if o.side == OrderSide.BUY])

    equity = float(acct.equity)
    last_equity = float(acct.last_equity)
    daily_pnl_pct = (equity - last_equity) / last_equity if last_equity else 0.0

    return AccountSnapshot(
        equity=equity,
        cash=float(acct.cash),
        open_positions=[
            {
                "ticker": p.symbol,
                "shares": int(p.qty),
                "entry": float(p.avg_entry_price),
                "current": float(p.current_price),
                "pnl_pct": float(p.unrealized_plpc),
            }
            for p in positions
        ],
        daily_pnl_pct=daily_pnl_pct,
        trades_today=len(filled_today),
        new_positions_today=new_positions_today,
        spy_above_ma200=market_regime.get("spy_above_ma200", True),
        spy_above_ma50=market_regime.get("spy_above_ma50", True),
        spy_high_vol_breakdown=market_regime.get("spy_high_vol_breakdown", False),
    )


def place_trade(
    ticker: str,
    entry_price: float,
    stop_price: float,
    shares: int,
    signal_tier: str,
    reason: str,
    market_regime: dict,
    config: RiskConfig | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Execute a trade. HARD GATE: goes through risk_manager first.
    Places a market buy + an OCO stop (automatic stop loss).
    """
    config = config or RiskConfig()
    intent = TradeIntent(
        ticker=ticker, side="buy",
        entry_price=entry_price, stop_price=stop_price,
        shares=shares, signal_tier=signal_tier, reason=reason,
    )
    account = get_account_snapshot(market_regime)
    decision = approve_trade(intent, account, config)
    log_decision(intent, decision)

    if not decision.approved:
        return {
            "executed": False,
            "reasons": decision.reasons,
            "adjusted": decision.adjusted,
        }

    if dry_run:
        return {"executed": False, "dry_run": True, "reasons": decision.reasons}

    c = _client()
    # Market buy only — stop is placed separately after fill (see place_pending_stops)
    buy = c.submit_order(MarketOrderRequest(
        symbol=ticker, qty=shares,
        side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
    ))
    # Log the pending stop for placement after fill
    pending = STATE_DIR / "pending_stops.jsonl"
    with pending.open("a") as f:
        import json as _json
        f.write(_json.dumps({
            "ticker": ticker, "shares": shares,
            "stop_price": round(stop_price, 2),
            "buy_order_id": str(buy.id),
            "date": date.today().isoformat(),
        }) + "\n")
    return {
        "executed": True,
        "reasons": decision.reasons,
        "buy_order_id": str(buy.id),
        "stop_order_id": "pending_after_fill",
    }


def close_position(ticker: str, reason: str) -> dict[str, Any]:
    """Exit a position (stop hit manually, or regime change)."""
    c = _client()
    try:
        c.close_position(ticker)
        # Audit
        with (STATE_DIR / "exits.jsonl").open("a") as f:
            f.write(json.dumps({
                "date": date.today().isoformat(),
                "ticker": ticker,
                "reason": reason,
            }) + "\n")
        return {"closed": True, "ticker": ticker, "reason": reason}
    except Exception as e:
        return {"closed": False, "error": str(e)}


def list_positions() -> dict[str, Any]:
    c = _client()
    positions = c.get_all_positions()
    return {
        "positions": [
            {
                "ticker": p.symbol,
                "shares": int(p.qty),
                "entry": float(p.avg_entry_price),
                "current": float(p.current_price),
                "pnl_pct": round(float(p.unrealized_plpc) * 100, 2),
                "pnl_usd": round(float(p.unrealized_pl), 2),
            }
            for p in positions
        ]
    }


def place_pending_stops() -> dict:
    """Call this after market open to attach stops to filled buy orders."""
    pending_path = STATE_DIR / "pending_stops.jsonl"
    if not pending_path.exists():
        return {"placed": 0}

    c = _client()
    positions = {p.symbol: p for p in c.get_all_positions()}

    # Get all open stop orders to avoid duplicate/wash trade error
    open_orders = c.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
    existing_stops = {
        o.symbol for o in open_orders
        if o.type is not None and str(o.type).lower() == "stop" and o.side == OrderSide.SELL
    }

    lines = pending_path.read_text().splitlines()
    placed = 0
    remaining = []

    for line in lines:
        if not line.strip():
            continue
        import json as _json
        rec = _json.loads(line)
        ticker = rec["ticker"]

        if ticker not in positions:
            remaining.append(line)  # not filled yet, keep pending
            continue

        if ticker in existing_stops:
            print(f"Stop already exists for {ticker}, skipping.")
            # Don't keep in pending — it's handled
            continue

        try:
            c.submit_order(StopOrderRequest(
                symbol=ticker,
                qty=rec["shares"],
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                stop_price=rec["stop_price"],
            ))
            placed += 1
        except Exception as e:
            print(f"Stop failed for {ticker}: {e}")
            remaining.append(line)

    pending_path.write_text("\n".join(remaining) + "\n" if remaining else "")
    return {"placed": placed, "still_pending": len(remaining)}
    pending_path.write_text("\n".join(remaining) + "\n" if remaining else "")
    return {"placed": placed, "still_pending": len(remaining)}
