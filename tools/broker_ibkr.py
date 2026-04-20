"""
Interactive Brokers implementation via ib_insync.

Requires IB Gateway or TWS running and reachable at IBKR_HOST:IBKR_PORT.
Paper port default: 7497 (TWS paper) or 4002 (Gateway paper).

Same interface as broker_alpaca, routes through risk_manager before execution.
"""
from __future__ import annotations
import os
from datetime import date
from pathlib import Path
import json
from typing import Any

try:
    from ib_insync import IB, Stock, MarketOrder, StopOrder
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False

from harness.risk_manager import (
    TradeIntent, AccountSnapshot, RiskConfig,
    approve_trade, log_decision,
)

STATE_DIR = Path(__file__).parent.parent / "state"


def _connect() -> "IB":
    if not IB_AVAILABLE:
        raise RuntimeError("ib_insync not installed. Run: pip install ib_insync")
    host = os.getenv("IBKR_HOST", "127.0.0.1")
    port = int(os.getenv("IBKR_PORT", "7497"))  # 7497 = TWS paper
    client_id = int(os.getenv("IBKR_CLIENT_ID", "1"))
    ib = IB()
    ib.connect(host, port, clientId=client_id, readonly=False, timeout=15)
    return ib


def _stock(symbol: str) -> "Stock":
    return Stock(symbol, "SMART", "USD")


def get_account_snapshot(market_regime: dict) -> AccountSnapshot:
    """Assemble the snapshot the risk manager needs."""
    ib = _connect()
    try:
        summary = {t.tag: t.value for t in ib.accountSummary()}
        positions = ib.positions()

        equity = float(summary.get("NetLiquidation", 0))
        cash = float(summary.get("TotalCashValue", 0))

        # Daily P&L — IBKR reports as RealizedPnL + UnrealizedPnL change
        # For simplicity use DailyPnL tag if present
        daily_pnl = float(summary.get("DailyPnL", 0) or 0)
        daily_pnl_pct = daily_pnl / equity if equity else 0.0

        # Today's fills — from executions
        executions = ib.reqExecutions()
        today = date.today()
        trades_today = sum(
            1 for e in executions
            if e.time and e.time.date() == today
        )
        new_positions_today = sum(
            1 for e in executions
            if e.time and e.time.date() == today and e.side == "BOT"
        )

        open_positions = [
            {
                "ticker": p.contract.symbol,
                "shares": int(p.position),
                "entry": float(p.avgCost),
                "current": float(p.avgCost),  # Updated via market data req
                "pnl_pct": 0.0,  # Filled in after marketPrice request if needed
            }
            for p in positions if p.position != 0
        ]

        return AccountSnapshot(
            equity=equity, cash=cash,
            open_positions=open_positions,
            daily_pnl_pct=daily_pnl_pct,
            trades_today=trades_today,
            new_positions_today=new_positions_today,
            spy_above_ma200=market_regime.get("spy_above_ma200", True),
            spy_above_ma50=market_regime.get("spy_above_ma50", True),
            spy_high_vol_breakdown=market_regime.get("spy_high_vol_breakdown", False),
        )
    finally:
        ib.disconnect()


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
    """Risk-gated buy with attached stop. Bracket order via IBKR parent-child pattern."""
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

    ib = _connect()
    try:
        contract = _stock(ticker)
        ib.qualifyContracts(contract)

        # Parent market buy
        parent = MarketOrder("BUY", shares)
        parent.transmit = False  # Don't send until child is attached

        parent_trade = ib.placeOrder(contract, parent)

        # Child stop loss
        stop = StopOrder("SELL", shares, round(stop_price, 2))
        stop.parentId = parent_trade.order.orderId
        stop.transmit = True  # Send both now

        stop_trade = ib.placeOrder(contract, stop)

        ib.sleep(1)  # Let orders register

        return {
            "executed": True,
            "reasons": decision.reasons,
            "buy_order_id": parent_trade.order.orderId,
            "stop_order_id": stop_trade.order.orderId,
        }
    finally:
        ib.disconnect()


def close_position(ticker: str, reason: str) -> dict[str, Any]:
    """Market sell to flatten position."""
    ib = _connect()
    try:
        positions = [p for p in ib.positions() if p.contract.symbol == ticker and p.position != 0]
        if not positions:
            return {"closed": False, "error": f"No open position in {ticker}"}

        pos = positions[0]
        side = "SELL" if pos.position > 0 else "BUY"
        shares = abs(int(pos.position))

        contract = _stock(ticker)
        ib.qualifyContracts(contract)

        # Cancel any existing stop orders first
        for order in ib.openOrders():
            if order.contract.symbol == ticker and order.orderType == "STP":
                ib.cancelOrder(order)

        order = MarketOrder(side, shares)
        trade = ib.placeOrder(contract, order)
        ib.sleep(1)

        # Audit
        with (STATE_DIR / "exits.jsonl").open("a") as f:
            f.write(json.dumps({
                "date": date.today().isoformat(),
                "ticker": ticker,
                "reason": reason,
                "broker": "ibkr",
                "order_id": trade.order.orderId,
            }) + "\n")

        return {"closed": True, "ticker": ticker, "reason": reason, "order_id": trade.order.orderId}
    except Exception as e:
        return {"closed": False, "error": str(e)}
    finally:
        ib.disconnect()


def list_positions() -> dict[str, Any]:
    ib = _connect()
    try:
        positions = [p for p in ib.positions() if p.position != 0]
        # Fetch current prices
        contracts = [_stock(p.contract.symbol) for p in positions]
        ib.qualifyContracts(*contracts)
        tickers_data = {
            c.symbol: ib.reqMktData(c, "", False, False)
            for c in contracts
        }
        ib.sleep(2)

        out = []
        for p in positions:
            sym = p.contract.symbol
            market = tickers_data.get(sym)
            current = float(market.last) if market and market.last else float(p.avgCost)
            entry = float(p.avgCost)
            pnl_pct = (current - entry) / entry * 100 if entry else 0
            out.append({
                "ticker": sym,
                "shares": int(p.position),
                "entry": round(entry, 2),
                "current": round(current, 2),
                "pnl_pct": round(pnl_pct, 2),
                "pnl_usd": round((current - entry) * p.position, 2),
            })
        return {"positions": out}
    finally:
        ib.disconnect()
