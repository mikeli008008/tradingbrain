"""
Broker abstraction. The agent calls these functions; the BROKER env var
selects Alpaca or IBKR under the hood.

IBKR integration uses ib_insync, which requires TWS or IB Gateway running
somewhere the GitHub Actions runner can reach. For cloud deployment, you
typically run IB Gateway on a small VM and expose it via a secure tunnel.
See the README for the deployment pattern.
"""
from __future__ import annotations
import os
from typing import Any

BROKER = os.getenv("BROKER", "alpaca").lower()


def get_account_snapshot(market_regime: dict):
    if BROKER == "ibkr":
        from tools.broker_ibkr import get_account_snapshot as impl
    else:
        from tools.broker_alpaca import get_account_snapshot as impl
    return impl(market_regime)


def place_trade(**kwargs) -> dict[str, Any]:
    if BROKER == "ibkr":
        from tools.broker_ibkr import place_trade as impl
    else:
        from tools.broker_alpaca import place_trade as impl
    return impl(**kwargs)


def close_position(ticker: str, reason: str) -> dict[str, Any]:
    if BROKER == "ibkr":
        from tools.broker_ibkr import close_position as impl
    else:
        from tools.broker_alpaca import close_position as impl
    return impl(ticker, reason)


def list_positions() -> dict[str, Any]:
    if BROKER == "ibkr":
        from tools.broker_ibkr import list_positions as impl
    else:
        from tools.broker_alpaca import list_positions as impl
    return impl()
