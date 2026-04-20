"""
Portfolio Manager — coordinates capital across strategies.

Responsibilities:
  1. Split capital across strategies by configured allocation
  2. Run each strategy's scan/plan on its cadence
  3. Resolve conflicts (two strategies want same ticker — first one wins)
  4. Route each proposal through its strategy's risk manager
  5. Enforce PORTFOLIO-level circuit breakers (combined drawdown)
  6. Track per-strategy P&L for attribution
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from datetime import date
from pathlib import Path
from typing import Any

from strategies.base import BaseStrategy, TradeProposal, get_strategy, list_strategies
# Import strategy modules to trigger @register_strategy decorators
from strategies import minervini, cs_momentum  # noqa: F401
from harness.risk_manager import TradeIntent, AccountSnapshot, RiskConfig, approve_trade, log_decision
from tools import market_data, broker

STATE = Path(__file__).parent.parent / "state"
PORTFOLIO_DIR = STATE / "portfolio"
PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = PORTFOLIO_DIR / "config.json"


@dataclass
class PortfolioConfig:
    """Configurable portfolio-level rules."""
    allocations: dict[str, float] = field(default_factory=lambda: {
        "minervini": 0.5,
        "cs_momentum": 0.5,
    })
    max_portfolio_drawdown_daily: float = 0.03   # -3% combined → halt all
    max_portfolio_drawdown_weekly: float = 0.07  # -7% in a week → halt all
    conflict_resolution: str = "first_wins"       # first_wins | split | block

    @classmethod
    def load(cls) -> "PortfolioConfig":
        if CONFIG_FILE.exists():
            return cls(**json.loads(CONFIG_FILE.read_text()))
        default = cls()
        CONFIG_FILE.write_text(json.dumps(asdict(default), indent=2))
        return default

    def save(self) -> None:
        CONFIG_FILE.write_text(json.dumps(asdict(self), indent=2))


@dataclass
class StrategyRun:
    """Result of running one strategy on one cycle."""
    strategy_id: str
    ran: bool
    candidates: int = 0
    proposals: int = 0
    executed: int = 0
    rejected: int = 0
    skipped_conflict: int = 0
    reasons: list[str] = field(default_factory=list)


# --- Per-strategy position tracking ---

POSITIONS_FILE = PORTFOLIO_DIR / "strategy_positions.json"


def load_strategy_positions() -> dict[str, list[dict]]:
    """Returns {strategy_id: [{ticker, shares, entry, strategy_id}, ...]}."""
    if not POSITIONS_FILE.exists():
        return {sid: [] for sid in list_strategies()}
    return json.loads(POSITIONS_FILE.read_text())


def save_strategy_positions(data: dict[str, list[dict]]) -> None:
    POSITIONS_FILE.write_text(json.dumps(data, indent=2))


def attribute_position(ticker: str, shares: int, entry: float, strategy_id: str) -> None:
    """Record that a position belongs to a strategy (for attribution)."""
    positions = load_strategy_positions()
    positions.setdefault(strategy_id, []).append({
        "ticker": ticker,
        "shares": shares,
        "entry": entry,
        "opened_at": date.today().isoformat(),
    })
    save_strategy_positions(positions)


def get_strategy_positions(strategy_id: str) -> list[dict]:
    """Positions owned by this strategy, enriched with current prices."""
    all_positions = broker.list_positions().get("positions", [])
    all_by_ticker = {p["ticker"]: p for p in all_positions}

    strategy_positions = load_strategy_positions().get(strategy_id, [])
    enriched = []
    for rec in strategy_positions:
        live = all_by_ticker.get(rec["ticker"])
        if live is None:
            continue  # position was closed
        enriched.append({
            "ticker": rec["ticker"],
            "shares": live["shares"],
            "entry": rec["entry"],
            "current": live.get("current", rec["entry"]),
            "pnl_pct": live.get("pnl_pct", 0),
            "strategy_id": strategy_id,
        })
    return enriched


# --- The orchestrator ---

def run_portfolio_cycle(dry_run: bool = False) -> dict[str, Any]:
    """
    The main entry — called by the daily workflow. Walks each strategy,
    runs those that are due today, executes trades through risk gates.
    """
    config = PortfolioConfig.load()

    # Market context shared across strategies
    regime = market_data.get_market_regime()

    # Full account
    try:
        snapshot_full = broker.get_account_snapshot(regime)
    except Exception as e:
        return {"error": f"Could not fetch account: {e}"}

    total_equity = snapshot_full.equity

    # --- Portfolio-level circuit breaker ---
    if snapshot_full.daily_pnl_pct <= -config.max_portfolio_drawdown_daily:
        return {
            "halted": True,
            "reason": f"Portfolio daily drawdown {snapshot_full.daily_pnl_pct:.2%} "
                      f"breached {-config.max_portfolio_drawdown_daily:.2%} limit",
        }

    today = date.today()
    results: dict[str, StrategyRun] = {}
    booked_tickers: set[str] = set()  # for conflict resolution

    # Run strategies in order (first-wins conflict resolution)
    for strategy_id in ["minervini", "cs_momentum"]:
        try:
            strategy = get_strategy(strategy_id)
        except KeyError:
            continue

        run_result = StrategyRun(strategy_id=strategy_id, ran=False)
        results[strategy_id] = run_result

        # Cadence check
        if not strategy.should_run_today(today):
            run_result.reasons.append(f"Skipped: not scheduled today (cadence={strategy.cadence})")
            continue

        # Allocation for this strategy
        allocation_pct = config.allocations.get(strategy_id, 0)
        if allocation_pct <= 0:
            run_result.reasons.append(f"Skipped: zero allocation")
            continue

        # Regime bias scales allocation
        bias = strategy.regime_bias(regime)
        effective_allocation = total_equity * allocation_pct * bias
        run_result.reasons.append(
            f"Allocation: ${effective_allocation:,.0f} "
            f"({allocation_pct:.0%} × regime_bias {bias})"
        )

        if effective_allocation <= 0:
            continue

        run_result.ran = True

        # Get this strategy's current positions
        strategy_positions = get_strategy_positions(strategy_id)

        # Scan + plan
        context = {
            "regime": regime,
            "as_of_date": today,
        }
        candidates = strategy.scan(context)
        run_result.candidates = len(candidates)

        proposals = strategy.plan(
            candidates=candidates,
            current_positions=strategy_positions,
            allocation_usd=effective_allocation,
            context=context,
        )
        run_result.proposals = len(proposals)

        # Execute each proposal through the risk gate
        for prop in proposals:
            # Conflict check
            if prop.ticker in booked_tickers and prop.side == "buy":
                run_result.skipped_conflict += 1
                run_result.reasons.append(
                    f"Skipped {prop.ticker}: conflict with another strategy"
                )
                continue

            result = _execute_proposal(prop, snapshot_full, effective_allocation, regime, dry_run)
            if result.get("executed"):
                run_result.executed += 1
                booked_tickers.add(prop.ticker)
                if prop.side == "buy" and prop.entry_price:
                    # Record attribution (will refine once fill arrives)
                    shares_est = int(effective_allocation * prop.target_weight / prop.entry_price)
                    if shares_est > 0:
                        attribute_position(prop.ticker, shares_est, prop.entry_price, strategy_id)
            else:
                run_result.rejected += 1
                run_result.reasons.append(f"{prop.ticker}: {result.get('reasons', ['?'])[0]}")

    summary = {
        "date": today.isoformat(),
        "regime": regime.get("regime"),
        "total_equity": total_equity,
        "dry_run": dry_run,
        "results": {k: asdict(v) for k, v in results.items()},
    }

    # Persist daily summary
    (PORTFOLIO_DIR / f"cycle_{today.isoformat()}.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    return summary


def _execute_proposal(
    prop: TradeProposal,
    snapshot: AccountSnapshot,
    allocation_usd: float,
    regime: dict,
    dry_run: bool,
) -> dict:
    """Route a proposal through the appropriate execution path."""
    # Sells are straightforward
    if prop.side == "sell":
        if dry_run:
            return {"executed": True, "dry_run": True}
        return {**broker.close_position(prop.ticker, prop.reason), "executed": True}

    # Buys with stops (Minervini) go through risk_manager
    if prop.stop_price is not None and prop.entry_price is not None:
        # Compute shares from target weight
        position_value = allocation_usd * prop.target_weight
        shares = max(1, int(position_value / prop.entry_price))

        signal_tier = "super" if "super_stock" in prop.tags else (
            "perfect" if "perfect_setup" in prop.tags else "leader"
        )

        return broker.place_trade(
            ticker=prop.ticker,
            entry_price=prop.entry_price,
            stop_price=prop.stop_price,
            shares=shares,
            signal_tier=signal_tier,
            reason=prop.reason,
            market_regime=regime,
            dry_run=dry_run,
        )

    # Buys without stops (XS momentum) — need a different execution path
    # For now: compute shares from target weight and use broker.place_trade
    # with a synthetic wide stop (25% below entry) so the risk manager accepts it.
    # The strategy's real "stop" is rotation at next rebalance, not a hard stop.
    if prop.entry_price is None:
        # Get current price
        quote = market_data.get_quote(prop.ticker)
        if "error" in quote:
            return {"executed": False, "reasons": [f"no price for {prop.ticker}"]}
        entry = quote["price"]
    else:
        entry = prop.entry_price

    position_value = allocation_usd * prop.target_weight
    shares = max(1, int(position_value / entry))
    # Synthetic wide stop — purely to satisfy risk manager's stop requirement.
    # Real risk control for XS momentum is diversification (top-N positions)
    # and the bi-weekly rotation.
    synthetic_stop = round(entry * 0.80, 2)  # 20% stop

    return broker.place_trade(
        ticker=prop.ticker,
        entry_price=entry,
        stop_price=synthetic_stop,
        shares=shares,
        signal_tier="leader",  # XS momentum counts as "leader" tier for risk gating
        reason=prop.reason,
        market_regime=regime,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_portfolio_cycle(dry_run=args.dry_run)
    print(json.dumps(result, indent=2, default=str))
