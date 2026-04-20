"""
Risk Manager — Non-negotiable gates before any trade executes.

These enforce Minervini's core rules at the harness level so the agent
CANNOT bypass them by reasoning around them. Every trade passes through
approve_trade() and any rejection is final.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Literal
import json
from pathlib import Path

STATE_DIR = Path(__file__).parent.parent / "state"
STATE_DIR.mkdir(exist_ok=True)


@dataclass
class RiskConfig:
    # Minervini core rules
    max_risk_per_trade_pct: float = 0.0075    # 0.75% account risk per trade (new trader: 0.25-1%)
    max_open_positions: int = 5                # "同时持仓 ≤ 5 只"
    max_stop_distance_pct: float = 0.25        # Stop max 25% below entry (XS momentum uses wide synthetic stops)
    min_stop_distance_pct: float = 0.03        # Stop at least 3% (not too tight)

    # Daily circuit breakers
    max_daily_loss_pct: float = 0.02           # -2% day → halt all new entries
    max_daily_trades: int = 10                 # Spam guard
    max_new_positions_per_day: int = 3         # No FOMO piling in

    # Regime gates (SPY rules from README)
    halt_if_spy_below_ma200: bool = True       # Reduce to half (handled by sizing)
    stop_all_if_spy_below_ma50_high_vol: bool = True  # Full halt

    # Never average down
    allow_averaging_down: bool = False

    # Signal quality gate — only trade Minervini-grade setups
    min_signal_tier: Literal["super", "perfect", "leader", "watchlist"] = "perfect"


# --- Presets for different strategies ---

def minervini_config() -> RiskConfig:
    """Strict Minervini rules — concentrated, tight stops, breakout-quality only."""
    return RiskConfig(
        max_risk_per_trade_pct=0.0075,
        max_open_positions=5,
        max_stop_distance_pct=0.08,   # Minervini's 7-8% hard stop
        min_stop_distance_pct=0.03,
        max_new_positions_per_day=3,
        min_signal_tier="perfect",
    )


def cs_momentum_config() -> RiskConfig:
    """Looser rules for diversified rotation strategy."""
    return RiskConfig(
        max_risk_per_trade_pct=0.02,  # larger per-position tolerance (diversified)
        max_open_positions=15,         # top-10 + buffer
        max_stop_distance_pct=0.25,    # wide synthetic stops (real exit is rotation)
        min_stop_distance_pct=0.05,
        max_new_positions_per_day=5,   # bi-weekly rebalance can move several at once
        min_signal_tier="leader",      # XS momentum doesn't have super/perfect tier
    )


@dataclass
class TradeIntent:
    ticker: str
    side: Literal["buy", "sell"]
    entry_price: float
    stop_price: float
    shares: int
    signal_tier: str   # "super" | "perfect" | "leader" | "watchlist"
    reason: str        # Agent's plain-English justification
    score: float = 0.0


@dataclass
class AccountSnapshot:
    equity: float
    cash: float
    open_positions: list[dict]   # [{ticker, shares, entry, current, pnl_pct}, ...]
    daily_pnl_pct: float
    trades_today: int
    new_positions_today: int
    spy_above_ma200: bool
    spy_above_ma50: bool
    spy_high_vol_breakdown: bool


@dataclass
class RiskDecision:
    approved: bool
    reasons: list[str]       # Why (both approvals and rejections)
    adjusted: dict | None = None   # Suggested adjustments (e.g. smaller size)


TIER_RANK = {"watchlist": 0, "leader": 1, "perfect": 2, "super": 3}


def approve_trade(
    intent: TradeIntent,
    account: AccountSnapshot,
    config: RiskConfig = RiskConfig(),
) -> RiskDecision:
    """Hard gate. Returns approved=True only if ALL checks pass."""
    reasons: list[str] = []

    # --- Signal quality gate ---
    if TIER_RANK.get(intent.signal_tier, -1) < TIER_RANK[config.min_signal_tier]:
        return RiskDecision(False, [
            f"REJECT: signal tier '{intent.signal_tier}' below minimum '{config.min_signal_tier}'"
        ])

    # --- Regime gate ---
    if config.stop_all_if_spy_below_ma50_high_vol and (
        not account.spy_above_ma50 and account.spy_high_vol_breakdown
    ):
        return RiskDecision(False, [
            "REJECT: SPY broke MA50 on high volume — Minervini rule says halt new entries"
        ])

    # --- Circuit breakers ---
    if account.daily_pnl_pct <= -config.max_daily_loss_pct:
        return RiskDecision(False, [
            f"REJECT: daily loss {account.daily_pnl_pct:.2%} exceeds limit {config.max_daily_loss_pct:.2%}"
        ])

    if account.trades_today >= config.max_daily_trades:
        return RiskDecision(False, [f"REJECT: hit daily trade limit ({config.max_daily_trades})"])

    if intent.side == "buy" and account.new_positions_today >= config.max_new_positions_per_day:
        return RiskDecision(False, [
            f"REJECT: already opened {account.new_positions_today} new positions today"
        ])

    # --- Position count gate ---
    if intent.side == "buy":
        open_tickers = {p["ticker"] for p in account.open_positions}
        if intent.ticker in open_tickers:
            if not config.allow_averaging_down:
                return RiskDecision(False, [
                    f"REJECT: already hold {intent.ticker} — no averaging down (Minervini rule)"
                ])
        elif len(account.open_positions) >= config.max_open_positions:
            return RiskDecision(False, [
                f"REJECT: at max positions ({config.max_open_positions})"
            ])

    # --- Stop placement sanity ---
    stop_dist_pct = abs(intent.entry_price - intent.stop_price) / intent.entry_price
    if stop_dist_pct > config.max_stop_distance_pct:
        return RiskDecision(False, [
            f"REJECT: stop {stop_dist_pct:.2%} below entry exceeds {config.max_stop_distance_pct:.2%} max"
        ])
    if stop_dist_pct < config.min_stop_distance_pct:
        return RiskDecision(False, [
            f"REJECT: stop {stop_dist_pct:.2%} too tight (min {config.min_stop_distance_pct:.2%})"
        ])

    # --- Position sizing ---
    dollar_risk_per_share = intent.entry_price - intent.stop_price
    if dollar_risk_per_share <= 0:
        return RiskDecision(False, ["REJECT: stop must be below entry for long"])

    max_dollar_risk = account.equity * config.max_risk_per_trade_pct
    # Halve risk budget in weak regime
    if config.halt_if_spy_below_ma200 and not account.spy_above_ma200:
        max_dollar_risk *= 0.5
        reasons.append("NOTE: SPY below MA200, risk budget halved")

    max_shares = int(max_dollar_risk / dollar_risk_per_share)

    if intent.shares > max_shares:
        return RiskDecision(
            False,
            [f"REJECT: {intent.shares} shares exceeds max {max_shares} (risk sizing)"],
            adjusted={"shares": max_shares},
        )

    # --- Cash check ---
    cost = intent.shares * intent.entry_price
    if intent.side == "buy" and cost > account.cash:
        return RiskDecision(False, [f"REJECT: need ${cost:,.0f} but only ${account.cash:,.0f} cash"])

    reasons.append(
        f"APPROVED: {intent.shares}sh {intent.ticker} @ ${intent.entry_price:.2f} "
        f"stop ${intent.stop_price:.2f} (risk ${intent.shares * dollar_risk_per_share:,.0f}, "
        f"{intent.shares * dollar_risk_per_share / account.equity:.2%} of account)"
    )
    return RiskDecision(True, reasons)


def log_decision(intent: TradeIntent, decision: RiskDecision) -> None:
    """Append every decision to a daily audit log."""
    log_path = STATE_DIR / f"risk_log_{date.today().isoformat()}.jsonl"
    with log_path.open("a") as f:
        f.write(json.dumps({
            "ticker": intent.ticker,
            "side": intent.side,
            "entry": intent.entry_price,
            "stop": intent.stop_price,
            "shares": intent.shares,
            "signal_tier": intent.signal_tier,
            "approved": decision.approved,
            "reasons": decision.reasons,
        }) + "\n")
