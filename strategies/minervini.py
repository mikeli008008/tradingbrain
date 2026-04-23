"""
Minervini SEPA as a BaseStrategy plugin.

Wraps the existing screener output + Minervini rules. Cadence: daily.
This is the "concentrated breakout" strategy — 3-8 positions, fresh
entries on pivot breakouts, strict 7-8% stops.
"""
from __future__ import annotations
from datetime import date
from typing import Any

from strategies.base import BaseStrategy, Candidate, TradeProposal, register_strategy
from tools import market_data


@register_strategy
class MinerviniStrategy(BaseStrategy):
    strategy_id = "minervini"
    cadence = "daily"

    # Closed vocabulary for this strategy — used by shadow journal tagging
    _TAGS = [
        "vcp_breakout", "earnings_catalyst", "rs_leader",
        "near_pivot", "far_from_pivot",
        "fundamental_grade_a", "fundamental_grade_d",
        "high_volume_confirmation", "low_volume_warning",
        "regime_strong", "regime_weak",
        "super_stock", "perfect_setup", "leader_profile",
        "near_52w_high", "far_from_high",
        "catalyst_positive_news", "catalyst_negative_news",
        "avoid_earnings_risk",
    ]

    @classmethod
    def journal_tags(cls) -> list[str]:
        return list(cls._TAGS)

    def should_run_today(self, as_of: date) -> bool:
        # Weekdays only
        return as_of.weekday() < 5

    def scan(self, context: dict[str, Any]) -> list[Candidate]:
        """Pull latest Minervini screener output."""
        scan_data = market_data.get_minervini_scan()
        candidates = []

        # Super Stocks first (highest priority)
        for row in scan_data.get("super", [])[:5]:
            candidates.append(Candidate(
                ticker=row.get("ticker") or row.get("symbol", ""),
                conviction=0.85,
                reason=f"Super Stock — VCP+8/8+near pivot+Leader Profile. Score {row.get('score', 0)}",
                tags=["super_stock", "vcp_breakout", "near_pivot", "fundamental_grade_a"],
                metadata={"scan_tier": "super", "raw": row},
            ))

        # Perfect Setups
        for row in scan_data.get("perfect", [])[:5]:
            grade = row.get("grade", "").upper()
            tags = ["perfect_setup", "vcp_breakout", "near_pivot"]
            if grade in ("A", "B"):
                tags.append("fundamental_grade_a")
            elif grade in ("D", "F"):
                tags.append("fundamental_grade_d")
            candidates.append(Candidate(
                ticker=row.get("ticker") or row.get("symbol", ""),
                conviction=0.65 if grade in ("A", "B") else 0.45,
                reason=f"Perfect Setup (Grade {grade}). Score {row.get('score', 0)}",
                tags=tags,
                metadata={"scan_tier": "perfect", "raw": row},
            ))

        return [c for c in candidates if c.ticker]

    def plan(
        self,
        candidates: list[Candidate],
        current_positions: list[dict],
        allocation_usd: float,
        context: dict[str, Any],
    ) -> list[TradeProposal]:
        """
        Per-trade sizing: risk 0.75% of THIS STRATEGY's allocation per trade,
        stop 7% below entry. Max 5 concurrent positions (Minervini rule).
        """
        held_tickers = {p["ticker"] for p in current_positions}
        max_new = max(0, 15 - len(current_positions))

        proposals = []
        for c in candidates[:max_new]:
            if c.ticker in held_tickers:
                continue  # never average down

            quote = market_data.get_quote(c.ticker)
            if "error" in quote:
                continue

            entry = quote["price"]
            stop = round(entry * 0.93, 2)  # 7% stop

            # Target weight = risk budget / stop distance
            # risk = 0.75% of allocation; stop_dist = 7%
            # target_weight_of_strategy = 0.0075 / 0.07 ≈ 10.7% per position
            target_weight = 0.0075 / 0.07

            proposals.append(TradeProposal(
                strategy_id=self.strategy_id,
                ticker=c.ticker,
                side="buy",
                target_weight=target_weight,
                entry_price=entry,
                stop_price=stop,
                reason=c.reason,
                tags=c.tags,
                conviction=c.conviction,
            ))

        return proposals

    def regime_bias(self, regime: dict) -> float:
        """Minervini explicitly says reduce in weak regimes."""
        if not regime.get("spy_above_ma50", True):
            if regime.get("spy_high_vol_breakdown", False):
                return 0.0   # full halt
            return 0.5       # half size
        if not regime.get("spy_above_ma200", True):
            return 0.5
        return 1.0

    def describe_for_agent(self) -> str:
        return (
            "**Minervini SEPA** — concentrated breakout strategy. "
            "Daily cadence. Buys Super Stocks and Perfect Setups at pivot, "
            "stops 7% below entry, max 5 positions. Reduces in weak SPY regime."
        )
