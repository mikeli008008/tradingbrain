"""
BaseStrategy — the abstract contract every strategy implements.

Design principle: strategies declare *what* they want to do; the agent
loop and harness decide *whether* and *how*. The strategy never talks
to the broker directly — it produces TradeIntents that flow through
the portfolio manager and risk manager.

The four required methods:
  - scan()        : find candidates (may differ wildly between strategies)
  - plan()        : turn candidates into TradeIntents
  - validate()    : strategy-specific pre-check (e.g. rebalance due?)
  - journal_tags(): tagging vocabulary for the shadow journal

Every strategy has:
  - An allocation (0-1, fraction of account assigned to it)
  - A cadence (daily / bi-weekly / monthly)
  - Its own rule ledger namespace
  - Its own shadow journal subdir
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal


@dataclass
class Candidate:
    """What a strategy's scan returns — a ticker it's interested in."""
    ticker: str
    conviction: float           # 0-1, how strongly the strategy wants this
    reason: str                 # plain English, for the journal
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)  # strategy-specific extras


@dataclass
class TradeProposal:
    """What a strategy proposes. Goes to portfolio manager, then risk gate."""
    strategy_id: str
    ticker: str
    side: Literal["buy", "sell"]
    target_weight: float        # 0-1, fraction of this strategy's allocation
    entry_price: float | None   # None for market orders
    stop_price: float | None    # None if strategy doesn't use stops (XS momentum)
    reason: str
    tags: list[str] = field(default_factory=list)
    conviction: float = 0.5


class BaseStrategy(ABC):
    """Every strategy subclasses this."""

    #: Unique string id, e.g. "minervini" or "cs_momentum"
    strategy_id: str = "base"

    #: How often this strategy runs: "daily" | "biweekly" | "monthly"
    cadence: Literal["daily", "biweekly", "monthly"] = "daily"

    #: The tag vocabulary this strategy uses. Shadow decisions must
    #: only use these tags (prevents cross-strategy contamination).
    @classmethod
    @abstractmethod
    def journal_tags(cls) -> list[str]:
        """Return the closed set of tags this strategy uses."""
        ...

    @abstractmethod
    def should_run_today(self, as_of: date) -> bool:
        """Cadence check. Minervini: every weekday. XS momentum: every other Friday."""
        ...

    @abstractmethod
    def scan(self, context: dict[str, Any]) -> list[Candidate]:
        """
        Find candidates. Context includes market_regime, current_positions,
        and anything else the strategy might need.
        """
        ...

    @abstractmethod
    def plan(
        self,
        candidates: list[Candidate],
        current_positions: list[dict],
        allocation_usd: float,
        context: dict[str, Any],
    ) -> list[TradeProposal]:
        """
        Turn candidates into concrete trade proposals.

        current_positions: positions this strategy currently holds
        allocation_usd: dollar amount assigned to this strategy
        """
        ...

    def regime_bias(self, regime: dict) -> float:
        """
        Optional: return a 0-2 multiplier on the strategy's allocation
        based on current regime. Default = 1.0 (no adjustment).

        E.g. Minervini might return 0.5 in bear regime, 1.5 in bull.
        """
        return 1.0

    def describe_for_agent(self) -> str:
        """Human-readable description injected into agent's morning context."""
        return f"Strategy '{self.strategy_id}' (cadence: {self.cadence})"


# Registry — strategies register themselves here so the portfolio manager
# can discover them without hardcoded imports.
_STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register_strategy(cls: type[BaseStrategy]) -> type[BaseStrategy]:
    """Decorator: @register_strategy above a strategy class."""
    _STRATEGY_REGISTRY[cls.strategy_id] = cls
    return cls


def get_strategy(strategy_id: str) -> BaseStrategy:
    """Factory: instantiate a registered strategy by id."""
    if strategy_id not in _STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy: {strategy_id}")
    return _STRATEGY_REGISTRY[strategy_id]()


def list_strategies() -> list[str]:
    return list(_STRATEGY_REGISTRY.keys())
