"""
Cross-sectional momentum strategy.

Bi-weekly partial rebalance: every other Friday, recompute rankings and
rotate 20-30% of the book toward current top-N. Partial rebalance reduces
transaction cost and avoids all-or-nothing regime whipsaws.

Config (strategies/cs_momentum_config.json):
  top_n: 5 or 10 (portfolio size)
  lookback_months: 8 or 12
  skip_months: 1 (skip most recent month — 12-1 momentum effect)
  max_drift: 0.3 (rebalance at most 30% of book per cycle)

Signals use risk-adjusted momentum: (lookback return) / (lookback std),
which reduces the tendency to overweight volatile names with lucky runs.
"""
from __future__ import annotations
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from strategies.base import BaseStrategy, Candidate, TradeProposal, register_strategy

CONFIG_PATH = Path(__file__).parent / "cs_momentum_config.json"
UNIVERSE_CACHE = Path(__file__).parent.parent / "state" / "bootstrap_cache"


@register_strategy
class CSMomentumStrategy(BaseStrategy):
    strategy_id = "cs_momentum"
    cadence = "biweekly"

    _TAGS = [
        "top_decile", "bottom_decile",
        "high_momentum_score", "low_momentum_score",
        "high_volatility", "low_volatility",
        "sector_tech", "sector_finance", "sector_consumer", "sector_energy", "sector_other",
        "rank_improved", "rank_deteriorated",
        "regime_strong", "regime_weak",
        "new_entry", "existing_holding", "rotation_out",
    ]

    def __init__(self):
        self.config = self._load_config()

    @classmethod
    def journal_tags(cls) -> list[str]:
        return list(cls._TAGS)

    def _load_config(self) -> dict:
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text())
        # Defaults
        default = {
            "top_n": 10,                # portfolio size
            "lookback_months": 12,
            "skip_months": 1,            # skip last month (12-1 momentum)
            "max_drift": 0.3,            # max % of book to rotate per cycle
            "min_price": 10.0,           # skip penny stocks
            "min_avg_dollar_volume": 10_000_000,  # liquidity filter
            "universe": "default",       # uses bootstrap cache
        }
        CONFIG_PATH.write_text(json.dumps(default, indent=2))
        return default

    def should_run_today(self, as_of: date) -> bool:
        # Bi-weekly: run on Fridays, every other Friday
        if as_of.weekday() != 4:  # Friday = 4
            return False
        # Use ISO week parity: odd weeks run
        iso_week = as_of.isocalendar()[1]
        return iso_week % 2 == 1

    def _load_universe(self) -> list[str]:
        """Load tickers from cached bootstrap universe, or fallback."""
        if UNIVERSE_CACHE.exists():
            tickers = [p.stem for p in UNIVERSE_CACHE.glob("*.parquet") if p.stem != "SPY"]
            if tickers:
                return tickers
        # Fallback: small default universe
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
            "ORCL", "CRM", "ADBE", "AMD", "INTC", "QCOM", "CSCO", "NFLX",
            "PYPL", "UBER", "SHOP", "JPM", "BAC", "V", "MA", "HD", "WMT",
            "KO", "PEP", "COST", "PG", "JNJ", "UNH", "XOM", "CVX",
        ]

    def _compute_momentum_score(self, ticker: str, as_of: date) -> float | None:
        """
        Risk-adjusted momentum: (lookback return) / (lookback daily std).
        Skips the last `skip_months` to capture the 12-1 effect.
        """
        try:
            start = as_of - timedelta(days=int((self.config["lookback_months"] + 2) * 30))
            df = yf.Ticker(ticker).history(start=start.isoformat(), end=as_of.isoformat(), auto_adjust=True)
            if df.empty or len(df) < 150:
                return None

            # Liquidity filter
            recent_vol = df.tail(20)
            avg_dollar_vol = (recent_vol["Close"] * recent_vol["Volume"]).mean()
            if avg_dollar_vol < self.config["min_avg_dollar_volume"]:
                return None
            if float(df["Close"].iloc[-1]) < self.config["min_price"]:
                return None

            # Define the momentum window: [start, end - skip]
            skip_days = int(self.config["skip_months"] * 21)  # ~21 trading days/month
            lookback_days = int(self.config["lookback_months"] * 21)

            if len(df) < lookback_days + skip_days:
                return None

            end_idx = len(df) - skip_days - 1
            start_idx = end_idx - lookback_days
            if start_idx < 0:
                return None

            window = df.iloc[start_idx:end_idx + 1]
            start_price = float(window["Close"].iloc[0])
            end_price = float(window["Close"].iloc[-1])
            total_return = (end_price - start_price) / start_price

            daily_returns = window["Close"].pct_change().dropna()
            vol = daily_returns.std()
            if vol == 0 or pd.isna(vol):
                return None

            # Risk-adjusted momentum (Sharpe-like)
            return float(total_return / (vol * (len(daily_returns) ** 0.5)))
        except Exception:
            return None

    def scan(self, context: dict[str, Any]) -> list[Candidate]:
        """Rank the universe by risk-adjusted momentum, return top N + bottom N."""
        as_of = context.get("as_of_date") or date.today()
        universe = self._load_universe()

        scores = {}
        for ticker in universe:
            score = self._compute_momentum_score(ticker, as_of)
            if score is not None:
                scores[ticker] = score

        if not scores:
            return []

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        top_n = self.config["top_n"]

        # Compute percentile thresholds for tagging
        n = len(ranked)
        top_cutoff = ranked[min(top_n - 1, n - 1)][1]
        bottom_cutoff = ranked[max(-top_n, -n)][1]

        candidates = []
        for i, (ticker, score) in enumerate(ranked[:top_n]):
            tags = ["top_decile", "high_momentum_score"]
            if score > top_cutoff * 1.5:
                tags.append("high_momentum_score")
            candidates.append(Candidate(
                ticker=ticker,
                conviction=max(0.5, min(0.9, 0.5 + (score - top_cutoff) * 0.1)),
                reason=f"Ranked #{i+1} by risk-adjusted momentum (score={score:.2f})",
                tags=tags,
                metadata={"rank": i + 1, "score": score, "total_universe": n},
            ))

        return candidates

    def plan(
        self,
        candidates: list[Candidate],
        current_positions: list[dict],
        allocation_usd: float,
        context: dict[str, Any],
    ) -> list[TradeProposal]:
        """
        Partial rebalance: rotate at most `max_drift` of the book.

        Steps:
         1. Compute target portfolio from top N (equal-weight within strategy)
         2. Compute current portfolio
         3. Sort differences: biggest overweights → sell, biggest underweights → buy
         4. Trade up to max_drift % of allocation, prioritizing largest moves
        """
        top_n = self.config["top_n"]
        if not candidates:
            return []

        # Equal-weight target: each position = 1/N of allocation
        target_weight = 1.0 / top_n
        target_tickers = {c.ticker for c in candidates[:top_n]}

        # Current positions by ticker
        current_by_ticker = {p["ticker"]: p for p in current_positions}
        current_usd = {t: p["shares"] * p.get("current", p.get("entry", 0))
                       for t, p in current_by_ticker.items()}
        total_current_value = sum(current_usd.values()) or allocation_usd
        current_weight_of_strategy = {
            t: v / total_current_value for t, v in current_usd.items()
        }

        proposals = []

        # Identify sells: positions NOT in target_tickers
        sells = []
        for ticker, p in current_by_ticker.items():
            if ticker not in target_tickers:
                sells.append((ticker, current_weight_of_strategy.get(ticker, 0)))
        sells.sort(key=lambda x: -x[1])  # largest positions first

        # Identify buys: target tickers with zero or underweight position
        buys = []
        for c in candidates[:top_n]:
            current_w = current_weight_of_strategy.get(c.ticker, 0)
            gap = target_weight - current_w
            if gap > 0.02:  # at least 2% gap to bother
                buys.append((c, gap))
        buys.sort(key=lambda x: -x[1])

        # Budget: max_drift of allocation
        budget_usd = allocation_usd * self.config["max_drift"]
        used_usd = 0.0

        # Execute sells first (frees capital)
        for ticker, weight in sells:
            sell_usd = weight * total_current_value
            if used_usd + sell_usd > budget_usd:
                break
            used_usd += sell_usd
            proposals.append(TradeProposal(
                strategy_id=self.strategy_id,
                ticker=ticker,
                side="sell",
                target_weight=0,
                entry_price=None,  # market
                stop_price=None,   # no stops in XS momentum
                reason=f"Rotated out of position (no longer top {top_n})",
                tags=["rotation_out", "low_momentum_score"],
                conviction=0.7,
            ))

        # Then buys
        for c, gap in buys:
            buy_usd = gap * allocation_usd
            if used_usd + buy_usd > budget_usd:
                # Partial fill
                buy_usd = max(0, budget_usd - used_usd)
                if buy_usd < allocation_usd * 0.02:
                    break
            used_usd += buy_usd
            is_existing = c.ticker in current_by_ticker
            proposals.append(TradeProposal(
                strategy_id=self.strategy_id,
                ticker=c.ticker,
                side="buy",
                target_weight=target_weight,
                entry_price=None,  # market order
                stop_price=None,   # no stops for XS momentum
                reason=c.reason,
                tags=c.tags + (["existing_holding"] if is_existing else ["new_entry"]),
                conviction=c.conviction,
            ))

        return proposals

    def regime_bias(self, regime: dict) -> float:
        """
        XS momentum is less regime-sensitive than Minervini because it's
        relative ranking, but still reduce in true bear markets.
        """
        if not regime.get("spy_above_ma200", True) and regime.get("spy_high_vol_breakdown", False):
            return 0.5
        return 1.0

    def describe_for_agent(self) -> str:
        c = self.config
        return (
            f"**Cross-sectional momentum** — bi-weekly rotation strategy. "
            f"Runs every other Friday. Top {c['top_n']} by {c['lookback_months']}-{c['skip_months']} "
            f"risk-adjusted momentum. Partial rebalance capped at {c['max_drift']*100:.0f}% per cycle. "
            f"No stops (relative strategy, exits by rotation)."
        )
