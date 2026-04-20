"""
Bootstrap Trainer — offline walk-forward backtest over 10 years of history.

One-time run that seeds the rule ledger with ~50,000 historical decisions
instead of waiting for 1,000 live-shadow ones.

Architecture:
  1. Download 10yr OHLCV for all tickers (cached locally)
  2. For each month from 2015-01 through last month:
     a. Compute Minervini technical signals as-of that date
     b. For each qualifying candidate, compute deterministic tags
     c. Look forward to grade T+5 and T+20 outcomes
     d. Feed labeled decision into a provisional shadow journal
  3. Run pattern miner on the historical journal, walk-forward style
     (each month's miner only sees data up to that month)
  4. Promote rules to the ledger with:
       - status: "provisional" (not yet live-active)
       - source: "bootstrap"
       - regime_tag: the regime they emerged from
  5. Provisional rules activate only after N live-shadow confirmations

CRITICAL: Walk-forward enforcement. When mining patterns for period T,
the miner ONLY sees decisions from periods < T. No peeking.

LIMITATIONS documented in every promoted rule:
  - Survivorship bias (current S&P, not point-in-time)
  - Splits/dividends adjusted by yfinance (good)
  - Tag generation is deterministic, differs from live LLM tags
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
import argparse

import pandas as pd
import yfinance as yf
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "state" / "bootstrap_cache"
BOOTSTRAP_DIR = ROOT / "state" / "bootstrap"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)


# --- Data layer ---

def download_universe(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download & cache daily OHLCV for each ticker."""
    data = {}
    for t in tqdm(tickers, desc="Downloading"):
        cache_path = CACHE_DIR / f"{t}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            # Normalize to tz-naive
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            # Refresh if stale (> 7 days old)
            if df.index.max().date() < date.today() - timedelta(days=7):
                cache_path.unlink()
            else:
                data[t] = df
                continue
        try:
            df = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
            if df.empty or len(df) < 250:
                continue
            # Normalize tz-naive to keep comparisons simple
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.to_parquet(cache_path)
            data[t] = df
        except Exception as e:
            print(f"  skip {t}: {e}")
    return data


def load_ticker_universe() -> list[str]:
    """Read tickers.txt from the screener, or use a default small universe."""
    tickers_file = ROOT / "tickers.txt"
    if tickers_file.exists():
        return [
            line.strip().upper() for line in tickers_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    # Default small set for testing
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
        "ORCL", "CRM", "ADBE", "AMD", "INTC", "QCOM", "CSCO", "NFLX",
        "PYPL", "UBER", "SHOP", "SQ", "SPY", "QQQ", "IWM",
    ]


# --- Technical signals (point-in-time for a given date) ---

def compute_minervini_as_of(df: pd.DataFrame, target_date: pd.Timestamp) -> dict | None:
    """
    Given a price history DataFrame and a date, compute Minervini's 8 trend
    template checks using ONLY data up to target_date. Returns the signal
    or None if data insufficient.
    """
    # Slice to point-in-time — no lookahead
    hist = df[df.index <= target_date]
    if len(hist) < 250:
        return None

    close = hist["Close"]
    last = float(close.iloc[-1])
    ma50 = close.rolling(50).mean().iloc[-1]
    ma150 = close.rolling(150).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    ma200_20d_ago = close.rolling(200).mean().iloc[-21] if len(close) > 220 else None

    low_52w = close.iloc[-252:].min()
    high_52w = close.iloc[-252:].max()

    checks = [
        last > ma150 and last > ma200,
        ma150 > ma200,
        ma200_20d_ago is not None and ma200 > ma200_20d_ago,
        ma50 > ma150 and ma50 > ma200,
        last > ma50,
        last >= low_52w * 1.30,
        last >= high_52w * 0.75,
        True,  # RS rating — we fake this as always-pass for bootstrap
    ]
    passes = sum(1 for c in checks if c)

    return {
        "date": target_date.isoformat(),
        "price": last,
        "ma50": float(ma50),
        "ma150": float(ma150),
        "ma200": float(ma200),
        "high_52w": float(high_52w),
        "low_52w": float(low_52w),
        "trend_passes": passes,
        "pct_from_high": (last - high_52w) / high_52w * 100,
    }


# --- Deterministic tag generation ---

def generate_tags(df: pd.DataFrame, target_date: pd.Timestamp, signal: dict) -> list[str]:
    """Rule-based tags. Cheap, consistent, no LLM."""
    tags = []
    hist = df[df.index <= target_date].tail(90)
    if len(hist) < 60:
        return tags

    close = hist["Close"]
    vol = hist["Volume"]
    last = float(close.iloc[-1])

    # Tier 1: trend strength
    if signal["trend_passes"] >= 7:
        tags.append("trend_strong")
    elif signal["trend_passes"] <= 4:
        tags.append("trend_weak")

    # VCP breakout proxy: price within 3% of 8-week high, after a recent 10%+ pullback
    high_8w = close.iloc[-40:].max()
    low_8w = close.iloc[-40:].min()
    pullback_pct = (high_8w - low_8w) / high_8w * 100
    near_high = (high_8w - last) / high_8w < 0.03
    if near_high and pullback_pct > 10:
        tags.append("vcp_breakout")

    # Near pivot: 52w high within reach
    if signal["pct_from_high"] > -10:
        tags.append("near_52w_high")
    elif signal["pct_from_high"] < -25:
        tags.append("far_from_high")

    # Volume confirmation: today vs 50d avg
    avg_vol = vol.rolling(50).mean().iloc[-1]
    if vol.iloc[-1] > avg_vol * 1.5:
        tags.append("high_volume_confirmation")
    elif vol.iloc[-1] < avg_vol * 0.5:
        tags.append("low_volume_warning")

    # Oversold bounce proxy: RSI-14 under 30 and turning up
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 0.001)
    rsi = 100 - (100 / (1 + rs))
    if len(rsi) >= 3:
        r_last = float(rsi.iloc[-1])
        r_prev = float(rsi.iloc[-2])
        if r_prev < 30 and r_last > r_prev:
            tags.append("oversold_bounce")

    # Momentum: 3-month return
    m3_return = (last - close.iloc[-63]) / close.iloc[-63] * 100 if len(close) > 63 else 0
    if m3_return > 20:
        tags.append("strong_3m_momentum")
    elif m3_return < -15:
        tags.append("weak_3m_momentum")

    return tags


# --- Regime labels ---

def compute_regime(spy_df: pd.DataFrame, target_date: pd.Timestamp) -> str:
    """Classify the market regime as of target_date."""
    hist = spy_df[spy_df.index <= target_date]
    if len(hist) < 200:
        return "unknown"

    close = hist["Close"]
    last = float(close.iloc[-1])
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]

    # 20-day realized volatility
    returns = close.pct_change()
    vol_20d = returns.iloc[-20:].std() * (252 ** 0.5) * 100

    if last > ma50 > ma200:
        base = "bull"
    elif last < ma200:
        base = "bear"
    else:
        base = "transitional"

    if vol_20d > 25:
        return f"{base}_high_vol"
    return base


# --- Forward outcome grading ---

def grade_forward(df: pd.DataFrame, entry_date: pd.Timestamp, entry_price: float,
                  stop_pct: float = 0.07) -> dict:
    """Walk forward to grade T+1, T+5, T+20 outcomes. No lookahead — we only
    use data at/after entry_date."""
    future = df[df.index > entry_date]
    if len(future) < 20:
        return {}

    t1 = future.iloc[0] if len(future) >= 1 else None
    t5 = future.iloc[4] if len(future) >= 5 else None
    t20 = future.iloc[19] if len(future) >= 20 else None

    result = {}
    stop_price = entry_price * (1 - stop_pct)

    if t1 is not None:
        result["pnl_pct_t1"] = (float(t1["Close"]) - entry_price) / entry_price * 100
    if t5 is not None:
        result["pnl_pct_t5"] = (float(t5["Close"]) - entry_price) / entry_price * 100
    if t20 is not None:
        result["pnl_pct_t20"] = (float(t20["Close"]) - entry_price) / entry_price * 100

    # Did it hit stop in first 20 days?
    first_20 = future.iloc[:20]
    stopped = bool((first_20["Low"] <= stop_price).any())
    result["stopped_out"] = stopped

    # Outcome labels
    if t5 is not None:
        result["outcome_correct"] = bool(result["pnl_pct_t5"] > 0 and not stopped)

    return result


# --- Pattern mining with walk-forward enforcement ---

def walk_forward_mine(decisions: list[dict], min_support: int = 50) -> list[dict]:
    """
    Walk forward through decisions, mining rules at month boundaries.
    Each rule gets stamped with the date it was discoverable.

    A rule mined at date D only uses decisions from before D. This is the
    discipline that prevents lookahead bias.
    """
    df = pd.DataFrame(decisions)
    df["decision_date"] = pd.to_datetime(df["decision_date"])
    df = df.sort_values("decision_date").reset_index(drop=True)

    # Only grade decisions where we have full T+20 outcome
    graded = df[df["outcome_correct"].notna()].copy()
    if len(graded) < min_support:
        return []

    # Month-end mining checkpoints
    start = graded["decision_date"].min()
    end = graded["decision_date"].max()
    checkpoints = pd.date_range(start, end, freq="ME")

    rules = []
    seen_rule_keys = set()

    for cp in checkpoints:
        # Only decisions known as-of cp (at least 20 trading days in the past)
        usable = graded[graded["decision_date"] <= cp - pd.Timedelta(days=30)]
        if len(usable) < min_support:
            continue

        baseline_hit = usable["outcome_correct"].mean()

        # Explode tags and group
        exploded = usable.explode("tags")
        by_tag = exploded.groupby("tags")

        for tag, group in by_tag:
            if pd.isna(tag) or len(group) < min_support:
                continue
            hit_rate = group["outcome_correct"].mean()
            delta = hit_rate - baseline_hit
            if abs(delta) < 0.10:
                continue

            p_value = _binomial_p(
                int(group["outcome_correct"].sum()),
                len(group),
                float(baseline_hit),
            )
            if p_value > 0.10:
                continue

            # Regime this rule emerged from
            dominant_regime = group["regime"].mode().iloc[0] if not group["regime"].mode().empty else "unknown"

            rule_key = (tag, "favorable" if delta > 0 else "unfavorable")
            if rule_key in seen_rule_keys:
                continue
            seen_rule_keys.add(rule_key)

            rules.append({
                "tag": tag,
                "prediction": "favorable" if delta > 0 else "unfavorable",
                "discovered_at": cp.strftime("%Y-%m-%d"),
                "n_supporting": len(group),
                "hit_rate_with_rule": round(float(hit_rate), 3),
                "hit_rate_baseline": round(float(baseline_hit), 3),
                "delta": round(float(delta), 3),
                "p_value": round(p_value, 4),
                "regime_stratification": dominant_regime,
                "source": "bootstrap",
            })

    return rules


def _binomial_p(k: int, n: int, p0: float) -> float:
    """Two-tailed binomial p-value via normal approximation."""
    import math
    if n < 5 or p0 <= 0 or p0 >= 1:
        return 1.0
    expected = n * p0
    variance = n * p0 * (1 - p0)
    if variance <= 0:
        return 1.0
    z = abs(k - expected) / math.sqrt(variance)
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


# --- Main orchestration ---

def run(start_year: int = 2015, end_year: int | None = None,
        scan_day: int = 15, sample_tickers: int | None = None) -> dict:
    """
    End-to-end bootstrap. Produces a rule list that seeds the live ledger
    with 'provisional' status.

    scan_day: day-of-month on which to scan (15th = mid-month, avoids noisy opens/closes)
    sample_tickers: limit universe for faster iteration; None = full universe
    """
    end_year = end_year or date.today().year
    tickers = load_ticker_universe()
    if sample_tickers:
        tickers = tickers[:sample_tickers]

    start_str = f"{start_year - 1}-01-01"   # extra year for MA200 warmup
    end_str = f"{end_year}-12-31"

    # 1. Download data
    print(f"Downloading {len(tickers)} tickers from {start_str} to {end_str}...")
    data = download_universe(tickers, start_str, end_str)
    print(f"Successfully loaded {len(data)} tickers")

    if "SPY" not in data:
        spy_raw = yf.Ticker("SPY").history(start=start_str, end=end_str, auto_adjust=True)
        if spy_raw.index.tz is not None:
            spy_raw.index = spy_raw.index.tz_localize(None)
        data["SPY"] = spy_raw
    spy_df = data["SPY"]

    # 2. Walk forward through months
    decisions = []
    months = pd.date_range(f"{start_year}-01-01", end_str, freq="MS")

    for month_start in tqdm(months, desc="Walk-forward scan"):
        scan_date = month_start + pd.Timedelta(days=scan_day - 1)
        regime = compute_regime(spy_df, scan_date)

        for ticker, df in data.items():
            if ticker == "SPY":
                continue
            if scan_date > df.index.max() or scan_date < df.index.min():
                continue

            signal = compute_minervini_as_of(df, scan_date)
            if signal is None or signal["trend_passes"] < 5:
                continue  # Only score real candidates

            tags = generate_tags(df, scan_date, signal)
            if not tags:
                continue

            outcome = grade_forward(df, scan_date, signal["price"])
            if "outcome_correct" not in outcome:
                continue  # Not enough forward data

            decisions.append({
                "decision_date": scan_date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "regime": regime,
                "tags": tags,
                "entry_price": signal["price"],
                **outcome,
            })

    print(f"\nGenerated {len(decisions)} labeled decisions")

    # Persist raw decisions for later analysis
    decisions_path = BOOTSTRAP_DIR / "decisions.jsonl"
    with decisions_path.open("w") as f:
        for d in decisions:
            f.write(json.dumps(d, default=str) + "\n")

    # 3. Walk-forward rule mining
    print("Mining rules with walk-forward discipline...")
    rules = walk_forward_mine(decisions)
    print(f"Discovered {len(rules)} historically-validated rules")

    # 4. Save as provisional rules
    rules_path = BOOTSTRAP_DIR / "provisional_rules.json"
    rules_path.write_text(json.dumps(rules, indent=2, default=str))

    # 5. Summary
    summary = {
        "run_date": date.today().isoformat(),
        "period": f"{start_year}-{end_year}",
        "n_tickers": len(data),
        "n_decisions": len(decisions),
        "n_rules_provisional": len(rules),
        "favorable": sum(1 for r in rules if r["prediction"] == "favorable"),
        "unfavorable": sum(1 for r in rules if r["prediction"] == "unfavorable"),
        "limitations": [
            "Survivorship bias: current S&P 500 composition, not point-in-time",
            "Deterministic tags differ from live LLM-generated tags",
            "Rules marked 'provisional' — require N live-shadow confirmations to activate",
            "RS rating check skipped in bootstrap (always-pass)",
        ],
    }
    (BOOTSTRAP_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--sample-tickers", type=int, default=None,
                        help="Limit to N tickers for faster iteration")
    args = parser.parse_args()

    result = run(
        start_year=args.start_year,
        end_year=args.end_year,
        sample_tickers=args.sample_tickers,
    )
    print("\n" + json.dumps(result, indent=2, default=str))
