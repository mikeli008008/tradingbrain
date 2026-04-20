"""
Outcome Grader + Forensics.

Grades shadow decisions (and real trades) against market outcomes at
T+1, T+5, T+20. Crucially, makes TWO separate judgments:

  - outcome_correct: did the P&L agree with action? (buy + up = correct)
  - thesis_correct: did the WHY play out? (e.g., if thesis was 'VCP
    breakout on volume', did price break above pivot with volume?)

You can be outcome-right for thesis-wrong reasons (lucky) or
outcome-wrong for thesis-right reasons (good process, bad luck). They
demand opposite lessons. This is the single biggest failure mode in
naive trading learning loops.
"""
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path
from typing import Any
import json
import yfinance as yf
import pandas as pd

from learning import shadow_journal as sj

FORENSICS_DIR = Path(__file__).parent.parent / "state" / "forensics"
FORENSICS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 5, 20]


def _price_on_or_after(ticker: str, target_date: date) -> float | None:
    """Get close price on target_date (or next trading day if weekend/holiday)."""
    try:
        hist = yf.Ticker(ticker).history(
            start=target_date.isoformat(),
            end=(target_date + timedelta(days=7)).isoformat(),
        )
        if hist.empty:
            return None
        return float(hist["Close"].iloc[0])
    except Exception:
        return None


def _hit_stop_or_target(
    ticker: str, entry_date: date, horizon_days: int,
    stop: float | None, target: float | None,
) -> tuple[bool, bool]:
    """Did price touch stop or target in the horizon window?"""
    try:
        hist = yf.Ticker(ticker).history(
            start=entry_date.isoformat(),
            end=(entry_date + timedelta(days=horizon_days + 5)).isoformat(),
        )
        if hist.empty:
            return False, False
        lo = hist["Low"].min()
        hi = hist["High"].max()
        stopped = bool(stop is not None and lo <= stop)
        reached = bool(target is not None and hi >= target)
        return stopped, reached
    except Exception:
        return False, False


def grade_at_horizon(horizon: int) -> dict[str, int]:
    """Grade all shadow decisions old enough for this horizon."""
    pending = sj.load_pending_grade(horizon)
    graded = 0
    skipped = 0

    for path, line_idx, d in pending:
        decision_date = date.fromisoformat(d.timestamp[:10])
        target_date = decision_date + timedelta(days=horizon)
        if target_date > date.today():
            continue

        price_then = _price_on_or_after(d.ticker, target_date)
        if price_then is None:
            skipped += 1
            continue

        pnl_pct = (price_then - d.entry_price) / d.entry_price * 100
        setattr(d, f"price_at_t{horizon}", round(price_then, 2))
        setattr(d, f"pnl_pct_t{horizon}", round(pnl_pct, 2))

        # At T+20 also check stop/target touches
        if horizon == 20:
            stopped, reached = _hit_stop_or_target(
                d.ticker, decision_date, horizon, d.stop_price, d.target_price
            )
            d.stopped_out = stopped
            d.reached_target = reached
            d.graded_at = date.today().isoformat()

        sj.update_decision(path, line_idx, d)
        graded += 1

    return {"horizon": horizon, "graded": graded, "skipped": skipped}


def run_forensics() -> dict[str, Any]:
    """
    Apply thesis_correct vs outcome_correct labels to fully-graded decisions.

    outcome_correct logic:
      action=buy:  outcome_correct = pnl_t5 > 0 (chose to buy, it went up)
      action=skip: outcome_correct = pnl_t5 <= 0 (chose to skip, it didn't rip without us)
      action=watch: outcome_correct = True only if we SHOULD have acted and didn't miss >5%

    thesis_correct logic:
      We ask: did the thesis MECHANISM play out?
      - If thesis mentions 'breakout', did it break above pivot?
      - If thesis mentions 'earnings catalyst', did it beat?
      - If thesis mentions 'bounce', did it bounce?
      Uses tag-based heuristics; the rule for each tag below.
    """
    all_decisions = sj.load_all()
    done = []

    for d in all_decisions:
        if d.pnl_pct_t5 is None or d.pnl_pct_t20 is None:
            continue
        if d.thesis_correct is not None and d.outcome_correct is not None:
            continue  # already labeled

        # --- outcome label ---
        if d.action == "buy":
            d.outcome_correct = d.pnl_pct_t5 > 0 and not (d.stopped_out or False)
        elif d.action == "skip":
            # Skipping was correct if price didn't run more than 5% in 5 days
            d.outcome_correct = d.pnl_pct_t5 < 5.0
        elif d.action == "watch":
            # Watching is always "neutral correct" — hard to grade
            d.outcome_correct = abs(d.pnl_pct_t5) < 5.0

        # --- thesis label (tag-driven heuristics) ---
        d.thesis_correct = _judge_thesis(d)

        # Persist
        for p in sorted(sj.SHADOW_DIR.glob("*.jsonl")):
            lines = p.read_text().splitlines()
            for i, ln in enumerate(lines):
                if json.loads(ln)["id"] == d.id:
                    sj.update_decision(p, i, d)
                    done.append(d.id)
                    break

    # Write a forensics summary
    summary = _summarize(all_decisions)
    (FORENSICS_DIR / f"{date.today().isoformat()}.json").write_text(
        json.dumps(summary, indent=2)
    )
    return {"labeled": len(done), "summary": summary}


def _judge_thesis(d: sj.ShadowDecision) -> bool | None:
    """Heuristic: did the thesis mechanism play out at T+5 or T+20?"""
    tags = set(d.thesis_tags or [])
    if not tags:
        # Fallback: thesis correct iff outcome correct (degrades to binary)
        return d.outcome_correct

    # Tag-specific rules
    if "vcp_breakout" in tags:
        # Valid breakout = up >3% in 5 days without touching stop
        return bool(d.pnl_pct_t5 and d.pnl_pct_t5 > 3.0 and not (d.stopped_out or False))
    if "earnings_catalyst" in tags:
        # Valid catalyst = gap up >5% at T+1
        return bool(d.pnl_pct_t1 and d.pnl_pct_t1 > 5.0)
    if "rs_leader" in tags:
        # RS leader thesis = outperforms over 20d (vs 0% proxy; full version would use SPY)
        return bool(d.pnl_pct_t20 and d.pnl_pct_t20 > 2.0)
    if "oversold_bounce" in tags:
        # Bounce = up >2% in 5d
        return bool(d.pnl_pct_t5 and d.pnl_pct_t5 > 2.0)
    if "avoid_earnings_risk" in tags:
        # Skip-earnings thesis = price moved >7% in either direction (would've been risky)
        return bool(d.pnl_pct_t5 and abs(d.pnl_pct_t5) > 7.0)
    # Unknown tags: defer to outcome
    return d.outcome_correct


def _summarize(decisions: list[sj.ShadowDecision]) -> dict[str, Any]:
    """Aggregate stats over all graded decisions."""
    graded = [d for d in decisions if d.outcome_correct is not None]
    if not graded:
        return {"graded_count": 0}

    buys = [d for d in graded if d.action == "buy"]
    skips = [d for d in graded if d.action == "skip"]

    hit_rate = sum(1 for d in graded if d.outcome_correct) / len(graded)
    buy_hit_rate = (
        sum(1 for d in buys if d.outcome_correct) / len(buys) if buys else None
    )
    avg_buy_pnl_t5 = (
        sum(d.pnl_pct_t5 for d in buys if d.pnl_pct_t5 is not None) / len(buys)
        if buys else None
    )

    # The "lucky vs skilled" matrix
    quadrants = {
        "skilled_win": 0,   # thesis + outcome both right (real skill)
        "lucky_win": 0,     # outcome right, thesis wrong (lucky)
        "unlucky_loss": 0,  # thesis right, outcome wrong (good process, bad luck)
        "mistake": 0,       # both wrong (learn from this)
    }
    for d in graded:
        if d.thesis_correct and d.outcome_correct:
            quadrants["skilled_win"] += 1
        elif not d.thesis_correct and d.outcome_correct:
            quadrants["lucky_win"] += 1
        elif d.thesis_correct and not d.outcome_correct:
            quadrants["unlucky_loss"] += 1
        else:
            quadrants["mistake"] += 1

    return {
        "graded_count": len(graded),
        "overall_hit_rate": round(hit_rate, 3),
        "buy_hit_rate": round(buy_hit_rate, 3) if buy_hit_rate else None,
        "avg_buy_pnl_t5_pct": round(avg_buy_pnl_t5, 2) if avg_buy_pnl_t5 else None,
        "n_buys": len(buys),
        "n_skips": len(skips),
        "quadrants": quadrants,
        "process_quality": round(
            (quadrants["skilled_win"] + quadrants["unlucky_loss"]) / len(graded), 3
        ),
    }


def run_all() -> dict[str, Any]:
    """Grade every pending horizon + run forensics. Called daily."""
    results = {}
    for h in HORIZONS:
        results[f"t{h}"] = grade_at_horizon(h)
    results["forensics"] = run_forensics()
    return results


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2))
