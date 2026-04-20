"""
Calibration Tracker.

The agent states a confidence on every shadow decision. Over time, we
check: when the agent says 70%, does it actually win 70%? Miscalibration
(over- or under-confidence) is the single most correctable bias.

Output: a calibration report injected into the agent's morning context
so it can adjust. E.g., "You said 80% on 45 decisions, actual hit rate
was 52% — you are significantly overconfident on high-conviction calls."
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
import json

from learning import shadow_journal as sj

CAL_DIR = Path(__file__).parent.parent / "state" / "calibration"
CAL_DIR.mkdir(parents=True, exist_ok=True)

# Confidence buckets (5 bins)
BUCKETS = [(0.0, 0.4), (0.4, 0.55), (0.55, 0.7), (0.7, 0.85), (0.85, 1.01)]


@dataclass
class BucketStat:
    low: float
    high: float
    n: int
    mean_stated: float
    actual_hit_rate: float
    calibration_error: float   # stated - actual (positive = overconfident)


def compute() -> dict[str, Any]:
    decisions = [
        d for d in sj.load_all()
        if d.outcome_correct is not None
    ]
    if not decisions:
        return {"ready": False, "reason": "no graded decisions yet"}

    stats = []
    for low, high in BUCKETS:
        bucket = [d for d in decisions if low <= d.confidence < high]
        if not bucket:
            continue
        mean_stated = sum(d.confidence for d in bucket) / len(bucket)
        actual = sum(1 for d in bucket if d.outcome_correct) / len(bucket)
        stats.append(BucketStat(
            low=low, high=high, n=len(bucket),
            mean_stated=round(mean_stated, 3),
            actual_hit_rate=round(actual, 3),
            calibration_error=round(mean_stated - actual, 3),
        ))

    # Brier score — overall calibration quality
    brier = sum(
        (d.confidence - (1 if d.outcome_correct else 0)) ** 2
        for d in decisions
    ) / len(decisions)

    # Overall bias
    mean_conf = sum(d.confidence for d in decisions) / len(decisions)
    mean_hit = sum(1 for d in decisions if d.outcome_correct) / len(decisions)

    report = {
        "ready": True,
        "n_decisions": len(decisions),
        "brier_score": round(brier, 4),          # lower = better, 0.25 = random
        "overall_bias": round(mean_conf - mean_hit, 3),
        "mean_confidence": round(mean_conf, 3),
        "mean_hit_rate": round(mean_hit, 3),
        "buckets": [
            {
                "range": f"{s.low:.0%}-{s.high:.0%}",
                "n": s.n,
                "stated": f"{s.mean_stated:.1%}",
                "actual": f"{s.actual_hit_rate:.1%}",
                "error": round(s.calibration_error, 3),
            }
            for s in stats
        ],
        "interpretation": _interpret(stats, mean_conf - mean_hit),
    }

    (CAL_DIR / f"{date.today().isoformat()}.json").write_text(
        json.dumps(report, indent=2)
    )
    return report


def _interpret(stats: list[BucketStat], overall_bias: float) -> str:
    if not stats:
        return "Not enough data."
    if overall_bias > 0.08:
        return (
            "SIGNIFICANTLY OVERCONFIDENT. Your stated confidence exceeds actual "
            "performance by > 8 pp. Dial down high-confidence calls and treat "
            "'80%+' setups as closer to 60%."
        )
    if overall_bias < -0.08:
        return (
            "UNDERCONFIDENT. You are winning more than you claim. You may be "
            "skipping good trades because you rate them below your threshold."
        )
    return "Reasonably calibrated overall. Keep reporting honest confidence."


def render_for_agent() -> str:
    """A paragraph the agent reads every morning."""
    r = compute()
    if not r.get("ready"):
        return "Calibration data not yet available."
    lines = [
        f"**Your calibration over {r['n_decisions']} decisions** "
        f"(Brier {r['brier_score']}, random=0.25):",
        r["interpretation"],
        "",
        "| Confidence | N | Stated | Actual |",
        "|---|---|---|---|",
    ]
    for b in r["buckets"]:
        lines.append(f"| {b['range']} | {b['n']} | {b['stated']} | {b['actual']} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render_for_agent())
