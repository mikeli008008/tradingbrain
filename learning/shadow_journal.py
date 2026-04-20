"""
Shadow Journal — the core of the learning engine.

Every hour during market hours the agent records a 'shadow decision':
what it would do right now for each candidate on the screener, even
if it won't actually trade. Each decision is stamped with forward
horizons (T+1, T+5, T+20) and graded later by outcome_grader.

This is how we get ~35 decisions/week instead of ~3. The agent learns
from all of them — not just the ones that cost money.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
import json
import uuid

SHADOW_DIR = Path(__file__).parent.parent / "state" / "shadow"
SHADOW_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ShadowDecision:
    """One hypothetical trading decision + its thesis + tracked outcomes."""
    id: str
    timestamp: str           # ISO, hour-granular
    ticker: str
    action: Literal["buy", "watch", "skip"]
    confidence: float         # 0.0 - 1.0 — this is what calibration tracks
    signal_tier: str
    entry_price: float
    stop_price: float | None
    target_price: float | None

    # Thesis — the agent's reasoning, for learning
    thesis: str               # 1-2 sentence plain English
    thesis_tags: list[str]    # ["catalyst_earnings", "vcp_breakout", "rs_leader", ...]
    market_regime: str        # "strong" | "transitional" | "weak"

    # Outcomes (filled in later by grader)
    price_at_t1: float | None = None
    price_at_t5: float | None = None
    price_at_t20: float | None = None
    pnl_pct_t1: float | None = None
    pnl_pct_t5: float | None = None
    pnl_pct_t20: float | None = None
    stopped_out: bool | None = None       # Did it hit stop in 20 days?
    reached_target: bool | None = None    # Did it hit target in 20 days?
    graded_at: str | None = None

    # Forensic labels (filled by forensics loop)
    thesis_correct: bool | None = None    # Did the WHY play out?
    outcome_correct: bool | None = None   # Did the P&L play out?


def record(decision: dict[str, Any]) -> ShadowDecision:
    """Agent-callable: persist a shadow decision."""
    d = ShadowDecision(
        id=str(uuid.uuid4())[:8],
        timestamp=datetime.utcnow().isoformat(timespec="seconds"),
        **decision,
    )
    path = SHADOW_DIR / f"{date.today().isoformat()}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(asdict(d)) + "\n")
    return d


def load_all() -> list[ShadowDecision]:
    """Load every shadow decision ever recorded."""
    out = []
    for p in sorted(SHADOW_DIR.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                out.append(ShadowDecision(**json.loads(line)))
    return out


def load_since(days_ago: int) -> list[ShadowDecision]:
    """Load decisions from the last N days."""
    cutoff = date.today() - timedelta(days=days_ago)
    out = []
    for p in sorted(SHADOW_DIR.glob("*.jsonl")):
        try:
            file_date = date.fromisoformat(p.stem)
        except ValueError:
            continue
        if file_date < cutoff:
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                out.append(ShadowDecision(**json.loads(line)))
    return out


def load_pending_grade(horizon_days: int) -> list[tuple[Path, int, ShadowDecision]]:
    """Decisions old enough to grade at this horizon but not yet graded."""
    cutoff = date.today() - timedelta(days=horizon_days)
    field_name = f"pnl_pct_t{horizon_days}"
    out = []
    for p in sorted(SHADOW_DIR.glob("*.jsonl")):
        try:
            file_date = date.fromisoformat(p.stem)
        except ValueError:
            continue
        if file_date > cutoff:
            continue
        lines = p.read_text().splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get(field_name) is None:
                out.append((p, i, ShadowDecision(**raw)))
    return out


def update_decision(path: Path, line_idx: int, decision: ShadowDecision) -> None:
    """Overwrite one line in a shadow jsonl with updated outcome fields."""
    lines = path.read_text().splitlines()
    lines[line_idx] = json.dumps(asdict(decision))
    path.write_text("\n".join(lines) + "\n")
