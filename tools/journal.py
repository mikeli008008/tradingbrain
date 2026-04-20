"""
Journal — the cross-session memory layer.

The core challenge of long-running agents is they start each session
with no memory. Our fix: the agent writes a structured journal each day
and loads yesterday's journal at the start of today's session.

The journal is also the '迭代' loop: the agent reviews past decisions,
flags mistakes, updates its own trading rules.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path
import json
from typing import Any

JOURNAL_DIR = Path(__file__).parent.parent / "state" / "journal"
JOURNAL_DIR.mkdir(parents=True, exist_ok=True)

RULES_FILE = Path(__file__).parent.parent / "state" / "learned_rules.md"


def write_journal(day: date, entry: dict[str, Any]) -> None:
    """Write the day's plan + outcome."""
    path = JOURNAL_DIR / f"{day.isoformat()}.json"
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text())
    existing.update(entry)
    path.write_text(json.dumps(existing, indent=2, default=str))


def read_recent_journals(n_days: int = 5) -> list[dict[str, Any]]:
    """Load last N days of journals for context."""
    today = date.today()
    out = []
    for i in range(1, n_days + 1):
        d = today - timedelta(days=i)
        path = JOURNAL_DIR / f"{d.isoformat()}.json"
        if path.exists():
            out.append(json.loads(path.read_text()))
    return out


def read_learned_rules() -> str:
    """The agent's self-authored rulebook. Persists across sessions."""
    if not RULES_FILE.exists():
        RULES_FILE.write_text(SEED_RULES)
    return RULES_FILE.read_text()


def append_learned_rule(rule: str, reason: str) -> dict[str, Any]:
    """Agent-callable: adds a new rule after a lesson."""
    current = read_learned_rules()
    addition = f"\n- **{date.today().isoformat()}**: {rule}\n  - Reason: {reason}\n"
    RULES_FILE.write_text(current + addition)
    return {"added": True, "rule": rule}


SEED_RULES = """# Learned Trading Rules

These are rules this agent has committed to after observing its own behavior.
Rules here override generic reasoning — if a rule says "do not X", do not X.

## Baseline (from Minervini SEPA)

- Never enter a position without a stop price defined upfront
- Never average down on a losing position
- Cut losses at -7% to -8% from entry, no exceptions
- Risk ≤ 0.75% of account per trade
- Maximum 5 concurrent positions
- No new entries if SPY broke MA50 on high volume
- Halve position sizes when SPY is below MA200
- Avoid new entries within 7 days of a company's earnings date

## Self-authored rules

(The agent appends rules below as it learns from mistakes)
"""


def summarize_yesterday() -> str:
    """Human-readable recap for the agent's morning briefing."""
    yesterday = date.today() - timedelta(days=1)
    path = JOURNAL_DIR / f"{yesterday.isoformat()}.json"
    if not path.exists():
        return "No journal entry from yesterday."
    j = json.loads(path.read_text())
    parts = [f"**Yesterday ({yesterday}):**"]
    if "plan" in j:
        parts.append(f"- Plan: {j['plan'].get('summary', 'n/a')}")
    if "trades_executed" in j:
        parts.append(f"- Trades executed: {len(j['trades_executed'])}")
        for t in j["trades_executed"]:
            parts.append(f"  - {t.get('ticker')}: {t.get('outcome', 'pending')}")
    if "reflection" in j:
        parts.append(f"- Reflection: {j['reflection']}")
    return "\n".join(parts)
