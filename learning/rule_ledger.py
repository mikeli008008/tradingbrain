"""
Rule Ledger — the A/B system that gates rule promotion.

Flow:
  1. Pattern Miner (below) proposes candidate rules from shadow journal
  2. Each candidate gets BACKTESTED against historical decisions
  3. Only promotes if:
       - ≥ 20 decisions matched the rule's condition
       - Effect size is meaningful (>10% hit rate delta)
       - Statistical significance (binomial test, p<0.10)
  4. Active rules get re-audited every 30 days against fresh data;
     if performance degrades, they're demoted.

This is what makes "aggressive auto-promotion" safe: the gate isn't
the agent's opinion — it's empirical evidence.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Callable
import json
import uuid
import math

from learning import shadow_journal as sj

RULES_DIR = Path(__file__).parent.parent / "state" / "rules"
RULES_DIR.mkdir(parents=True, exist_ok=True)
RULES_FILE = RULES_DIR / "ledger.json"


@dataclass
class Rule:
    id: str
    description: str                      # Plain English for the agent
    condition_tags: list[str]             # Shadow decisions with these tags trigger it
    prediction: Literal["favorable", "unfavorable"]  # 'favorable' = take the trade, 'unfavorable' = skip
    status: Literal["candidate", "provisional", "active", "demoted"]
    created_at: str
    promoted_at: str | None = None
    demoted_at: str | None = None

    # Evidence at promotion
    n_supporting: int = 0                 # How many decisions matched
    hit_rate_with_rule: float | None = None
    hit_rate_baseline: float | None = None
    p_value: float | None = None

    # Post-promotion tracking
    n_since_promotion: int = 0
    hit_rate_since_promotion: float | None = None
    notes: list[str] = field(default_factory=list)

    # Bootstrap-specific fields
    source: Literal["live", "bootstrap"] = "live"
    regime_stratification: str | None = None
    live_confirmations: int = 0            # How many live decisions confirmed this
    live_confirmations_needed: int = 10    # Before provisional → active


def load_ledger() -> list[Rule]:
    if not RULES_FILE.exists():
        return []
    return [Rule(**r) for r in json.loads(RULES_FILE.read_text())]


def save_ledger(rules: list[Rule]) -> None:
    RULES_FILE.write_text(json.dumps([asdict(r) for r in rules], indent=2))


# --- Statistics ---

def _binomial_p_value(k: int, n: int, p0: float) -> float:
    """
    Two-tailed binomial test: what's the prob of seeing k or more hits
    out of n trials if true rate were p0?

    Uses normal approximation (valid for n > 20).
    """
    if n < 5:
        return 1.0
    expected = n * p0
    variance = n * p0 * (1 - p0)
    if variance <= 0:
        return 1.0
    z = abs(k - expected) / math.sqrt(variance)
    # Two-tailed p from standard normal
    p = 2 * (1 - _phi(z))
    return p


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# --- Pattern mining ---

def mine_patterns(min_support: int = 20, lookback_days: int = 60) -> list[dict[str, Any]]:
    """
    Look for thesis tags that systematically over- or under-perform
    the overall hit rate. Propose them as candidate rules.
    """
    decisions = [
        d for d in sj.load_since(lookback_days)
        if d.outcome_correct is not None
    ]
    if len(decisions) < min_support:
        return []

    baseline_hit = sum(1 for d in decisions if d.outcome_correct) / len(decisions)

    # Group by each tag
    by_tag: dict[str, list[sj.ShadowDecision]] = {}
    for d in decisions:
        for tag in (d.thesis_tags or []):
            by_tag.setdefault(tag, []).append(d)

    proposals = []
    for tag, dcs in by_tag.items():
        if len(dcs) < min_support:
            continue
        hits = sum(1 for d in dcs if d.outcome_correct)
        rate = hits / len(dcs)
        delta = rate - baseline_hit
        if abs(delta) < 0.10:
            continue  # effect too small to care about
        p = _binomial_p_value(hits, len(dcs), baseline_hit)
        if p > 0.10:
            continue  # not significant enough
        proposals.append({
            "tag": tag,
            "n_supporting": len(dcs),
            "rate_with": round(rate, 3),
            "rate_baseline": round(baseline_hit, 3),
            "delta": round(delta, 3),
            "p_value": round(p, 4),
            "prediction": "favorable" if delta > 0 else "unfavorable",
        })

    return proposals


def promote_proposals(proposals: list[dict[str, Any]]) -> list[Rule]:
    """Promote qualifying proposals to candidate status in the ledger."""
    existing = load_ledger()
    existing_tags = {tuple(sorted(r.condition_tags)): r for r in existing if r.status == "active"}
    new_rules = []
    today = date.today().isoformat()

    for p in proposals:
        key = (p["tag"],)
        if key in existing_tags:
            continue  # already an active rule covers this

        direction = (
            f"PREFER trading setups tagged '{p['tag']}'"
            if p["prediction"] == "favorable"
            else f"AVOID or deprioritize setups tagged '{p['tag']}'"
        )
        rule = Rule(
            id=str(uuid.uuid4())[:8],
            description=(
                f"{direction} — empirically {p['rate_with']:.0%} win rate vs "
                f"{p['rate_baseline']:.0%} baseline over {p['n_supporting']} decisions "
                f"(p={p['p_value']})"
            ),
            condition_tags=[p["tag"]],
            prediction=p["prediction"],
            status="active",   # aggressive policy: skip candidate stage
            created_at=today,
            promoted_at=today,
            n_supporting=p["n_supporting"],
            hit_rate_with_rule=p["rate_with"],
            hit_rate_baseline=p["rate_baseline"],
            p_value=p["p_value"],
        )
        new_rules.append(rule)

    if new_rules:
        existing.extend(new_rules)
        save_ledger(existing)
    return new_rules


def audit_active_rules() -> list[dict[str, Any]]:
    """
    Re-check every active rule. If decisions since promotion show the
    rule no longer holds (hit rate reverted toward baseline), demote it.
    """
    rules = load_ledger()
    changes = []

    for r in rules:
        if r.status != "active" or not r.promoted_at:
            continue
        promoted = date.fromisoformat(r.promoted_at)
        post_promotion = [
            d for d in sj.load_all()
            if d.outcome_correct is not None
            and date.fromisoformat(d.timestamp[:10]) > promoted
            and any(tag in (d.thesis_tags or []) for tag in r.condition_tags)
        ]
        if len(post_promotion) < 10:
            continue  # not enough new data to judge

        hits = sum(1 for d in post_promotion if d.outcome_correct)
        rate = hits / len(post_promotion)
        r.n_since_promotion = len(post_promotion)
        r.hit_rate_since_promotion = round(rate, 3)

        # Demotion: rate reverted to baseline OR flipped direction
        baseline = r.hit_rate_baseline or 0.5
        if r.prediction == "favorable" and rate < baseline + 0.03:
            r.status = "demoted"
            r.demoted_at = date.today().isoformat()
            r.notes.append(f"Demoted: post-promotion rate {rate:.2%} collapsed to baseline")
            changes.append({"id": r.id, "change": "demoted", "tag": r.condition_tags})
        elif r.prediction == "unfavorable" and rate > baseline - 0.03:
            r.status = "demoted"
            r.demoted_at = date.today().isoformat()
            r.notes.append(f"Demoted: post-promotion rate {rate:.2%} collapsed to baseline")
            changes.append({"id": r.id, "change": "demoted", "tag": r.condition_tags})

    save_ledger(rules)
    return changes


def render_active_rules_for_agent() -> str:
    """Format active rules for injection into the agent's system prompt."""
    rules = load_ledger()
    active = [r for r in rules if r.status == "active"]
    provisional = [r for r in rules if r.status == "provisional"]

    if not active and not provisional:
        return "No data-backed rules yet."

    lines = []
    if active:
        lines.append("# Active evidence-backed rules (highest weight)\n")
        for r in active:
            tag = "🧪 bootstrap" if r.source == "bootstrap" else "✓ live"
            lines.append(f"- **[{r.id}]** {tag} — {r.description}")

    if provisional:
        lines.append("\n# Provisional rules from historical backtest (lower weight)")
        lines.append(f"*These emerged from 2015-present data. Weight them lower until live-confirmed "
                     f"({min(r.live_confirmations_needed for r in provisional)} decisions needed).*\n")
        for r in provisional:
            regime_note = f" [regime: {r.regime_stratification}]" if r.regime_stratification else ""
            lines.append(f"- **[{r.id}]** {r.description}{regime_note} "
                         f"(confirmed {r.live_confirmations}/{r.live_confirmations_needed})")

    return "\n".join(lines)


def confirm_provisional_rules() -> list[dict]:
    """
    Walk recent live-shadow decisions, increment confirmation counts on
    provisional rules whose tags were present AND whose prediction matched
    the outcome. Promote to active once threshold reached.
    """
    from learning import shadow_journal as sj
    rules = load_ledger()
    provisional = [r for r in rules if r.status == "provisional"]
    if not provisional:
        return []

    # Look at decisions since each rule's creation
    changes = []
    for r in provisional:
        created = date.fromisoformat(r.created_at)
        matching = [
            d for d in sj.load_all()
            if d.outcome_correct is not None
            and date.fromisoformat(d.timestamp[:10]) > created
            and any(tag in (d.thesis_tags or []) for tag in r.condition_tags)
        ]
        if not matching:
            continue

        # Count confirmations: decisions where prediction aligns with outcome
        confirmations = 0
        for d in matching:
            if r.prediction == "favorable" and d.outcome_correct:
                confirmations += 1
            elif r.prediction == "unfavorable" and not d.outcome_correct:
                confirmations += 1

        r.live_confirmations = confirmations
        if confirmations >= r.live_confirmations_needed:
            r.status = "active"
            r.promoted_at = date.today().isoformat()
            r.notes.append(
                f"Promoted from provisional: {confirmations} live confirmations"
            )
            changes.append({
                "id": r.id,
                "change": "promoted_to_active",
                "confirmations": confirmations,
            })

    save_ledger(rules)
    return changes


def import_bootstrap_rules(bootstrap_rules_path: str | None = None) -> int:
    """
    Import rules from bootstrap_trainer output into the ledger as provisional.
    Idempotent: re-importing won't create duplicates.
    """
    import uuid
    path = Path(bootstrap_rules_path) if bootstrap_rules_path else (
        RULES_DIR.parent / "bootstrap" / "provisional_rules.json"
    )
    if not path.exists():
        return 0

    bootstrap_rules = json.loads(path.read_text())
    existing = load_ledger()
    existing_keys = {
        (tuple(sorted(r.condition_tags)), r.prediction, r.source)
        for r in existing
    }

    added = 0
    for b in bootstrap_rules:
        key = ((b["tag"],), b["prediction"], "bootstrap")
        if key in existing_keys:
            continue

        direction = (
            f"PREFER tag '{b['tag']}'"
            if b["prediction"] == "favorable"
            else f"AVOID tag '{b['tag']}'"
        )
        rule = Rule(
            id=str(uuid.uuid4())[:8],
            description=(
                f"{direction} — historical win rate {b['hit_rate_with_rule']:.0%} vs "
                f"{b['hit_rate_baseline']:.0%} baseline, n={b['n_supporting']}, "
                f"p={b['p_value']} (discovered {b['discovered_at']})"
            ),
            condition_tags=[b["tag"]],
            prediction=b["prediction"],
            status="provisional",
            created_at=date.today().isoformat(),
            n_supporting=b["n_supporting"],
            hit_rate_with_rule=b["hit_rate_with_rule"],
            hit_rate_baseline=b["hit_rate_baseline"],
            p_value=b["p_value"],
            source="bootstrap",
            regime_stratification=b.get("regime_stratification"),
            notes=["Imported from walk-forward bootstrap. Requires live confirmation."],
        )
        existing.append(rule)
        added += 1

    save_ledger(existing)
    return added


def run_cycle() -> dict[str, Any]:
    """The nightly cycle: confirm provisional → mine → promote → audit."""
    confirmations = confirm_provisional_rules()
    audit_changes = audit_active_rules()
    proposals = mine_patterns()
    new_rules = promote_proposals(proposals)
    return {
        "provisional_confirmations": confirmations,
        "demoted": audit_changes,
        "proposals_found": len(proposals),
        "rules_promoted": [asdict(r) for r in new_rules],
    }


if __name__ == "__main__":
    print(json.dumps(run_cycle(), indent=2, default=str))
