# Trading Brain 🧠 — Learning Edition

An autonomous AI trading agent running the Minervini SEPA strategy, engineered for **maximum rate of self-improvement** while preserving capital.

**Strategy source**: [mikeli008008/minervini-screener](https://github.com/mikeli008008/minervini-screener)
**Runtime**: Claude Agent SDK (Anthropic API)
**Broker**: Alpaca (paper-first)
**Scheduler**: GitHub Actions

---

## Why this isn't just another LLM trading bot

Most LLM trading agents have one loop: decide → execute → maybe reflect next run. That gives you ~3 data points per week. This one runs **five parallel loops** so the brain gets ~35 training signals per week — without risking money on them.

### The Five Loops

```
┌─ Loop 1: SHADOW DECISIONS (hourly, market hours)
│   Agent records what it WOULD do, does not trade. ~35/week.
│   → state/shadow/YYYY-MM-DD.jsonl
│
├─ Loop 2: OUTCOME GRADING (nightly, T+1/T+5/T+20)
│   Pulls actual market prices, scores every shadow decision.
│   Labels outcome_correct AND thesis_correct separately.
│   → state/forensics/
│
├─ Loop 3: PATTERN MINING (nightly)
│   Groups decisions by thesis tag, finds tags with statistically
│   significant edge (p<0.10, |Δ| >10%, n≥20).
│   → Promotes to active rules if significant
│   → state/rules/ledger.json
│
├─ Loop 4: RULE AUDIT (nightly)
│   Re-tests active rules against post-promotion decisions.
│   Demotes rules whose edge collapsed (regime change / overfit).
│
├─ Loop 5: CALIBRATION (nightly)
│   Measures stated confidence vs actual hit rate by bucket.
│   Reports over/underconfidence to the agent next morning.
│   → state/calibration/
│
└─ Loop 6: OVERNIGHT SENTINEL (7pm ET + 8am ET)
    Scans open positions + watchlist for catastrophic news.
    Triggers emergency agent if SEC / fraud / bankruptcy / etc.
    Emergency agent cancels resting stops and preps market exits.
    → state/overnight_alerts/ and state/emergency_actions.jsonl
```

Each morning the main trading agent wakes up with yesterday's journal, active evidence-backed rules (loop 3-4 output), the calibration report (e.g., "you are 8pp overconfident on 70%+ calls"), hand-written rules, and a recent shadow activity summary.

### The thesis_correct vs outcome_correct insight

This is the single most important thing in the learning layer. We grade two labels per decision:

- **outcome_correct** — did the P&L agree with the action?
- **thesis_correct** — did the *mechanism* the agent named actually play out?

You can be outcome-right for thesis-wrong reasons (lucky) or outcome-wrong for thesis-right reasons (good process, bad luck). These demand opposite lessons. Most naive "reflection" loops collapse both into one signal and teach the wrong lesson half the time.

Forensics reports a four-quadrant matrix: skilled_win (both correct), lucky_win (outcome only), unlucky_loss (thesis only), mistake (both wrong). **`process_quality = (skilled_win + unlucky_loss) / total`** is your leading indicator — it stabilizes faster than P&L.

---

## Aggressive promotion, safe promotion

You chose aggressive auto-promotion. To make that safe, the promotion gate has three filters:

1. **n ≥ 20** matching decisions (avoid small-sample noise)
2. **|Δ| > 10%** absolute hit-rate difference (effect size floor)
3. **p < 0.10** binomial test vs baseline (statistical significance)

Plus a **demotion mechanism** — every active rule re-audits against post-promotion decisions. If its edge collapsed, it gets demoted automatically. Catches overfit rules and regime-specific rules that stopped working.

**Tested on simulated data** with two real patterns (75% and 35% hit rates) and one null pattern: correctly promoted both real rules, ignored the null, and correctly flagged seeded overconfidence.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  GitHub Actions (3 schedules)                                   │
│  • daily.yml @ 16:15 ET — main trading agent                    │
│  • hourly_shadow.yml @ hourly mkt hrs — shadow decisions        │
│  • nightly_learning.yml @ 23:00 UTC — grade + mine + audit      │
└─────────────────────────────────────────────────────────────────┘
         ↓                     ↓                    ↓
  agent/run.py         agent/shadow_run.py   learning/*.py
  (Claude Opus 4.7)    (Claude Haiku 4.5)   (pure statistics, no LLM)
         ↓                     ↓                    ↓
  ┌──────────────────────────────────────────────────────┐
  │  Tools (agent's senses + hands)                      │
  │  market_data, news_research, broker, journal         │
  └──────────────────────────────────────────────────────┘
         ↓
  ┌──────────────────────────────────────────────────────┐
  │  harness/risk_manager.py  ← HARD GATE                │
  │  Only the main agent goes through here.              │
  │  Shadow decisions never touch this — no capital risk.│
  └──────────────────────────────────────────────────────┘
         ↓
   Alpaca (paper → live only after validation)
```

---

## Multi-strategy architecture

The brain now runs multiple trading strategies in parallel, each with its own rule ledger, capital allocation, and risk profile. The portfolio manager coordinates them.

### Current strategies

**Minervini SEPA** (`strategies/minervini.py`)
- Daily cadence (every weekday)
- Concentrated breakout strategy: max 5 positions
- Strict 7-8% stops, 0.75% per-trade risk
- Reduces in weak SPY regime

**Cross-sectional momentum** (`strategies/cs_momentum.py`)
- Bi-weekly cadence (every other Friday)
- Diversified rotation: top 10 by 12-1 risk-adjusted momentum
- Partial rebalance capped at 30% per cycle (avoid whipsaw)
- No hard stops — exits via rotation
- Configurable: edit `strategies/cs_momentum_config.json` to tune `top_n`, `lookback_months`, `skip_months`, `max_drift`

### Capital allocation

Configured in `state/portfolio/config.json`:
```json
{
  "allocations": {"minervini": 0.5, "cs_momentum": 0.5},
  "max_portfolio_drawdown_daily": 0.03,
  "conflict_resolution": "first_wins"
}
```

Each strategy treats its allocation as its sub-account for position sizing. Minervini's 0.75%-per-trade is 0.75% of its 50% slice, not the whole account.

### Conflict resolution

When two strategies want the same ticker, "first_wins" — Minervini runs first, so its picks block XS momentum from buying the same name. The skip is logged in the cycle summary so you can see how often it happens.

### Portfolio-level circuit breakers

Independent of per-strategy daily loss limits, the portfolio manager halts ALL strategies if combined daily P&L breaches -3%. This catches correlated failure (e.g., a market-wide crash hitting both books at once).

### Adding a third strategy

1. Create `strategies/your_strategy.py` subclassing `BaseStrategy`
2. Add `@register_strategy` decorator
3. Implement `journal_tags()`, `should_run_today()`, `scan()`, `plan()`, optionally `regime_bias()`
4. Add to `state/portfolio/config.json` allocations (rebalance the existing weights)
5. Add to `agent/portfolio_manager.py` import line and the strategy iteration list

Each strategy gets its own shadow journal namespace (`state/shadow/{strategy_id}/`), its own rule ledger, and its own risk config preset. Pattern miner only finds patterns within a strategy's own data — no cross-contamination.

---



Same as base version (see prior README) plus:

### Broker selection

Set the `BROKER` secret to `alpaca` or `ibkr`.

**For IBKR paper trading** (your chosen starting point):

IBKR requires TWS or IB Gateway running somewhere reachable. GitHub Actions runners can't reach your home machine, so you have two options:

**Option A — Deploy IB Gateway on a small VM (recommended):**
1. Spin up a $5/mo DigitalOcean or Hetzner VM
2. Install IB Gateway in paper mode, enable API (port 4002)
3. Use `ngrok` or a Tailscale tunnel to expose the port privately
4. Set secrets:
   - `BROKER=ibkr`
   - `IBKR_HOST=<tunnel address>`
   - `IBKR_PORT=4002`
   - `IBKR_CLIENT_ID=1`

**Option B — Run Gateway locally + use self-hosted runner:**
If you don't want a VM, register your own Mac as a GitHub Actions self-hosted runner. Slightly more involved, but no VM cost. IB Gateway runs on your Mac; the runner on your Mac triggers when GitHub schedules fire.

IBKR paper accounts reset weekly on Sundays — historical journal data survives, but position state doesn't. Plan around this.

### Cost estimate

- Daily main agent (Opus): ~$0.20-0.50/run × 5/wk = ~$1-2.50/wk
- Hourly shadow (Haiku): ~$0.05-0.10/run × 35/wk = ~$2-4/wk
- Nightly learning: $0 (pure statistics, no LLM)

**Total: ~$3-7/week** at current pricing.

### Recommended ramp

- **Day 0**: Run bootstrap trainer once — seeds rule ledger with historical patterns
- **Week 1-2**: Only hourly shadow + nightly learning. No trading. Inspect `state/shadow/` by hand.
- **Week 3-4**: Enable daily agent in dry-run mode. Verify risk gates.
- **Week 5-6**: Paper trading. Watch `process_quality` in forensics.
- **Week 7+**: Bootstrap rules start getting live-confirmed (10 matching live decisions → provisional becomes active).
- **Month 3+**: Consider live if process_quality > 0.6 and |calibration bias| < 0.05.

---

## The bootstrap trainer (one-time offline run)

Instead of waiting 4-6 weeks of live shadow for the pattern miner to have enough data, run the bootstrap trainer once — it replays 10 years of historical prices through the same logic and seeds the ledger with statistically-validated rules from the start.

```bash
cd trading-brain
python -m learning.bootstrap_trainer --start-year 2015 --end-year 2025
```

Takes ~30 minutes on a decent machine (mostly yfinance downloads; cached afterward). Produces `state/bootstrap/decisions.jsonl` (~30-50k labeled decisions), `state/bootstrap/provisional_rules.json` (historically-validated rules), and a run summary.

Then import into the live ledger:

```python
from learning.rule_ledger import import_bootstrap_rules
import_bootstrap_rules()
```

### How the bootstrap stays honest (critical)

Three safeguards make this not-a-lying-backtest:

1. **Walk-forward mining** — when mining rules for period T, the miner only sees data from before T. A rule "discovered 2020-01-31" only used 2015-2019 data. No lookahead bias.
2. **Regime stratification** — every rule gets tagged with the regime it emerged from (bull / bear / transitional / high-vol). When live trading starts in a different regime, the agent weights these lower.
3. **Provisional status** — bootstrap rules don't activate immediately. They stay provisional until 10 live-shadow decisions confirm the pattern in current market conditions. Only then do they promote to active and influence the daily agent's decisions.

### Known limitations of the bootstrap

- **Survivorship bias** — uses current S&P composition, not point-in-time. Companies that went bankrupt are missing. This overstates historical edge.
- **Deterministic tags** — the bootstrap uses rule-based tags (e.g., `vcp_breakout` = price within 3% of 8w high after 10%+ pullback) rather than LLM-generated tags. They're consistent and cheap but less nuanced than the live agent's tags.
- **RS rating skipped** — the bootstrap's 8th trend check always passes. The live agent has proper RS from the screener.
- **yfinance data quality** — occasional errors, split adjustments assumed correct.

Interpret bootstrap rules as "here's where to look, not here's the answer." The live shadow journal is the ground truth.

---

## State directory — what to read weekly

```
state/
├── shadow/YYYY-MM-DD.jsonl      # Hourly shadow decisions
├── forensics/YYYY-MM-DD.json    # Daily forensics summary ← read this
├── rules/ledger.json            # Active / demoted rules ← read this
├── calibration/YYYY-MM-DD.json  # Daily calibration report
├── journal/YYYY-MM-DD.json      # Main agent's plan + reflection
├── traces/YYYY-MM-DD.json       # Agent tool-call traces
├── risk_log_*.jsonl             # Risk manager audit trail
└── learned_rules.md             # Agent's narrative rulebook
```

The two most valuable: `forensics/{latest}.json` tells you if the agent is actually improving; `rules/ledger.json` tells you what it learned.

---

## Known limitations

- **Single-tag mining** — won't find "VCP breakout AND weak regime → skip" conjunctions. Extension: mine pairs of tags.
- **Tags are LLM-generated** — if the agent stops using a tag, patterns for it stop being minable. Prompt pins the vocabulary but drift is possible.
- **No regime-conditional rules** — a bull-market rule may fail in a bear. Demotion catches this after the fact, not preemptively.
- **Shadow ≠ real** — shadow decisions don't include slippage, spread, or execution pressure. Learning transfers imperfectly.

---

## Disclaimer

Infrastructure, not investment advice. Shadow learning tells you the agent is getting better at *predicting*. It does not tell you it will make money at *trading*. Those are different things. Paper-trade until boring.
