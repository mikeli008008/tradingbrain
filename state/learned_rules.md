# Learned Trading Rules

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
