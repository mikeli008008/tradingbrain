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

- **2026-04-27**: Before entering any position, verify earnings date is >7 days out AND >14 days from today if possible. The 7-day rule applies at entry, but if a name reports 8-14 days out, your trade has no time to build a cushion before binary risk arrives. Aim for fresh entries with at least 2-3 weeks of runway before earnings.
  - Reason: Today I had to close TER (+2.37%) and APH (+0.7%) because they reported within 1-2 days with insufficient cushion. The trades were technically compliant with the 7-day rule at entry but provided no time to build a profit buffer. Forcing premature exits on otherwise valid setups.

- **2026-04-29**: Run earnings_on_deck() against ALL open positions at the start of every session, not just candidates. Any position reporting within 24-48 hours must be exited unless there is a substantial profit cushion (>10%) AND a clear bullish thesis for the print. The 7-day rule applies at entry; at exit, the standard tightens to 1-2 days.
  - Reason: Today APH was carried into its earnings date because I checked earnings only on new candidates, not on existing holdings. This is a process gap, not bad luck — it will repeat unless codified. Yesterday's KLAC and today's APH both came uncomfortably close to binary risk because I wasn't sweeping the book daily.
