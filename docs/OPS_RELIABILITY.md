# Ops reliability — paper/shadow redesign (2026-09-22)

## Why this exists

Nightly Learning was failing every weekday after successful grade/mine/calibrate
because `git pull --rebase` ran on a dirty `state/` tree. PR #2 fixed that
ordering. Remaining fragility:

1. Multiple workflows still race on `state/` (no concurrency lock).
2. If git push fails after learning, artifacts disappear with the runner.
3. Daily Trading Agent was disabled_manually (~2026-09-10) for week 1–2
   shadow-only ramp — and its schedule path did **not** force `--dry-run`.
4. Overnight Sentinel checked `trading-brain/state/EMERGENCY_TRIGGER` (wrong
   path); the Python writer uses `state/EMERGENCY_TRIGGER`.

## What we changed

| Change | Files |
|--------|-------|
| Shared concurrency group `tradingbrain-state` (no cancel) | all `.github/workflows/*.yml` |
| Shared `commit_state.sh` (commit → rebase w/ retry → push) | `.github/scripts/commit_state.sh` |
| Upload `state/` artifacts even if git fails (`if: always()`) | nightly, hourly, sentinel, daily |
| Daily: schedule always dry-run; refuse orders when `ramp.mode=shadow_only` | `daily.yml`, `state/ramp.json` |
| Fix emergency trigger path check | `overnight_sentinel.yml` |
| Docs: re-enable criteria, schedule map | this file + README |

## Schedule map (UTC, summer ET)

| Workflow | Cron | Purpose | Touches broker? |
|----------|------|---------|-----------------|
| Hourly Shadow | `30 14-20 * * 1-5` | Record would-be decisions | No |
| Overnight Sentinel | `0 23 * * 1-5` + `0 12 * * 2-6` | News / catastrophic scan | Read-only (+ emergency agent if triggered, paper) |
| Nightly Learning | `30 23 * * 1-5` | Grade / mine / calibrate | No |
| Daily Trading Agent | `15 20 * * 1-5` | Portfolio cycle + optional LLM journal | **Yes if not dry-run** — keep disabled until criteria below |

Concurrency serializes these so two writers never rebase/push `state/` at once.

## Ramp (`state/ramp.json`)

Current (do not advance without Han):

```json
{
  "mode": "shadow_only",
  "alpaca_paper": true,
  "phase": "week_1_2_shadow_learning"
}
```

Modes (advance in order):

1. `shadow_only` — hourly + nightly + sentinel only. Daily must stay disabled or dry-run.
2. `dry_run` — Daily may be enabled; portfolio manager runs with `--dry-run` only.
3. `paper_trading` — Daily may place Alpaca **paper** orders (`ALPACA_PAPER=true` forever until explicit live decision).
4. `live` — **blocked by policy in workflows**; requires a separate intentional change.

Paper execution via TradingView is out of band (alpha stays in tradingbrain/Grok).
Do not wire live broker keys into these workflows for that path.

## When to re-enable Daily Trading Agent

Do **not** re-enable until **all** of:

1. Nightly Learning has succeeded (green commit step) for ≥5 consecutive weekdays.
2. `state/forensics/` has fresh files and `process_quality` ≥ **0.50** (current calibration was ~0.29 / overconfident).
3. Buy hit rate in calibration is not dominated by overconfidence (bias magnitude &lt; ~0.10 preferred).
4. `state/ramp.json` `mode` advanced to at least `dry_run`.
5. First re-enable week: workflow_dispatch with `dry_run=true` only; inspect `state/portfolio/cycle_*.json`.
6. Second week: advance ramp to `paper_trading`, keep `ALPACA_PAPER` secret unset or `true`.

Re-enable path: GitHub → Actions → Daily Trading Agent → Enable workflow.
Do not merge a PR that flips the workflow to live or sets `ALPACA_PAPER=false`.

## Recovering lost learning from artifacts

If a Nightly Learning run fails at Commit but Upload succeeded:

1. Actions → run → Artifacts → `learning-state-<run_id>`
2. Unpack over a checkout, review diffs, commit manually or re-run workflow_dispatch.

## Known non-goals (this redesign)

- No greenfield agent rewrite.
- No live trading enablement.
- No TradingView broker integration in-repo (paper exec stays external).
- No change to promotion statistics thresholds.
