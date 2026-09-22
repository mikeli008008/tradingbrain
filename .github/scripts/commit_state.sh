#!/usr/bin/env bash
# Robust commit+push for state/ artifacts written by tradingbrain workflows.
# Usage: commit_state.sh "<commit message>"
# - Commits local state/ changes FIRST (avoids dirty-tree pull --rebase failure)
# - Rebases onto origin/main with retries
# - Pushes to main
# Exit 0 if nothing to commit. Exit non-zero only if push ultimately fails.
set -euo pipefail

MSG="${1:?commit message required}"
MAX_RETRIES="${COMMIT_STATE_RETRIES:-3}"

git config user.name "trading-brain"
git config user.email "bot@users.noreply.github.com"

git add state/
if git diff --quiet && git diff --staged --quiet; then
  echo "commit_state: no state/ changes to commit"
  exit 0
fi

git commit -m "$MSG"

# Prefer originating branch when present; scheduled runs are on detached HEAD → main
TARGET_REF="${GITHUB_REF_NAME:-main}"
if [[ "$TARGET_REF" == "main" ]] || [[ -z "$TARGET_REF" ]]; then
  TARGET_REF="main"
fi

attempt=1
while true; do
  echo "commit_state: pull --rebase attempt ${attempt}/${MAX_RETRIES}"
  if git pull --rebase origin main; then
    break
  fi
  if (( attempt >= MAX_RETRIES )); then
    echo "commit_state: rebase failed after ${MAX_RETRIES} attempts" >&2
    git rebase --abort 2>/dev/null || true
    exit 1
  fi
  git rebase --abort 2>/dev/null || true
  sleep $(( attempt * 5 ))
  attempt=$(( attempt + 1 ))
done

git push origin "HEAD:${TARGET_REF}"
echo "commit_state: pushed to origin/${TARGET_REF}"
