#!/usr/bin/env bash
# Persist pipeline state across ephemeral GitHub Actions runners.
#
# The runner is destroyed after every job, so the Parquet mirror and the model
# artefacts have nowhere to live between runs. Hopsworks is the primary store,
# but the mirror is what keeps the system working when Hopsworks is not
# reachable, so it needs somewhere durable too.
#
# A dedicated orphan branch is that place. It keeps binary churn out of main's
# history, needs no external service, and has a useful side effect: the commits
# count as repository activity, which stops GitHub from auto-disabling the
# scheduled workflows after 60 days of quiet.
set -euo pipefail

BRANCH="${DATA_BRANCH:-data}"
DIR="${DATA_DIR:-data}"

restore() {
  git fetch origin "$BRANCH" --depth=1 2>/dev/null || {
    echo "no $BRANCH branch yet; starting from empty state"
    mkdir -p "$DIR"; return 0
  }
  mkdir -p "$DIR"
  git checkout "origin/$BRANCH" -- "$DIR" 2>/dev/null || echo "no $DIR on $BRANCH yet"
  git reset -- "$DIR" >/dev/null 2>&1 || true
  echo "restored $(find "$DIR" -type f 2>/dev/null | wc -l) files from $BRANCH"
}

save() {
  local message="${1:-pipeline run}"
  if [ ! -d "$DIR" ] || [ -z "$(ls -A "$DIR" 2>/dev/null)" ]; then
    echo "nothing to save"; return 0
  fi
  # The HTTP response cache is a local speed-up for backfills, not state.
  rm -rf "${DIR:?}/http_cache"

  git config user.name "aqi-pipeline[bot]"
  git config user.email "aqi-pipeline@users.noreply.github.com"

  local tmp; tmp="$(mktemp -d)"
  cp -r "$DIR/." "$tmp/"

  if git fetch origin "$BRANCH" --depth=1 2>/dev/null; then
    git checkout -B "$BRANCH" "origin/$BRANCH"
  else
    git checkout --orphan "$BRANCH"
    git rm -rf --cached . >/dev/null 2>&1 || true
    find . -maxdepth 1 ! -name . ! -name .git ! -name "$DIR" -exec rm -rf {} + 2>/dev/null || true
  fi

  mkdir -p "$DIR"
  cp -r "$tmp/." "$DIR/"
  git add -f "$DIR"
  if git diff --cached --quiet; then
    echo "no changes to persist"; return 0
  fi
  git commit -m "data: $message [skip ci]"
  git push -f origin "$BRANCH"
  echo "persisted state to $BRANCH"
}

case "${1:-}" in
  restore) restore ;;
  save)    save "${2:-pipeline run}" ;;
  *) echo "usage: $0 {restore|save [message]}" >&2; exit 1 ;;
esac
