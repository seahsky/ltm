#!/bin/bash
# scripts/notify-run.sh — run any command, then email the run report.
#
# Generic wrapper for the race-*.sh drivers (zero edits to the drivers):
# tees the wrapped command's output to runs/notify-<tag>-<ts>.log and, on
# EXIT (normal finish, crash, Ctrl-C/SIGTERM), calls scripts/notify_email.py
# which emails a markdown report + the gzipped log via Resend.
#
# On RACE, launch under nohup (or inside tmux) so the run AND the
# notification survive the SSH session disconnecting:
#
#   nohup bash scripts/notify-run.sh bash scripts/race-revisit.sh --tag wide-1 &
#   tail -f runs/notify-wide-1-*.log
#
# Or via the alias from `source scripts/race-setup.sh`:
#
#   nohup nrun bash scripts/race-revisit.sh --tag wide-1 &
#
# Config: RESEND_API_KEY / NOTIFY_EMAIL_TO in .env at the repo root (see
# .env.example). Unconfigured -> the run still works, just no email.
#
# Exit code: ALWAYS the wrapped command's exit code — a notifier failure
# never changes it.
#
# Env knobs:
#   NOTIFY_RUN_LOG_DIR  where to write the tee'd log (default: REPO_ROOT/runs)
#   NOTIFY_DISABLE=1    skip the email entirely

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

if [ "$#" -eq 0 ]; then
  echo "usage: bash scripts/notify-run.sh <command> [args...]"
  echo "e.g.:  nohup bash scripts/notify-run.sh bash scripts/race-revisit.sh --tag t &"
  exit 2
fi

# --- derive a tag: first --tag value in the args, else the script basename -
TAG=""
prev=""
for a in "$@"; do
  if [ "$prev" = "--tag" ]; then TAG="$a"; break; fi
  case "$a" in --tag=*) TAG="${a#--tag=}"; break ;; esac
  prev="$a"
done
if [ -z "$TAG" ]; then
  for a in "$@"; do
    case "$a" in
      bash|sh|python|python3|-m|nohup) continue ;;
      *) TAG="$(basename "$a")"; TAG="${TAG%.*}"; break ;;
    esac
  done
fi
TAG="${TAG:-run}"

LOG_DIR="${NOTIFY_RUN_LOG_DIR:-$REPO_ROOT/runs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/notify-${TAG}-$(date +%Y%m%d-%H%M%S).log"

START_TS=$(date +%s)
GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
CMD="$*"
EC=0

# --- EXIT trap: notify on normal exit, crash, and Ctrl-C/SIGTERM ----------
notify() {
  # `|| true`: the notifier must never change the wrapped exit code.
  python3 "$REPO_ROOT/scripts/notify_email.py" \
      --exit-code "$EC" \
      --log "$LOG" \
      --command "$CMD" \
      --start-ts "$START_TS" \
      --commit "$GIT_COMMIT" \
      --tag "$TAG" || true
}
trap notify EXIT
# Ctrl-C/SIGTERM: record the conventional code, then exit (fires the EXIT trap).
trap 'EC=130; exit 130' INT
trap 'EC=143; exit 143' TERM

echo "[notify-run] tag=$TAG commit=$GIT_COMMIT log=$LOG"
echo "[notify-run] running: $CMD"

"$@" 2>&1 | tee "$LOG"
EC=${PIPESTATUS[0]}

exit "$EC"
