#!/bin/bash
# scripts/notify-run.sh — run any command, then email the run report; AND the
# canonical home of the `nrun` self-detaching wrapper function.
#
# TWO usage modes:
#
#  1. SOURCE it to get `nrun` (SAFE in an interactive shell — it only defines the
#     function and returns; it NEVER exits or changes your shell options):
#
#       source scripts/notify-run.sh
#       nrun bash scripts/race-revisit.sh --tag wide-1   # self-detaches (nohup+bg)
#       tail -f runs/nrun-*.out
#
#     (`source scripts/race-setup.sh` also defines nrun — it sources this file.)
#
#  2. EXECUTE it to wrap a command in the FOREGROUND, tee its output to
#     runs/notify-<tag>-<ts>.log, and email a markdown report + gzipped log via
#     scripts/notify_email.py on EXIT (normal finish, crash, Ctrl-C/SIGTERM):
#
#       bash scripts/notify-run.sh bash scripts/race-revisit.sh --tag wide-1
#       nohup bash scripts/notify-run.sh bash scripts/race-revisit.sh --tag t &
#
# Config: RESEND_API_KEY / NOTIFY_EMAIL_TO in .env at the repo root (see
# .env.example). Unconfigured -> the run still works, just no email.
#
# Exit code (execute mode): ALWAYS the wrapped command's exit code — a notifier
# failure never changes it.
#
# Env knobs:
#   NOTIFY_RUN_LOG_DIR  where to write the tee'd log (default: REPO_ROOT/runs)
#   NOTIFY_DISABLE=1    skip the email entirely

# --- nrun: self-detaching, email-notified wrapper (SAFE to source) --------
# Defined BEFORE any `set`/`exit` so that sourcing this file only ever defines
# the function and returns — it can never kill the caller's interactive shell
# (the old version hit `exit 2` on a no-arg source and closed the session).
nrun() {
  local repo_root log_dir out
  repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  log_dir="${NOTIFY_RUN_LOG_DIR:-$repo_root/runs}"
  mkdir -p "$log_dir"
  out="$log_dir/nrun-$(date +%Y%m%d-%H%M%S).out"
  # nohup + background so the run survives an SSH disconnect; do NOT prefix
  # nohup yourself — nrun is a function and nohup cannot launch functions.
  nohup bash "$repo_root/scripts/notify-run.sh" "$@" > "$out" 2>&1 &
  disown 2>/dev/null || true
  echo "[nrun] detached (pid $!) — follow with: tail -f $out"
}

# When SOURCED, stop here: nrun is defined, and we must NOT set shell options or
# run/exit the wrapper in the caller's shell. ${BASH_SOURCE[0]} != $0 ⇔ sourced.
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
  return 0
fi

# --- EXECUTED below: foreground wrapper -----------------------------------
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
