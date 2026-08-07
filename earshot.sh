#!/bin/bash
# earshot.sh — the box operator menu: set the env up, see what nrun is running, stop it.
#
#   ./earshot.sh
#
# BOX ONLY. `bootstrap_ss2.sh` needs Linux + CUDA, and a running nrun task only exists on
# the machine it was launched on. On a Mac every item is either meaningless or actively
# wrong, so this refuses rather than half-works.
#
# TTY REQUIRED. A read-driven menu with stdin closed spins on EOF forever, which under
# `nrun` or CI is a background process burning a core until somebody notices. So it checks.
#
# It SOURCES earshot/tools/notify/notify-run.sh to get the real `nrun` rather than
# re-implementing its nohup/disown dance — that file is safe to source by construction
# (it returns early) and copying four lines of detachment logic is how the two drift.
#
# Discovery, the log lookup and the kill-safety rule live in `earshot/tools/nrun_tasks.py`,
# unit-tested on a Mac against captured `ps` output; this file reads its `--plan` output
# and sends the signals. Everything with a real failure mode is on the tested side.
#
# No env knobs (ADR-0008).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

NOTIFY_RUN="$REPO_ROOT/earshot/tools/notify/notify-run.sh"
LOG_DIR="$REPO_ROOT/runs"
PY="${PYTHON:-python3}"
TASKS=("$PY" -m earshot.tools.nrun_tasks --log-dir "$LOG_DIR")

# --- refusals, before anything else ---------------------------------------

if [ "$(uname -s)" != "Linux" ]; then
  echo "earshot.sh is box-only: it needs Linux \`ps\`, /proc, and a CUDA habitat-sim."
  echo "The Mac suite is:  conda activate earshot-mac && PYTHONPATH=. python -m unittest discover earshot/tests/mac"
  exit 2
fi

if [ ! -t 0 ]; then
  echo "earshot.sh is an interactive menu and stdin is not a terminal — refusing to run."
  echo "For a non-interactive listing:  python -m earshot.tools.nrun_tasks --json"
  exit 2
fi

if [ ! -f "$NOTIFY_RUN" ]; then
  echo "FATAL: no notify-run.sh at $NOTIFY_RUN — broken checkout."
  exit 1
fi
# shellcheck source=earshot/tools/notify/notify-run.sh
source "$NOTIFY_RUN"

pause() {
  echo
  read -r -p "  press enter to continue "
}

# --- env setup submenu -----------------------------------------------------

env_menu() {
  local choice
  while true; do
    cat <<'EOF'

  env setup
  ---------
  1) env_check --strict     seconds, read-only — "is this env fine?"
  2) bootstrap_ss2.sh       the ss2 rebuild, detached under nrun
  3) stage ESC-50 clips     into data/anomaly_audio, detached
  4) box_gate.sh            the box suite, detached
  b) back
EOF
    read -r -p "  > " choice
    case "$choice" in
      1)
        "$PY" -m earshot.env_check --strict
        echo "  [earshot] env_check exit $?"
        pause
        ;;
      2) nrun bash earshot/tools/bootstrap_ss2.sh; pause ;;
      3) nrun "$PY" -m earshot.audio.clips --out-dir data/anomaly_audio; pause ;;
      4) nrun bash earshot/tools/box_gate.sh; pause ;;
      b|B) return 0 ;;
      *) echo "  ? $choice" ;;
    esac
  done
}

# --- tasks -----------------------------------------------------------------

# Fills PLAN_* from `--plan N`. Returns 1 when there is no task N, so every caller can
# stop before printing a confirmation about a task that no longer exists.
PLAN_PID=""; PLAN_STARTED=""; PLAN_ELAPSED=""; PLAN_COMMAND=""
PLAN_LOG=""; PLAN_MODE=""; PLAN_TARGETS=""
load_plan() {
  local out key rest
  out="$("${TASKS[@]}" --plan "$1" 2>/dev/null)" || return 1
  PLAN_PID=""; PLAN_STARTED=""; PLAN_ELAPSED=""; PLAN_COMMAND=""
  PLAN_LOG=""; PLAN_MODE=""; PLAN_TARGETS=""
  while read -r key rest; do
    case "$key" in
      PID) PLAN_PID="$rest" ;;
      STARTED) PLAN_STARTED="$rest" ;;
      ELAPSED) PLAN_ELAPSED="$rest" ;;
      COMMAND) PLAN_COMMAND="$rest" ;;
      LOG) PLAN_LOG="$rest" ;;
      MODE) PLAN_MODE="$rest" ;;
      TARGETS) PLAN_TARGETS="$rest" ;;
    esac
  done <<< "$out"
  [ -n "$PLAN_PID" ]
}

tail_task() {
  if ! load_plan "$1"; then echo "  no task $1."; return 1; fi
  if [ -z "$PLAN_LOG" ] || [ ! -f "$PLAN_LOG" ]; then
    echo "  no log file for task $1 (its stdout is not a regular file)."
    return 0
  fi
  echo "  tailing $PLAN_LOG — Ctrl-C returns to the menu."
  # Without this trap the INT reaches the script and quits the whole menu, which is more
  # than "stop tailing" should cost.
  trap ' ' INT
  tail -f "$PLAN_LOG"
  trap - INT
  echo
}

send_signal() {
  local sig="$1" t
  for t in $PLAN_TARGETS; do
    if [ "$PLAN_MODE" = "group" ]; then
      kill "-$sig" -- "-$t" 2>/dev/null || true
    else
      kill "-$sig" "$t" 2>/dev/null || true
    fi
  done
}

# TERM the whole task, wait, then KILL what is left.
#
# TERM first is not politeness: notify-run.sh traps it, records exit 143 and fires its EXIT
# trap, which flushes the tee'd log and sends the run-report email. A KILL skips all of it.
# The escalation is what stops a wedged process from making this menu item a lie.
kill_task() {
  local assume_yes="${2:-}" ok i
  if ! load_plan "$1"; then echo "  no task $1."; return 1; fi

  echo
  echo "  about to stop:"
  echo "    pid      $PLAN_PID"
  echo "    started  $PLAN_STARTED"
  echo "    elapsed  $PLAN_ELAPSED"
  echo "    command  $PLAN_COMMAND"
  echo "    log      ${PLAN_LOG:-(stdout is not a file)}"
  if [ "$PLAN_MODE" = "group" ]; then
    echo "    kill     process group -$PLAN_TARGETS (the wrapper leads its own group)"
  else
    echo "    kill     pid tree: $PLAN_TARGETS"
    echo "             this task shares its caller's process group, so a group kill would"
    echo "             take that caller down too — walking the tree instead"
  fi

  if [ "$assume_yes" != "yes" ]; then
    read -r -p "  stop it? [y/N] " ok
    case "$ok" in y|Y) ;; *) echo "  left alone."; return 0 ;; esac
  fi

  send_signal TERM
  echo "  TERM sent — waiting up to 10s for the run report to be written..."
  for i in $(seq 1 10); do
    if ! kill -0 "$PLAN_PID" 2>/dev/null; then
      echo "  stopped cleanly after ${i}s."
      return 0
    fi
    sleep 1
  done
  echo "  still alive after 10s — sending KILL."
  send_signal KILL
  sleep 1
  if kill -0 "$PLAN_PID" 2>/dev/null; then
    echo "  WARNING: pid $PLAN_PID survived SIGKILL — uninterruptible sleep, likely stuck in"
    echo "           a CUDA or NFS call. It is NOT stopped.  ps -o stat= -p $PLAN_PID"
    return 1
  fi
  echo "  killed."
}

kill_all() {
  local n i confirm
  n="$("${TASKS[@]}" --count)"
  if [ "$n" = "0" ]; then echo "  nothing running."; return 0; fi
  echo
  "${TASKS[@]}"
  echo
  echo "  this stops ALL $n running nrun task(s) — possibly hours of box time."
  read -r -p "  type 'yes' in full to confirm: " confirm
  if [ "$confirm" != "yes" ]; then echo "  aborted."; return 0; fi
  # Descending, because each kill renumbers the list and ascending would skip tasks.
  for i in $(seq "$n" -1 1); do
    kill_task "$i" yes
  done
}

tasks_menu() {
  local choice arg
  while true; do
    echo
    "${TASKS[@]}"
    cat <<'EOF'

  t <n>  tail task n's log        k <n>  kill task n
  A      kill ALL tasks           r      refresh          b  back
EOF
    read -r -p "  > " choice arg
    case "$choice" in
      t|T) tail_task "${arg:-0}" ;;
      k) kill_task "${arg:-0}"; pause ;;
      A) kill_all; pause ;;
      r|R|"") ;;
      b|B) return 0 ;;
      *) echo "  ? $choice" ;;
    esac
  done
}

# --- main menu -------------------------------------------------------------

while true; do
  cat <<'EOF'

  earshot — box operator menu
  ===========================
  1) env setup
  2) nrun tasks — check / tail / kill
  q) quit
EOF
  read -r -p "  > " choice
  case "$choice" in
    1) env_menu ;;
    2) tasks_menu ;;
    q|Q) exit 0 ;;
    *) echo "  ? $choice" ;;
  esac
done
