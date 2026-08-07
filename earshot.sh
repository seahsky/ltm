#!/bin/bash
# earshot.sh — the box operator menu: set the env up, see what nrun is running, stop it.
#
#   source ./earshot.sh     # RECOMMENDED — env setup can then actually leave you in ss2
#   ./earshot.sh            # works, but nothing it does can change your shell
#
# BOX ONLY. `bootstrap_ss2.sh` needs Linux + CUDA, and a running nrun task only exists on
# the machine it was launched on. On a Mac every item is either meaningless or actively
# wrong, so this refuses rather than half-works.
#
# TTY REQUIRED. A read-driven menu with stdin closed spins on EOF forever, which under
# `nrun` or CI is a background process burning a core until somebody notices.
#
# --- why env setup runs in the FOREGROUND, and why sourcing matters --------
#
# The first cut ran bootstrap detached under `nrun`. It built the env fine and left the
# operator in `base`, which is the wrong end state: "set up the env" means being able to
# use it, not owning a directory under ~/miniconda3. Detachment could never deliver that
# — a detached process has no shell to hand back — so the env items are plain foreground
# bash now, and `nrun` is kept for the long unattended jobs it was written for.
#
# Foreground is not sufficient either. A child process cannot change its parent's
# environment; only `source` crosses that boundary. So this file is written to be SAFE TO
# SOURCE, the way notify-run.sh is: no `exit` anywhere (a sourced `exit` closes your
# session — the exact bug notify-run.sh's own comments record paying for), no `set` that
# leaks into your shell, everything through `return`. Sourced, option 1 activates ss2 in
# YOUR shell and it stays activated after you quit the menu. Executed, it prints the
# `conda activate` line instead and says why it cannot run it for you.
#
# Discovery, the log lookup and the kill-safety rule live in `earshot/tools/nrun_tasks.py`,
# unit-tested on a Mac against captured `ps` output; this file reads its `--plan` output
# and sends the signals. Everything with a real failure mode is on the tested side.
#
# No env knobs (ADR-0008).

# ${BASH_SOURCE[0]} != $0 ⇔ sourced. Everything below branches on this, and nothing below
# calls `exit`.
_earshot_sourced() { [ "${BASH_SOURCE[0]}" != "${0}" ]; }

# Shell options are NOT set when sourced: `set -u` in an interactive shell turns a typo'd
# variable into a closed session. Executed, they are worth having.
_earshot_sourced || set -uo pipefail

EARSHOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EARSHOT_NOTIFY_RUN="$EARSHOT_ROOT/earshot/tools/notify/notify-run.sh"
EARSHOT_LOG_DIR="$EARSHOT_ROOT/runs"
EARSHOT_MINICONDA="${HOME}/miniconda3"
EARSHOT_SS2_ENV="ss2"

# The task scan is deliberately NOT run inside ss2: it reads `ps` and `/proc` with the
# standard library only, and requiring the env would make "what is running?" unanswerable
# on exactly the broken-env day you most need to ask it.
EARSHOT_PY="${PYTHON:-python3}"

_earshot_tasks() {
  (cd "$EARSHOT_ROOT" && "$EARSHOT_PY" -m earshot.tools.nrun_tasks \
      --log-dir "$EARSHOT_LOG_DIR" "$@")
}

_earshot_pause() {
  echo
  read -r -p "  press enter to continue "
}

# --- refusals --------------------------------------------------------------

_earshot_refuse() {
  if [ "$(uname -s)" != "Linux" ]; then
    echo "earshot.sh is box-only: it needs Linux \`ps\`, /proc, and a CUDA habitat-sim."
    echo "The Mac suite is:  conda activate earshot-mac && PYTHONPATH=. python -m unittest discover earshot/tests/mac"
    return 2
  fi
  if [ ! -t 0 ]; then
    echo "earshot.sh is an interactive menu and stdin is not a terminal — refusing."
    echo "For a non-interactive listing:  python -m earshot.tools.nrun_tasks --json"
    return 2
  fi
  if [ ! -x "$EARSHOT_MINICONDA/bin/conda" ]; then
    echo "FATAL: no conda at $EARSHOT_MINICONDA/bin/conda — every env item needs it."
    echo "See docs/race-box-runbook.md for how the box is provisioned."
    return 1
  fi
  if [ ! -f "$EARSHOT_NOTIFY_RUN" ]; then
    echo "FATAL: no notify-run.sh at $EARSHOT_NOTIFY_RUN — broken checkout."
    return 1
  fi
  return 0
}

# --- env setup -------------------------------------------------------------

_earshot_ss2_exists() { [ -d "$EARSHOT_MINICONDA/envs/$EARSHOT_SS2_ENV" ]; }

# Bring conda's shell functions into THIS shell. Sourced, "this shell" is yours, which is
# what makes activation stick. `set +u` around it is carried from bootstrap_ss2.sh: conda's
# compiler hooks dereference unset CONDA_BACKUP_* vars and would kill the script under -u.
_earshot_conda_hook() {
  if ! command -v conda >/dev/null 2>&1 || [ -z "${CONDA_SHLVL:-}" ]; then
    set +u
    eval "$("$EARSHOT_MINICONDA/bin/conda" shell.bash hook)" || return 1
    _earshot_sourced || set -u
  fi
  return 0
}

_earshot_activate() {
  if ! _earshot_ss2_exists; then
    echo "  no $EARSHOT_SS2_ENV env yet — build it with option 2 first."
    return 1
  fi
  _earshot_conda_hook || { echo "  conda hook failed."; return 1; }
  set +u
  conda activate "$EARSHOT_SS2_ENV"
  local rc=$?
  _earshot_sourced || set -u
  if [ $rc -ne 0 ]; then echo "  conda activate $EARSHOT_SS2_ENV failed."; return 1; fi
  echo "  active: $(python -V 2>&1) at $(command -v python)"
  if _earshot_sourced; then
    echo "  this shell is now in $EARSHOT_SS2_ENV, and stays there after you quit."
  else
    echo
    echo "  ...but only inside this script. It was EXECUTED, not sourced, and a child"
    echo "  process cannot activate an env in your shell. To get it in yours:"
    echo "      conda activate $EARSHOT_SS2_ENV"
    echo "  or re-enter this menu with:   source ./earshot.sh"
  fi
  return 0
}

# Run something inside ss2 in the FOREGROUND, from the repo root, with output live on the
# terminal. Activating first (rather than `conda run`) keeps one activation path in this
# file, and it means the sourced case leaves the caller in the env afterwards, which is
# the whole point of option 1.
_earshot_in_ss2() {
  _earshot_activate >/dev/null || { _earshot_activate; return 1; }
  (cd "$EARSHOT_ROOT" && "$@")
}

_earshot_env_menu() {
  local choice rc
  while true; do
    echo
    if _earshot_ss2_exists; then
      echo "  env setup — ss2 present; active env: ${CONDA_DEFAULT_ENV:-<none>}"
    else
      echo "  env setup — ss2 DOES NOT EXIST yet (option 2 builds it)"
    fi
    cat <<'EOF'
  ---------------------------------------------------------------
  1) activate ss2           and report which python you just got
  2) bootstrap_ss2.sh       build/repair the env — FOREGROUND, long
  3) env_check --strict     seconds, read-only — "is ss2 fine?"
  4) stage ESC-50 clips     into data/anomaly_audio
  5) box_gate.sh            the box suite, a few minutes
  b) back
EOF
    read -r -p "  > " choice
    case "$choice" in
      1) _earshot_activate; _earshot_pause ;;
      2)
        # NOT inside ss2: bootstrap creates the env and does its own activation, so
        # running it in the env it is about to build is circular.
        echo "  running in the foreground — this is long, and it dies if your SSH drops."
        echo "  in a fresh session, run it under tmux, or use nrun for an unattended build:"
        echo "      source earshot/tools/notify/notify-run.sh && nrun bash earshot/tools/bootstrap_ss2.sh"
        read -r -p "  start it here? [y/N] " choice
        case "$choice" in
          y|Y)
            (cd "$EARSHOT_ROOT" && bash earshot/tools/bootstrap_ss2.sh)
            rc=$?
            echo "  [earshot] bootstrap exit $rc"
            # Only offer the env once the build says it is good; activating a red env is
            # how you end up debugging the wrong python.
            if [ $rc -eq 0 ]; then _earshot_activate; fi
            ;;
          *) echo "  not started." ;;
        esac
        _earshot_pause
        ;;
      3)
        _earshot_in_ss2 python -m earshot.env_check --strict
        echo "  [earshot] env_check exit $?"
        _earshot_pause
        ;;
      4)
        _earshot_in_ss2 python -m earshot.audio.clips --out-dir data/anomaly_audio
        echo "  [earshot] clips exit $?"
        _earshot_pause
        ;;
      5)
        # box_gate.sh activates ss2 itself, so it runs as plain bash.
        (cd "$EARSHOT_ROOT" && bash earshot/tools/box_gate.sh)
        echo "  [earshot] box_gate exit $?"
        _earshot_pause
        ;;
      b|B) return 0 ;;
      *) echo "  ? $choice" ;;
    esac
  done
}

# --- tasks -----------------------------------------------------------------

# Filled from `--plan N`. Returns 1 when there is no task N, so no caller prints a
# confirmation about a task that no longer exists.
_earshot_load_plan() {
  local out key rest
  out="$(_earshot_tasks --plan "$1" 2>/dev/null)" || return 1
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

_earshot_tail_task() {
  if ! _earshot_load_plan "$1"; then echo "  no task $1."; return 1; fi
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

_earshot_signal() {
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
# trap, which flushes the tee'd log and sends the run report. A KILL skips all of it. The
# escalation is what stops a wedged process from making this menu item a lie.
_earshot_kill_task() {
  local assume_yes="${2:-}" ok i
  if ! _earshot_load_plan "$1"; then echo "  no task $1."; return 1; fi

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

  _earshot_signal TERM
  echo "  TERM sent — waiting up to 10s for the run report to be written..."
  for i in $(seq 1 10); do
    if ! kill -0 "$PLAN_PID" 2>/dev/null; then
      echo "  stopped cleanly after ${i}s."
      return 0
    fi
    sleep 1
  done
  echo "  still alive after 10s — sending KILL."
  _earshot_signal KILL
  sleep 1
  if kill -0 "$PLAN_PID" 2>/dev/null; then
    echo "  WARNING: pid $PLAN_PID survived SIGKILL — uninterruptible sleep, likely stuck"
    echo "           in a CUDA or NFS call. It is NOT stopped.  ps -o stat= -p $PLAN_PID"
    return 1
  fi
  echo "  killed."
}

_earshot_kill_all() {
  local n i confirm
  n="$(_earshot_tasks --count)"
  if [ "$n" = "0" ]; then echo "  nothing running."; return 0; fi
  echo
  _earshot_tasks
  echo
  echo "  this stops ALL $n running nrun task(s) — possibly hours of box time."
  read -r -p "  type 'yes' in full to confirm: " confirm
  if [ "$confirm" != "yes" ]; then echo "  aborted."; return 0; fi
  # Descending, because each kill renumbers the list and ascending would skip tasks.
  for i in $(seq "$n" -1 1); do
    _earshot_kill_task "$i" yes
  done
}

_earshot_tasks_menu() {
  local choice arg
  while true; do
    echo
    _earshot_tasks
    cat <<'EOF'

  t <n>  tail task n's log        k <n>  kill task n
  A      kill ALL tasks           r      refresh          b  back
EOF
    read -r -p "  > " choice arg
    case "$choice" in
      t|T) _earshot_tail_task "${arg:-0}" ;;
      k) _earshot_kill_task "${arg:-0}"; _earshot_pause ;;
      A) _earshot_kill_all; _earshot_pause ;;
      r|R|"") ;;
      b|B) return 0 ;;
      *) echo "  ? $choice" ;;
    esac
  done
}

# --- main ------------------------------------------------------------------

_earshot_main() {
  local choice
  _earshot_refuse || return $?

  # Sourced, this also puts the real `nrun` in the caller's shell — which is the tool for
  # the long unattended jobs this menu deliberately does not detach for you.
  # shellcheck source=earshot/tools/notify/notify-run.sh
  source "$EARSHOT_NOTIFY_RUN"

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
      1) _earshot_env_menu ;;
      2) _earshot_tasks_menu ;;
      q|Q) return 0 ;;
      *) echo "  ? $choice" ;;
    esac
  done
}

if _earshot_sourced; then
  _earshot_main
  # No `exit`: this is your shell. Whatever option 1 activated stays activated, and `nrun`
  # is now defined here too.
else
  _earshot_main
  exit $?
fi
