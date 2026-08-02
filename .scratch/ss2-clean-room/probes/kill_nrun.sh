#!/bin/bash
# Ticket 15 — stop every `nrun`-launched job and free the GPU.
#
# Dry run by default. Add --yes to actually signal.
#
#   bash .scratch/ss2-clean-room/probes/kill_nrun.sh            # show the plan
#   bash .scratch/ss2-clean-room/probes/kill_nrun.sh --yes      # do it
#
# WHY THE ORDER MATTERS. `nrun` builds this tree:
#
#   nohup bash scripts/notify-run.sh <cmd...>     <- wrapper, traps TERM
#     |__ <cmd...> 2>&1 | tee <log>               <- workload, plus a tee
#
# The wrapper is `"$@" | tee "$LOG"` under `trap notify EXIT`. Signal the wrapper
# first and bash exits while the workload keeps running, orphaned and still
# holding the GPU — the opposite of the intent. So this signals the WORKLOAD
# first: the pipeline ends, the wrapper falls through to its EXIT trap, and the
# run-report email goes out with a complete log. `notify_email.py` is protected
# for the same reason, since it is spawned during that teardown.
#
# Nothing is SIGKILLed automatically. A hard kill on a process holding a CUDA
# context can leave the device in a worse state than the job did.
#
# Deliberately POSIX-ish (no associative arrays): the tree walk is an awk
# fixpoint so this runs and can be tested on bash 3.2 as well as the box's 5.x.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIRM=0
GRACE=30
PROTECT_DEFAULT='vram_probe|rendercost|oneenv_gate|audioguard|kill_nrun|notify_email'
PROTECT="$PROTECT_DEFAULT"
PSFILE=""
FAKE_PS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --yes) CONFIRM=1 ;;
    --grace) GRACE="$2"; shift ;;
    --no-protect) PROTECT='kill_nrun|notify_email' ;;
    --fake-ps) FAKE_PS="$2"; shift ;;   # test hook: read the table from a file
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

cleanup() { [ -n "$PSFILE" ] && rm -f "$PSFILE"; }
trap cleanup EXIT

SELF=$$
RECORD="$REPO_ROOT/runs/ss2-vram/killed-$(date +%Y%m%d-%H%M%S).txt"
mkdir -p "$(dirname "$RECORD")" 2>/dev/null || RECORD=/dev/null

# --- one process-table snapshot, so the tree cannot shift under the walk ----
PSFILE="$(mktemp "${TMPDIR:-/tmp}/kill_nrun.XXXXXX")"
if [ -n "$FAKE_PS" ]; then
  cat "$FAKE_PS" > "$PSFILE"
else
  ps -eo pid=,ppid=,etime=,pcpu=,rss=,args= > "$PSFILE" 2>/dev/null
fi
if [ ! -s "$PSFILE" ]; then
  echo "FATAL: could not read the process table" >&2
  exit 1
fi

field() {  # field <pid> <n>   (n: 3=etime 4=pcpu 5=rss)
  awk -v p="$1" -v n="$2" '$1==p {print $n; exit}' "$PSFILE"
}
argsof() {
  awk -v p="$1" '$1==p {for(i=6;i<=NF;i++) printf "%s%s", $i, (i<NF?" ":""); exit}' "$PSFILE"
}
ppidof() {
  awk -v p="$1" '$1==p {print $2; exit}' "$PSFILE"
}

# --- wrappers: processes whose cmdline mentions notify-run.sh ---------------
WRAPPERS="$(awk -v self="$SELF" '
  /notify-run\.sh/ && $1 != self && !/kill_nrun/ { print $1 }
' "$PSFILE")"

# --- descendants: awk fixpoint over (pid, ppid) -----------------------------
descendants() {
  local roots="$1"
  [ -z "$roots" ] && return 0
  awk -v roots="$roots" '
    BEGIN { n=split(roots, r, /[ \n]+/); for (i=1;i<=n;i++) if (r[i]!="") sel[r[i]]=1 }
    { pid[NR]=$1; par[NR]=$2; total=NR }
    END {
      changed=1
      while (changed) {
        changed=0
        for (i=1;i<=total;i++)
          if (!(pid[i] in sel) && (par[i] in sel)) { sel[pid[i]]=1; changed=1 }
      }
      # roots themselves are wrappers, not workload
      m=split(roots, r2, /[ \n]+/); for (i=1;i<=m;i++) if (r2[i]!="") delete sel[r2[i]]
      for (k in sel) print k
    }
  ' "$PSFILE"
}

WORKLOAD=""
SKIPPED=""
for d in $(descendants "$WRAPPERS"); do
  a="$(argsof "$d")"
  case "$a" in
    tee\ *|*/tee\ *) SKIPPED="$SKIPPED$d|tee, dies with the pipeline"$'\n'; continue ;;
  esac
  if printf '%s' "$a" | grep -Eq "$PROTECT"; then
    SKIPPED="$SKIPPED$d|protected: $a"$'\n'; continue
  fi
  WORKLOAD="$WORKLOAD$d"$'\n'
done

# --- orphans: known drivers with no live nrun wrapper above them ------------
ORPHANS=""
for pid in $(awk '/run_hm3d_pol|race-[a-z0-9-]*\.sh|run_msc_/ {print $1}' "$PSFILE"); do
  [ "$pid" = "$SELF" ] && continue
  a="$(argsof "$pid")"
  printf '%s' "$a" | grep -Eq "$PROTECT" && continue
  printf '%s\n' "$WORKLOAD" | grep -qx "$pid" && continue
  # A wrapper's own cmdline repeats the command it wraps, so it matches these
  # patterns too. It is already handled as a wrapper and must not be signalled.
  printf '%s\n' "$WRAPPERS" | grep -qx "$pid" && continue
  ORPHANS="$ORPHANS$pid"$'\n'
done

nonempty() { printf '%s' "$1" | grep -c . 2>/dev/null || true; }

show() {
  printf '  %-8s %-8s %-12s %6s  %9s KB  %s\n' \
    "$1" "$(ppidof "$1")" "$(field "$1" 3)" "$(field "$1" 4)" "$(field "$1" 5)" \
    "$(argsof "$1" | cut -c1-105)"
}

{
echo "=== nrun teardown ==="
echo "repo root : $REPO_ROOT"
echo "protected : $PROTECT"
echo
echo "--- wrappers ($(nonempty "$WRAPPERS")) — NOT signalled; they exit via their own EXIT trap"
printf '  %-8s %-8s %-12s %6s  %12s  %s\n' PID PPID ELAPSED CPU RSS COMMAND
for p in $WRAPPERS; do show "$p"; done

echo
echo "--- workload to SIGTERM ($(nonempty "$WORKLOAD"))"
printf '  %-8s %-8s %-12s %6s  %12s  %s\n' PID PPID ELAPSED CPU RSS COMMAND
for p in $WORKLOAD; do show "$p"; done

if [ -n "$(printf '%s' "$ORPHANS" | tr -d '[:space:]')" ]; then
  echo
  echo "--- orphaned drivers, no live nrun wrapper ($(nonempty "$ORPHANS")) — ALSO signalled"
  for p in $ORPHANS; do show "$p"; done
fi

if [ -n "$(printf '%s' "$SKIPPED" | tr -d '[:space:]')" ]; then
  echo
  echo "--- skipped"
  printf '%s' "$SKIPPED" | while IFS='|' read -r p why; do
    [ -n "$p" ] && echo "  $p  $why" | cut -c1-120
  done
fi

echo
echo "--- output directories these runs were writing to"
for p in $WORKLOAD $ORPHANS; do
  d="$(argsof "$p" | grep -o -- '--out-dir[= ][^ ]*' | head -1 | sed 's/--out-dir[= ]//')"
  [ -z "$d" ] && continue
  full="$REPO_ROOT/$d"; [ -d "$d" ] && full="$d"
  n="$(find "$full" -type f 2>/dev/null | wc -l | tr -d ' ')"
  echo "  pid $p -> $d  ($n files, newest: $(ls -t "$full" 2>/dev/null | head -1))"
done
} | tee "$RECORD"

TARGETS="$(printf '%s\n%s\n' "$WORKLOAD" "$ORPHANS" | grep -E '^[0-9]+$' | sort -u)"
NTARGETS="$(nonempty "$TARGETS")"

echo
if [ "$NTARGETS" -eq 0 ]; then
  echo "Nothing to signal."
  exit 0
fi

if [ "$CONFIRM" != 1 ]; then
  echo "DRY RUN — $NTARGETS process(es) would be signalled."
  echo "Re-run with --yes to do it. Plan recorded at $RECORD"
  exit 0
fi

echo "Signalling $NTARGETS process(es) with SIGTERM..."
for p in $TARGETS; do
  if kill -TERM "$p" 2>/dev/null; then echo "  TERM -> $p"
  else echo "  TERM -> $p FAILED (already gone, or not ours)"; fi
done

echo "Waiting up to ${GRACE}s for wrappers to reap and email their reports..."
deadline=$(( $(date +%s) + GRACE ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  alive=0
  for p in $TARGETS; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
  [ "$alive" = 0 ] && break
  sleep 2
done

echo
SURVIVORS=""
for p in $TARGETS; do kill -0 "$p" 2>/dev/null && SURVIVORS="$SURVIVORS $p"; done
if [ -n "$(printf '%s' "$SURVIVORS" | tr -d '[:space:]')" ]; then
  echo "STILL ALIVE after ${GRACE}s:$SURVIVORS"
  echo "Not escalating to SIGKILL automatically — a hard kill on a process holding"
  echo "a CUDA context can leave the device worse than the job did. To force:"
  echo "  kill -9$SURVIVORS"
else
  echo "All targets exited."
fi

echo
echo "--- GPU after"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv 2>/dev/null \
  || echo "  nvidia-smi unavailable"

echo
echo "Record: $RECORD"
