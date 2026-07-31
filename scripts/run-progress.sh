#!/bin/bash
# scripts/run-progress.sh — $0 live progress + ETA for a long RACE run.
#
# Reads the per-episode JSONs the runner writes as it goes (episode_runner.py:978,
# one file per completed episode) and turns their mtimes into the three numbers
# that decide whether to wait or to kill:
#
#   * how many episodes are actually done, per arm;
#   * the REAL per-episode rate on THIS run's scenes (not extrapolated from a
#     val_mini smoke), both overall and over the trailing window;
#   * whether the run is still advancing or has stalled (age of the newest
#     episode file measured against the recent rate).
#
# The rate/ETA above come from file MTIMES, so they only hold on the box that
# wrote them (a copied or rsynced run dir carries copy times). The metrics block
# below re-derives wall-clock from each episode's own `finished_at`, which
# travels with the JSON — if the two disagree, trust `finished_at`.
#
# Then the partial metrics via progress_metrics.py — CONTEXT.md's headline set:
# Cost (steps + wall-clock), Benchmark SPL @0.1 m with SR, soft-SPL, Find-SR, and
# Anomaly-response SR, plus the reach@1m diagnostic labelled as NOT a success
# rate. Metrics absent on this task print n/a, never 0.0.
#
# No GPU and no conda (progress_metrics.py is stdlib-only and runs on a bare
# system python3; without one, the metrics block is skipped and progress still
# prints). Never writes to the run directories, so it is safe to run repeatedly
# while the job holds the card.
#
#   bash scripts/run-progress.sh --tag r1v1
#   bash scripts/run-progress.sh --tag r1v1 --total 2000 --arms 2
#   watch -n 600 bash scripts/run-progress.sh --tag r1v1
#
# --total defaults to the driver's own "n_episodes=<N>" banner in
# runs/<tag>-*.log, so the ETA is against what the run was actually asked for.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1

TAG=""
TOTAL=""            # episodes per arm; empty => read from the driver log
ARMS=""             # arms the driver runs; empty => 2 (the S1/S1+ A/B shape)
RECENT=10           # episodes in the trailing-rate window
STALE_MULT=6        # newest file older than this many recent-rates => STALLED?
LOG_LINES=3

while [ $# -gt 0 ]; do
  case "$1" in
    --tag)        TAG="$2"; shift 2 ;;
    --total)      TOTAL="$2"; shift 2 ;;
    --arms)       ARMS="$2"; shift 2 ;;
    --recent)     RECENT="$2"; shift 2 ;;
    --log-lines)  LOG_LINES="$2"; shift 2 ;;
    -h|--help)    sed -n '2,34p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown arg '$1' (see --help)"; exit 1 ;;
  esac
done
[ -z "$TAG" ] && { echo "FATAL: --tag <name> required (e.g. --tag r1v1)"; exit 1; }

# stat(1) is GNU on the VM and BSD on a mac laptop — pick the working spelling once.
if stat -c %Y . >/dev/null 2>&1; then STAT_FLAG="-c%Y"; else STAT_FLAG="-f%m"; fi

banner() { printf '\n========== %s ==========\n' "$1"; }

ARM_DIRS=$(find runs -maxdepth 1 -type d -name "${TAG}-*" 2>/dev/null | sort)
[ -z "$ARM_DIRS" ] && { echo "FATAL: no run dirs match runs/${TAG}-*"; exit 1; }
N_ARM_DIRS=$(printf '%s\n' "$ARM_DIRS" | grep -c .)

# Episodes-per-arm: prefer what the driver itself printed at its [4/7] banner.
if [ -z "$TOTAL" ]; then
  for lg in runs/"${TAG}"-*.log; do
    [ -f "$lg" ] || continue
    v=$(sed -n 's/.*n_episodes=\([0-9][0-9]*\).*/\1/p' "$lg" | head -1)
    [ -n "$v" ] && { TOTAL="$v"; break; }
  done
fi
[ -z "$TOTAL" ] && echo "NOTE: no 'n_episodes=' banner in runs/${TAG}-*.log — pass --total <N> for an ETA."
[ -z "$ARMS" ] && ARMS=2   # the R1 / A-B driver shape; override with --arms

NOW=$(date +%s)
# Explicit template: BSD mktemp with no template ignores TMPDIR and lands in
# /var/folders, which a sandboxed laptop shell cannot write.
TMP="$(mktemp -d "${TMPDIR:-/tmp}/runprog.XXXXXX" 2>/dev/null)"
[ -n "$TMP" ] && [ -d "$TMP" ] || { echo "FATAL: mktemp -d failed (TMPDIR=${TMPDIR:-unset} not writable?)"; exit 1; }
trap 'rm -rf "$TMP"' EXIT
: > "$TMP/left"; : > "$TMP/rate"

banner "arms (tag=$TAG, ${TOTAL:-?} episodes/arm)"
for d in $ARM_DIRS; do
  done_list=$(find "$d" -maxdepth 1 -type f -name 'episode_*.json' ! -name '*_error.json' 2>/dev/null)
  n_done=$(printf '%s\n' "$done_list" | grep -c .)
  n_err=$(find "$d" -maxdepth 1 -type f -name 'episode_*_error.json' 2>/dev/null | grep -c .)
  [ -f "$d/summary.json" ] && fin=" [summary.json written]" || fin=""

  printf '%-30s %5d done  %3d errored%s\n' "$d" "$n_done" "$n_err" "$fin"

  if [ "$n_done" -lt 2 ]; then
    echo "                               (need >=2 episodes for a rate)"
    [ -n "$TOTAL" ] && echo "$TOTAL" >> "$TMP/left"
    continue
  fi

  printf '%s\n' "$done_list" | tr '\n' '\0' \
    | xargs -0 -n 500 stat "$STAT_FLAG" 2>/dev/null | sort -n > "$TMP/mt"

  awk -v n="$n_done" -v total="${TOTAL:-0}" -v recent="$RECENT" -v now="$NOW" \
      -v stale_mult="$STALE_MULT" -v tmp="$TMP" '
    { t[NR] = $1 + 0 }
    END {
      span    = t[NR] - t[1]
      overall = span / 60 / (NR - 1)
      k       = (NR - 1 < recent ? NR - 1 : recent)
      rec     = (t[NR] - t[NR - k]) / 60 / k
      age     = (now - t[NR]) / 60

      printf "                               %.1f h elapsed | %.2f min/ep overall | %.2f min/ep last %d  (file mtimes)\n",
             span / 3600, overall, rec, k
      if (rec > 0 && age > stale_mult * rec)
        printf "                               STALLED? newest episode is %.0f min old (>%dx the recent rate)\n",
               age, stale_mult
      else
        printf "                               advancing (newest episode %.0f min old)\n", age

      if (total > 0) {
        left = total - n
        if (left <= 0) { print "                               arm complete"; left = 0 }
        else printf "                               %d left here => %.1f d at the recent rate\n",
                    left, left * rec / 1440
        print left >> (tmp "/left")
        printf "%.6f\n", rec >> (tmp "/rate")
      }
    }' "$TMP/mt"
done

# Whole-job ETA: what is left in the arms that exist, plus every arm not started.
if [ -n "$TOTAL" ]; then
  banner "whole job (assuming $ARMS arms; override with --arms)"
  RATE=$(tail -1 "$TMP/rate" 2>/dev/null)
  awk -v arms="$ARMS" -v seen="$N_ARM_DIRS" -v total="$TOTAL" -v rate="${RATE:-0}" '
    { left += $1 }
    END {
      unstarted = (arms > seen ? arms - seen : 0)
      left += unstarted * total
      if (rate + 0 <= 0) { printf "  %d episode-runs left (no rate yet)\n", left; exit }
      printf "  %d episode-runs left at %.2f min/ep => %.1f d (%.0f h)\n",
             left, rate, left * rate / 1440, left * rate / 60
      if (unstarted > 0)
        printf "  includes %d arm(s) not started yet, %d episodes each\n", unstarted, total
    }' "$TMP/left"
fi

banner "metrics so far (CONTEXT.md headline set)"
PY=""
for c in python3 python; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done
if [ -n "$PY" ]; then
  # progress_metrics.py is stdlib-only and is invoked as a FILE, so it never
  # imports embodied_memory/__init__.py (numpy/habitat) — no conda env needed.
  # shellcheck disable=SC2086
  "$PY" embodied_memory/scripts/progress_metrics.py $ARM_DIRS
else
  echo "  (no python3 on PATH — metrics block skipped)"
fi

banner "process"
if pgrep -af run_hm3d_pol >/dev/null 2>&1; then pgrep -af run_hm3d_pol | head -3
else echo "  no run_hm3d_pol process — the run is NOT active"; fi

banner "gpu"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
else echo "  (no nvidia-smi)"; fi

banner "log tail"
for lg in runs/"${TAG}"-*.log; do
  [ -f "$lg" ] || continue
  echo "--- $lg"; tail -n "$LOG_LINES" "$lg"
done
